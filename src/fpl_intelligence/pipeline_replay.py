"""End-to-end historical execution of the production forecast and multiweek MIP.

This is a later reconstruction with deadline-pinned markets, not a fabricated
prospective track record. Live account/model/forecast/plan pointers are read-only.
Only the first action of each full plan is executed before the next observation.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

from .availability import moment, train_availability
from .consensus import paired_interval, replay_packets
from .decision import canonical_rows, learned_forecast, paired_bootstrap, score_metrics
from .decision_training import reprice
from .discipline import build_discipline
from .domain import Action, Lot, TeamState
from .objectives import SELECTION_VERSION
from .observations import completed_before
from .optimization import optimize, select_squad
from .replay import deadline_rows, fit_deadline
from .replay_benchmarks import baseline_signals
from .rules import Rules, score_gameweek, transition, validate_squad
from .scenarios import action_from_row
from .storage import Store, atomic_json, digest, now
from .strategy import solve_action


def link_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        try:
            target.hardlink_to(source.resolve())
        except OSError:
            shutil.copy2(source, target)


def attach_observations(store, archive, packet):
    cutoff = packet["deadline"]
    group = f"{packet['season']}-GW{packet['event']:02d}"
    store.put(
        "replay_context",
        {"information_cutoff": cutoff, "group": group, "track": "historical_statistical_reconstruction"},
    )
    target = store.root / "features" / archive["path"].name
    link_file(archive["path"], target)
    store.put(
        "observation_frame",
        {"path": str(target), "sha256": archive["sha256"], "cutoff": cutoff, "group": group},
    )


def attach_market(store, packet, originals):
    cutoff = packet["deadline"]
    group = f"{packet['season']}-GW{packet['event']:02d}"
    completed = {(r["season"], int(r["fixture"])) for r in completed_before(originals, cutoff, group)}
    fixtures = [
        {**f, "finished": (packet["season"], f["id"]) in completed}
        for f in packet["fixture_proof"]["fixtures"]
    ]
    players, _ = deadline_rows([], packet["bootstrap"], fixtures, packet["season"], packet["event"], cutoff)
    snapshot = store.snapshot(
        "bootstrap",
        "verified-deadline-replay://" + packet["id"],
        packet["bootstrap"],
        packet["bootstrap_proof"]["source_captured_at"],
    )
    market = {
        "season": packet["season"],
        "players": [p.model_dump() for p in players.values()],
        "teams": packet["bootstrap"]["teams"],
        "events": packet["bootstrap"]["events"],
        "fixtures": fixtures,
        "source_hash": snapshot["hash"],
        "fixture_hash": digest(fixtures),
        "known_at": cutoff,
        "replay_input_id": packet["id"],
    }
    identity = store.put("market", market)
    return players, market | {"id": identity}


def past_frame(frame, observations, group):
    completed = {(r["season"], int(r["fixture"])) for r in observations}
    eligible = [
        (r["season"], r["fixture_id"]) in completed for r in frame.select("season", "fixture_id").to_dicts()
    ]
    return frame.filter(pl.Series(eligible)).filter(pl.col("deadline_group") < group)


def install_feature_view(store, frame, parent):
    path = store.root / "features" / "decision-past.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)
    feature = {
        "path": str(path),
        "hash": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_feature_id": parent["id"],
        "seasons": sorted(frame["season"].unique().to_list()),
        "rows": len(frame),
    }
    return feature | {"id": store.put("decision_features", feature)}


def prior_gate(model, group):
    rows = [r for r in model["predictions"] if r["deadline_group"] < group]
    if not rows:
        raise ValueError("No strictly earlier out-of-fold reference acceptance evidence")
    ci = paired_bootstrap(rows, "predicted")
    regular = [r for r in rows if r["prior_start"] >= 0.6]
    actual = np.array([r["actual"] for r in regular])
    reference = score_metrics(actual, np.array([r["predicted"] for r in regular]))["rmse"]
    baseline = score_metrics(actual, np.array([r["baseline"] for r in regular]))["rmse"]
    return {
        "through": max(r["deadline_group"] for r in rows),
        "rows": len(rows),
        "paired_rmse_95ci": ci,
        "regular_reference_rmse": reference,
        "regular_baseline_rmse": baseline,
        "passed": bool(ci[1] < 0 and reference < baseline),
    }


def availability_epoch(parent, root, packet, archive, originals, frame, source_feature):
    epoch = Store(root)
    attach_observations(epoch, archive, packet)
    attach_market(epoch, packet, originals)
    group = epoch.get("replay_context")["group"]
    observed = completed_before(originals, packet["deadline"], group)
    install_feature_view(epoch, past_frame(frame, observed, group), source_feature)
    context = parent.get("context_archive")
    entries = [r for r in context["entries"] if moment(r["deadline"]) < moment(packet["deadline"])]
    for row in entries:
        body, meta = parent.latest(row["source"], content_hash=row["hash"])
        epoch.snapshot(row["source"], meta["url"], body, row["source_captured_at"])
    epoch.put(
        "context_archive", {"entries": entries, "source_id": context["id"], "cutoff": packet["deadline"]}
    )
    if epoch.list("availability_model", 1):
        return epoch, epoch.get("availability_model")
    result = train_availability(epoch)
    return epoch, result


def attach_availability(target, epoch, spec):
    for model in spec["models"].values():
        original = epoch.artifact_path(model["path"])
        relative = original.relative_to(epoch.root)
        link_file(original, target.root / relative)
    target.put("availability_model", {k: v for k, v in spec.items() if k != "id"})


def attach_dated_sources(parent, target, cutoff):
    counts = {}
    for kind in ("ban_decision", "ban_window", "official_cautions"):
        count = 0
        for item in parent.list(kind, 2000):
            record = parent.get(kind, item["id"])
            if (
                record.get("known_at")
                and moment(record["known_at"]) <= moment(cutoff)
                and moment(item["created_at"]) <= moment(cutoff)
            ):
                target.put(kind, {k: v for k, v in record.items() if k != "id"})
                count += 1
        counts[kind] = count
    return counts


def fit_reference(store, source_frame, source_feature, deadline, group, gate):
    observations, _, _ = canonical_rows(store)
    frame = past_frame(source_frame, observations, group)
    feature = install_feature_view(store, frame, source_feature)
    fitted, pools, dependence, minutes, evidence = fit_deadline(
        store, frame, deadline, group, store.root / "models" / "primary-reference"
    )
    spec = {
        "promoted": gate["passed"],
        "prior_acceptance": gate,
        "version": "learned-points-v1",
        "directory": "primary-reference",
        "model_hashes": evidence["model_hashes"],
        "curves": fitted["curves"],
        "feature_id": feature["id"],
        "seasons": feature["seasons"],
        "residuals": pools,
        "dependence": dependence,
        "minute_quantiles": {"0": [0.0], **minutes},
        "production_fit_before": sorted(frame["deadline_group"].unique().to_list())[-8],
        "fit_evidence": evidence,
    }
    spec["id"] = store.put("decision_model", spec)
    return spec


def initial_account(packet, players, rules, seconds):
    _, rows = deadline_rows(
        [],
        packet["bootstrap"],
        packet["fixture_proof"]["fixtures"],
        packet["season"],
        packet["event"],
        packet["deadline"],
    )
    signals = baseline_signals(players, "ownership", {r["player_id"] for r in rows})
    diagnostics: dict = {}
    squad = select_squad(
        None,
        players,
        rules,
        signals,
        seconds=seconds,
        initial_budget=1000,
        bench_weight=0,
        relative_gap=0,
        diagnostics=diagnostics,
    )
    state = TeamState(
        season=packet["season"],
        event=packet["event"],
        bank=1000 - sum(players[p].price for p in squad),
        free_transfers=1,
        initial=packet["event"] == 1,
        status="dated_seed",
        known_at=packet["deadline"],
        lots=[
            Lot(
                player_id=p,
                purchase_price=players[p].price,
                sell_price=players[p].price,
                quote_market_price=players[p].price,
            )
            for p in squad
        ],
    )
    return state, diagnostics


def verify_pipeline_replay(parent, report):
    """Recheck sealed inputs, model files, money, FT/chips and actual scores."""
    originals, _, _ = canonical_rows(parent)
    packets = replay_packets(parent)
    checked = {}
    for row in report["weeks"]:
        key = row["season"], row["event"]
        if key not in checked:
            path = parent.artifact_path(row["decision_file"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["decision_sha256"]:
                raise ValueError("Frozen pipeline decision checksum mismatch")
            body = json.loads(path.read_text())
            packet = packets[key]
            if body["input_id"] != row["input_id"] or body["information_cutoff"] != packet["deadline"]:
                raise ValueError("Replay decision uses a different deadline input")
            training = body["model"]["fit_evidence"]
            group = f"{key[0]}-GW{key[1]:02d}"
            if not training["fit_max_group"] < training["calibration_max_group"] < group:
                raise ValueError("Replay training reaches its target deadline")
            for artifact in body["model_artifacts"].values():
                if (
                    hashlib.sha256(parent.artifact_path(artifact["path"]).read_bytes()).hexdigest()
                    != artifact["hash"]
                ):
                    raise ValueError("Frozen replay model checksum mismatch")
            players, _ = deadline_rows(
                [],
                packet["bootstrap"],
                packet["fixture_proof"]["fixtures"],
                key[0],
                key[1],
                packet["deadline"],
            )
            rules = Rules.from_bootstrap(packet["bootstrap"])
            actual: dict = defaultdict(float)
            minutes: dict = defaultdict(int)
            for original in originals:
                if (original["season"], int(original["round"])) == key:
                    actual[int(original["element"])] += float(original.get("total_points") or 0)
                    minutes[int(original["element"])] += int(original.get("minutes") or 0)
            checked[key] = body, players, rules, actual, minutes
        body, players, rules, actual, minutes = checked[key]
        before = TeamState.model_validate(body["states_before"][row["policy"]])
        action = Action.model_validate(row["action"])
        assert action.lineup is not None
        following, ledger = transition(before, action, players, rules)
        for field in ("hit", "bank_after", "free_transfers_after", "playing_squad", "permanent_squad"):
            if ledger[field] != row["ledger"][field]:
                raise ValueError("Independent replay ledger disagrees: " + field)
        if (
            score_gameweek(actual, minutes, action.lineup, players, rules, action.chip) - ledger["hit"]
            != row["net_points"]
        ):
            raise ValueError("Independent actual score disagrees")
        if row["forecast_cutoff"] != body["information_cutoff"]:
            raise ValueError("Forecast and observations use different cutoffs")
    from .pipeline_verification import audit_continuity

    return {
        "sealed_deadlines": len(checked),
        "validated_actions": len(report["weeks"]),
        "models": "checksums passed",
        "ledger_and_actual_scoring": "passed",
        **audit_continuity(parent, report),
    }


def run_pipeline_replay(
    parent: Store,
    *,
    worlds=4000,
    seconds=60,
    periods=None,
    resume_protocol_id=None,
    reference_family_only=False,
):
    """Execute the full production primary path, never a hindsight-guided policy."""
    if not 100 <= worlds <= 20000 or not 1 <= seconds <= 300:
        raise ValueError("Invalid replay budget")
    active = parent.get("decision_model")
    if active.get("fusion_model_id") or (active.get("future_horizon_model_id") and not reference_family_only):
        raise ValueError("This reference-family replay cannot certify a different promoted model family")
    live_before = {
        k: parent.get(k)["id"] for k in ("state", "market", "decision_model", "forecast", "selected_plan")
    }
    packets = replay_packets(parent)
    season = parent.get("market")["season"]
    last = max((e["id"] for e in parent.get("market")["events"] if e.get("data_checked")), default=0)
    periods = periods or [(season, list(range(1, last + 1))), ("2025/26", list(range(20, 39)))]
    protocol = {
        "version": "integrated-primary-v1",
        "policy_selection_version": SELECTION_VERSION,
        "worlds": worlds,
        "seconds": seconds,
        "seed": 11,
        "objective": "expected_points",
        "horizon": "Complete chip window (GW19/GW38), plus four terminal weeks; never the end of the scored interval",
        "execution": "Production learned_forecast → full-market multiweek optimize → independent transition/scoring → next observation",
        "models": "Reference refit each deadline on completed earlier labels; recovery and joint-DGW gates frozen before each evaluation period",
        "prices": "Observed next-deadline prices update lots; future prices stay fixed as in the production primary forecast",
        "initial": "Identical £100m ownership seed and chip inventory; GW1 permits unlimited initial selection and grants one FT for GW2",
        "controls": ["same-forecast one-week ordinary policy", "ownership free transfers only"],
        "track": "Historical statistical reconstruction; imported corrected labels, not prospective or blindly untouched research",
    }
    originals, _, _ = canonical_rows(parent)
    last_group = max(f"{s}-GW{max(games):02d}" for s, games in periods)
    originals = [r for r in originals if f"{r['season']}-GW{int(r['round']):02d}" <= last_group]
    source_feature = parent.get("decision_features")
    source_frame = pl.read_parquet(parent.artifact_path(source_feature["path"])).filter(
        pl.col("deadline_group") <= last_group
    )
    context_entries = parent.get("context_archive")["entries"]
    last_cutoff = max(packets[s, g]["deadline"] for s, games in periods for g in games)

    def context_signature(cutoff):
        return sorted(
            (r["source"], r["hash"], r["deadline"])
            for r in context_entries
            if moment(r["deadline"]) < moment(cutoff)
        )

    code_hash = digest(
        {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))
        }
    )
    identity = digest(
        {
            "protocol": protocol,
            "inputs": [packets[s, g]["id"] for s, games in periods for g in games],
            "features": digest(source_frame.to_dicts()),
            "observations": digest(originals),
            "context": context_signature(last_cutoff),
            "discipline_rules": hashlib.sha256(
                (Path(__file__).resolve().parents[2] / "configs/discipline-2026-27.json").read_bytes()
            ).hexdigest(),
            "implementation": code_hash,
        }
    )
    for item in parent.list("pipeline_replay", 100):
        complete = parent.get("pipeline_replay", item["id"])
        if complete["protocol_id"] == identity:
            verify_pipeline_replay(parent, complete)
            parent.put(
                "pipeline_replay_progress",
                {
                    "at": now(),
                    "stage": "completed",
                    "season": None,
                    "event": None,
                    "protocol_id": identity,
                    "completed_deadlines": len({(r["season"], r["event"]) for r in complete["weeks"]}),
                    "total_deadlines": sum(len(games) for _, games in periods),
                },
            )
            print("Reused independently verified complete integrated replay", identity, flush=True)
            return complete
    cache = Path("artifacts/runtime/integrated-replay") / identity[:20]
    cache.mkdir(parents=True, exist_ok=True)
    resumed = []
    if resume_protocol_id is not None:
        if not re.fullmatch(r"[0-9a-f]{64}", resume_protocol_id):
            raise ValueError("Invalid source replay identity")
        source_cache = cache.parent / resume_protocol_id[:20]
        trusted = set()
        source_report = next(
            (
                parent.get("pipeline_replay", r["id"])
                for r in parent.list("pipeline_replay", 100)
                if parent.get("pipeline_replay", r["id"])["protocol_id"] == resume_protocol_id
            ),
            None,
        )
        if source_report:
            from .pipeline_checkpoints import recover_portable_report

            verify_pipeline_replay(parent, source_report)
            recover_portable_report(parent, source_report, source_cache, packets, originals)
            trusted = {
                (str(parent.artifact_path(r["decision_file"]).resolve()), r["decision_sha256"])
                for r in source_report["weeks"]
            }
        for prior_season, games in periods:
            for game in games:
                relative = Path(prior_season.replace("/", "-")) / f"gw{game}.json"
                source = source_cache / relative
                if not source.exists():
                    continue
                checkpoint_body = json.loads(source.read_text())
                first_row = checkpoint_body["weeks"][0]
                sealed = json.loads(parent.artifact_path(first_row["decision_file"]).read_text())
                forecast_spec = sealed["actions"][0]["forecast_summary"]
                if (
                    (
                        sealed["protocol_id"] != resume_protocol_id
                        and (
                            str(parent.artifact_path(first_row["decision_file"]).resolve()),
                            first_row["decision_sha256"],
                        )
                        not in trusted
                    )
                    or forecast_spec["worlds"] != worlds
                    or forecast_spec["seed"] != 11
                    or sealed["input_id"] != packets[prior_season, game]["id"]
                ):
                    raise ValueError("Source checkpoint has different forecast inputs or worlds")
                resumed.extend(checkpoint_body["weeks"])
        if not resumed:
            raise ValueError("No completed deadlines available in the source replay")
        verify_pipeline_replay(parent, {"weeks": resumed})
        for prior_season, games in periods:
            for game in games:
                relative = Path(prior_season.replace("/", "-")) / f"gw{game}.json"
                if (source_cache / relative).exists() and not (cache / relative).exists():
                    atomic_json(cache / relative, json.loads((source_cache / relative).read_text()))

    def progress(stage, season=None, event=None):
        body = {
            "at": now(),
            "stage": stage,
            "season": season,
            "event": event,
            "protocol_id": identity,
            "completed_deadlines": len({(r["season"], r["event"]) for r in all_weeks}),
            "total_deadlines": sum(len(games) for _, games in periods),
        }
        parent.put("pipeline_replay_progress", body)
        print("Integrated replay", stage, season, event, flush=True)

    archive_path = parent.root / "features" / ("replay-history-" + digest(originals)[:24] + ".parquet")
    if not archive_path.exists():
        pl.DataFrame(originals, infer_schema_length=None, strict=False).write_parquet(archive_path)
    archive = {"path": archive_path, "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest()}
    all_weeks = []
    epochs = []
    for season, events in periods:
        first = packets[season, events[0]]
        group = f"{season}-GW{events[0]:02d}"
        gate = prior_gate(parent.get("decision_model"), group)
        if not gate["passed"]:
            raise ValueError("Primary reference has no passing strictly earlier acceptance evidence")
        progress("fitting earlier-only availability gates", season, events[0])
        epoch_code = digest(
            {
                name: hashlib.sha256((Path(__file__).parent / (name + ".py")).read_bytes()).hexdigest()
                for name in ("availability", "decision", "features", "models", "observations")
            }
        )
        epoch_key = digest(
            {
                "group": group,
                "observed": completed_before(originals, first["deadline"], group),
                "context": context_signature(first["deadline"]),
                "features": digest(
                    past_frame(
                        source_frame, completed_before(originals, first["deadline"], group), group
                    ).to_dicts()
                ),
                "implementation": epoch_code,
            }
        )[:20]
        epoch, availability = availability_epoch(
            parent,
            Path("artifacts/runtime/integrated-replay-epochs") / epoch_key,
            first,
            archive,
            originals,
            source_frame,
            source_feature,
        )
        epochs.append(
            {"season": season, "before_group": group, "reference_gate": gate, "availability": availability}
        )
        states: dict[str, TeamState] = {}
        for event in events:
            packet = packets[season, event]
            progress("building frozen deadline inputs", season, event)
            checkpoint = cache / season.replace("/", "-") / f"gw{event}.json"
            frozen_models = (
                parent.root
                / "models"
                / ("integrated-" + identity[:16])
                / season.replace("/", "-")
                / f"gw{event}"
            )
            retained = frozen_models / "frozen-decision.json"
            if not checkpoint.exists() and retained.exists():
                from .pipeline_checkpoints import recover_deadline

                atomic_json(
                    checkpoint, recover_deadline(parent, retained, packet, identity, originals, states)
                )
            if checkpoint.exists():
                saved = json.loads(checkpoint.read_text())
                all_weeks.extend(saved["weeks"])
                states = {k: TeamState.model_validate(v) for k, v in saved["next_states"].items()}
                print("Restored complete integrated deadline", season, event, flush=True)
                continue
            scratch = Path("artifacts/runtime/pipeline-scratch")
            scratch.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="fie-primary-replay-", dir=scratch) as temporary:
                store = Store(Path(temporary) / "data")
                attach_observations(store, archive, packet)
                players, market = attach_market(store, packet, originals)
                rules = Rules.from_bootstrap(packet["bootstrap"])
                if not states:
                    seed, _ = initial_account(packet, players, rules, seconds)
                    states = {
                        name: seed.model_copy(deep=True)
                        for name in ("primary_multiweek", "reference_one_week", "ownership_no_hits")
                    }
                for name in states:
                    states[name] = reprice(states[name], players, rules)
                    states[name].known_at = packet["deadline"]
                    validate_squad(states[name], players, rules)
                attach_availability(store, epoch, availability)
                coverage = attach_dated_sources(parent, store, packet["deadline"])
                build_discipline(store)
                model = fit_reference(
                    store, source_frame, source_feature, packet["deadline"], f"{season}-GW{event:02d}", gate
                )
                end = 19 if event <= 19 else 38
                forecast = learned_forecast(store, event, min(38, end + 4), worlds, 11)
                progress("solving full chip-window policy", season, event)
                print("Integrated forecast ready", season, event, "planning through", end, flush=True)
                state = states["primary_multiweek"]
                state_id = store.put("state", state.model_dump())
                plan = optimize(store, state_id, forecast["id"], end, seconds, use_decision_critic=False)
                from .objectives import choose_recommendation

                plan, selection = choose_recommendation(store, [plan], forecast)
                action = action_from_row(plan["ledger"][0])
                following, ledger = transition(state, action, players, rules)
                first_means = {r["player_id"]: r["mean"] for r in forecast["players"] if r["event"] == event}
                scheduled = {
                    r["player_id"]
                    for r in forecast["players"]
                    if r["event"] == event and r["fixture_count"] > 0
                }
                frozen = [
                    {
                        "policy": "primary_multiweek",
                        "action": action.model_dump(),
                        "ledger": ledger,
                        "next_state": following.model_dump(),
                        "solver": plan["solver"],
                        "planned_ledger": plan["ledger"],
                        "selection": selection,
                        "forecast_summary": {
                            k: v
                            for k, v in forecast.items()
                            if k not in ("players", "fixtures", "match_audit")
                        },
                    }
                ]
                for name, means, limit in (
                    ("reference_one_week", first_means, 15),
                    (
                        "ownership_no_hits",
                        baseline_signals(players, "ownership", scheduled),
                        max(
                            0,
                            states["ownership_no_hits"].free_transfers
                            - states["ownership_no_hits"].transfers_made,
                        ),
                    ),
                ):
                    choice = solve_action(
                        states[name], players, rules, means, max_transfers=limit, seconds=seconds
                    )
                    frozen.append(
                        {
                            "policy": name,
                            "action": choice.action.model_dump(),
                            "ledger": choice.ledger,
                            "next_state": choice.state.model_dump(),
                            "solver": choice.solver,
                        }
                    )
                # Freeze every action before touching target points/minutes.
                frozen_record = {
                    "season": season,
                    "event": event,
                    "input_id": packet["id"],
                    "information_cutoff": packet["deadline"],
                    "constructed_at": now(),
                    "protocol_id": identity,
                    "coverage": coverage,
                    "model": model,
                    "actions": frozen,
                    "states_before": {name: state.model_dump() for name, state in states.items()},
                    "first_week_predictions": [r for r in forecast["players"] if r["event"] == event],
                }
                frozen_models = (
                    parent.root
                    / "models"
                    / ("integrated-" + identity[:16])
                    / season.replace("/", "-")
                    / f"gw{event}"
                )
                if (frozen_models / "frozen-decision.json").exists():
                    raise ValueError("A sealed replay decision already exists; refusing to overwrite it")
                shutil.copytree(store.root / "models", frozen_models, dirs_exist_ok=True)
                frozen_record["model_artifacts"] = {
                    name: {"path": str(frozen_models / "primary-reference" / (name + ".txt")), "hash": value}
                    for name, value in model["model_hashes"].items()
                }
                frozen_record["availability"] = availability
                from .worlds import load_worlds

                with load_worlds(store, forecast) as samples:
                    np.savez_compressed(
                        frozen_models / "observed-worlds.npz",
                        points=samples[f"points_{event}"],
                        minutes=samples[f"minutes_{event}"],
                    )
                frozen[0]["forecast_summary"].pop("scenario_path", None)
                frozen[0]["forecast_summary"]["world_retention"] = (
                    "First-week samples preserved with the frozen model; future worlds regenerated from frozen inputs/model/seed"
                )
                decision_file = frozen_models / "frozen-decision.json"
                atomic_json(decision_file, frozen_record)
                decision_hash = hashlib.sha256(decision_file.read_bytes()).hexdigest()
                actual: dict = defaultdict(float)
                minutes: dict = defaultdict(int)
                for row in originals:
                    if (row["season"], int(row["round"])) == (season, event):
                        actual[int(row["element"])] += float(row.get("total_points") or 0)
                        minutes[int(row["element"])] += int(row.get("minutes") or 0)
                from .domain import Lineup

                results = []
                for row in frozen:
                    score = (
                        score_gameweek(
                            actual,
                            minutes,
                            Lineup.model_validate(row["action"]["lineup"]),
                            players,
                            rules,
                            row["action"]["chip"],
                        )
                        - row["ledger"]["hit"]
                    )
                    states[row["policy"]] = TeamState.model_validate(row["next_state"])
                    states[row["policy"]].points += int(score)
                    results.append(
                        {
                            "season": season,
                            "event": event,
                            "policy": row["policy"],
                            "net_points": score,
                            "action": row["action"],
                            "ledger": row["ledger"],
                            "solver": row["solver"],
                            "decision_file": str(decision_file),
                            "decision_sha256": decision_hash,
                            "input_id": packet["id"],
                            "model_id": model["id"],
                            "forecast_id": forecast["id"],
                            "forecast_cutoff": forecast["cutoff"],
                        }
                    )
                # Preserve calibrated model files; large future worlds are
                # reproducible from these inputs/seed rather than copied forever.
                atomic_json(
                    checkpoint,
                    {
                        "weeks": results,
                        "next_states": {k: v.model_dump() for k, v in states.items()},
                        "frozen_models": str(frozen_models),
                    },
                )
                all_weeks.extend(results)
                print(
                    "Completed integrated deadline",
                    season,
                    event,
                    {r["policy"]: r["net_points"] for r in results},
                    flush=True,
                )
    summaries = {}
    for season, _ in periods:
        rows = [r for r in all_weeks if r["season"] == season]
        summaries[season] = {
            name: {
                "net_points": sum(r["net_points"] for r in rows if r["policy"] == name),
                "hit_points": sum(r["ledger"]["hit"] for r in rows if r["policy"] == name),
                "chips": [
                    {"event": r["event"], "chip": r["action"]["chip"]}
                    for r in rows
                    if r["policy"] == name and r["action"]["chip"]
                ],
            }
            for name in ("primary_multiweek", "reference_one_week", "ownership_no_hits")
        }
    report: dict = {
        "at": now(),
        "protocol": protocol,
        "protocol_id": identity,
        "epochs": epochs,
        "resumed_from": resume_protocol_id,
        "resumed_deadlines": len({(r["season"], r["event"]) for r in resumed}),
        "resume_scope": "Completed pre-scoring actions retained after interruption and independently reverified; source lineage remains in each immutable decision file"
        if resumed
        else None,
        "weeks": all_weeks,
        "summaries": summaries,
        "live_pointers_unchanged": all(parent.get(k)["id"] == v for k, v in live_before.items()),
        "limits": [
            "Later reconstruction of corrected match facts; not prospective performance or guaranteed future scores.",
            "Runs the deployed primary expected-points engine. Rejected neural/critic models and optional research sensitivities are not injected as if accepted.",
            "No hindsight future prices, fixtures, rival choices or news. Missing original cup/discipline statements remain explicit coverage gaps.",
            "Full-window additive MIP candidates are independently transitioned and rescored in shared worlds. Actual solver gaps and time limits remain visible.",
            "Current-season outcomes are already inspected; they do not select architecture, thresholds or hyperparameters here.",
        ],
    }
    for season, events in periods:
        values = {(r["event"], r["policy"]): r["net_points"] for r in all_weeks if r["season"] == season}
        report["summaries"][season]["paired_primary_minus_ownership_95ci"] = (
            paired_interval([values[g, "primary_multiweek"] - values[g, "ownership_no_hits"] for g in events])
            if len(events) >= 2
            else None
        )
    report["independent_verification"] = verify_pipeline_replay(parent, report)
    publish = worlds >= 4000 and seconds >= 60 and all(len(games) >= 9 for _, games in periods)
    report["production_budget"] = publish
    complete_scope = len(periods) >= 2
    report["id"] = parent.put(
        "pipeline_replay"
        if publish and complete_scope
        else "pipeline_replay_current"
        if publish
        else "pipeline_replay_check",
        report,
    )
    atomic_json(
        Path("artifacts/evaluations/integrated-pipeline-replay.json")
        if publish and complete_scope
        else Path("artifacts/evaluations/integrated-current-season.json")
        if publish
        else Path("artifacts/runtime/pipeline-smoke.json"),
        report,
    )
    progress("completed")
    return report
