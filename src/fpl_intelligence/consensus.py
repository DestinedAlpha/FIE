"""Deadline-safe crowd/football fusion, selected on a separate earlier season.

The inspected current season never selects weights or promotion. Published EP
is a weekly quantity: it is allocated across the *announced* fixtures, never
multiplied by the number of matches. Missing values have explicit masks.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits

from .availability import moment
from .decision import canonical_rows, fit_components, player_key, predict_models, score_metrics
from .enrichment import EnrichmentIndex
from .fusion import calibrate, corrected, fit_tree, load_data, tree_predict
from .fusion_data import sequence, source_features, tree_matrix
from .replay import deadline_rows
from .replay_benchmarks import BASELINES, baseline_signals
from .rules import Rules, best_lineup, score_gameweek
from .storage import Store, atomic_json, digest, now

PUBLISHED = ("ep_next", "points_per_game", "form")
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


async def prepare_inputs(store):
    """Cache proven schedules/registries without refreshing today's account."""
    import asyncio
    import json

    from .context_archive import MANIFEST
    from .replay import capture_replay_inputs

    market = store.get("market")
    current = market["season"]
    completed = max((e["id"] for e in market["events"] if e.get("data_checked")), default=0)
    keys = {
        (r["season"], r["event"])
        for r in json.loads(MANIFEST.read_text())["snapshots"]
        if r["season"] < current or r["event"] <= completed
    }
    keys.update((current, g) for g in range(1, completed + 1))
    semaphore = asyncio.Semaphore(3)

    async def one(key):
        async with semaphore:
            return (await capture_replay_inputs(store, key[0], [key[1]]))[0]["id"]

    return await asyncio.gather(*(one(key) for key in sorted(keys)))


def published_vector(packet, fixture_count):
    result: list[float] = []
    for name in PUBLISHED:
        value = packet.get(name)
        known = value not in (None, "")
        number = float(value) if known else 0.0
        if not np.isfinite(number):
            raise ValueError("Nonfinite published forecast feature")
        if name == "ep_next":
            number = number / max(1, fixture_count)
        result.extend((number, float(known)))
    return result + [float(fixture_count)]


def matrix_for(arrays, published):
    return np.column_stack(
        [
            tree_matrix(
                arrays["tab"],
                arrays["news"],
                arrays["market"],
                arrays["workload"],
                arrays["text"],
                arrays["sequence"],
                enrichment=arrays.get("enrichment"),
            ),
            np.asarray(published, dtype=np.float32),
        ]
    )


def replay_packets(store):
    result = {}
    # Prefer the newest complete provenance record for each deadline.
    for row in reversed(store.list("replay_inputs", 1000)):
        body = store.get("replay_inputs", row["id"])
        result[body["season"], body["event"]] = body
    return result


class DeadlineInputs:
    """Historical serving tensors, including players with no previous matches."""

    def __init__(self, store, feature_id):
        from .elite import elite_features

        self.originals, _, _ = canonical_rows(store)
        feature = store.get("fusion_features", feature_id)
        self.workload_version = feature.get("workload_transform", "legacy-v1")
        self.enrichment = EnrichmentIndex(
            store, self.originals, feature.get("lineage", {}).get("temporal_model")
        )
        self.experts, _ = elite_features(store)
        self.samples = {(s, e): v["sample"] for (s, e, _), v in self.experts.items()}
        self.history = defaultdict(list)
        for row in self.originals:
            self.history[player_key(row)].append(row)
        self.workload = defaultdict(list)
        from .workload import records as workload_records

        for record in workload_records(store):
            self.workload[record["player_code"]].append(record)

    def build(self, rows, players, packet):
        deadline, season, event = packet["deadline"], packet["season"], packet["event"]
        counts = Counter(r["player_id"] for r in rows)
        arrays = defaultdict(list)
        published = []
        for row in rows:
            player = players[row["player_id"]]
            past = [
                p
                for p in self.history["code", player.code]
                if p.get("kickoff_time")
                and moment(p["kickoff_time"]) + timedelta(hours=4) < moment(deadline)
                and (p["season"], int(p["round"])) < (season, event)
            ]
            seq, mask = sequence(past)
            sample = self.samples.get((season, event))
            expert = self.experts.get((season, event, player.id), {"sample": sample} if sample else {})
            workload = [r for r in self.workload[player.code] if moment(r["known_after"]) < moment(deadline)]
            text, news, market, load, quality = source_features(
                player.raw, expert, workload, player.code, deadline, self.workload_version
            )
            enrichment = self.enrichment.vector(player.code, player.position, deadline, season, player.raw)
            quality = np.append(quality, float(enrichment[-2] > 0 or enrichment[-1] > 0))
            quality[0] = mask.any()
            for name, value in (
                ("tab", row["x"]),
                ("sequence", seq),
                ("sequence_mask", mask),
                ("text", text),
                ("news", news),
                ("market", market),
                ("workload", load),
                ("quality", quality),
                ("enrichment", enrichment),
            ):
                arrays[name].append(value)
            published.append(published_vector(player.raw, counts[player.id]))
        return {
            key: np.asarray(
                values, dtype=np.int64 if key == "text" else bool if key == "sequence_mask" else np.float32
            )
            for key, values in arrays.items()
        }, published


def mix_distributions(reference, challenger, weight):
    """Mixture preserves probability mass and conditional expected points."""
    rp, rc, rs = reference
    cp, cc, cs = challenger
    p = (1 - weight) * rp + weight * cp
    joint = (1 - weight) * rp * rc + weight * cp * cc
    c = np.divide(joint, p, out=np.zeros_like(joint), where=p > 0)
    return p, c, (1 - weight) * rs + weight * cs


def apply_reference_availability(probability, rows):
    p = probability.copy()
    p[:, 1:] *= np.asarray([r["availability"] for r in rows])[:, None]
    p[:, 0] = 1 - p[:, 1:].sum(1)
    return p


def expected_points(distribution, rows, players):
    p, c, _ = distribution
    hard_absent = np.array([r["availability"] == 0 or not players[r["player_id"]].selectable for r in rows])
    means = (p * (c + 2 * np.asarray([r["dc_probability"] for r in rows]))).sum(1)
    means[hard_absent] = 0
    result = {pid: 0.0 for pid in players}
    for row, mean in zip(rows, means, strict=True):
        result[row["player_id"]] += float(mean)
    return result


def paired_interval(delta, seed=81):
    values = np.asarray(delta, dtype=float)
    if len(values) < 2:
        raise ValueError("At least two deadlines are required")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(values), size=(4000, (len(values) + 1) // 2))
    indices = ((starts[:, :, None] + np.arange(2)) % len(values)).reshape(4000, -1)[:, : len(values)]
    return np.quantile(values[indices].mean(1), [0.025, 0.975]).tolist()


def select_weight(decisions, season):
    """The selector has no dependence on later-season/current diagnostic scores."""
    rows = [r for r in decisions if r["season"] == season and 1 <= r["event"] <= 19]
    names = [f"mix_{w:g}" for w in WEIGHTS] + list(BASELINES)
    for name in names:
        if sorted(r["event"] for r in rows if r["policy"] == name) != list(range(1, 20)):
            raise ValueError("Incomplete or duplicate earlier-season selection decisions")
    totals = {name: sum(r["actual_points"] for r in rows if r["policy"] == name) for name in names}
    selected = max((f"mix_{w:g}" for w in WEIGHTS), key=lambda p: (totals[p], -float(p[4:])))
    return selected, totals


def reusable_decision(cached, signals, input_id):
    """Reuse a solve only after reproducing every current input utility."""
    if cached is None or cached.get("input_id") != input_id:
        return False
    previous = {int(p): float(v) for p, v in cached["signals"].items()}
    return previous.keys() == signals.keys() and all(
        np.isclose(previous[p], v, rtol=1e-12, atol=1e-12) for p, v in signals.items()
    )


def fit_consensus(store: Store, seconds=10):
    """Fit before 2025, select on GW1–19, accept on GW20–38; diagnose 2026 later."""
    from .optimization import select_squad

    if not 1 <= seconds <= 300:
        raise ValueError("Invalid solver budget")
    state_id = store.get("state")["id"]
    arrays, metadata, feature = load_data(store)
    packets = replay_packets(store)
    current_season = store.get("market")["season"]
    previous_season = "2025/26"
    required = [(previous_season, e) for e in range(1, 39)] + [(current_season, e) for e in range(1, 10)]
    if missing := sorted(set(required) - packets.keys()):
        raise ValueError(f"Missing proven replay packets: {missing}")
    lookups = {key: {p["id"]: p for p in v["bootstrap"]["elements"]} for key, v in packets.items()}
    counts = {}
    fixture_ids = {}
    for key, packet in packets.items():
        fixtures = [f for f in packet["fixture_proof"]["fixtures"] if f.get("event") == packet["event"]]
        counts[key] = Counter(t for f in fixtures for t in (f["team_h"], f["team_a"]))
        fixture_ids[key] = {f["id"] for f in fixtures}
    published, eligible = [], []
    for row in metadata:
        key = row["season"], row["event"]
        packet = lookups.get(key, {}).get(row["player_id"], {})
        published.append(published_vector(packet, counts.get(key, {}).get(packet.get("team"), 0)))
        eligible.append(bool(packet) and row["fixture_id"] in fixture_ids[key])
    matrix = matrix_for(arrays, published)
    groups = np.array([r["deadline_group"] for r in metadata])
    eligible_mask = np.asarray(eligible)
    prior = sorted(set(groups[(groups < previous_season) & eligible_mask]))
    cut = prior[-4]
    train = np.flatnonzero((groups < cut) & eligible_mask)
    cal = np.flatnonzero((groups >= cut) & (groups < previous_season) & eligible_mask)
    if len(train) < 10000 or len(cal) < 1000:
        raise ValueError("Insufficient proven training/calibration coverage")
    protocol = {
        "version": "published-fusion-v1",
        "train_before": cut,
        "calibration_through": prior[-1],
        "weight_selection": f"{previous_season} GW1–19",
        "acceptance": f"{previous_season} GW20–38",
        "current_diagnostic": f"{current_season} GW1–9; previously inspected, never weight selection",
        "weight_grid": WEIGHTS,
        "seed": 47,
        "seconds_per_solve": seconds,
        "coverage": "Full archived registered market, pre-deadline fixture schedules and prices, explicit missing-source masks",
        "promotion": "Positive lower block-bootstrap interval for legal weekly-team returns and negative upper interval for per-deadline MSE against reference; no current-season selection",
        "next_deadline_only": True,
        "baseline_schedule": "v2: announced blank gameweeks have zero utility",
    }
    directory = (
        store.root
        / "models"
        / (
            "consensus-"
            + digest(
                {
                    "feature": feature["id"],
                    "inputs": [packets[k]["id"] for k in sorted(packets)],
                    "protocol": protocol,
                }
            )[:20]
        )
    )
    directory.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        tree = fit_tree(arrays, matrix, train, cal)
        curves, offsets = calibrate(*tree_predict(tree, matrix[cal]), arrays, cal)
    tree_spec = {"curves": curves, "offsets": offsets, "models": {}}
    for name, model in tree.items():
        path = directory / (name + ".txt")
        model.booster_.save_model(str(path))
        tree_spec["models"][name] = {"path": str(path), "hash": hashlib.sha256(path.read_bytes()).hexdigest()}
    # Both architectures have the same chronological fitting boundary.
    frame = pl.read_parquet(store.artifact_path(store.get("decision_features")["path"]))
    earlier = frame.filter(pl.col("deadline_group") < previous_season)
    ref_groups = sorted(earlier["deadline_group"].unique().to_list())
    ref_cut = ref_groups[-8]
    with threadpool_limits(limits=1):
        reference = fit_components(
            earlier.filter(pl.col("deadline_group") < ref_cut),
            earlier.filter(pl.col("deadline_group") >= ref_cut),
        )
    reference_spec = {"curves": reference["curves"], "models": {}, "fit_before": ref_cut}
    for name in ("minutes", "points", "starter"):
        path = directory / ("reference-" + name + ".txt")
        reference[name].booster_.save_model(str(path))
        reference_spec["models"][name] = {
            "path": str(path),
            "hash": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    builder = DeadlineInputs(store, feature["id"])
    policies = [f"mix_{w:g}" for w in WEIGHTS] + list(BASELINES)
    frozen: list[dict] = []
    errors: dict = defaultdict(dict)
    for key in required:
        if key == (previous_season, 20):
            selected, totals = select_weight(frozen, previous_season)
            selection_id = store.put(
                "consensus_weight_selection",
                {
                    "season": previous_season,
                    "through_event": 19,
                    "selected": selected,
                    "totals": totals,
                    "decision_ids": [r["decision_id"] for r in frozen],
                    "protocol": protocol,
                },
            )
        packet = packets[key]
        checkpoint = directory / (key[0].replace("/", "-") + f"-gw{key[1]}.json")
        cached = {}
        if checkpoint.exists():
            import json

            restored = json.loads(checkpoint.read_text())
            cached = {r["policy"]: r for r in restored["decisions"]}
        players, rows = deadline_rows(
            builder.originals,
            packet["bootstrap"],
            packet["fixture_proof"]["fixtures"],
            key[0],
            key[1],
            packet["deadline"],
        )
        tensors, pub = builder.build(rows, players, packet)
        with threadpool_limits(limits=1):
            challenger = corrected(*tree_predict(tree, matrix_for(tensors, pub)), curves, offsets)
            _, rp, rc = predict_models(
                reference, np.asarray([r["x"] for r in rows]), [r["dc_probability"] for r in rows]
            )
        rp = apply_reference_availability(rp, rows)
        # Start probabilities are not used by this independent weekly policy.
        base = rp, rc, np.zeros(len(rows))
        signals = {
            f"mix_{w:g}": expected_points(mix_distributions(base, challenger, w), rows, players)
            for w in WEIGHTS
        }
        signals.update(
            {
                method: baseline_signals(players, method, {r["player_id"] for r in rows})
                for method in BASELINES
            }
        )
        rules = Rules.from_bootstrap(packet["bootstrap"])
        event_decisions = []
        for policy in policies:
            cached_row = cached.get(policy)
            if cached_row is not None and reusable_decision(cached_row, signals[policy], packet["id"]):
                event_decisions.append({k: v for k, v in cached_row.items() if k != "actual_points"})
                continue
            solver: dict = {}
            squad = select_squad(
                None,
                players,
                rules,
                signals[policy],
                seconds=seconds,
                initial_budget=1000,
                bench_weight=0,
                relative_gap=0,
                diagnostics=solver,
            )
            lineup = best_lineup(squad, signals[policy], players, rules)
            decision = {
                "season": key[0],
                "event": key[1],
                "policy": policy,
                "input_id": packet["id"],
                "lineup": lineup.model_dump(),
                "solver": solver,
                "players": [{"id": p, "name": players[p].web_name, "price": players[p].price} for p in squad],
                "bank": 1000 - sum(players[p].price for p in squad),
                "signals": {str(p): v for p, v in signals[policy].items()},
                "frozen_at": now(),
            }
            decision["decision_id"] = store.put("consensus_decision", decision)
            event_decisions.append(decision)
        # None of the functions above accept the target-week labels.
        actual: dict = defaultdict(float)
        minutes: dict = defaultdict(int)
        for row in builder.originals:
            if (row["season"], int(row["round"])) == key:
                actual[int(row["element"])] += float(row.get("total_points") or 0)
                minutes[int(row["element"])] += int(row.get("minutes") or 0)
        from .domain import Lineup

        for decision in event_decisions:
            decision["actual_points"] = score_gameweek(
                actual, minutes, Lineup.model_validate(decision["lineup"]), players, rules
            )
        for weight in WEIGHTS:
            name = f"mix_{weight:g}"
            errors[key][name] = score_metrics(
                np.array([actual[p] for p in players]), np.array([signals[name][p] for p in players])
            )
        atomic_json(checkpoint, {"decisions": event_decisions, "errors": errors[key]})
        frozen.extend(event_decisions)
        print(
            "Consensus decisions", key, {r["policy"]: r["actual_points"] for r in event_decisions}, flush=True
        )
    # Selection is complete before the untouched prior-season acceptance slice.
    holdout = {
        (r["event"], r["policy"]): r for r in frozen if r["season"] == previous_season and r["event"] >= 20
    }
    delta = [
        holdout[g, selected]["actual_points"] - holdout[g, "mix_0"]["actual_points"] for g in range(20, 39)
    ]
    mse_delta = [
        errors[previous_season, g][selected]["rmse"] ** 2 - errors[previous_season, g]["mix_0"]["rmse"] ** 2
        for g in range(20, 39)
    ]
    team_ci, error_ci = paired_interval(delta), paired_interval(mse_delta)
    passed = float(selected[4:]) > 0 and team_ci[0] > 0 and error_ci[1] < 0
    result = {
        "at": now(),
        "version": "published-fusion-v1",
        "feature_id": feature["id"],
        "reference_feature_id": store.get("decision_features")["id"],
        "training_input_ids": [packets[k]["id"] for k in sorted(packets) if k[0] < previous_season],
        "protocol": protocol,
        "directory": str(directory),
        "tree_spec": tree_spec,
        "reference_spec": reference_spec,
        "training_rows": len(train),
        "calibration_rows": len(cal),
        "selected": selected,
        "weight": float(selected[4:]),
        "selection_totals": totals,
        "selection_id": selection_id,
        "acceptance": {
            "weekly_return_delta_95ci": team_ci,
            "deadline_mse_delta_95ci": error_ci,
            "return_delta": sum(delta),
            "passed": bool(passed),
        },
        "promoted": False,
        "current_totals": {
            name: sum(
                r["actual_points"] for r in frozen if r["season"] == current_season and r["policy"] == name
            )
            for name in policies
        },
        "weeks": frozen,
        "account_state_unchanged": store.get("state")["id"] == state_id,
        "limits": [
            "Independent weekly £100m teams with no chips; not a continuous account or a prospective track record.",
            "Predictions are evaluated on full dated registries and fixture schedules. Historical training match labels and enrichment were imported retrospectively.",
            "The crowd model here is frozen before 2025/26. Current-season rows update historical player inputs but do not tune its parameters or ensemble weights.",
            "This comparison does not certify optimal season-long transfer policy or guarantee 550 points.",
            "Top-manager history is from a prior-qualified cohort; retrospective current top-1000 selection never enters earlier deadlines.",
            "This is chronological acceptance for this configuration on a repeatedly studied research archive, not a blinded prospective experiment.",
        ],
    }
    result["id"] = store.put("consensus_model", result)
    atomic_json(Path("artifacts/evaluations/published-fusion.json"), result)
    return result


def load_verified_models(store, config):
    result = {}
    for key, reference in config["models"].items():
        path = store.artifact_path(reference["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != reference["hash"]:
            raise ValueError("Consensus model integrity verification failed")
        result[key] = lgb.Booster(model_file=str(path))
    return result


def comparison_is_current(report, market, state_id, primary_id, consensus_id, at):
    if (
        report["market_id"] != market["id"]
        or report["state_id"] != state_id
        or report["primary_model_id"] != primary_id
        or report["consensus_model_id"] != consensus_id
    ):
        return False
    deadline = next((e["deadline_time"] for e in market["events"] if e["id"] == report["event"]), None)
    return deadline is not None and moment(at) < moment(deadline)


def compare_live_consensus(store: Store, seconds=15):
    """Connect the accepted weekly challenger to the real squad and legal moves.

    This one-deadline analysis is explicitly distinct from the multiweek primary
    plan. Account, forecast and selected-plan pointers are never changed.
    """
    from .decision import future_inputs
    from .domain import TeamState
    from .fusion_data import live_inputs
    from .ingestion import load_market
    from .strategy import solve_action

    model = store.get("consensus_model")
    state_body = store.get("state")
    state = TeamState.model_validate({k: v for k, v in state_body.items() if k != "id"})
    players, market = load_market(store)
    event = next(e["id"] for e in market["events"] if e.get("is_next"))
    if state.event != event:
        raise ValueError("Reconcile the account before comparing current decisions")
    deadline = next(e["deadline_time"] for e in market["events"] if e["id"] == event)
    if moment(now()) >= moment(deadline):
        raise ValueError("Deadline passed; refresh the account and upcoming gameweek before comparing")
    rows, _ = future_inputs(store, market, players, event, event)
    arrays = live_inputs(store, rows, market, players, model["feature_id"])
    counts = Counter(r["player_id"] for r in rows)
    pub = [published_vector(players[r["player_id"]].raw, counts[r["player_id"]]) for r in rows]
    with threadpool_limits(limits=1):
        challenger = corrected(
            *tree_predict(load_verified_models(store, model["tree_spec"]), matrix_for(arrays, pub)),
            model["tree_spec"]["curves"],
            model["tree_spec"]["offsets"],
        )
        ref = load_verified_models(store, model["reference_spec"]) | {
            "curves": model["reference_spec"]["curves"]
        }
        _, p, c = predict_models(ref, np.asarray([r["x"] for r in rows]), [r["dc_probability"] for r in rows])
    reference = apply_reference_availability(p, rows), c, np.zeros(len(rows))
    selected = mix_distributions(reference, challenger, model["weight"])
    means = expected_points(selected, rows, players)
    reference_means = expected_points(reference, rows, players)
    rules = Rules.from_bootstrap(store.latest("bootstrap", content_hash=market["source_hash"])[0])
    choices = []
    for name, limit in (
        ("ordinary", 15),
        ("no_hit", max(0, state.free_transfers - state.transfers_made)),
        ("hold", 0),
    ):
        decision = solve_action(state, players, rules, means, max_transfers=limit, seconds=seconds)
        choices.append(
            {
                "name": name,
                "action": decision.action.model_dump(),
                "ledger": decision.ledger,
                "solver": decision.solver,
                "additive_expected_net": decision.utility,
            }
        )
    primary = store.get("decision_model")
    report = {
        "at": now(),
        "consensus_model_id": model["id"],
        "market_id": market["id"],
        "state_id": state_body["id"],
        "primary_model_id": primary["id"],
        "event": event,
        "selected_weight": model["weight"],
        "historical_gate_passed": model["acceptance"]["passed"],
        "production_active": False,
        "scope": "Legal one-deadline additive comparison; not the primary multiweek recommendation. Prior-season weekly acceptance does not certify the full transfer/chip policy.",
        "players": sorted(
            [
                {
                    "player_id": pid,
                    "name": p.web_name,
                    "consensus": means[pid],
                    "frozen_reference": reference_means[pid],
                    "published": {name: p.raw.get(name) for name in PUBLISHED},
                    "ownership": p.ownership,
                }
                for pid, p in players.items()
            ],
            key=lambda r: (-r["consensus"], r["player_id"]),
        ),
        "decisions": choices,
        "account_state_unchanged": store.get("state")["id"] == state_body["id"],
    }
    report["id"] = store.put("consensus_comparison", report)
    atomic_json(Path("artifacts/runtime/published-fusion-live.json"), report)
    return report


def validate_continuous_policy(store: Store, seconds=10):
    """Ordinary-transfer acceptance with exact lots/FT/bank and frozen forecasts.

    Policies start from the same £100m ownership squad, then must finance every
    move and pay hits. Later weekly forecasts are revealed only at their own
    deadline. Chips are deliberately excluded from this component test.
    """
    from .domain import Lineup, Lot, TeamState
    from .rules import selling_price, validate_squad
    from .strategy import solve_action

    model = store.get("consensus_model")
    packets = replay_packets(store)
    original_state_id = store.get("state")["id"]
    originals, _, _ = canonical_rows(store)
    current_season = store.get("market")["season"]
    seasons = [("2025/26", list(range(20, 39))), (current_season, list(range(1, 10)))]
    policies = list(dict.fromkeys([model["selected"], "mix_0", "ownership_no_hits", "official_expected"]))
    frozen = {(r["season"], r["event"], r["policy"]): r for r in model["weeks"]}
    ledgers = []
    for season, events in seasons:
        for policy in policies:
            source_policy = "ownership" if policy == "ownership_no_hits" else policy
            first = frozen[season, events[0], "ownership"]
            packet = packets[season, events[0]]
            players, _ = deadline_rows(
                [],
                packet["bootstrap"],
                packet["fixture_proof"]["fixtures"],
                season,
                events[0],
                packet["deadline"],
            )
            squad = [p["id"] for p in first["players"]]
            state = TeamState(
                season=season,
                event=events[0],
                bank=first["bank"],
                free_transfers=1,
                initial=events[0] == 1,
                lots=[
                    Lot(
                        player_id=p,
                        purchase_price=players[p].price,
                        sell_price=players[p].price,
                        quote_market_price=players[p].price,
                    )
                    for p in squad
                ],
                known_at=packet["deadline"],
                status="dated_seed",
            )
            for event in events:
                packet = packets[season, event]
                players, _ = deadline_rows(
                    [],
                    packet["bootstrap"],
                    packet["fixture_proof"]["fixtures"],
                    season,
                    event,
                    packet["deadline"],
                )
                rules = Rules.from_bootstrap(packet["bootstrap"])
                for lot in state.lots:
                    if lot.player_id not in players:
                        raise ValueError(
                            "Owned player absent from dated registry; exact sale cannot be inferred"
                        )
                    assert lot.purchase_price is not None
                    lot.sell_price = selling_price(lot.purchase_price, players[lot.player_id].price)
                    lot.quote_market_price = players[lot.player_id].price
                if event > events[0]:
                    state.grandfathered_clubs = [
                        club
                        for club, n in Counter(players[lot.player_id].team for lot in state.lots).items()
                        if n > rules.club_limit
                    ]
                state.known_at = packet["deadline"]
                validate_squad(state, players, rules)
                signals = {int(p): v for p, v in frozen[season, event, source_policy]["signals"].items()}
                # The first squad was chosen within the initial £100m budget.
                decision = solve_action(
                    state,
                    players,
                    rules,
                    signals,
                    max_transfers=0
                    if event == events[0]
                    else (
                        max(0, state.free_transfers - state.transfers_made)
                        if policy == "ownership_no_hits"
                        else 15
                    ),
                    seconds=seconds,
                )
                body = {
                    "season": season,
                    "event": event,
                    "policy": policy,
                    "state": state.model_dump(),
                    "action": decision.action.model_dump(),
                    "solver": decision.solver,
                    "forecast_decision_id": frozen[season, event, source_policy]["decision_id"],
                    "common_initial_decision_id": first["decision_id"],
                    "input_id": packet["id"],
                    "frozen_at": now(),
                }
                identity = store.put("continuous_consensus_decision", body)
                actual: dict = defaultdict(float)
                minutes: dict = defaultdict(int)
                for row in originals:
                    if (row["season"], int(row["round"])) == (season, event):
                        actual[int(row["element"])] += float(row.get("total_points") or 0)
                        minutes[int(row["element"])] += int(row.get("minutes") or 0)
                points = score_gameweek(
                    actual, minutes, Lineup.model_validate(body["action"]["lineup"]), players, rules
                )
                net = points - decision.ledger["hit"]
                ledgers.append(
                    body
                    | {
                        "decision_id": identity,
                        "gross_points": points,
                        "net_points": net,
                        "hit": decision.ledger["hit"],
                        "bank_after": decision.state.bank,
                        "ft_next": decision.state.free_transfers,
                    }
                )
                state = decision.state
                state.points += int(net)
            print(
                "Continuous ordinary policy",
                season,
                policy,
                sum(w["net_points"] for w in ledgers if w["season"] == season and w["policy"] == policy),
                flush=True,
            )

    def totals(season):
        return {
            p: sum(w["net_points"] for w in ledgers if w["season"] == season and w["policy"] == p)
            for p in policies
        }

    previous = {(w["event"], w["policy"]): w for w in ledgers if w["season"] == "2025/26"}
    delta = [
        previous[g, model["selected"]]["net_points"] - previous[g, "mix_0"]["net_points"]
        for g in range(20, 39)
    ]
    interval = paired_interval(delta)
    result = {
        "at": now(),
        "consensus_model_id": model["id"],
        "policies": policies,
        "weeks": ledgers,
        "previous_season_totals": totals("2025/26"),
        "current_totals": totals(current_season),
        "acceptance": {"ordinary_return_delta_95ci": interval, "passed": interval[0] > 0},
        "account_state_unchanged": store.get("state")["id"] == original_state_id,
        "scope": "Separate continuous ordinary-transfer component test: identical initial £100m ownership squads, bank, purchase lots and FT; then exact selling-price profit sharing, bank, FT accumulation, paid hits, captain fallback and autosubs. GW1 initialization grants one FT for GW2. No chips or future-forecast lookahead.",
        "limits": [
            "This tests a one-deadline ordinary-transfer policy, not the primary full multiweek/chip/rival pipeline.",
            "Current-season scores are diagnostics and do not select policy weights.",
            "The ownership ranking baseline makes only free transfers; ownership percentages are not expected points and are never compared numerically against a four-point hit.",
        ],
    }
    result["id"] = store.put("consensus_policy_validation", result)
    atomic_json(Path("artifacts/evaluations/published-fusion-continuous.json"), result)
    return result
