"""Frozen earlier-year selection and legal-account tests for horizon forecasts."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits

from .availability import moment
from .consensus import mix_distributions, paired_interval, replay_packets
from .decision import NAMES, PARAMETERS, canonical_rows, fit_components, predict_models, score_metrics
from .decision_training import reprice
from .fixture_horizon import FEATURES, PARAMS, build_data, fit_models, player_bootstrap, predict
from .pipeline_replay import initial_account
from .replay import deadline_rows
from .replay_benchmarks import baseline_signals
from .rules import Rules, score_gameweek
from .storage import atomic_json, digest, now
from .strategy import solve_action


def fitted(store, frame, before, weighting, directory):
    # A cutoff/weighting name alone can silently reuse a different dataset.
    contract = {
        "before": before,
        "weighting": weighting,
        "parameters": PARAMS,
        "features": FEATURES,
        "columns": frame.columns,
        "polars": pl.__version__,
        "frame_hash": hashlib.sha256(frame.hash_rows(seed=137).to_numpy().tobytes()).hexdigest(),
    }
    folder = directory / (before.replace("/", "-") + "-" + weighting + "-" + digest(contract)[:20])
    specfile = folder / "model.json"
    if specfile.exists():
        spec = json.loads(specfile.read_text())
        models = {}
        for name, checksum in spec["hashes"].items():
            path = folder / (name + ".txt")
            if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
                raise ValueError("Horizon trial model checksum mismatch")
            models[name] = lgb.Booster(model_file=str(path))
        return {**spec, **models}
    with threadpool_limits(limits=1):
        bundle = fit_models(frame, before, weighting)
    folder.mkdir(parents=True, exist_ok=True)
    spec = {k: v for k, v in bundle.items() if k not in ("minutes", "points", "starter")}
    spec.update(
        before=before, weighting=weighting, parameters=PARAMS, features=FEATURES, hashes={}, contract=contract
    )
    from .models import save_booster_atomic

    for name in ("minutes", "points", "starter"):
        path = folder / (name + ".txt")
        save_booster_atomic(bundle[name].booster_, path)
        spec["hashes"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_json(specfile, spec)
    print("Fitted horizon expert", before, weighting, spec["training_rows"], flush=True)
    return bundle


def reference(store, before, directory):
    feature = store.get("decision_features")
    source = store.artifact_path(feature["path"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != feature["hash"]:
        raise ValueError("Horizon reference feature checksum mismatch")
    contract = {
        "before": before,
        "feature_hash": feature["hash"],
        "parameters": PARAMETERS,
        "features": NAMES,
    }
    path = directory / ("reference-" + before.replace("/", "-") + "-" + digest(contract)[:20])
    if (path / "model.json").exists():
        spec = json.loads((path / "model.json").read_text())
        for name in ("minutes", "points", "starter"):
            if hashlib.sha256((path / (name + ".txt")).read_bytes()).hexdigest() != spec["hashes"][name]:
                raise ValueError("Horizon reference model checksum mismatch")
        return {
            **spec,
            **{
                name: lgb.Booster(model_file=str(path / (name + ".txt")))
                for name in ("minutes", "points", "starter")
            },
        }
    core = pl.read_parquet(source).filter(pl.col("deadline_group") < before)
    groups = sorted(core["deadline_group"].unique().to_list())
    cut = groups[-8]
    with threadpool_limits(limits=1):
        bundle = fit_components(
            core.filter(pl.col("deadline_group") < cut), core.filter(pl.col("deadline_group") >= cut)
        )
    path.mkdir(parents=True, exist_ok=True)
    from .models import save_booster_atomic

    hashes = {}
    for name in ("minutes", "points", "starter"):
        weight = path / (name + ".txt")
        save_booster_atomic(bundle[name].booster_, weight)
        hashes[name] = hashlib.sha256(weight.read_bytes()).hexdigest()
    atomic_json(
        path / "model.json", {"curves": bundle["curves"], "cut": cut, "hashes": hashes, "contract": contract}
    )
    return bundle


def prediction(bundle, frame):
    """The same predictor accepts fitted estimators and immutable text boosters."""
    return predict(bundle, frame)


def predict_period(frame, season, baseline, experts):
    data = frame.filter(pl.col("season") == season)
    x = data.select(NAMES).to_numpy()
    dc = data["dc_prior"].to_list()
    with threadpool_limits(limits=1):
        _, rp, rc = predict_models(baseline, x, dc)
        # The reference's future injury prior is its declared recovery fallback.
        availability = 1 - (1 - data["availability"].to_numpy()) * np.exp(-data["lead_weeks"].to_numpy() / 5)
        rp[:, 1:] *= availability[:, None]
        rp[:, 0] = 1 - rp[:, 1:].sum(1)
        distributions = {"reference": (rp, rc, np.zeros(len(data)))}
        for name, bundle in experts.items():
            _, p, c = prediction(bundle, data)
            for weight in (0.5, 1.0):
                distributions[f"{name}_{weight:g}"] = mix_distributions(
                    (rp, rc, np.zeros(len(data))), (p, c, np.zeros(len(data))), weight
                )
    signals = {}
    metrics: dict = {}
    priors = np.asarray(dc)
    actual = data["target"].to_numpy()
    for name, distribution in distributions.items():
        mean = (distribution[0] * (distribution[1] + 2 * priors)).sum(1)
        hard = data["selectable"].to_numpy() == 0
        mean[hard] = 0
        one = data.filter(pl.col("lead_weeks") == 0).select("event", "player_id", "fixture_id")
        first = mean[data["lead_weeks"].to_numpy() == 0]
        aggregate: dict = defaultdict(lambda: defaultdict(float))
        for row, value in zip(one.iter_rows(named=True), first, strict=True):
            aggregate[row["event"]][row["player_id"]] += float(value)
        signals[name] = {event: dict(values) for event, values in aggregate.items()}
        metrics[name] = {}
        for label, mask in (
            ("immediate", data["lead_weeks"].to_numpy() == 0),
            ("future", data["lead_weeks"].to_numpy() > 0),
            (
                "relevant_immediate",
                (data["lead_weeks"].to_numpy() == 0) & (data["now_cost"].to_numpy() >= 60),
            ),
        ):
            metrics[name][label] = score_metrics(actual[mask], mean[mask])
        appeared = data["minutes_target"].to_numpy() > 0
        pany = np.clip(1 - distribution[0][:, 0], 1e-7, 1 - 1e-7)
        metrics[name]["participation"] = {
            "brier": float(np.mean((pany - appeared) ** 2)),
            "log_loss": float(-np.mean(appeared * np.log(pany) + (1 - appeared) * np.log(1 - pany))),
        }
    return signals, metrics


def legal_period(store, season, events, signals, originals, packets, directory, seconds=2):
    packet = packets[season, events[0]]
    packet = {**packet, "bootstrap": player_bootstrap(packet["bootstrap"])}
    players, _ = deadline_rows([], packet["bootstrap"], [], season, events[0], packet["deadline"])
    rules = Rules.from_bootstrap(packet["bootstrap"])
    seed, _ = initial_account(packet, players, rules, seconds)
    truth: dict = defaultdict(lambda: defaultdict(float))
    minutes: dict = defaultdict(lambda: defaultdict(int))
    known_at = {}
    for row in originals:
        if row["season"] == season:
            known_at[int(row["fixture"])] = (moment(row["kickoff_time"]) + timedelta(hours=4)).timestamp()
        if row["season"] == season and int(row["round"]) in events:
            event, p = int(row["round"]), int(row["element"])
            truth[event][p] += float(row.get("total_points") or 0)
            minutes[event][p] += int(row.get("minutes") or 0)
    outcomes = []
    for name in [*signals, "ownership_no_hits"]:
        state = seed.model_copy(deep=True)
        for event in events:
            packet = packets[season, event]
            packet = {**packet, "bootstrap": player_bootstrap(packet["bootstrap"])}
            players, _ = deadline_rows([], packet["bootstrap"], [], season, event, packet["deadline"])
            rules = Rules.from_bootstrap(packet["bootstrap"])
            state = reprice(state, players, rules)
            state.known_at = packet["deadline"]
            scheduled = {
                p["id"]
                for p in packet["bootstrap"]["elements"]
                if any(
                    f.get("event") == event
                    and p["team"] in (f["team_h"], f["team_a"])
                    and not f.get("finished")
                    and known_at.get(int(f["id"]), float("inf")) >= moment(packet["deadline"]).timestamp()
                    for f in packet["fixture_proof"]["fixtures"]
                )
            }
            means = (
                baseline_signals(players, "ownership", scheduled)
                if name == "ownership_no_hits"
                else {p: signals[name][event].get(p, 0.0) for p in players}
            )
            identity = digest(
                {
                    "input": packet["id"],
                    "state": state.model_dump(),
                    "signals": means,
                    "seconds": seconds,
                    "version": "legal-policy-v2",
                }
            )
            path = directory / (season.replace("/", "-") + f"-{events[0]}-{name}-gw{event}.json")
            if path.exists():
                cached = json.loads(path.read_text())
            else:
                cached = {}
            if cached.get("key") == identity:
                from .domain import Action
                from .rules import transition

                action = Action.model_validate(cached["action"])
                following, ledger = transition(state, action, players, rules)
                solver = cached["solver"]
            else:
                choice = solve_action(
                    state,
                    players,
                    rules,
                    means,
                    max_transfers=(
                        max(0, state.free_transfers - state.transfers_made)
                        if name == "ownership_no_hits"
                        else 15
                    ),
                    seconds=seconds,
                )
                action, following, ledger, solver = choice.action, choice.state, choice.ledger, choice.solver
                atomic_json(
                    path,
                    {
                        "key": identity,
                        "at": now(),
                        "state": state.model_dump(),
                        "action": action.model_dump(),
                        "solver": solver,
                        "input_id": packet["id"],
                    },
                )
            assert action.lineup is not None
            score = (
                score_gameweek(truth[event], minutes[event], action.lineup, players, rules) - ledger["hit"]
            )
            outcomes.append(
                {
                    "season": season,
                    "event": event,
                    "policy": name,
                    "net_points": score,
                    "hit": ledger["hit"],
                    "action": action.model_dump(),
                    "solver": solver,
                }
            )
            state = following
            state.points += int(score)
        print(
            "Horizon legal policy",
            season,
            events[0],
            name,
            sum(r["net_points"] for r in outcomes if r["policy"] == name),
            flush=True,
        )
    return outcomes


def run_trial(store):
    from .fixture_horizon import FEATURES

    path = build_data(store)
    frame = pl.read_parquet(path)
    key = digest(
        {
            "data": str(path),
            "params": PARAMS,
            "weights": ["uniform", "squad"],
            "code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    )
    directory = store.root / "models" / ("horizon-trial-" + key[:20])
    directory.mkdir(parents=True, exist_ok=True)
    protocol = {
        "at": now(),
        "fit_before": "2024/25-GW01",
        "selection": "2024/25 GW1-19 legal net returns",
        "acceptance": "2024/25 GW20-38 then frozen-configuration 2025/26 replication",
        "current_season": "diagnostic only; never selects weights",
        "features": FEATURES,
        "leads": "0,1,2,3,5,8,12,18 announced weeks",
        "purging": "Training label completion precedes calibration source start; all calibration labels precede test start",
        "candidate_weights": [0.5, 1.0],
        "families": ["uniform", "squad"],
        "parameters": PARAMS,
        "acceptance_gate": "Nonnegative legal-return gains in each old-year acceptance period; positive pooled deadline-block gain interval; no immediate or future RMSE degradation versus matched reference",
        "historical_scope": "Ordinary 15-player policies; the 2024 Assistant Manager chip is excluded for every policy",
    }
    atomic_json(directory / "protocol.json", protocol)
    experts = {name: fitted(store, frame, "2024/25-GW01", name, directory) for name in ("uniform", "squad")}
    baseline = reference(store, "2024/25-GW01", directory)
    signals, metrics = predict_period(frame, "2024/25", baseline, experts)
    originals, _, _ = canonical_rows(store)
    packets = replay_packets(store)
    selection = legal_period(store, "2024/25", list(range(1, 20)), signals, originals, packets, directory)
    totals = {name: sum(r["net_points"] for r in selection if r["policy"] == name) for name in signals}
    selected = max((name for name in signals if name != "reference"), key=lambda name: totals[name])
    atomic_json(directory / "selection.json", {"selected": selected, "totals": totals, "protocol": protocol})
    acceptance = legal_period(
        store,
        "2024/25",
        list(range(20, 39)),
        {"reference": signals["reference"], selected: signals[selected]},
        originals,
        packets,
        directory,
    )
    diff = [
        next(r["net_points"] for r in acceptance if r["event"] == g and r["policy"] == selected)
        - next(r["net_points"] for r in acceptance if r["event"] == g and r["policy"] == "reference")
        for g in range(20, 39)
    ]
    report = {
        "at": now(),
        "protocol": protocol,
        "selected": selected,
        "selection_totals": totals,
        "forecast_metrics": metrics,
        "acceptance_totals": {
            name: sum(r["net_points"] for r in acceptance if r["policy"] == name)
            for name in (selected, "reference", "ownership_no_hits")
        },
        "return_delta_95ci": paired_interval(diff),
        "weeks": acceptance,
        "promoted": False,
        "directory": str(directory),
    }
    # Configuration is already selected and saved before the replication year.
    family, weight_text = selected.rsplit("_", 1)
    replication_expert = fitted(store, frame, "2025/26-GW01", family, directory)
    replication_reference = reference(store, "2025/26-GW01", directory)
    replication_signals, replication_metrics = predict_period(
        frame, "2025/26", replication_reference, {family: replication_expert}
    )
    replication = legal_period(
        store,
        "2025/26",
        list(range(1, 39)),
        {name: replication_signals[name] for name in ("reference", selected)},
        originals,
        packets,
        directory,
    )
    replication_diff = [
        next(r["net_points"] for r in replication if r["event"] == g and r["policy"] == selected)
        - next(r["net_points"] for r in replication if r["event"] == g and r["policy"] == "reference")
        for g in range(1, 39)
    ]
    report["replication_totals"] = {
        name: sum(r["net_points"] for r in replication if r["policy"] == name)
        for name in (selected, "reference", "ownership_no_hits")
    }
    report["replication_metrics"] = replication_metrics
    report["pooled_return_delta_95ci"] = paired_interval(diff + replication_diff)
    report["ordinary_gate_passed"] = bool(
        sum(diff) >= 0
        and sum(replication_diff) >= 0
        and report["pooled_return_delta_95ci"][0] > 0
        and all(
            metric[selected][h]["rmse"] <= metric["reference"][h]["rmse"]
            for metric in (metrics, replication_metrics)
            for h in ("immediate", "future")
        )
    )
    report["weeks"].extend(replication)
    report["id"] = store.put("horizon_trial", report)
    atomic_json(Path("artifacts/evaluations/fixture-horizon-trial.json"), report)
    return report
