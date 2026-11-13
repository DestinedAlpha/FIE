"""Deadline-snapshot fixture forecasts with purged future-label windows.

Unlike repeating a next-match profile, lead time is a training input. Every
registered player is represented, including nonappearances. This is a challenger
until forecast, distribution and legal-policy acceptance all pass.
"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from .availability import moment
from .consensus import replay_packets
from .decision import NAMES, canonical_rows, minute_band
from .features import number
from .models import curve_apply, curve_fit
from .replay import deadline_rows
from .storage import digest

LEADS = (0, 1, 2, 3, 5, 8, 12, 18)
DATA_CONTRACT = "base-target-missing-strength-v6"
# Exactly the published v5 transform; other contracts rebuild from sealed inputs.
LEGACY_V5_TRANSFORM = "14f778c158820a55dbdddffa2fd277a0038fde777e44d7d84ad94d2065b0f55c"
RAW = (
    "now_cost",
    "selected_by_percent",
    "form",
    "points_per_game",
    "total_points",
    "minutes",
    "starts",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "saves",
    "bonus",
    "bps",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "expected_goals_per_90",
    "expected_assists_per_90",
    "expected_goal_involvements_per_90",
    "expected_goals_conceded_per_90",
    "saves_per_90",
    "clean_sheets_per_90",
    "starts_per_90",
    "goals_conceded_per_90",
    "clearances_blocks_interceptions",
    "tackles",
    "recoveries",
    "defensive_contribution",
    "transfers_in_event",
    "transfers_out_event",
    "cost_change_start",
    "cost_change_event",
    "penalties_order",
    "direct_freekicks_order",
    "corners_and_indirect_freekicks_order",
    "chance_of_playing_next_round",
)
EXTRA = (
    "lead_weeks",
    "lead_days",
    "source_week",
    "target_week",
    "fixture_count",
    "home_difficulty",
    "away_difficulty",
    "official_own_attack",
    "official_opponent_defence",
    "rest_scheduled",
    "ep_next",
    "ep_available",
    "selectable",
    "status_a",
    "status_d",
    "status_i",
    "status_s",
    "status_u",
)
FEATURES = (*NAMES, *RAW, *("observed_" + n for n in RAW), *EXTRA)
PARAMS: dict[str, Any] = {
    "n_estimators": 260,
    "num_leaves": 24,
    "learning_rate": 0.035,
    "min_child_samples": 100,
    "reg_lambda": 20,
    "verbosity": -1,
    "n_jobs": 1,
    "random_state": 137,
}


def player_bootstrap(bootstrap):
    """The 2024 Assistant Manager asset is outside the ordinary 15-player squad."""
    return {
        **bootstrap,
        "chips": bootstrap.get("chips", []),
        "elements": [p for p in bootstrap["elements"] if p["element_type"] in (1, 2, 3, 4)],
        "element_types": [p for p in bootstrap["element_types"] if p["id"] in (1, 2, 3, 4)],
    }


def feature_vector(player, row, fixture, schedule, teams, source_event, cutoff):
    target = int(fixture["event"])
    core = list(row["x"])
    core[NAMES.index("season_week")] = target
    kickoff = fixture.get("kickoff_time")
    previous = [
        f["kickoff_time"]
        for f in schedule
        if f.get("kickoff_time")
        and kickoff
        and player.team in (f["team_h"], f["team_a"])
        and f["kickoff_time"] < kickoff
    ]
    rest = (
        min(30.0, max(0.0, (moment(kickoff) - moment(max(previous))).total_seconds() / 86400))
        if previous
        else 7.0
    )
    core[NAMES.index("rest_days")] = rest
    raw = player.raw
    fields, observed = [], []
    for name in RAW:
        known = raw.get(name) not in (None, "")
        fields.append(float(raw[name]) if known else np.nan)
        observed.append(float(known))
    home = player.team == fixture["team_h"]
    own, opp = teams[player.team], teams[fixture["team_a"] if home else fixture["team_h"]]
    count = sum(
        f.get("event") == target and not f.get("finished") and player.team in (f["team_h"], f["team_a"])
        for f in schedule
    )
    ep = raw.get("ep_next")
    use_ep = target == source_event and ep not in (None, "")
    extra = [
        target - source_event,
        (moment(kickoff) - moment(cutoff)).total_seconds() / 86400
        if kickoff
        else 7.0 * (target - source_event),
        source_event,
        target,
        count,
        number(fixture, "team_h_difficulty", 3),
        number(fixture, "team_a_difficulty", 3),
        positive_strength(own, "strength_attack_home" if home else "strength_attack_away"),
        positive_strength(opp, "strength_defence_away" if home else "strength_defence_home"),
        rest,
        float(ep) / max(1, count) if use_ep else np.nan,
        float(use_ep),
        float(player.selectable),
        *(float(player.status == s) for s in ("a", "d", "i", "s", "u")),
    ]
    return core + fields + observed + extra


def positive_strength(team, field):
    """Zero/absent provider ratings are unknown, not an extremely weak club."""
    value = number(team, field, np.nan)
    return value if np.isfinite(value) and value > 0 else np.nan


def repair_cached_data(store, path, source, packets, transform, signature):
    """Apply only the two reviewed column corrections to an exact v5 cache.

    Reuse requires identical sealed packets, labels, transform and file checksum.
    Unknown old contracts or unobserved ratings rebuild through the normal path.
    """
    import json

    from .storage import atomic_json

    if not store.list("future_horizon_model", 1):
        return False
    previous = store.get("future_horizon_model")
    if not previous.get("data_path") or not previous.get("data_hash"):
        return False
    old_path = store.artifact_path(previous["data_path"])
    lineage_path = old_path.parent / "lineage.json"
    if not lineage_path.is_file():
        return False
    lineage = json.loads(lineage_path.read_text())
    expected = digest(
        {
            "packets": [p["id"] for _, p in sorted(packets.items())],
            "labels": source["hash"],
            "transform": lineage["transform"],
        }
    )
    if digest(lineage["transform"]) != LEGACY_V5_TRANSFORM or lineage["label_hash"] != source["hash"]:
        return False
    ratings = (
        "strength_attack_home",
        "strength_attack_away",
        "strength_defence_home",
        "strength_defence_away",
    )
    if any(
        team.get(k) is None
        for packet in packets.values()
        for team in packet["bootstrap"]["teams"]
        for k in ratings
    ):
        return False  # An old fallback of 1000 cannot be distinguished from a real rating.
    if hashlib.sha256(old_path.read_bytes()).hexdigest() != previous["data_hash"]:
        raise ValueError("Prior future training cache checksum mismatch")
    old = pl.read_parquet(old_path)
    if lineage["signature"] != expected:
        # v5 also hashed the changing, unlabelled upcoming packet. It was never
        # consumed. The exact productive packet set still proves input identity.
        if "input_id" not in old.columns or set(old["input_id"]) != {p["id"] for p in packets.values()}:
            return False
    labels = pl.read_parquet(store.artifact_path(source["path"])).select(
        "season", "fixture_id", "player_id", pl.col("base_target").alias("canonical_base")
    )
    repaired = old.join(
        labels, on=["season", "fixture_id", "player_id"], how="left", validate="m:1", maintain_order="left"
    )
    if repaired.filter(pl.col("canonical_base").is_null() & (pl.col("target") != 0)).height:
        raise ValueError("A nonzero horizon label lacks its canonical scoring target")
    repaired = repaired.with_columns(
        pl.col("canonical_base").fill_null(0).alias("base_target"),
        *[
            pl.when(pl.col(k).is_finite() & (pl.col(k) > 0)).then(pl.col(k)).otherwise(float("nan")).alias(k)
            for k in ("official_own_attack", "official_opponent_defence")
        ],
    ).select(old.columns)
    temporary = path.with_suffix(".temporary.parquet")
    repaired.write_parquet(temporary)
    temporary.replace(path)
    atomic_json(
        path.parent / "lineage.json",
        {
            "signature": signature,
            "transform": transform,
            "label_hash": source["hash"],
            "migrated_from": {"path": str(old_path), "hash": previous["data_hash"]},
        },
    )
    return True


def build_data(store):
    """No later snapshot fields enter an earlier source/target feature row."""
    packets = replay_packets(store)
    originals, _, _ = canonical_rows(store)
    source = store.get("decision_features")
    labels = (
        pl.read_parquet(store.artifact_path(source["path"]))
        .select(
            "season",
            "fixture_id",
            "player_id",
            "target",
            "base_target",
            "minutes_target",
            "start_target",
            "deadline_group",
        )
        .to_dicts()
    )
    # A newly captured upcoming deadline has no finalized future labels. It
    # cannot contribute a row and must not invalidate the entire training cache.
    last_label = max((r["deadline_group"] for r in labels), default="")
    packets = {
        key: packet for key, packet in packets.items() if f"{key[0]}-GW{int(key[1]):02d}" <= last_label
    }
    truth = {(r["season"], int(r["fixture_id"]), int(r["player_id"])): r for r in labels}
    from .observations import finalization_clocks, label_available_at

    finalization = finalization_clocks(originals)
    completed = {}
    for r in originals:
        key = r["season"], int(r["fixture"])
        completed[key] = {
            "end_group": f"{r['season']}-GW{int(r['round']):02d}",
            "kickoff": r["kickoff_time"],
            "known": label_available_at(r, finalization).timestamp(),
        }
    from .decision import context_vector, dc_probability, player_key, strength

    transform = {
        "features": FEATURES,
        "leads": LEADS,
        "data_version": 6,
        "data_contract": DATA_CONTRACT,
        "functions": [
            inspect.getsource(f)
            for f in (
                feature_vector,
                positive_strength,
                deadline_rows,
                context_vector,
                dc_probability,
                player_key,
                strength,
                player_bootstrap,
            )
        ],
        "features_module": hashlib.sha256((Path(__file__).parent / "features.py").read_bytes()).hexdigest(),
    }
    signature = digest(
        {
            "packets": [p["id"] for _, p in sorted(packets.items())],
            "labels": source["hash"],
            "transform": transform,
        }
    )
    directory = store.root / "features" / ("fixture-horizon-" + signature[:20])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "data.parquet"
    if path.exists():
        return path
    if repair_cached_data(store, path, source, packets, transform, signature):
        return path
    cache = store.root / "features" / "fixture-horizon-chunks"
    cache.mkdir(parents=True, exist_ok=True)
    seasons = sorted({s for s, _ in packets})
    history_hashes = {s: digest([r for r in originals if r["season"] <= s]) for s in seasons}
    label_hashes = {s: digest([r for r in labels if r["season"] == s]) for s in seasons}
    chunks = []
    for (season, event), packet in sorted(packets.items()):
        chunk_id = digest(
            {
                "packet": packet["id"],
                "history": history_hashes[season],
                "labels": label_hashes[season],
                "transform": transform,
            }
        )[:20]
        chunk = cache / (season.replace("/", "-") + f"-gw{event}-" + chunk_id + ".parquet")
        if chunk.exists():
            chunks.append(chunk)
            continue
        schedule = []
        cutoff = moment(packet["deadline"]).timestamp()
        for fixture in packet["fixture_proof"]["fixtures"]:
            prior = completed.get((season, int(fixture["id"])))
            if prior is not None and prior["known"] < cutoff:
                fixture = {
                    **fixture,
                    "finished": True,
                    "kickoff_time": prior["kickoff"],
                    "event": int(prior["end_group"].split("GW")[1]),
                }
            schedule.append(fixture)
        chosen = [
            f
            for f in schedule
            if f.get("event") is not None
            and not f.get("finished")
            and f["event"] - event in LEADS
            and (season, int(f["id"])) in completed
        ]
        if not chosen:
            continue
        # Reuse the core historical transform once for all selected fixtures.
        players, rows = deadline_rows(
            originals,
            player_bootstrap(packet["bootstrap"]),
            [{**f, "event": event} for f in chosen],
            season,
            event,
            packet["deadline"],
        )
        fixtures = {f["id"]: f for f in chosen}
        teams = {t["id"]: t for t in packet["bootstrap"]["teams"]}
        x, meta = [], []
        for row in rows:
            player = players[row["player_id"]]
            fixture = fixtures[row["fixture_id"]]
            known = completed[season, fixture["id"]]
            label = truth.get((season, fixture["id"], player.id))
            # A player with no appearance row in a completed fixture earns zero.
            actual, base, minutes, start = (
                (
                    label["target"],
                    label["base_target"],
                    label["minutes_target"],
                    label["start_target"],
                )
                if label
                else (0.0, 0.0, 0.0, 0.0)
            )
            x.append(feature_vector(player, row, fixture, schedule, teams, event, packet["deadline"]))
            meta.append(
                {
                    "season": season,
                    "event": event,
                    "source_group": f"{season}-GW{event:02d}",
                    "label_end": known["end_group"],
                    "label_known": known["known"],
                    "source_deadline": moment(packet["deadline"]).timestamp(),
                    "target_event": fixture["event"],
                    "fixture_id": fixture["id"],
                    "player_id": player.id,
                    "input_id": packet["id"],
                    "target": float(actual),
                    "base_target": float(base),
                    "minutes_target": float(minutes),
                    "start_target": None if start is None else float(start),
                    "dc_prior": row["dc_probability"] if season >= "2025/26" else [0.0] * 4,
                    "availability": row["availability"],
                }
            )
        matrix = np.asarray(x, dtype=np.float32)
        frame = pl.DataFrame(matrix, schema=list(FEATURES)).hstack(
            pl.DataFrame(meta, infer_schema_length=None)
        )
        temporary = chunk.with_suffix(".temporary.parquet")
        frame.write_parquet(temporary)
        temporary.replace(chunk)
        chunks.append(chunk)
        print("Built horizon snapshot", season, event, len(frame), flush=True)
    temporary = path.with_suffix(".temporary.parquet")
    pl.concat([pl.scan_parquet(p) for p in chunks], how="vertical_relaxed").sink_parquet(temporary)
    temporary.replace(path)
    from .storage import atomic_json

    atomic_json(
        directory / "lineage.json",
        {
            "signature": signature,
            "transform": transform,
            "label_hash": source["hash"],
            "chunks": [str(p) for p in chunks],
        },
    )
    return path


def partitions(frame, before, before_clock=None, calibration_groups=8):
    clocks = dict(frame.select("source_group", "source_deadline").unique().iter_rows())
    boundary = before_clock if before_clock is not None else clocks.get(before)
    if boundary is None:
        raise ValueError("A missing target deadline requires an explicit information cutoff")
    eligible = frame.filter(
        (pl.col("source_group") < before)
        & (pl.col("label_end") < before)
        & (pl.col("label_known") < boundary)
    )
    groups = sorted(eligible["source_group"].unique().to_list())
    if calibration_groups < 1 or len(groups) <= calibration_groups:
        raise ValueError("Future expert requires fitting history before its calibration deadlines")
    cut = groups[-calibration_groups]
    train = eligible.filter((pl.col("label_end") < cut) & (pl.col("label_known") < clocks[cut]))
    cal = eligible.filter(pl.col("source_group") >= cut)
    return train, cal, cut


def fit_models(frame, before, weighting, before_clock=None, calibration_groups=8):
    train, cal, cut = partitions(frame, before, before_clock, calibration_groups)
    x, xc = train.select(FEATURES).to_numpy(), cal.select(FEATURES).to_numpy()

    def weights(data):
        ownership = data["selected_by_percent"].fill_nan(0).fill_null(0).to_numpy()
        return np.ones(len(data)) if weighting == "uniform" else 1 + 4 * np.clip(ownership / 20, 0, 1)

    weight = weights(train)
    band = minute_band(train["minutes_target"].to_numpy())
    minute = lgb.LGBMClassifier(**PARAMS).fit(x, band, sample_weight=weight)
    present = train["minutes_target"].to_numpy() > 0
    point = lgb.LGBMRegressor(**PARAMS).fit(
        np.column_stack([x[present], band[present]]),
        train["base_target"].to_numpy()[present],
        sample_weight=weight[present],
    )
    observed = train["start_target"].is_not_null().to_numpy()
    starter = lgb.LGBMClassifier(**PARAMS).fit(
        x[observed],
        train["start_target"].fill_null(0).to_numpy()[observed] > 0,
        sample_weight=weight[observed],
    )
    raw = np.asarray(minute.predict_proba(xc))
    bands = minute_band(cal["minutes_target"].to_numpy())
    curves = {str(k): curve_fit(raw[:, k], bands == k) for k in range(4)}
    known = cal["start_target"].is_not_null().to_numpy()
    curves["start"] = curve_fit(
        np.asarray(starter.predict_proba(xc[known]))[:, 1],
        cal["start_target"].fill_null(0).to_numpy()[known] > 0,
    )
    return {
        "minutes": minute,
        "data_contract": DATA_CONTRACT,
        "points": point,
        "starter": starter,
        "curves": curves,
        "fit_max_label": train["label_end"].max(),
        "cal_max_label": cal["label_end"].max(),
        "cut": cut,
        "training_rows": len(train),
        "calibration_rows": len(cal),
    }


def predict(bundle, frame):
    x = frame.select(FEATURES).to_numpy()
    minute = bundle["minutes"]
    raw = np.asarray(
        minute.predict(x, num_threads=1) if isinstance(minute, lgb.Booster) else minute.predict_proba(x)
    )
    p = np.column_stack([curve_apply(raw[:, k], bundle["curves"][str(k)]) for k in range(4)])
    empty = p.sum(1) <= 1e-12
    p[empty] = raw[empty]
    p /= np.maximum(p.sum(1, keepdims=True), 1e-12)
    c = np.zeros_like(p)
    for k in range(1, 4):
        matrix = np.column_stack([x, np.full(len(x), k)])
        point = bundle["points"]
        c[:, k] = (
            point.predict(matrix, num_threads=1) if isinstance(point, lgb.Booster) else point.predict(matrix)
        )
    dc = np.asarray(frame["dc_prior"].to_list())
    mean = (p * (c + 2 * dc)).sum(1)
    hard = (frame["selectable"].to_numpy() == 0) | (
        (frame["lead_weeks"].to_numpy() == 0) & (frame["chance_of_playing_next_round"].to_numpy() == 0)
    )
    p[hard] = [1, 0, 0, 0]
    mean[hard] = 0
    return mean, p, c
