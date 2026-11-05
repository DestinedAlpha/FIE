"""Weekly budgeted teams with deadline-pinned registries and fixture schedules.

Every team is selected before the scorer reads the target week's outcomes.
This is a reconstructed historical experiment, not a prospective track record.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import lzma
from collections import defaultdict
from datetime import timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import urlencode

import httpx
import numpy as np
import polars as pl
from scipy.special import ndtr
from threadpoolctl import threadpool_limits

from .availability import moment
from .context_archive import MANIFEST, check_proof
from .decision import (
    BAND_MINUTES,
    canonical_rows,
    context_vector,
    dc_probability,
    fit_components,
    minute_band,
    player_key,
    predict_models,
    residual_contract,
    sample_point_outcomes,
    strength,
)
from .domain import Lot, Player, TeamState
from .optimization import select_squad
from .rules import Rules, best_lineup, optimize_bench, score_gameweek, validate_squad
from .storage import atomic_json, digest, now
from .transport import PublicTransport


def archive_integer(value):
    """CSV archives sometimes serialize nullable integer columns as '1.0'."""
    if value in (None, "", "None", "nan"):
        return None
    try:
        number = Decimal(str(value))
        if not number.is_finite() or number != number.to_integral_value():
            raise ValueError("Non-integral archive identifier")
        return int(number)
    except InvalidOperation as error:
        raise ValueError("Invalid archive identifier") from error


async def capture_replay_inputs(store, season, events):
    """Fetch immutable pre-deadline Git versions; never use today's prices."""
    manifest = json.loads(MANIFEST.read_text())
    transport = PublicTransport(store, 2, 0.25)
    results = []
    async with httpx.AsyncClient(timeout=35, follow_redirects=False) as client:
        for event in events:
            row = next(
                (r for r in manifest["snapshots"] if r["season"] == season and r["event"] == event), None
            )
            if row is None:
                # Locally frozen observations cover deadlines newer than the
                # upstream Git manifest. Both sources must precede the deadline.
                context, _ = store.latest(f"context_live_{season.replace('/', '-')}_{event}")
                deadline = context["deadline"]
                bootstrap, bm = store.latest("bootstrap", cutoff=deadline)
                fixtures, fm = store.latest("fixtures", cutoff=deadline)
                from .ingestion import season_of

                if (
                    season_of(bootstrap) != season
                    or next(e["deadline_time"] for e in bootstrap["events"] if e["id"] == event) != deadline
                    or any(moment(m["captured_at"]) >= moment(deadline) for m in (bm, fm))
                ):
                    raise ValueError("Local replay sources do not precede the requested deadline")
                result = {
                    "season": season,
                    "event": event,
                    "deadline": deadline,
                    "bootstrap": bootstrap,
                    "bootstrap_hash": bm["hash"],
                    "bootstrap_proof": {
                        "source_captured_at": bm["captured_at"],
                        "temporal_track": "Locally observed before deadline",
                    },
                    "fixture_proof": {
                        "fixtures": fixtures,
                        "source_captured_at": fm["captured_at"],
                        "file_sha256": fm["hash"],
                        "temporal_track": "Locally observed before deadline",
                    },
                }
                result["id"] = store.put("replay_inputs", result)
                results.append(result)
                continue
            check_proof(row)
            source = f"replay_bootstrap_{season.replace('/', '-')}_{event}"
            try:
                bootstrap, bootstrap_meta = store.latest(source)
            except ValueError:
                compressed = await transport.get(client, row["url"], source, binary=True, max_bytes=2_000_000)
                if hashlib.sha256(compressed).hexdigest() != row["sha256"]:
                    raise ValueError("Replay bootstrap checksum mismatch")
                decoder = lzma.LZMADecompressor()
                raw = decoder.decompress(compressed, max_length=16_000_000)
                if not decoder.eof:
                    raise ValueError("Replay bootstrap exceeds decompression budget")
                bootstrap = json.loads(raw)
                bootstrap_meta = store.snapshot(source, row["url"], bootstrap, row["source_captured_at"])
            fixture_source = f"replay_fixtures_{season.replace('/', '-')}_{event}"
            try:
                fixture_data, _ = store.latest(fixture_source)
            except ValueError:
                path = "data/" + season.replace("/", "-") + "/fixtures.csv"
                until = (moment(row["deadline"]) - timedelta(seconds=1)).isoformat()
                url = "https://api.github.com/repos/vaastav/Fantasy-Premier-League/commits?" + urlencode(
                    {"path": path, "until": until, "per_page": 1}
                )
                commits = await transport.get(client, url, fixture_source + "_commit")
                if not commits:
                    raise ValueError("No pre-deadline fixture archive commit")
                commit = commits[0]
                stamp = commit["commit"]["committer"]["date"]
                if moment(stamp) >= moment(row["deadline"]):
                    raise ValueError("Fixture schedule was not committed before deadline")
                fixture_url = (
                    f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/{commit['sha']}/{path}"
                )
                text = await transport.get(
                    client, fixture_url, fixture_source, json_body=False, max_bytes=2_000_000
                )
                fixtures = []
                for r in csv.DictReader(io.StringIO(text)):
                    fixtures.append(
                        {
                            "id": archive_integer(r["id"]),
                            "event": archive_integer(r.get("event")),
                            "team_h": archive_integer(r["team_h"]),
                            "team_a": archive_integer(r["team_a"]),
                            "kickoff_time": r.get("kickoff_time") or None,
                        }
                    )
                fixture_data = {
                    "fixtures": fixtures,
                    "commit": commit["sha"],
                    "git_committed_at": stamp,
                    "source_url": fixture_url,
                    "file_sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
                store.snapshot(fixture_source, fixture_url, fixture_data, stamp)
            result = {
                "season": season,
                "event": event,
                "deadline": row["deadline"],
                "bootstrap": bootstrap,
                "bootstrap_hash": bootstrap_meta["hash"],
                "bootstrap_proof": row,
                "fixture_proof": fixture_data,
            }
            result["id"] = store.put("replay_inputs", result)
            results.append(result)
    return results


def capture_observed_inputs(store, bootstrap, fixtures):
    """Retain a real predeadline registry and schedule for later learning."""
    from .ingestion import season_of

    event = next((e for e in bootstrap["events"] if e.get("is_next")), None)
    if event is None or moment(now()) >= moment(event["deadline_time"]):
        return None
    _, bm = store.latest("bootstrap", content_hash=digest(bootstrap))
    _, fm = store.latest("fixtures", content_hash=digest(fixtures))
    deadline = event["deadline_time"]
    if any(moment(m["captured_at"]) >= moment(deadline) for m in (bm, fm)):
        raise ValueError("Observed replay sources must precede their target deadline")
    body = {
        "season": season_of(bootstrap),
        "event": event["id"],
        "deadline": deadline,
        "bootstrap": bootstrap,
        "bootstrap_hash": bm["hash"],
        "bootstrap_proof": {
            "source_captured_at": bm["captured_at"],
            "temporal_track": "Locally observed before deadline",
        },
        "fixture_proof": {
            "fixtures": fixtures,
            "source_captured_at": fm["captured_at"],
            "file_sha256": fm["hash"],
            "temporal_track": "Locally observed before deadline",
        },
    }
    return body | {"id": store.put("replay_inputs", body)}


def deadline_rows(originals, bootstrap, schedule, season, event, deadline):
    """The same core feature transform as serving, using only completed past facts."""
    players = {
        p["id"]: Player(
            id=p["id"],
            code=p["code"],
            season=season,
            name=p["first_name"] + " " + p["second_name"],
            web_name=p["web_name"],
            team=p["team"],
            position=p["element_type"],
            price=p["now_cost"],
            selectable=p.get("can_select", True) and not p.get("removed", False),
            ownership=float(p.get("selected_by_percent") or 0),
            status=p["status"],
            chance=p["chance_of_playing_next_round"] / 100
            if p.get("chance_of_playing_next_round") is not None
            else None,
            news=p.get("news") or "",
            raw=p,
        )
        for p in bootstrap["elements"]
    }
    names = {t["id"]: t["name"] for t in bootstrap["teams"]}
    histories = defaultdict(list)
    clubs = defaultdict(list)
    seen = set()
    for r in originals:
        if not r.get("kickoff_time") or moment(r["kickoff_time"]) + timedelta(hours=4) >= moment(deadline):
            continue
        if (r["season"], int(r["round"])) >= (season, event):
            continue
        histories[player_key(r)].append(r)
        key = (r["season"], r["fixture"])
        if key not in seen:
            seen.add(key)
            home, away = (r["team"], r["opponent_name"]) if r["was_home"] else (r["opponent_name"], r["team"])
            clubs[home].append((float(r["team_h_score"]), float(r["team_a_score"])))
            clubs[away].append((float(r["team_a_score"]), float(r["team_h_score"])))
    output = []
    for fixture in schedule:
        if fixture.get("event") != event:
            continue
        for player in players.values():
            if player.team not in (fixture["team_h"], fixture["team_a"]):
                continue
            home = player.team == fixture["team_h"]
            past = histories["code", player.code]
            opponent = names[fixture["team_a"] if home else fixture["team_h"]]
            x, profile = context_vector(
                player.position,
                home,
                past,
                fixture.get("kickoff_time"),
                strength(clubs[names[player.team]]),
                strength(clubs[opponent]),
                event,
            )
            availability = (
                player.chance if player.chance is not None else (1 if player.status == "a" else 0.5)
            )
            if not player.selectable:
                availability = 0
            output.append(
                {
                    "player_id": player.id,
                    "fixture_id": fixture["id"],
                    "team": player.team,
                    "position": player.position,
                    "x": x,
                    "dc_probability": dc_probability(player.position, profile, BAND_MINUTES),
                    "availability": availability,
                    "kickoff_time": fixture.get("kickoff_time"),
                }
            )
    return players, output


def fit_deadline(store, frame, deadline, group, directory):
    # Final labels are reconstructed later, but the cutoff excludes every match
    # not completed at the decision time, even if its final GW label is earlier.
    originals, _, _ = canonical_rows(store)
    completed = {
        (r["season"], int(r["fixture"]))
        for r in originals
        if r.get("kickoff_time") and moment(r["kickoff_time"]) + timedelta(hours=4) < moment(deadline)
    }
    allowed = np.array(
        [(r["season"], r["fixture_id"]) in completed for r in frame.select("season", "fixture_id").to_dicts()]
    )
    past = frame.filter(pl.Series(allowed)).filter(pl.col("deadline_group") < group)
    groups = sorted(past["deadline_group"].unique().to_list())
    if len(groups) < 12:
        raise ValueError("Insufficient prior deadlines for weekly refitting")
    cut = groups[-8]
    train = past.filter(pl.col("deadline_group") < cut)
    cal = past.filter(pl.col("deadline_group") >= cut)
    assert train["deadline_group"].max() < cal["deadline_group"].min() < group
    with threadpool_limits(limits=1):
        fitted = fit_components(train, cal, False)
    hashes = {}
    directory.mkdir(parents=True, exist_ok=True)
    for name, model in fitted.items():
        if hasattr(model, "booster_"):
            path = directory / (name + ".txt")
            model.booster_.save_model(str(path))
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    pools, dependence = residual_contract(fitted, cal)
    minute_pools = {
        str(k): np.quantile(
            cal.filter(pl.Series(minute_band(cal["minutes_target"].to_numpy()) == k))[
                "minutes_target"
            ].to_numpy(),
            np.linspace(0, 1, 101),
        ).tolist()
        for k in range(1, 4)
    }
    return (
        fitted,
        pools,
        dependence,
        minute_pools,
        {
            "fit_max_group": train["deadline_group"].max(),
            "calibration_max_group": cal["deadline_group"].max(),
            "training_rows": len(train),
            "calibration_rows": len(cal),
            "model_hashes": hashes,
            "models": {
                name: {"path": str(directory / (name + ".txt")), "hash": value}
                for name, value in hashes.items()
            },
            "curves": fitted["curves"],
            "trained_on_target_week": False,
            "updates_from_previous_weeks": True,
        },
    )


def weekly_worlds(rows, fitted, pools, dependence, minute_pools, players, worlds, seed):
    ids = sorted(players)
    index = {p: i for i, p in enumerate(ids)}
    points = np.zeros((worlds, len(ids)), dtype=np.int16)
    minutes = np.zeros_like(points)
    _, probabilities, conditional = predict_models(
        fitted, np.array([r["x"] for r in rows]), [r["dc_probability"] for r in rows]
    )
    factors = {}
    shared = np.random.default_rng(seed).normal(size=(worlds, len(ids)))
    for i, row in enumerate(rows):
        pid = row["player_id"]
        column = index[pid]
        position = players[pid].position
        rng = np.random.default_rng([seed, row["fixture_id"], pid])
        p = probabilities[i].copy()
        p[1:] *= row["availability"]
        p[0] = 1 - p[1:].sum()
        u = ndtr(0.3 * shared[:, column] + np.sqrt(0.91) * rng.normal(size=worlds))
        role = "defence" if position <= 2 else "attack"
        key = (row["fixture_id"], row["team"], role)
        if key not in factors:
            factors[key] = np.random.default_rng(
                [seed, row["fixture_id"], row["team"], 9107 if role == "defence" else 9109]
            ).normal(size=worlds)
        rho = dependence[role]
        residual = ndtr(np.sqrt(rho) * factors[key] + np.sqrt(1 - rho) * rng.normal(size=worlds))
        values, band, _ = sample_point_outcomes(
            rng, p, conditional[i], row["dc_probability"], pools, position, u, residual
        )
        m = np.zeros(worlds, dtype=np.int16)
        for k in range(1, 4):
            mask = band == k
            pool = minute_pools[str(k)]
            m[mask] = np.rint(np.interp(rng.random(mask.sum()), np.linspace(0, 1, len(pool)), pool)).astype(
                np.int16
            )
        points[:, column] += values
        minutes[:, column] += m
    return ids, points, minutes


def run_replay(store, events=None, worlds=4000, seconds=60, publish=True):
    """Independent £100m squads each week, not eight weeks of free account transfers."""
    import asyncio

    events = events or list(range(1, 10))
    season = store.get("market")["season"]
    if not events or any(g not in range(1, 39) for g in events) or not 100 <= worlds <= 20000:
        raise ValueError("Invalid replay request")
    source_state = store.get("state")["id"]
    inputs = asyncio.run(capture_replay_inputs(store, season, events))
    feature = store.get("decision_features")
    frame = pl.read_parquet(store.artifact_path(feature["path"]))
    originals, _, _ = canonical_rows(store)
    directory = (
        store.root
        / "models"
        / (
            "weekly-replay-"
            + digest(
                {
                    "inputs": [r["id"] for r in inputs],
                    "features": feature["hash"],
                    "worlds": worlds,
                    "seconds": seconds,
                    "protocol": "weekly-refit-v1",
                }
            )[:20]
        )
    )
    weeks = []
    for item in inputs:
        event = item["event"]
        group = f"{season}-GW{event:02d}"
        checkpoint = directory / f"gw{event}.json"
        if checkpoint.exists():
            weeks.append(json.loads(checkpoint.read_text()))
            print("Restored weekly replay", event, flush=True)
            continue
        players, rows = deadline_rows(
            originals, item["bootstrap"], item["fixture_proof"]["fixtures"], season, event, item["deadline"]
        )
        rules = Rules.from_bootstrap(item["bootstrap"])
        fitted, pools, dependence, minute_pools, training = fit_deadline(
            store, frame, item["deadline"], group, directory / f"gw{event}"
        )
        ids, points, minutes = weekly_worlds(
            rows, fitted, pools, dependence, minute_pools, players, worlds, 11
        )
        means = {p: float(points[:, i].mean()) for i, p in enumerate(ids)}
        diagnostics: dict = {}
        squad = select_squad(
            None,
            players,
            rules,
            means,
            seconds=seconds,
            initial_budget=1000,
            bench_weight=0,
            relative_gap=0,
            diagnostics=diagnostics,
        )
        lineup = best_lineup(squad, means, players, rules)
        lineup = optimize_bench(lineup, points, minutes, ids, players, rules)
        state = TeamState(
            season=season,
            event=event,
            bank=1000 - sum(players[p].price for p in squad),
            free_transfers=1,
            lots=[
                Lot(
                    player_id=p,
                    purchase_price=players[p].price,
                    sell_price=players[p].price,
                    quote_market_price=players[p].price,
                )
                for p in squad
            ],
            known_at=item["deadline"],
            status="dated_seed",
        )
        validate_squad(state, players, rules)
        from .simulation import lineup_worlds

        distribution = lineup_worlds(lineup, points, minutes, ids, players, rules)
        frozen = {
            "event": event,
            "season": season,
            "deadline": item["deadline"],
            "players": [players[p].model_dump(exclude={"raw"}) for p in squad],
            "lineup": lineup.model_dump(),
            "bank": state.bank,
            "training": training,
            "solver": diagnostics,
            "input_id": item["id"],
            "forecast_mean": float(distribution.mean()),
            "forecast_10_90": np.quantile(distribution, [0.1, 0.9]).tolist(),
            "protocol": "Full pre-deadline registered market and £100m; XI/captain additive MIP with exact autosub/bench rescoring; no chips",
        }
        decision_id = store.put("weekly_replay_decision", frozen)
        # Only after freezing the decision do target outcomes enter scoring.
        actual: dict[int, float] = defaultdict(float)
        actual_minutes: dict[int, int] = defaultdict(int)
        for r in originals:
            if r["season"] == season and int(r["round"]) == event:
                actual[int(r["element"])] += float(r.get("total_points") or 0)
                actual_minutes[int(r["element"])] += int(r.get("minutes") or 0)
        score = score_gameweek(actual, actual_minutes, lineup, players, rules)
        oracle_info: dict = {}
        oracle = select_squad(
            None,
            players,
            rules,
            actual,
            seconds=seconds,
            initial_budget=1000,
            bench_weight=0,
            relative_gap=0,
            starting_eligible={p for p, m in actual_minutes.items() if m > 0},
            diagnostics=oracle_info,
        )
        oracle_lineup = best_lineup(
            oracle,
            {p: actual.get(p, 0) if actual_minutes.get(p, 0) > 0 else -10000 for p in oracle},
            players,
            rules,
        )
        oracle_score = score_gameweek(actual, actual_minutes, oracle_lineup, players, rules)
        result = {
            **frozen,
            "decision_id": decision_id,
            "actual_points": int(score),
            "hindsight_points": int(oracle_score),
            "hindsight_solver": oracle_info,
            "hindsight_lineup": oracle_lineup.model_dump(),
            "hindsight_players": [
                {"player_id": p, "name": players[p].web_name, "price": players[p].price} for p in oracle
            ],
            "outcome_percentile": float(np.mean(distribution <= score)),
            "realized_player_points": {str(p): actual[p] for p in squad},
            "oracle_scope": "Separate hindsight £100m budgeted XI/captain with all 15 squad constraints. Solver gap retained; never used as a forecast input or next-week feature.",
        }
        atomic_json(checkpoint, result)
        weeks.append(result)
        print("Completed weekly replay", event, "actual", score, "hindsight", oracle_score, flush=True)
    report = {
        "at": now(),
        "season": season,
        "weeks": weeks,
        "total_points": sum(r["actual_points"] for r in weeks),
        "feature_id": feature["id"],
        "worlds": worlds,
        "account_state_unchanged": store.get("state")["id"] == source_state,
        "learning": "Each week refits on completed earlier matches, updates player history and reserves eight earlier deadlines for calibration. Target-week points are read only after its team is frozen.",
        "limits": [
            "Reconstructed historical test, not prospective performance; finalized historical labels and match facts were imported later.",
            "Teams are independent weekly £100m selections, not a continuous account receiving unlimited transfers.",
            "Additive squad-selection optimum is distinct from a global optimum of all nonlinear autosub distributions or an entire season.",
            "Observed minus forecast points includes football randomness, model error and omitted information; it is not identified as pure luck.",
        ],
    }
    report["id"] = store.put("weekly_replay" if publish else "weekly_replay_segment", report)
    atomic_json(Path(f"artifacts/evaluations/gw{min(events)}-gw{max(events)}-replay.json"), report)
    return report


def extend_replay(store, end_event=9, seconds=60):
    """Append newly completed weeks without retraining already frozen decisions."""
    if not 1 <= end_event <= 38:
        raise ValueError("Invalid replay end event")
    try:
        previous = store.get("weekly_replay")
    except ValueError:
        return run_replay(store, list(range(1, end_event + 1)), seconds=seconds)
    if previous["feature_id"] != store.get("decision_features")["id"]:
        raise ValueError("Feature lineage changed; request an explicit new replay instead of mixing runs")
    existing = sorted(w["event"] for w in previous["weeks"])
    if existing != list(range(1, len(existing) + 1)):
        raise ValueError("Only a contiguous replay starting GW1 can be extended")
    if end_event <= existing[-1]:
        return previous
    market = store.get("market")
    finalized = {e["id"] for e in market["events"] if e.get("data_checked")}
    needed = list(range(existing[-1] + 1, end_event + 1))
    if not set(needed) <= finalized:
        raise ValueError("Replay scoring requires finalized target gameweeks")
    segment = run_replay(store, needed, previous["worlds"], seconds, publish=False)
    weeks = previous["weeks"] + segment["weeks"]
    report = {k: v for k, v in segment.items() if k != "id"}
    report.update(
        weeks=weeks,
        total_points=sum(w["actual_points"] for w in weeks),
        parent_replay_id=previous["id"],
        segment_id=segment["id"],
    )
    report["id"] = store.put("weekly_replay", report)
    atomic_json(Path(f"artifacts/evaluations/gw1-gw{end_event}-replay.json"), report)
    return report
