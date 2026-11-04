"""Free dated market adapter; exhaustive de-vigging and gated goal constraints."""

from __future__ import annotations

import csv
import io
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import numpy as np
from pydantic import Field
from scipy.optimize import brentq, minimize
from scipy.stats import poisson

from .domain import Strict
from .shot_data import club_key
from .storage import atomic_json, now
from .transport import PublicTransport


class OddsObservation(Strict):
    fixture_id: int = Field(ge=1)
    known_at: str
    source_url: str = Field(pattern="^https://", max_length=2000)
    bookmaker: str = Field(default="Declared exhaustive market", max_length=150)
    result_odds: list[float] = Field(min_length=3, max_length=3)
    total_2_5_odds: list[float] | None = Field(default=None, min_length=2, max_length=2)


def devig(decimal_odds):
    odds = np.asarray(decimal_odds, dtype=float)
    if len(odds) < 2 or not np.all(np.isfinite(odds)) or np.any(odds <= 1):
        raise ValueError("An exhaustive market needs at least two finite decimal prices above one")
    raw = 1 / odds
    if raw.sum() <= 1:
        return raw / raw.sum()
    exponent = brentq(lambda k: float(np.sum(raw**k) - 1), 1, 20)
    return raw**exponent


def score_markets(home, away):
    n = max(16, int(np.ceil(home + away + 12)))
    h = poisson.pmf(np.arange(n), home)
    a = poisson.pmf(np.arange(n), away)
    joint = h[:, None] * a[None, :]
    joint /= joint.sum()
    return np.array([np.tril(joint, -1).sum(), np.trace(joint), np.triu(joint, 1).sum()]), float(
        poisson.sf(2, home + away)
    )


def goal_constraints(result_odds, total_odds=None):
    target = devig(result_odds)
    over = devig(total_odds)[0] if total_odds is not None else None

    def loss(log_rates):
        rates = np.exp(log_rates)
        p, o = score_markets(*rates)
        value = float(-np.dot(target, np.log(np.clip(p, 1e-9, 1))))
        if over is not None:
            value -= float(over * np.log(max(o, 1e-9)) + (1 - over) * np.log(max(1 - o, 1e-9)))
        return value

    fit = minimize(loss, np.log([1.5, 1.2]), bounds=[(-4, 2.4), (-4, 2.4)], method="L-BFGS-B")
    if not fit.success:
        raise ValueError("Goal-intensity market fit failed")
    rates = np.exp(fit.x)
    fitted, fitted_over = score_markets(*rates)
    discrepancy = max(
        float(np.max(np.abs(fitted - target))), abs(fitted_over - over) if over is not None else 0
    )
    if discrepancy > 0.12:
        raise ValueError("Market constraints conflict with the declared goal model")
    return {
        "home_goals": float(rates[0]),
        "away_goals": float(rates[1]),
        "result_probabilities": target.tolist(),
        "over_2_5_probability": float(over) if over is not None else None,
        "fit_discrepancy": discrepancy,
        "method": "Power de-vig within one exhaustive market; joint Poisson intensity fit",
    }


def scorer_rate(yes_no_odds, minutes_if_available, p_available=1):
    if not 0 < minutes_if_available <= 120 or not 0 < p_available <= 1:
        raise ValueError("Scorer conversion requires explicit minutes and availability")
    probability = float(devig(yes_no_odds)[0])
    if probability >= p_available:
        raise ValueError("Scorer probability is incompatible with availability")
    return float(-np.log1p(-probability / p_available) * 90 / minutes_if_available)


def import_market(store, record):
    from .ingestion import load_market

    _, market = load_market(store)
    fixture = next((f for f in market["fixtures"] if f["id"] == record["fixture_id"]), None)
    if fixture is None:
        raise ValueError("Unknown registered fixture")
    event = next((e for e in market["events"] if e["id"] == fixture.get("event")), None)
    if event is None:
        raise ValueError("Unassigned fixture cannot claim a deadline-known market")
    known = datetime.fromisoformat(record["known_at"].replace("Z", "+00:00"))
    captured = datetime.fromisoformat(now())
    deadline = datetime.fromisoformat(event["deadline_time"].replace("Z", "+00:00"))
    if known.tzinfo is None or known > captured or known >= deadline:
        raise ValueError("Market observation must be known before the FPL deadline")
    if not record.get("source_url", "").startswith("https://"):
        raise ValueError("Market needs an original HTTPS source")
    fitted = goal_constraints(record["result_odds"], record.get("total_2_5_odds"))
    result = {
        **record,
        **fitted,
        "market_id": market["id"],
        "event": event["id"],
        "deadline": event["deadline_time"],
        "captured_at": now(),
        "provenance": "Caller-supplied dated market; source capture is retained, not closing-odds backfill",
    }
    result["id"] = store.put("odds_observation", result)
    return result


async def collect_odds(store):
    from .ingestion import load_market

    _, market = load_market(store)
    teams = {t["id"]: t["name"] for t in market["teams"]}
    url = "https://football-data.co.uk/fixtures.csv"
    async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
        text = await PublicTransport(store).get(
            client, url, "odds_public", json_body=False, max_bytes=2_000_000
        )
    store.snapshot("odds_public", url, {"text": text})
    captured = now()
    observations = []
    rejected = []
    pl_rows = 0
    for row in csv.DictReader(io.StringIO(text.lstrip("\ufeff"))):
        if row.get("Div") != "E0":
            continue
        pl_rows += 1
        candidates = [
            f
            for f in market["fixtures"]
            if not f.get("finished")
            and f.get("kickoff_time")
            and club_key(teams[f["team_h"]]) == club_key(row["HomeTeam"])
            and club_key(teams[f["team_a"]]) == club_key(row["AwayTeam"])
        ]
        if len(candidates) != 1:
            rejected.append(
                {
                    "fixture": row["HomeTeam"] + " v " + row["AwayTeam"],
                    "reason": "Ambiguous or unavailable future fixture",
                }
            )
            continue
        fixture = candidates[0]
        try:
            published_date = datetime.strptime(row["Date"], "%d/%m/%Y").date()
            if (
                published_date
                != datetime.fromisoformat(fixture["kickoff_time"].replace("Z", "+00:00")).date()
            ):
                raise ValueError("Fixture date conflict")
            prices = [float(row[k]) for k in ("B365H", "B365D", "B365A")]
            total = (
                [float(row[k]) for k in ("B365>2.5", "B365<2.5")]
                if row.get("B365>2.5") and row.get("B365<2.5")
                else None
            )
            record = import_market(
                store,
                {
                    "fixture_id": fixture["id"],
                    "known_at": captured,
                    "source_url": url,
                    "bookmaker": "B365",
                    "result_odds": prices,
                    "total_2_5_odds": total,
                },
            )
            observations.append(record["id"])
        except (ValueError, KeyError) as exc:
            rejected.append({"fixture_id": fixture["id"], "reason": str(exc)})
    result = {
        "at": captured,
        "market_id": market["id"],
        "source_url": url,
        "premier_league_rows": pl_rows,
        "observations": observations,
        "rejected": rejected,
        "status": "Captured compatible pre-deadline markets"
        if observations
        else "Free feed currently has no compatible pre-deadline Premier League markets",
        "fallback": "Statistical forecast remains active; no closing odds or market probabilities are invented",
    }
    result["id"] = store.put("odds_refresh", result)
    atomic_json(Path("artifacts/evaluations/odds_coverage.json"), result)
    return result


def calibrate_odds(store):
    """Fit reliability only from genuinely archived pre-deadline observations."""
    from .features import intensity, team_rates
    from .ingestion import load_market

    _, market = load_market(store)
    fixtures = {f["id"]: f for f in market["fixtures"]}
    finalized = {e["id"] for e in market["events"] if e.get("data_checked")}
    entries = []
    for item in store.list("odds_observation", 10000):
        r = store.get("odds_observation", item["id"])
        f = fixtures.get(r["fixture_id"])
        if (
            not f
            or f.get("event") not in finalized
            or not f.get("finished")
            or f.get("team_h_score") is None
            or r.get("market_id") is None
        ):
            continue
        if datetime.fromisoformat(r["known_at"].replace("Z", "+00:00")) >= datetime.fromisoformat(
            r["deadline"].replace("Z", "+00:00")
        ):
            continue
        if datetime.fromisoformat(r["captured_at"].replace("Z", "+00:00")) >= datetime.fromisoformat(
            r["deadline"].replace("Z", "+00:00")
        ):
            continue
        past = [
            p
            for p in fixtures.values()
            if p.get("finished")
            and p.get("kickoff_time")
            and datetime.fromisoformat(p["kickoff_time"].replace("Z", "+00:00")) + timedelta(hours=4)
            < datetime.fromisoformat(r["known_at"].replace("Z", "+00:00"))
        ]
        rates = team_rates(past, market["teams"])
        stat = intensity(f["team_h"], f["team_a"], rates, 0)
        entries.append(
            {
                "event": r["event"],
                "horizon": horizon_key(
                    (
                        datetime.fromisoformat(f["kickoff_time"].replace("Z", "+00:00"))
                        - datetime.fromisoformat(r["known_at"].replace("Z", "+00:00"))
                    ).total_seconds()
                    / 86400
                ),
                "fixture_id": f["id"],
                "actual": [f["team_h_score"], f["team_a_score"]],
                "statistical": stat,
                "market": [r["home_goals"], r["away_goals"]],
            }
        )
    # Latest compatible snapshot per fixture, never repeated matches as independent evidence.
    unique: dict = {}
    for r in entries:
        unique.setdefault(r["fixture_id"], r)
    entries = list(unique.values())
    entries.sort(key=lambda r: (r["event"], r["fixture_id"]))
    result: dict = {
        "at": now(),
        "observations": len(entries),
        "weights": {},
        "promoted": False,
        "reason": "Needs 20 independently completed captured fixtures for calibration and later testing",
        "horizons": [[0, 3], [3, 8]],
    }
    from .availability import block_interval

    for horizon in ("0-3d", "3-8d", "8-21d", "21d+"):
        rows = [r for r in entries if r["horizon"] == horizon]
        if len(rows) < 20:
            result["weights"][horizon] = {"observations": len(rows), "weight": 0.0, "promoted": False}
            continue
        # Split by whole deadline so fixtures from one decision cannot straddle calibration/test.
        groups = sorted({r["event"] for r in rows})
        split = groups[len(groups) // 2]
        cal = [r for r in rows if r["event"] < split]
        test = [r for r in rows if r["event"] >= split]
        if len(cal) < 10 or len(test) < 10 or len({r["event"] for r in test}) < 4:
            result["weights"][horizon] = {"observations": len(rows), "weight": 0.0, "promoted": False}
            continue

        def losses(items, w):
            return np.array(
                [
                    np.sum(
                        (1 - w) * np.asarray(r["statistical"])
                        + w * np.asarray(r["market"])
                        - np.asarray(r["actual"])
                        * np.log(
                            np.maximum(
                                0.01, (1 - w) * np.asarray(r["statistical"]) + w * np.asarray(r["market"])
                            )
                        )
                    )
                    for r in items
                ]
            )

        weight = min((0, 0.25, 0.5, 0.75, 1), key=lambda w: float(losses(cal, w).mean()))
        ci = block_interval(losses(test, weight) - losses(test, 0), [r["event"] for r in test])
        accepted = bool(weight > 0 and ci[1] < 0)
        result["weights"][horizon] = {
            "observations": len(rows),
            "weight": weight if accepted else 0.0,
            "selected_weight": weight,
            "block_nll_difference_95ci": ci,
            "promoted": accepted,
        }
    result["promoted"] = any(r["promoted"] for r in result["weights"].values())
    if result["promoted"]:
        result["reason"] = (
            "Deadline-separated, lead-time-specific statistical/market blending passed later captured-fixture testing"
        )
    result["id"] = store.put("odds_model", result)
    return result


def live_constraints(store, fixture_id, cutoff=None):
    from .ingestion import load_market

    cutoff = cutoff or now()
    try:
        model = store.get("odds_model")
    except ValueError:
        return None
    if not model["promoted"]:
        return None
    for item in store.list("odds_observation", 500):
        r = store.get("odds_observation", item["id"])
        if r["fixture_id"] == fixture_id and r["known_at"] <= cutoff and r["known_at"] < r["deadline"]:
            age = (
                datetime.fromisoformat(cutoff) - datetime.fromisoformat(r["known_at"])
            ).total_seconds() / 3600
            if 0 <= age <= 24:
                _, market = load_market(store)
                fixture = next((f for f in market["fixtures"] if f["id"] == fixture_id), None)
                if fixture is None or not fixture.get("kickoff_time"):
                    return None
                lead = (
                    datetime.fromisoformat(fixture["kickoff_time"].replace("Z", "+00:00"))
                    - datetime.fromisoformat(r["known_at"].replace("Z", "+00:00"))
                ).total_seconds() / 86400
                reliability = model["weights"].get(horizon_key(lead), {})
                if reliability.get("promoted"):
                    return r | {"weight": reliability["weight"]}
    return None


def horizon_key(days):
    return "0-3d" if days < 3 else "3-8d" if days < 8 else "8-21d" if days < 21 else "21d+"
