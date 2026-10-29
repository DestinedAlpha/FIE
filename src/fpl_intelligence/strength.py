"""Dynamic regularized Poisson team strengths with deadline-held-out promotion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from .features import intensity, team_rates
from .ingestion import load_market
from .storage import Store, atomic_json, now


def fit_rates(fixtures, teams):
    complete = [f for f in fixtures if f.get("finished") and f.get("team_h_score") is not None]
    if len(complete) < 20:
        return team_rates(fixtures, teams)
    ids = sorted(t["id"] for t in teams)
    index = {v: i for i, v in enumerate(ids)}
    n = len(ids)
    x = np.zeros((2 * len(complete), 2 * n + 2))
    y = np.zeros(len(x))
    weights = np.zeros(len(x))
    latest = max(f.get("event") or 0 for f in complete)
    for i, f in enumerate(complete):
        for away in (0, 1):
            row = 2 * i + away
            attack = f["team_a"] if away else f["team_h"]
            defend = f["team_h"] if away else f["team_a"]
            x[row, 0] = 1
            x[row, 1] = 1 if not away else -1
            x[row, 2 + index[attack]] = 1
            x[row, 2 + n + index[defend]] = 1
            y[row] = f["team_a_score"] if away else f["team_h_score"]
            weights[row] = 0.94 ** max(0, latest - (f.get("event") or latest))
    initial = np.zeros(x.shape[1])
    initial[0] = np.log(max(0.3, float(np.mean(y))))
    initial[1] = 0.12
    penalty = np.array([0, 0.5] + [5.0] * (2 * n))

    def objective(beta):
        eta = x @ beta
        mu = np.exp(np.clip(eta, -5, 5))
        return float(np.sum(weights * (mu - y * eta)) + 0.5 * np.sum(penalty * beta**2)), x.T @ (
            weights * (mu - y)
        ) + penalty * beta

    fit = minimize(objective, initial, jac=True, method="L-BFGS-B", options={"maxiter": 200})
    if not fit.success:
        return team_rates(fixtures, teams)
    b = fit.x
    return {
        club: {
            "attack": float(np.exp(b[2 + index[club]])),
            "defence": float(np.exp(b[2 + n + index[club]])),
            "mean": float(np.exp(b[0])),
            "home_factor": float(np.exp(b[1])),
            "away_factor": float(np.exp(-b[1])),
        }
        for club in ids
    }


def train_strength(store: Store) -> dict:
    _, market = load_market(store)
    complete = [f for f in market["fixtures"] if f.get("finished") and f.get("team_h_score") is not None]
    events = sorted({f["event"] for f in complete if f.get("event")})
    rows = []
    for event in events[3:]:
        past = [f for f in complete if f["event"] < event]
        a = team_rates(past, market["teams"])
        b = fit_rates(past, market["teams"])
        for fixture in [f for f in complete if f["event"] == event]:
            baseline = intensity(fixture["team_h"], fixture["team_a"], a, 0)
            candidate = intensity(fixture["team_h"], fixture["team_a"], b, 0)
            actual = [fixture["team_h_score"], fixture["team_a_score"]]
            for truth, prior, predicted in zip(actual, baseline, candidate):
                rows.append({"event": event, "actual": truth, "baseline": prior, "candidate": predicted})
    if not rows:
        raise ValueError("At least four completed gameweeks are required for team-strength validation")
    y = np.array([r["actual"] for r in rows])
    p = np.array([r["baseline"] for r in rows])
    q = np.array([r["candidate"] for r in rows])

    def nll(means):
        return float(np.mean(means - y * np.log(means) + gammaln(y + 1)))

    report = {
        "at": now(),
        "version": "regularized-poisson-v1",
        "source_fixture_hash": market["fixture_hash"],
        "rows": len(rows),
        "baseline_poisson_nll": nll(p),
        "candidate_poisson_nll": nll(q),
        "promoted": bool(nll(q) < nll(p)),
        "rates": fit_rates(complete, market["teams"]),
        "method": "recency-weighted Poisson MAP; attack/defence and home effect; L2 shrinkage",
        "test_events": sorted({r["event"] for r in rows}),
        "training_max_event": max(events),
    }
    report["id"] = store.put("team_strength", report)
    atomic_json(Path("artifacts/evaluations/team_strength.json"), report)
    return report
