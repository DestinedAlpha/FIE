"""Deadline-tested score dependence, time decay and shared Laplace uncertainty.

Dixon--Coles changes only the four low score cells; a Gamma--Poisson
alternative tests shared match tempo. Neither is enabled by architecture alone.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln, logsumexp
from threadpoolctl import threadpool_limits

from .availability import block_interval, moment
from .shot_data import club_key
from .storage import atomic_json, now


def fixture_facts(store):
    from .decision import canonical_rows

    rows, metadata, _ = canonical_rows(store)
    dates = {(r["season"], int(r["fixture"])): r.get("kickoff_time") for r in rows}
    facts = []
    for key, item in metadata.items():
        if dates.get(key):
            facts.append(
                {
                    **item,
                    "season": key[0],
                    "fixture_id": key[1],
                    "kickoff": dates[key],
                    "home": club_key(item["home"]),
                    "away": club_key(item["away"]),
                    "group": f"{key[0]}-GW{item['event']:02d}",
                }
            )
    return sorted(facts, key=lambda r: (r["season"], r["event"], r["kickoff"], r["fixture_id"]))


def design(fixtures, clubs):
    index = {v: i for i, v in enumerate(clubs)}
    n = len(clubs)
    x = np.zeros((len(fixtures), 2, 2 + 2 * n))
    for i, r in enumerate(fixtures):
        for side in range(2):
            attack, defence = (r["home"], r["away"]) if side == 0 else (r["away"], r["home"])
            x[i, side, 0] = 1
            x[i, side, 1] = 1 if side == 0 else -1
            if attack in index:
                x[i, side, 2 + index[attack]] = 1
            if defence in index:
                x[i, side, 2 + n + index[defence]] = 1
    return x


def fit_strength(fixtures, clubs, at, decay):
    xx = design(fixtures, clubs).reshape(-1, 2 + 2 * len(clubs))
    y = np.array([[r["home_goals"], r["away_goals"]] for r in fixtures]).ravel()
    ages = np.array([max(0, (moment(at) - moment(r["kickoff"])).total_seconds() / 604800) for r in fixtures])
    weights = np.repeat(decay**ages, 2)
    penalty = np.array([0.5, 0.5] + [4.0] * (2 * len(clubs)))
    initial = np.zeros(xx.shape[1])
    initial[0] = np.log(max(0.2, float(y.mean())))
    initial[1] = 0.1

    def objective(beta):
        eta = xx @ beta
        mu = np.exp(np.clip(eta, -5, 5))
        return float(np.sum(weights * (mu - y * eta)) + 0.5 * np.sum(penalty * beta**2)), xx.T @ (
            weights * (mu - y)
        ) + penalty * beta

    fit = minimize(objective, initial, jac=True, method="L-BFGS-B", options={"maxiter": 120, "ftol": 1e-8})
    if not fit.success and np.linalg.norm(fit.jac) > 1:
        raise ValueError("Team score-process fit did not converge")
    mu = np.exp(np.clip(xx @ fit.x, -5, 5))
    hessian = xx.T @ ((weights * mu)[:, None] * xx) + np.diag(penalty)
    covariance = np.linalg.inv(hessian)
    return fit.x, covariance


def log_scores(actual, rates, kind="independent", parameter=0):
    y = np.asarray(actual)
    rates = np.maximum(0.01, np.asarray(rates))
    base = np.sum(y * np.log(rates) - rates - gammaln(y + 1), axis=-1)
    h, a = y[..., 0], y[..., 1]
    lh, la = rates[..., 0], rates[..., 1]
    if kind == "dixon_coles":
        rho = parameter
        tau = np.where(
            (h == 0) & (a == 0),
            1 - lh * la * rho,
            np.where(
                (h == 0) & (a == 1),
                1 + lh * rho,
                np.where((h == 1) & (a == 0), 1 + la * rho, np.where((h == 1) & (a == 1), 1 - rho, 1)),
            ),
        )
        return base + np.log(np.maximum(1e-12, tau))
    if kind == "gamma_poisson":
        k = parameter
        return (
            gammaln(k + h + a)
            - gammaln(k)
            - gammaln(h + 1)
            - gammaln(a + 1)
            + k * np.log(k)
            - (k + h + a) * np.log(k + lh + la)
            + h * np.log(lh)
            + a * np.log(la)
        )
    return base


def _predictive_log(actual, x, beta, cov, scale, kind, parameter, seed=822):
    if scale == 0:
        return log_scores(actual, np.exp(np.clip(x @ beta, np.log(0.1), np.log(5))), kind, parameter)
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal(np.zeros(len(beta)), cov, size=64) * scale
    rates = np.exp(np.clip(np.einsum("nsd,kd->kns", x, beta + z), np.log(0.1), np.log(5)))
    return logsumexp(log_scores(np.asarray(actual)[None, :, :], rates, kind, parameter), axis=0) - np.log(
        len(z)
    )


def train_score_process(store):
    from .features import intensity, team_rates

    facts = fixture_facts(store)
    clubs = sorted({r[k] for r in facts for k in ("home", "away")})
    groups = defaultdict(list)
    for r in facts:
        groups[r["group"]].append(r)
    predictions: dict = {decay: [] for decay in (0.94, 0.98, 1.0)}
    posterior = []
    shrinkage_baseline: list[dict] = []
    club_ids = {name: i + 1 for i, name in enumerate(clubs)}
    for group, items in sorted(groups.items()):
        if group < "2024/25-GW25":
            continue
        at = min(r["kickoff"] for r in items)
        # Entire target deadline is withheld, including the first DGW match.
        past = [r for r in facts if r["group"] < group and moment(r["kickoff"]) < moment(at)]
        x = design(items, clubs)
        actual = np.array([[r["home_goals"], r["away_goals"]] for r in items])
        seasonal = [r for r in past if r["season"] == items[0]["season"]]
        observed = [
            {
                "team_h": club_ids[r["home"]],
                "team_a": club_ids[r["away"]],
                "team_h_score": r["home_goals"],
                "team_a_score": r["away_goals"],
                "finished": True,
            }
            for r in seasonal
        ]
        simple = team_rates(observed, [{"id": p} for p in club_ids.values()])
        shrinkage_baseline.extend(
            {
                "group": group,
                "actual": y.tolist(),
                "rates": list(intensity(club_ids[r["home"]], club_ids[r["away"]], simple, 0)),
            }
            for r, y in zip(items, actual)
        )
        for decay in predictions:
            with threadpool_limits(limits=1):
                beta, cov = fit_strength(past, clubs, at, decay)
            rates = np.exp(np.clip(x @ beta, np.log(0.1), np.log(5)))
            predictions[decay].extend(
                {"group": group, "rates": p.tolist(), "actual": y.tolist()} for p, y in zip(rates, actual)
            )
            posterior.append(
                {"group": group, "decay": decay, "x": x, "beta": beta, "cov": cov, "actual": actual}
            )
    if not predictions[1.0]:
        raise ValueError("Score-process comparison requires later completed deadlines")

    def entries(decay, cal):
        return [r for r in predictions[decay] if (r["group"] < "2025/26-GW01") == cal]

    def losses(items, kind="independent", parameter=0):
        return -log_scores([r["actual"] for r in items], [r["rates"] for r in items], kind, parameter)

    selected_decay = min(predictions, key=lambda d: float(losses(entries(d, True)).mean()))
    test = entries(selected_decay, False)
    baseline = entries(1.0, False)
    decay_ci = block_interval(losses(test) - losses(baseline), [r["group"] for r in test])
    decay_promoted = bool(selected_decay != 1 and decay_ci[1] < 0)
    adopted_decay = selected_decay if decay_promoted else 1.0
    cal = entries(adopted_decay, True)
    test = entries(adopted_decay, False)
    simple_test = [r for r in shrinkage_baseline if r["group"] >= "2025/26-GW01"]
    strength_ci = block_interval(losses(test) - losses(simple_test), [r["group"] for r in test])
    strength_accepted = bool(strength_ci[1] < 0)
    parameters = {
        "independent": 0.0,
        "dixon_coles": float(
            minimize_scalar(
                lambda rho: float(losses(cal, "dixon_coles", rho).mean()),
                bounds=(-0.15, 0.039),
                method="bounded",
            ).x
        ),
        "gamma_poisson": float(
            np.exp(
                minimize_scalar(
                    lambda logk: float(losses(cal, "gamma_poisson", np.exp(logk)).mean()),
                    bounds=(0, 12),
                    method="bounded",
                ).x
            )
        ),
    }
    selected_kind = min(parameters, key=lambda k: float(losses(cal, k, parameters[k]).mean()))
    kind_ci = block_interval(
        losses(test, selected_kind, parameters[selected_kind]) - losses(test), [r["group"] for r in test]
    )
    kind_promoted = bool(selected_kind != "independent" and kind_ci[1] < 0)
    adopted_kind = selected_kind if kind_promoted else "independent"
    parameter = parameters[adopted_kind]
    predictive: dict = {scale: [] for scale in (0.0, 0.5, 1.0)}
    for item in posterior:
        if item["decay"] != adopted_decay:
            continue
        for scale in predictive:
            logp = _predictive_log(
                item["actual"], item["x"], item["beta"], item["cov"], scale, adopted_kind, parameter
            )
            predictive[scale].extend({"group": item["group"], "loss": float(-v)} for v in logp)
    selected_scale = min(
        predictive, key=lambda s: np.mean([r["loss"] for r in predictive[s] if r["group"] < "2025/26-GW01"])
    )
    ptest = [r for r in predictive[selected_scale] if r["group"] >= "2025/26-GW01"]
    btest = [r for r in predictive[0.0] if r["group"] >= "2025/26-GW01"]
    uncertainty_ci = block_interval(
        [a["loss"] - b["loss"] for a, b in zip(ptest, btest)], [r["group"] for r in ptest]
    )
    posterior_promoted = bool(selected_scale > 0 and uncertainty_ci[1] < 0)
    report = {
        "at": now(),
        "training_max_group": max(groups),
        "matches": len(facts),
        "test_matches": len(test),
        "strength": {
            "promoted": strength_accepted,
            "map_test_nll": float(losses(test).mean()),
            "shrinkage_test_nll": float(losses(simple_test).mean()),
            "block_nll_difference_95ci": strength_ci,
        },
        "decay": {
            "selected": selected_decay,
            "adopted": adopted_decay,
            "block_nll_difference_95ci": decay_ci,
            "promoted": decay_promoted,
        },
        "score_dependence": {
            "selected": selected_kind,
            "adopted": adopted_kind,
            "parameters": parameters,
            "block_nll_difference_95ci": kind_ci,
            "promoted": kind_promoted,
            "test_nll": {k: float(losses(test, k, p).mean()) for k, p in parameters.items()},
        },
        "parameter_uncertainty": {
            "selected_scale": selected_scale,
            "adopted_scale": selected_scale if posterior_promoted else 0.0,
            "block_nll_difference_95ci": uncertainty_ci,
            "promoted": posterior_promoted,
        },
        "protocol": "Expanding earlier-deadline team fits; 2024/25 GW25–38 select decay/dependence/posterior scale, later seasons test; shared Laplace coefficient draws separate from match noise",
        "limits": [
            "Regularized club strength, not a guarantee about future results",
            "Posterior conditional on fitted model; independent structural alternatives remain visible",
        ],
    }
    report["id"] = store.put("score_process_model", report)
    atomic_json(Path("artifacts/evaluations/score_process.json"), report)
    return report


def match_goal_draws(store, market, fixtures, worlds, seed):
    """One shared parameter draw per world, then independent event randomness."""
    if not store.list("score_process_model", 1):
        return {}, None
    spec = store.get("score_process_model")
    if not spec.get("strength", {}).get("promoted", False):
        # The dependence candidates were evaluated on the fitted strength
        # model. They cannot be transplanted to another goal-rate baseline.
        return {}, spec
    facts = fixture_facts(store)
    names = {t["id"]: club_key(t["name"]) for t in market["teams"]}
    clubs = sorted({r[k] for r in facts for k in ("home", "away")} | set(names.values()))
    with threadpool_limits(limits=1):
        beta, cov = fit_strength(facts, clubs, now(), spec["decay"]["adopted"])
    rng = np.random.default_rng([seed, 822])
    scale = spec["parameter_uncertainty"]["adopted_scale"]
    coefficients = beta[None, :] + (
        rng.multivariate_normal(np.zeros(len(beta)), cov, size=worlds) * scale
        if scale
        else np.zeros((worlds, len(beta)))
    )
    kind = spec["score_dependence"]["adopted"]
    parameter = spec["score_dependence"]["parameters"][kind]
    output = {}
    for fixture in sorted(fixtures, key=lambda f: f["id"]):
        x = design([{"home": names[fixture["team_h"]], "away": names[fixture["team_a"]]}], clubs)[0]
        rates = np.exp(np.clip(coefficients @ x.T, np.log(0.1), np.log(5)))
        from .odds import live_constraints

        observation = live_constraints(store, fixture["id"])
        if observation:
            rates = (1 - observation["weight"]) * rates + observation["weight"] * np.array(
                [observation["home_goals"], observation["away_goals"]]
            )[None, :]
        fixture_rng = np.random.default_rng([seed, fixture["id"], 823])
        tempo = (
            fixture_rng.gamma(parameter, 1 / parameter, worlds)
            if kind == "gamma_poisson"
            else np.ones(worlds)
        )
        values = fixture_rng.poisson(rates * tempo[:, None])
        if kind == "dixon_coles":
            # Rejection of independent draws with a finite envelope exactly
            # samples the normalized four-cell Dixon--Coles correction.
            envelope = 1 + abs(parameter) * np.maximum(1, np.prod(rates, axis=1) + rates.max(axis=1))
            pending = np.ones(worlds, dtype=bool)
            for _ in range(100):
                indexes = np.where(pending)[0]
                if not len(indexes):
                    break
                proposed = fixture_rng.poisson(rates[indexes])
                tau = np.exp(
                    log_scores(proposed, rates[indexes], kind, parameter)
                    - log_scores(proposed, rates[indexes])
                )
                accept = fixture_rng.random(len(indexes)) < tau / envelope[indexes]
                values[indexes[accept]] = proposed[accept]
                pending[indexes[accept]] = False
            if pending.any():
                raise ValueError("Score rejection sampler exceeded its convergence budget")
        output[fixture["id"]] = (
            {fixture["team_h"]: values[:, 0], fixture["team_a"]: values[:, 1]},
            float(rates[:, 0].mean()),
            float(rates[:, 1].mean()),
        )
    return output, spec
