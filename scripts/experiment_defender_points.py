"""Controlled defender-only point-head comparison; never publishes live model weights."""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from fpl_intelligence.decision import NAMES, PARAMETERS, minute_band
from fpl_intelligence.storage import Store, atomic_json, now


def main():
    s = Store()
    dataset = s.get("decision_features")
    df = pl.read_parquet(s.artifact_path(dataset["path"]))
    pop = s.get("population_diagnostics")
    directory = s.artifact_path(pop["directory"])
    out = []
    constraints = [
        1 if n in ("goals90", "assists90") else -1 if n in ("team_defence", "opponent_attack") else 0
        for n in NAMES
    ] + [0]
    params = PARAMETERS | {"monotone_constraints": constraints}
    for receipt in pop["folds"]:
        fold = receipt["fold"]
        block = fold["test"]
        if not block[0].startswith(("2024/25", "2025/26")):
            continue
        cutoff = fold["calibration_min"]
        before = block[0]
        tr = df.filter(
            (pl.col("deadline_group") < cutoff) & (pl.col("position") == 2) & (pl.col("minutes_target") > 0)
        )
        ca = df.filter(
            (pl.col("deadline_group") >= cutoff)
            & (pl.col("deadline_group") < before)
            & (pl.col("position") == 2)
            & (pl.col("minutes_target") > 0)
        )
        key = block[0].replace("/", "-")
        reference = pl.read_parquet(directory / (key + ".parquet")).filter(pl.col("position") == 2)
        test = df.filter(pl.col("deadline_group").is_in(block) & (pl.col("position") == 2))
        keys = ["season", "fixture_id", "player_id"]
        joined = reference.join(test.select(*keys, *NAMES), on=keys, how="inner", suffix="_input")
        assert len(joined) == len(reference)
        model = lgb.LGBMRegressor(**params).fit(
            np.column_stack([tr.select(NAMES).to_numpy(), minute_band(tr["minutes_target"].to_numpy())]),
            tr["base_target"].to_numpy(),
        )
        calband = minute_band(ca["minutes_target"].to_numpy())
        pred = model.predict(np.column_stack([ca.select(NAMES).to_numpy(), calband]))
        offset = {
            k: float(
                (ca["base_target"].to_numpy()[calband == k] - pred[calband == k]).sum()
                / ((calband == k).sum() + 100)
            )
            for k in (1, 2, 3)
        }
        # Joined tab columns have suffix only when already present in population metadata.
        x = joined.select([n + "_input" if n + "_input" in joined.columns else n for n in NAMES]).to_numpy()
        c = np.column_stack(
            [np.zeros(len(x))]
            + [model.predict(np.column_stack([x, np.full(len(x), k)])) + offset[k] for k in (1, 2, 3)]
        )
        p = joined.select([f"p{k}" for k in range(4)]).to_numpy()
        dc = joined.select([f"dc{k}" for k in range(4)]).to_numpy()
        means = (p * (c + 2 * dc)).sum(1)
        for row, value in zip(
            joined.select(
                "season", "event", "deadline_group", "player_id", "actual", "reference", "prior_start"
            ).to_dicts(),
            means,
        ):
            row["candidate"] = float(value)
            out.append(row)
        print("Completed", key, len(joined), flush=True)
    selection = [r for r in out if r["season"] == "2024/25" and r["event"] <= 19]
    weights = (0, 0.25, 0.5, 1)
    weight = min(
        weights,
        key=lambda w: np.mean(
            [(r["reference"] + w * (r["candidate"] - r["reference"]) - r["actual"]) ** 2 for r in selection]
        ),
    )
    metrics = {}
    selection_losses = {
        str(w): float(
            np.mean(
                [
                    (r["reference"] + w * (r["candidate"] - r["reference"]) - r["actual"]) ** 2
                    for r in selection
                ]
            )
        )
        for w in weights
    }
    for name, part in [
        ("acceptance_2024", [r for r in out if r["season"] == "2024/25" and r["event"] > 19]),
        ("replication_2025", [r for r in out if r["season"] == "2025/26"]),
    ]:
        a = np.array([r["actual"] for r in part])
        b = np.array([r["reference"] for r in part])
        v = b + weight * (np.array([r["candidate"] for r in part]) - b)
        regular = np.array([r["prior_start"] >= 0.6 for r in part])
        groups = sorted({r["deadline_group"] for r in part})
        delta = np.array(
            [np.mean(((v - a) ** 2 - (b - a) ** 2)[[r["deadline_group"] == g for r in part]]) for g in groups]
        )
        rng = np.random.default_rng(473)
        ci = np.quantile(
            delta[rng.integers(0, len(delta), (2000, len(delta)))].mean(1), [0.005, 0.995]
        ).tolist()
        metrics[name] = {
            "rows": len(part),
            "reference_rmse": float(np.sqrt(np.mean((b - a) ** 2))),
            "candidate_rmse": float(np.sqrt(np.mean((v - a) ** 2))),
            "regular_reference_rmse": float(np.sqrt(np.mean((b[regular] - a[regular]) ** 2))),
            "regular_candidate_rmse": float(np.sqrt(np.mean((v[regular] - a[regular]) ** 2))),
            "deadline_delta_mse_99ci": ci,
        }
    result = {
        "at": now(),
        "protocol": "Defender-only conditional points with positive goals90/assists90 and negative team-defence/opponent-attack monotonicity. Existing role probabilities, DC and all other positions unchanged. Select blend on 2024/25 GW1-19; independent acceptance GW20-38 and 2025/26 replication. Eight prior calibration deadlines excluded from fitting. Current player preferences do not select a blend.",
        "feature_hash": dataset["hash"],
        "selected_weight": weight,
        "selection_losses": selection_losses,
        "metrics": metrics,
        "passed": bool(
            weight
            and all(
                v["deadline_delta_mse_99ci"][1] < 0
                and v["regular_candidate_rmse"] <= v["regular_reference_rmse"]
                for v in metrics.values()
            )
        ),
        "promoted": False,
    }
    atomic_json(Path("artifacts/evaluations/defender_points_0_14_7.json"), result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
