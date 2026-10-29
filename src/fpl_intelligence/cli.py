"""Local operations; every command executes a real transformation or check."""

from __future__ import annotations

import asyncio
import json
import shutil
import sqlite3
from pathlib import Path

import typer

from .settings import load_environment
from .storage import Store, atomic_json, now

load_environment()

app = typer.Typer(no_args_is_help=True, help="FPL Intelligence · free local numerical decision tools")


def show(value):
    typer.echo(json.dumps(value, indent=2, default=str))


@app.command()
def doctor():
    """Check local state integrity, source freshness, installed versions and free-cost policy."""
    import importlib.metadata

    from .ingestion import freshness

    store = Store()
    with store.db() as db:
        integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
    result = {
        "at": now(),
        "database_integrity": integrity,
        "data_dir": str(store.root.resolve()),
        "paid_provider_calls_allowed": False,
        "local_llm": False,
        "versions": {
            name: importlib.metadata.version(name) for name in ("numpy", "highspy", "fastapi", "lightgbm")
        },
    }
    for source in ("bootstrap", "fixtures"):
        try:
            _, meta = store.latest(source)
            result[source] = {"age_hours": freshness(meta), "hash": meta["hash"]}
        except ValueError:
            result[source] = "not yet ingested"
    show(result)


@app.command()
def ingest(entry: int = 9375678, league: int = 1873551, histories: bool = True):
    """Refresh official sources and optional public comparator with raw provenance."""
    from .ingestion import Official

    show(asyncio.run(Official(Store()).ingest(entry=entry, league=league, histories=histories)))


@app.command()
def snapshot(destination: Path = Path("artifacts/runtime/snapshot-manifest.json")):
    """Freeze the current source manifest without altering any raw snapshots."""
    store = Store()
    with store.db() as db:
        records = [dict(r) for r in db.execute("SELECT * FROM snapshots ORDER BY id")]
    body = {"created_at": now(), "snapshots": records}
    atomic_json(destination, body)
    show({"path": str(destination), "count": len(records)})


@app.command("reconcile-team")
def reconcile_team(path: Path = Path("research/04_USER_CONTEXT.json"), confirmed: bool = False):
    """Import the dated seed or a structured exact account export; validate money and identity."""
    from .domain import TeamState
    from .ingestion import load_market, reconcile_seed
    from .rules import Rules, validate_squad

    store = Store()
    body = json.loads(path.read_text())
    if "team" in body:
        state = reconcile_seed(store, body)
        if confirmed:
            raise typer.BadParameter(
                "Dated research seed cannot be marked confirmed; import an exact current state"
            )
    else:
        parsed_state = TeamState.model_validate({k: v for k, v in body.items() if k != "id"})
        if confirmed:
            parsed_state.status = "confirmed"
        players, _ = load_market(store)
        validate_squad(parsed_state, players, Rules.from_bootstrap(store.latest("bootstrap")[0]))
        identity = store.put("state", parsed_state.model_dump())
        store.event("account_import", identity, {"path": str(path), "status": parsed_state.status})
        state = parsed_state.model_dump() | {"id": identity}
    show(
        {
            "state_id": state["id"],
            "status": state["status"],
            "bank": state["bank"],
            "selling_value": sum(lot["sell_price"] for lot in state["lots"]),
        }
    )


@app.command("build-features")
def build_features():
    from .features import build_features as build
    from .ingestion import load_market

    store = Store()
    players, market = load_market(store)
    show(build(store, players, market))


@app.command()
def train(legacy: bool = False, neural: bool = True):
    """Train the actual decision forecaster; --legacy reproduces the old shadow benchmark."""
    if legacy:
        from .evaluation import train_evaluate

        result = train_evaluate(Store())
    else:
        from threadpoolctl import threadpool_limits

        from .decision import train_decisions

        with threadpool_limits(limits=1):
            result = train_decisions(Store(), neural)
    show({k: v for k, v in result.items() if k not in ("predictions", "residuals", "curves")})


@app.command("uncertainty-audit")
def uncertainty_audit():
    from threadpoolctl import threadpool_limits

    from .decision import validate_uncertainty

    with threadpool_limits(limits=1):
        show(validate_uncertainty(Store()))


@app.command("manager-benchmark")
def manager_benchmark():
    from .manager_audit import benchmark_managers

    result = asyncio.run(benchmark_managers(Store()))
    show({k: v for k, v in result.items() if k not in ("comparisons", "errors")})


@app.command("train-components")
def train_components():
    """Chronologically train/calibrate minutes and test dynamic team strengths."""
    from .models import fit_ensemble, fit_model_suite
    from .strength import train_strength

    store = Store()
    minutes = fit_model_suite(store)
    strength = train_strength(store)
    ensemble = fit_ensemble(store)
    show(
        {
            "minutes_model_id": minutes["id"],
            "minutes_promoted": minutes["promoted"],
            "strength_model_id": strength["id"],
            "strength_promoted": strength["promoted"],
            "ensemble_promoted": ensemble["promoted"],
        }
    )


@app.command("open-benchmark")
def open_benchmark():
    """Reproduce hash-pinned Dastan outputs and the released historical OpenFPL benchmark."""
    from .open_models import reproduce

    show(reproduce(Store()))


@app.command("policy-replay")
def policy_replay():
    """Replay the out-of-fold production model through a continuously evolving account."""
    from .decision_audit import replay_policy

    result = replay_policy(Store())
    show(
        {
            **result,
            "policies": {
                name: {k: v for k, v in row.items() if k != "ledger"}
                for name, row in result["policies"].items()
            },
        }
    )


@app.command("decision-audit")
def decision_audit():
    """Compare held-out captains and budgeted teams with honest hindsight diagnostics."""
    from .decision_audit import evaluate_choices

    result = evaluate_choices(Store())
    show({k: v for k, v in result.items() if k not in ("weeks", "captains")})


@app.command("research-refresh")
def refresh_research(benchmark: bool = False):
    """Refresh original-source evidence or benchmark eligible free hosted extractors."""
    from .context import benchmark_extraction, research_refresh

    store = Store()
    show(asyncio.run(benchmark_extraction(store) if benchmark else research_refresh(store)))


@app.command("configure-providers")
def configure_providers(unbilled: bool = False):
    """Verify each optional free provider independently without printing credentials."""
    from .providers import configure

    show(configure(Store(), unbilled))


@app.command("train-decisions")
def train_decisions(neural: bool = True):
    """Train, calibrate and evaluate the points model that drives recommendations."""
    from threadpoolctl import threadpool_limits

    from .decision import train_decisions as run

    with threadpool_limits(limits=1):
        result = run(Store(), neural)
    show({k: v for k, v in result.items() if k not in ("predictions", "residuals", "curves")})


@app.command("scenario")
def scenario(path: Path, plan_id: str = ""):
    """Evaluate observable injury/role/fixture/price branches with an identical action prefix."""
    from .scenarios import ScenarioSpec, evaluate_scenarios

    store = Store()
    body = json.loads(path.read_text())
    show(
        evaluate_scenarios(
            store,
            plan_id or store.get("selected_plan")["plan_id"],
            [ScenarioSpec.model_validate(row) for row in body["scenarios"]],
        )
    )


@app.command("import-history")
def import_history(season: str = "2025-26"):
    """Import a free historical archive with season-scoped identities and statistical provenance."""
    from .history import import_season

    show(import_season(Store(), season))


@app.command("select-initial")
def select_initial(budget: int = 1000, event: int = 1, seconds: float = 15):
    """Select a legal initial/late-entry squad with all 15 players inside the supplied budget."""
    from .domain import Lot, TeamState
    from .ingestion import load_market
    from .optimization import select_squad
    from .rules import Rules, validate_squad

    store = Store()
    players, market = load_market(store)
    forecast = store.get("forecast")
    if not forecast["start_event"] <= event <= forecast["end_event"]:
        raise typer.BadParameter("Requested initial event is outside the frozen forecast")
    rules = Rules.from_bootstrap(store.latest("bootstrap")[0])
    means = {r["player_id"]: r["mean"] for r in forecast["players"] if r["event"] == event}
    picked = select_squad(None, players, rules, means, seconds, initial_budget=budget)
    state = TeamState(
        season=market["season"],
        event=event,
        bank=budget - sum(players[p].price for p in picked),
        free_transfers=1,
        lots=[
            Lot(
                player_id=p,
                purchase_price=players[p].price,
                sell_price=players[p].price,
                quote_market_price=players[p].price,
                provenance="initial_selection",
            )
            for p in picked
        ],
        initial=True,
        known_at=now(),
        status="public_deadline",
    )
    validate_squad(state, players, rules)
    identity = store.put("state", state.model_dump())
    show({"state_id": identity, "bank": state.bank, "squad": picked, "status": state.status})


@app.command()
def calibrate():
    """Measure held-out start-probability reliability in fixed bins."""
    import numpy as np

    store = Store()
    result = store.get("evaluation")
    rows = result["predictions"]
    bins = []
    for lo in np.arange(0, 1, 0.1):
        selected = [r for r in rows if lo <= r["prior_start"] < lo + 0.1]
        if selected:
            bins.append(
                {
                    "low": float(lo),
                    "high": float(lo + 0.1),
                    "n": len(selected),
                    "predicted": float(np.mean([r["prior_start"] for r in selected])),
                    "observed": float(np.mean([r["started"] for r in selected])),
                }
            )
    body = {
        "at": now(),
        "evaluation_id": result["id"],
        "kind": "held_out_reliability_diagnostic",
        "bins": bins,
        "applied_to_model": False,
    }
    body["id"] = store.put("calibration", body)
    show(body)


@app.command()
def forecast(end: int = 38, worlds: int = 4000, seed: int = 11, beliefs: bool = False):
    """Freeze complete player-fixture distributions in coherent shared match worlds."""
    from .simulation import forecast as run

    result = run(Store(), end_event=end, worlds=worlds, seed=seed, use_beliefs=beliefs)
    show({k: v for k, v in result.items() if k not in ("players", "fixtures", "match_audit", "player_ids")})


@app.command()
def research(
    url: str = "", claims: Path | None = None, belief: Path | None = None, capability: Path | None = None
):
    """Retrieve official source text, import supported claims, or extract using verified free hosted AI."""
    from .ingestion import load_market
    from .research import Belief, Claim, FreeCapability, extract, retrieve, save_claim

    store = Store()
    if belief:
        item = Belief.model_validate(json.loads(belief.read_text()))
        players, _ = load_market(store)
        if item.player_id not in players:
            raise typer.BadParameter("Unknown player")
        show({"belief_id": store.put("belief", item.model_dump())})
    elif claims:
        body = json.loads(claims.read_text())
        show(
            {
                "claim_ids": [
                    save_claim(store, Claim.model_validate(c), body["source_text"]) for c in body["claims"]
                ]
            }
        )
    elif url:
        source = asyncio.run(retrieve(store, url))
        if capability:
            players, _ = load_market(store)
            cap = FreeCapability.model_validate(json.loads(capability.read_text()))
            show(
                {
                    "claim_ids": asyncio.run(
                        extract(
                            store,
                            source,
                            cap,
                            [
                                {"id": p.id, "name": p.name}
                                for p in players.values()
                                if p.web_name.lower() in source["text"].lower()
                            ],
                        )
                    )
                }
            )
        else:
            show(source)
    else:
        show(
            {
                "claims": store.list("claim"),
                "beliefs": store.list("belief"),
                "hosted_inference": "Optional; verified free capability and account key required",
            }
        )


@app.command()
def simulate(plan_id: str = ""):
    """Rescore an actual plan with exact autosubs/captain fallback in frozen shared worlds."""
    from .simulation import score_plan

    store = Store()
    plan = store.get("plan", plan_id or None)
    score_plan(store, plan, store.get("forecast", plan["forecast_id"]))
    show(plan["simulation"])


@app.command()
def optimize(
    end: int = 19,
    seconds: float = 60,
    state_id: str = "",
    forecast_id: str = "",
    chips: str = "auto",
    max_transfers: int = 3,
):
    """Joint full-market transfer/chip MIP with independent legality and scenario validation."""
    from .optimization import hold_plan
    from .optimization import optimize as run

    store = Store()
    state_id = state_id or store.get("state")["id"]
    forecast_id = forecast_id or store.get("forecast")["id"]
    fixed = None if chips == "auto" else {int(k): v for k, v in json.loads(chips).items()}
    result = run(store, state_id, forecast_id, end, seconds, chips=fixed, max_transfers=max_transfers)
    store.put("selected_plan", {"plan_id": result["id"], "at": now()})
    hold = hold_plan(store, state_id, forecast_id, end)
    show(
        {
            "plan_id": result["id"],
            "solver": result["solver"],
            "simulation": result["simulation"],
            "hold": hold["simulation"],
            "execution_ready": result["execution_ready"],
        }
    )


@app.command()
def rivals(league: int = 1873551, entry: int = 9375678, case: bool = False, seconds: float = 30):
    """Retrieve actual mini-league membership and compare candidate policies in common worlds."""
    from .rivals import case_comparison, discover, effective_ownership

    store = Store()
    result = asyncio.run(discover(store, league, entry))
    if case:
        result = case_comparison(store, store.get("state")["id"], store.get("forecast")["id"], seconds)
    else:
        result["effective_ownership"] = effective_ownership(store, result)
    show(result)


@app.command()
def explain(plan_id: str = ""):
    """Render a numerical plan as a human-readable report without language-model invention."""
    from .reports import render

    store = Store()
    typer.echo(render(store, store.get("plan", plan_id or store.get("selected_plan")["plan_id"])))


@app.command()
def replay(plan_id: str = ""):
    """Independently replay every exported action against its frozen state and market."""
    from .domain import Action, Lineup, TeamState
    from .ingestion import load_market
    from .rules import Rules, transition

    store = Store()
    plan = store.get("plan", plan_id or store.get("selected_plan")["plan_id"])
    state = TeamState.model_validate(
        {k: v for k, v in store.get("state", plan["state_id"]).items() if k != "id"}
    )
    players, market = load_market(store, plan["market_id"])
    rules = Rules.from_bootstrap(store.latest("bootstrap", content_hash=market["source_hash"])[0])
    for row in plan["ledger"]:
        action = Action(
            sell=[r["player_id"] for r in row["sell"]],
            buy=[r["player_id"] for r in row["buy"]],
            chip=row["chip"],
            lineup=Lineup.model_validate(row["lineup"]),
        )
        state, checked = transition(state, action, players, rules)
        for field in ("bank_after", "free_transfers_after", "hit", "playing_squad", "permanent_squad"):
            if (
                (set(checked[field]) != set(row[field]))
                if field.endswith("squad")
                else (checked[field] != row[field])
            ):
                raise ValueError(f"Ledger replay mismatch: GW{row['event']} {field}")
    show({"legal": True, "events_replayed": len(plan["ledger"]), "final_bank": state.bank})


@app.command()
def evaluate(prospective: bool = False):
    from .evaluation import evaluate_frozen

    store = Store()
    result = evaluate_frozen(store) if prospective else store.get("evaluation")
    if prospective:
        from .prospective import score_frozen_decisions

        result["policy"] = score_frozen_decisions(store)
    show({k: v for k, v in result.items() if k != "predictions"})


@app.command("freeze")
def freeze_decision(plan_id: str = ""):
    """Archive the actual current recommendation before its deadline for shadow-policy evaluation."""
    from .prospective import freeze

    store = Store()
    show(freeze(store, plan_id or store.get("selected_plan")["plan_id"]))


@app.command("chip-comparison")
def chip_comparison(plan_id: str = "", seconds: float = 30):
    """Re-optimize the whole horizon without each chosen chip under the same football worlds."""
    from .chips import compare_chips

    store = Store()
    show(compare_chips(store, plan_id or store.get("selected_plan")["plan_id"], seconds))


@app.command()
def export(destination: Path = Path("artifacts/runtime/report"), plan_id: str = ""):
    """Export JSON, CSV, Markdown and authoritative deadline ICS."""
    from .reports import export as run

    store = Store()
    show(run(store, store.get("plan", plan_id or store.get("selected_plan")["plan_id"]), destination))


@app.command()
def backup(destination: Path = Path("artifacts/runtime/backup")):
    """Create a portable database and immutable raw/scenario backup."""
    store = Store()
    if destination.resolve().is_relative_to(store.root.resolve()):
        raise typer.BadParameter("Backup destination must be outside the data directory")
    destination.mkdir(parents=True, exist_ok=True)
    store.backup(destination / "state.sqlite")
    for name in ("raw", "scenarios", "models", "normalized", "features"):
        if (store.root / name).exists():
            shutil.copytree(store.root / name, destination / name, dirs_exist_ok=True)
    show(
        {
            "backup": str(destination),
            "database_integrity": sqlite3.connect(destination / "state.sqlite")
            .execute("PRAGMA integrity_check")
            .fetchone()[0],
        }
    )


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000):
    """Run the private API and production-built React interface."""
    import uvicorn

    from .api import create_app

    if host not in ("127.0.0.1", "::1", "localhost"):
        raise typer.BadParameter("Personal API binds to loopback only")
    uvicorn.run(create_app(), host=host, port=port)
