"""Private localhost API. Long operations run in a serial durable job queue."""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .domain import Action, OptimizeRequest, Strict, TeamState
from .ingestion import Official, freshness, load_market, reconcile_seed
from .optimization import hold_plan, optimize
from .research import Belief, Claim, save_claim
from .rules import Rules, transition, validate_squad
from .settings import load_environment, provider_configuration, redact
from .storage import Store, now

executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fpl-job")
state_lock = threading.Lock()


class ForecastRequest(Strict):
    end_event: int = Field(default=38, ge=1, le=38)
    worlds: int = Field(default=4000, ge=100, le=20000)
    seed: int = 11
    use_beliefs: bool = False


class StateImport(Strict):
    state: TeamState
    expected_state_id: str | None = None


class ClaimInput(Strict):
    claim: Claim
    source_text: str = Field(max_length=60000)


class TransferCheck(Strict):
    state_id: str
    action: Action


class RefreshSchedule(Strict):
    enabled: bool
    interval_minutes: int = Field(default=240, ge=30, le=1440)


class RivalRequest(Strict):
    league_id: int = Field(default=1873551, ge=1)
    entry_id: int = Field(default=9375678, ge=1)
    compare_case: bool = False


class ResearchRequest(Strict):
    urls: list[str] = Field(default_factory=list, max_length=5)
    limit: int = Field(default=3, ge=1, le=5)


class ScenarioRequest(Strict):
    plan_id: str
    scenarios: list[dict] = Field(min_length=1, max_length=8)


def create_app(root=None):
    load_environment()

    @asynccontextmanager
    async def lifespan(application):
        from .scheduling import start_scheduler

        stop, thread = start_scheduler(store, launch)
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=1)

    app = FastAPI(title="FPL Intelligence", version="0.5.0", lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]"])

    @app.middleware("http")
    async def local_origin(request, call_next):
        from urllib.parse import urlparse

        origin = request.headers.get("origin")
        if origin and urlparse(origin).hostname not in ("localhost", "127.0.0.1", "::1"):
            from fastapi.responses import JSONResponse

            return JSONResponse({"detail": "Cross-origin access denied"}, status_code=403)
        return await call_next(request)

    store = Store(root)
    app.state.store = store
    if store.list("claim", 1) and store.list("market", 1) and not store.list("claim_identity_migration", 1):
        from .research import audit_claim_identities

        review = audit_claim_identities(store)
        store.put("claim_identity_migration", {"version": 1, "review_id": review["id"], "at": now()})
    # Restart recovery is explicit; an interrupted job is never shown as running forever.
    with store.db() as db:
        db.execute(
            "UPDATE jobs SET status='interrupted',error='Process restarted; submit job again',updated_at=? "
            "WHERE status IN ('queued','running')",
            (now(),),
        )

    @app.exception_handler(ValueError)
    async def bad_request(request, exc):
        from fastapi.responses import JSONResponse

        return JSONResponse({"detail": str(exc)}, status_code=422)

    def launch(kind, request, work):
        identity = uuid.uuid4().hex
        body = {"kind": kind, **request}
        with store.db() as db:
            db.execute(
                "INSERT INTO jobs VALUES(?,?,?,?,?,?,?)",
                (identity, "queued", now(), now(), json.dumps(body), None, None),
            )

        def execute():
            with store.db() as db:
                db.execute("UPDATE jobs SET status='running',updated_at=? WHERE id=?", (now(), identity))
            try:
                result = work()
                with store.db() as db:
                    db.execute(
                        "UPDATE jobs SET status='completed',result=?,updated_at=? WHERE id=?",
                        (json.dumps(result, allow_nan=False), now(), identity),
                    )
            except Exception as exc:
                with store.db() as db:
                    db.execute(
                        "UPDATE jobs SET status='failed',error=?,updated_at=? WHERE id=?",
                        (f"{type(exc).__name__}: {redact(exc)[:400]}", now(), identity),
                    )

        executor.submit(execute)
        return {"job_id": identity, "status": "queued"}

    @app.get("/api/health")
    def health():
        with store.db() as db:
            integrity = db.execute("PRAGMA quick_check").fetchone()[0]
            sources = [
                dict(r)
                for r in db.execute(
                    "SELECT source,max(captured_at) AS captured_at FROM snapshots GROUP BY source"
                )
            ]
            quota = [dict(r) for r in db.execute("SELECT * FROM quota ORDER BY day DESC LIMIT 10")]
        return {
            "status": "ok" if integrity == "ok" else "degraded",
            "integrity": integrity,
            "sources": sources,
            "quota": quota,
            "at": now(),
            "paid_provider_calls_allowed": False,
            "hosted_ai_configured": any(
                v for k, v in provider_configuration().items() if k.endswith("key_present")
            ),
            "providers": provider_configuration(),
            "mode": "local private",
        }

    @app.get("/api/dashboard")
    def dashboard(event: int | None = None):
        result: dict[str, Any] = {
            "at": now(),
            "market": None,
            "state": None,
            "forecast": None,
            "plan": None,
            "evaluation": None,
            "rivals": None,
            "league": None,
            "evidence": [],
            "beliefs": [],
            "model_suite": None,
            "policy_evaluation": None,
            "scenario_evaluation": None,
            "extraction_benchmark": None,
            "open_model_reproduction": None,
            "cohort": None,
            "chip_counterfactual": None,
            "decision_model": None,
            "decision_audit": None,
            "decision_policy": None,
            "provider_setup": None,
            "decision_uncertainty": None,
            "manager_audit": None,
            "hit_comparison": None,
            "manager_behavior": None,
            "horizon_model": None,
            "transfer_policy_model": None,
            "personalized_policy_analysis": None,
            "decision_critic": None,
            "state_value_model": None,
            "horizon_consistency": None,
            "market_model": None,
            "coherent_model": None,
            "coherent_comparison": None,
            "manager_policy": None,
            "context_ablation": None,
            "decision_analysis": None,
            "timing_analysis": None,
            "price_bridge": None,
            "population_forecast": None,
            "training_workflow": None,
            "refresh_schedule": None,
            "source_coverage": {},
            "fusion_model": None,
            "fusion_comparison": None,
        }
        try:
            players, market = load_market(store)
            event = event or next((e["id"] for e in market["events"] if e.get("is_next")), 38)
            _, meta = store.latest("bootstrap")
            result["market"] = {
                "id": market["id"],
                "season": market["season"],
                "teams": market["teams"],
                "events": market["events"],
                "known_at": market["known_at"],
                "age_hours": freshness(meta),
                "fixtures": market["fixtures"],
                "players": [
                    {k: v for k, v in p.model_dump().items() if k != "raw"} for p in players.values()
                ],
            }
        except ValueError:
            return result
        for kind in (
            "state",
            "forecast",
            "evaluation",
            "league",
            "model_suite",
            "policy_evaluation",
            "scenario_evaluation",
            "extraction_benchmark",
            "open_model_reproduction",
            "cohort",
            "chip_counterfactual",
            "decision_model",
            "decision_audit",
            "decision_policy",
            "provider_setup",
            "decision_uncertainty",
            "manager_audit",
            "hit_comparison",
            "manager_behavior",
            "horizon_model",
            "transfer_policy_model",
            "personalized_policy_analysis",
            "decision_critic",
            "state_value_model",
            "horizon_consistency",
            "market_model",
            "coherent_model",
            "coherent_comparison",
            "manager_policy",
            "context_ablation",
            "decision_analysis",
            "timing_analysis",
            "price_bridge",
            "population_forecast",
            "training_workflow",
            "refresh_schedule",
        ):
            try:
                artifact = store.get(kind)
                if kind in (
                    "decision_audit",
                    "decision_policy",
                    "decision_uncertainty",
                    "manager_audit",
                ) and artifact.get("decision_model_id") != (result["decision_model"] or {}).get("id"):
                    continue
                if kind in ("market_model", "coherent_model", "manager_policy", "decision_critic"):
                    artifact = {
                        k: v
                        for k, v in artifact.items()
                        if k not in ("models", "magnitudes", "bps_residual_quantiles")
                    }
                if kind in ("decision_analysis", "timing_analysis", "price_bridge") and (
                    artifact.get("state_id") != (result["state"] or {}).get("id")
                    or artifact.get("market_id") != market["id"]
                ):
                    continue
                if kind in ("decision_analysis", "horizon_consistency") and artifact.get("forecast_id") != (
                    result["forecast"] or {}
                ).get("id"):
                    continue
                if kind == "horizon_consistency" and artifact.get("state_id") != (result["state"] or {}).get(
                    "id"
                ):
                    continue
                if kind == "manager_behavior":
                    artifact = {k: v for k, v in artifact.items() if k not in ("predictions", "models")}
                if kind == "personalized_policy_analysis" and (
                    artifact.get("state_id") != (result["state"] or {}).get("id")
                    or artifact.get("market_id") != market["id"]
                    or artifact.get("policy_model_id") != (result["transfer_policy_model"] or {}).get("id")
                    or artifact.get("horizon_model_id") != (result["horizon_model"] or {}).get("id")
                ):
                    continue
                if kind == "forecast":
                    artifact = {k: v for k, v in artifact.items() if k not in ("match_audit", "fixtures")}
                    artifact["players"] = [r for r in artifact["players"] if r["event"] == event]
                if kind == "decision_model":
                    artifact = {
                        k: v
                        for k, v in artifact.items()
                        if k not in ("predictions", "residuals", "curves", "minute_quantiles")
                    }
                if kind == "manager_audit":
                    artifact = {k: v for k, v in artifact.items() if k not in ("comparisons", "errors")}
                if kind == "decision_audit":
                    artifact = {k: v for k, v in artifact.items() if k not in ("weeks", "captains")}
                if kind == "evaluation":
                    artifact = {k: v for k, v in artifact.items() if k != "predictions"}
                if kind in ("policy_evaluation", "decision_policy"):
                    artifact = {
                        **artifact,
                        "policies": {
                            name: {k: v for k, v in row.items() if k != "ledger"}
                            for name, row in artifact["policies"].items()
                        },
                    }
                result[kind] = artifact
            except ValueError:
                pass
        try:
            selected = store.get("selected_plan")
            selected_plan = store.get("plan", selected["plan_id"])
            selected_plan["stale"] = (
                selected_plan["state_id"] != (result["state"] or {}).get("id")
                or selected_plan["forecast_id"] != (result["forecast"] or {}).get("id")
                or selected_plan["market_id"] != market["id"]
                or (result["forecast"] or {}).get("model_version") != "learned-points-v1"
                or (result["forecast"] or {}).get("decision_model_id")
                != (result["decision_model"] or {}).get("id")
            )
            if selected_plan["stale"]:
                selected_plan["execution_ready"] = False
            result["plan"] = selected_plan
        except ValueError:
            pass
        for kind in ("coherent_comparison", "population_forecast", "timing_analysis", "price_bridge"):
            scoped_artifact = result.get(kind)
            if scoped_artifact and scoped_artifact.get("plan_id") != (result["plan"] or {}).get("id"):
                result[kind] = None
        for kind in ("scenario_evaluation", "chip_counterfactual", "hit_comparison"):
            scoped_artifact = result.get(kind)
            if scoped_artifact and scoped_artifact.get("base_plan_id") != (result["plan"] or {}).get("id"):
                result[kind] = None
        try:
            result["rivals"] = store.get("rival_comparison")
            if result["rivals"].get("forecast_id") != (result["forecast"] or {}).get("id"):
                result["rivals"] = None
        except ValueError:
            pass
        from .intelligence import source_coverage

        result["source_coverage"] = source_coverage(store)
        try:
            fusion = store.get("fusion_model")
            result["fusion_model"] = {
                k: v for k, v in fusion.items() if k not in ("tree_spec", "network_spec")
            }
            result["fusion_model"]["promoted"] = (result["decision_model"] or {}).get(
                "fusion_model_id"
            ) == fusion["id"]
            for field, kind in (
                ("decision_validation", "fusion_decisions"),
                ("review", "fusion_review"),
                ("uncertainty", "fusion_uncertainty"),
            ):
                try:
                    checked = store.get(kind)
                    if checked.get("fusion_model_id") == fusion["id"]:
                        result["fusion_model"][field] = checked
                except ValueError:
                    pass
            comparison = store.get("fusion_comparison")
            if comparison.get("fusion_model_id") == fusion["id"] and comparison.get("decision_model_id") == (
                result["decision_model"] or {}
            ).get("id"):
                result["fusion_comparison"] = comparison
        except ValueError:
            pass
        for name, kind in [("evidence", "claim"), ("beliefs", "belief")]:
            if kind == "claim":
                from .research import active_claims

                result[name] = active_claims(store)
            else:
                result[name] = [store.get(kind, r["id"]) for r in store.list(kind, 100)]
        return result

    @app.post("/api/refresh")
    def refresh():
        try:
            state = store.get("state")
        except ValueError:
            state = {}
        return launch(
            "ingest",
            {},
            lambda: asyncio.run(Official(store).ingest(entry=state.get("entry_id"), histories=True)),
        )

    @app.post("/api/forecast")
    def forecasts(request: ForecastRequest):
        from .simulation import forecast

        return launch(
            "forecast",
            request.model_dump(),
            lambda: {"forecast_id": forecast(store, **request.model_dump())["id"]},
        )

    @app.post("/api/train")
    def train():
        def work():
            from threadpoolctl import threadpool_limits

            from .decision import train_decisions

            with threadpool_limits(limits=1):
                active = store.get("decision_model") if store.list("decision_model", 1) else {}
                if active.get("fusion_model_id"):
                    from .context_archive import import_context
                    from .fusion import refit_active_fusion

                    asyncio.run(import_context(store))
                    result = refit_active_fusion(store)
                else:
                    result = train_decisions(store)
            return {"decision_model_id": result["id"]}

        return launch("train", {}, work)

    @app.post("/api/decision-audit")
    def decision_audit():
        from .decision_audit import evaluate_choices

        return launch("decision-audit", {}, lambda: evaluate_choices(store))

    @app.post("/api/uncertainty-audit")
    def uncertainty_audit():
        from threadpoolctl import threadpool_limits

        from .decision import validate_uncertainty

        def work():
            with threadpool_limits(limits=1):
                return validate_uncertainty(store)

        return launch("uncertainty-audit", {}, work)

    @app.post("/api/managers/benchmark")
    def manager_benchmark():
        from .manager_audit import benchmark_managers

        return launch("manager-benchmark", {}, lambda: asyncio.run(benchmark_managers(store)))

    @app.post("/api/research/refresh")
    def refresh_research(request: ResearchRequest):
        from .context import research_refresh

        return launch(
            "research",
            request.model_dump(),
            lambda: asyncio.run(research_refresh(store, request.urls, request.limit)),
        )

    @app.post("/api/managers/refresh")
    def manager_refresh():
        from .manager_behavior import collect_top_managers, train_behavior

        def work():
            asyncio.run(collect_top_managers(store))
            result = train_behavior(store)
            return {"manager_behavior_id": result["id"]}

        return launch("top-managers", {}, work)

    @app.post("/api/coherent/compare")
    def compare_football_worlds():
        from .coherent import compare_football_worlds as run

        return launch("football-world-comparison", {}, lambda: run(store))

    @app.post("/api/strategy/analyze")
    def strategy_analysis():
        from .decision_runtime import analyze_decisions

        return launch("strategy-analysis", {}, lambda: analyze_decisions(store))

    @app.post("/api/strategy/timing")
    def strategy_timing():
        from .decision_runtime import timing_analysis

        return launch("timing-analysis", {}, lambda: timing_analysis(store))

    @app.post("/api/strategy/refine")
    def strategy_refine():
        from .decision_runtime import refine_frontier

        return launch("strategy-refine", {}, lambda: refine_frontier(store))

    @app.post("/api/strategy/train")
    def strategy_train():
        from .workflow import full_training

        return launch("complete-training", {}, lambda: full_training(store))

    @app.post("/api/refresh-schedule")
    def refresh_schedule(request: RefreshSchedule):
        identity = store.put("refresh_schedule", request.model_dump())
        return {"id": identity, **request.model_dump()}

    @app.post("/api/refresh-cycle")
    def refresh_all():
        from .workflow import refresh_cycle

        return launch("refresh-cycle", {}, lambda: refresh_cycle(store))

    @app.post("/api/population/analyze")
    def population_analysis():
        from .population import collect_population, evaluate_population

        def work():
            asyncio.run(collect_population(store))
            return evaluate_population(store)

        return launch("population-analysis", {}, work)

    @app.post("/api/policy/analyze")
    def policy_analysis():
        from .policy_learning import personalized_analysis

        return launch("policy-analysis", {}, lambda: personalized_analysis(store))

    @app.post("/api/intelligence/refresh")
    def refresh_intelligence_sources():
        from .intelligence import refresh_sources

        return launch("sources-refresh", {}, lambda: asyncio.run(refresh_sources(store)))

    @app.post("/api/fusion/train")
    def fusion_train():
        def work():
            try:
                __import__("torch")
            except ImportError:
                raise ValueError("Advanced CPU training requires: uv sync --extra neural") from None
            from .context_archive import import_context
            from .fusion import compare_live, review_fusion, train_fusion

            asyncio.run(import_context(store))
            result = train_fusion(store)
            review_fusion(store)
            compare_live(store)
            return {"fusion_model_id": result["id"], "selected": result["selected"]}

        return launch("fusion-train", {}, work)

    @app.post("/api/scenarios")
    def scenarios(request: ScenarioRequest):
        from .scenarios import ScenarioSpec, evaluate_scenarios

        specs = [ScenarioSpec.model_validate(row) for row in request.scenarios]
        return launch(
            "scenarios", request.model_dump(), lambda: evaluate_scenarios(store, request.plan_id, specs)
        )

    @app.post("/api/open-benchmark")
    def open_benchmark():
        from .open_models import reproduce

        return launch("open-models", {}, lambda: reproduce(store))

    @app.post("/api/cohort/refresh")
    def refresh_cohort():
        from .cohorts import sample_current_rank

        return launch("cohort", {}, lambda: asyncio.run(sample_current_rank(store)))

    @app.post("/api/freeze")
    def freeze_current():
        from .prospective import freeze

        return freeze(store, store.get("selected_plan")["plan_id"])

    @app.post("/api/chips/compare")
    def chip_comparison():
        from .chips import compare_chips

        return launch(
            "chip-counterfactual", {}, lambda: compare_chips(store, store.get("selected_plan")["plan_id"])
        )

    @app.post("/api/compare-hits")
    def compare_hits():
        from .chips import compare_immediate_hit

        return launch(
            "compare-hits", {}, lambda: compare_immediate_hit(store, store.get("selected_plan")["plan_id"])
        )

    @app.post("/api/policy-replay")
    def policy_replay():
        from .decision_audit import replay_policy

        def work():
            result = replay_policy(store)
            return {
                "evaluation_id": result["id"],
                "policies": {k: v["realized_net_points"] for k, v in result["policies"].items()},
            }

        return launch("policy-replay", {}, work)

    @app.post("/api/rivals")
    def refresh_rivals(request: RivalRequest):
        def work():
            from .rivals import case_comparison, discover

            league = asyncio.run(discover(store, request.league_id, request.entry_id))
            if request.compare_case:
                comparison = case_comparison(store, store.get("state")["id"], store.get("forecast")["id"], 30)
                return {"comparison_id": comparison["id"]}
            return {"league_id": league["id"]}

        return launch("rivals", request.model_dump(), work)

    @app.post("/api/optimize")
    def optimization(request: OptimizeRequest):
        def work():
            plan = optimize(
                store,
                request.state_id,
                request.forecast_id,
                request.end_event,
                request.time_limit,
                chips=request.chips or None,
                max_transfers=request.max_transfers,
                first_week_hit_budget=request.first_week_hit_budget,
            )
            hold = hold_plan(store, request.state_id, request.forecast_id, request.end_event)
            if request.objective == "beat_both":
                from .rivals import compare

                state = store.get("state", request.state_id)
                comparison = compare(
                    store,
                    [plan, hold],
                    request.forecast_id,
                    store.get("league")["id"],
                    state["points"],
                    state["entry_id"],
                    request.sacrifice_budget,
                )
                plan = store.get("plan", comparison["selected_plan_id"])
            with state_lock:
                current = store.get("state")["id"]
                current_forecast = store.get("forecast")["id"]
                stale = current != request.state_id or current_forecast != request.forecast_id
                if not stale:
                    store.put("selected_plan", {"plan_id": plan["id"], "at": now()})
            return {"plan_id": plan["id"], "hold_id": hold["id"], "stale": stale}

        return launch("optimize", request.model_dump(), work)

    @app.get("/api/jobs/{identity}")
    def job(identity: str):
        with store.db() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (identity,)).fetchone()
        if row is None:
            raise HTTPException(404, "Unknown job")
        value = dict(row)
        value["result"] = json.loads(value["result"]) if value["result"] else None
        return value

    @app.post("/api/state")
    def save_state(request: StateImport):
        players, _ = load_market(store)
        rules = Rules.from_bootstrap(store.latest("bootstrap")[0])
        validate_squad(request.state, players, rules)
        with state_lock:
            if request.expected_state_id and store.get("state")["id"] != request.expected_state_id:
                raise HTTPException(409, "Account changed; reload before saving")
            identity = store.put("state", request.state.model_dump())
            store.event("account_import", identity, {"status": request.state.status})
        return {"state_id": identity}

    @app.post("/api/state/seed")
    def seed_state():
        path = Path(__file__).resolve().parents[2] / "research" / "04_USER_CONTEXT.json"
        return reconcile_seed(store, json.loads(path.read_text()))

    @app.post("/api/transfers/check")
    def check_transfers(request: TransferCheck):
        players, _ = load_market(store)
        rules = Rules.from_bootstrap(store.latest("bootstrap")[0])
        state = TeamState.model_validate(
            {k: v for k, v in store.get("state", request.state_id).items() if k != "id"}
        )
        next_state, ledger = transition(state, request.action, players, rules)
        return {"valid": True, "next_state": next_state.model_dump(), "ledger": ledger}

    @app.post("/api/evidence")
    def evidence(request: ClaimInput):
        return {"claim_id": save_claim(store, request.claim, request.source_text)}

    @app.post("/api/beliefs")
    def belief(request: Belief):
        players, _ = load_market(store)
        if request.player_id not in players:
            raise ValueError("Unknown player")
        for identity in request.evidence_ids:
            store.get("claim", identity)
        return {"belief_id": store.put("belief", request.model_dump())}

    @app.get("/api/players/{identity}/history")
    def player_history(identity: int):
        return store.latest(f"history_{identity}")[0]

    @app.get("/api/export/{identity}/{format}", response_class=PlainTextResponse)
    def export_artifact(identity: str, format: str):
        from .reports import calendar, export, render

        if format == "ics":
            return PlainTextResponse(calendar(store), media_type="text/calendar")
        plan = read_plan(identity)
        if format == "json":
            return PlainTextResponse(json.dumps(plan, indent=2), media_type="application/json")
        if format == "md":
            return PlainTextResponse(render(store, plan))
        if format == "csv":
            import tempfile

            with tempfile.TemporaryDirectory() as path:
                export(store, plan, Path(path))
                return PlainTextResponse((Path(path) / "ledger.csv").read_text(), media_type="text/csv")
        raise HTTPException(404, "Unknown format")

    @app.get("/api/plans/{identity}")
    def get_plan(identity: str):
        return read_plan(identity)

    def read_plan(identity: str):
        try:
            return store.get("plan", identity)
        except ValueError:
            return store.get("scenario_plan", identity)

    @app.get("/api/schema")
    def schema():
        return {
            "state": TeamState.model_json_schema(),
            "action": Action.model_json_schema(),
            "claim": Claim.model_json_schema(),
            "belief": Belief.model_json_schema(),
        }

    dist = Path(__file__).resolve().parents[2] / "apps" / "web" / "dist"
    if dist.exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="web")
    return app
