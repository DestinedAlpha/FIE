import { useEffect, useState } from "react";
import { Activity, FlaskConical, RefreshCw, ShieldCheck } from "lucide-react";
import { api, Dashboard, Plan, money, percent } from "./types";

const chipNames: Record<string, string> = { freehit: "Free Hit", bboost: "Bench Boost", wildcard: "Wildcard", "3xc": "Triple Captain" };

type Props = {
  data: Dashboard;
  busy: boolean;
  act: (path: string, body?: unknown) => Promise<unknown>;
};

type TrainingStatus = {
  decision_model_id?: string;
  status: string;
  warning: string | null;
  finalized_through?: string | null;
  explanation?: string;
  components: {name: string; id: string; fit_max: string; calibration_min: string; calibration_max: string; training_rows: number; calibration_rows: number}[];
};

export function TrainingStatusPanel({modelId}: {modelId?: string}) {
  const [result, setResult] = useState<TrainingStatus | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    setResult(null); setError("");
    api<TrainingStatus>("/models/training").then(value => {if(active)setResult(value);}).catch(e=>{if(active)setError(String(e));});
    return () => {active = false;};
  }, [modelId]);
  return <section className="panel">
    <h3>Automatic learning and model freshness</h3>
    {!result && <p role="status">{error || "Loading training provenance…"}</p>}
    {result && <>
      <p>{result.status}. Official finalized results: {result.finalized_through || "none"}.</p>
      {result.warning && <p role="alert">Last update failed; accepted composition retained: {result.warning}</p>}
      <div className="table-scroll"><table><thead><tr><th>Component</th><th>Weights fitted through</th><th>Calibration labels</th><th>Fit / calibration rows</th></tr></thead><tbody>{result.components.map(c=><tr key={c.id}><td title={c.id}>{c.name}</td><td>{c.fit_max}</td><td>{c.calibration_min} – {c.calibration_max}</td><td>{c.training_rows.toLocaleString()} / {c.calibration_rows.toLocaleString()}</td></tr>)}</tbody></table></div>
      <p>{result.explanation}</p>
      <a href="/api/models/training" target="_blank" rel="noreferrer">Exact model IDs and training provenance</a>
    </>}
  </section>;
}

type ForecastDiagnostics = {
  population: {
    forecast_id: string;
    cohorts: {position: number; lead: number; players: number}[];
    independent_status: string;
    aligned_cohorts: {position: number; lead: number; fixtures: number; point_bias: number; minute_bias: number}[];
    material_disagreements: {player_id: number; event: number; points_difference: number; minutes_difference: number; classification: string}[];
    historical: {compatible: boolean; periods: Record<string, {all: {points: {n: number; rmse: number}; any: {brier: number}; sixty: {brier: number}; mean_crps: number; interval_80_coverage: number}}> } | null;
    challenger: {decision: string; promoted: boolean} | null;
    interpretation: string;
  };
};

export function ForecastDiagnosticsPanel({data}: {data: Dashboard}) {
  const [result,setResult] = useState<ForecastDiagnostics | null>(null);
  const [error,setError] = useState("");
  useEffect(() => {
    let active = true;
    setResult(null); setError("");
    if (data.forecast?.id) api<ForecastDiagnostics>("/forecast/diagnostics").then(value => {if(active)setResult(value);}).catch(e=>{if(active)setError(String(e));});
    return () => {active=false;};
  },[data.forecast?.id]);
  const report=result?.population;
  return <section className="panel">
    <h3>Population forecast diagnostics</h3>
    {!report && <p role="status">{error || (data.forecast ? "Loading forecast diagnostics…" : "Generate a forecast to inspect its population.")}</p>}
    {report && <>
      <p>{report.cohorts.reduce((n,r)=>n+r.players,0).toLocaleString()} player-gameweek projections checked. {report.independent_status}.</p>
      {report.historical && <>
        <p>{report.historical.compatible ? "Historical evaluation matches the accepted immediate model contract." : "Historical evaluation uses an earlier model contract; it does not certify this forecast."}</p>
        <div className="table-scroll"><table><thead><tr><th>Season</th><th>Fixtures</th><th>Points RMSE</th><th>Appearance Brier</th><th>60-minute Brier</th><th>CRPS</th><th>80% coverage</th></tr></thead><tbody>{Object.entries(report.historical.periods).map(([season,p])=><tr key={season}><td>{season}</td><td>{p.all.points.n.toLocaleString()}</td><td>{p.all.points.rmse.toFixed(3)}</td><td>{p.all.any.brier.toFixed(3)}</td><td>{p.all.sixty.brier.toFixed(3)}</td><td>{p.all.mean_crps.toFixed(3)}</td><td>{percent(p.all.interval_80_coverage)}</td></tr>)}</tbody></table></div>
      </>}
      {report.challenger && <p>Football component challenger: {report.challenger.decision}.</p>}
      {!!report.material_disagreements.length && <details><summary>{report.material_disagreements.length} material independent projection disagreements</summary><div className="table-scroll"><table><thead><tr><th>Player</th><th>GW</th><th>FIE − independent xPts</th><th>Minutes difference</th><th>Assumption to inspect</th></tr></thead><tbody>{report.material_disagreements.slice(0,30).map(r=><tr key={`${r.player_id}-${r.event}`}><td>{data.market?.players.find(p=>p.id===r.player_id)?.web_name || r.player_id}</td><td>{r.event}</td><td>{r.points_difference.toFixed(2)}</td><td>{r.minutes_difference.toFixed(1)}</td><td>{r.classification.replaceAll('_',' ')}</td></tr>)}</tbody></table></div></details>}
      <p>{report.interpretation}</p>
      <a href="/api/forecast/diagnostics" download="forecast-diagnostics.json">Download all cohorts, calibration and disagreements</a>
    </>}
  </section>;
}

export function ResearchControls({ data, busy, act }: Props) {
  const selection = data.extraction_benchmark?.selected;
  return (
    <section className="panel analysis-controls">
      <div>
        <h3>Original sources, checked claims.</h3>
        <p>
          {selection
            ? `${selection.model} selected from measured extraction tests. Provider authentication status appears below.`
            : "Configure an eligible free provider to automate interpretation."}
        </p>
      </div>
      <button
        className="button primary"
        disabled={busy}
        onClick={() => act("/research/refresh", {})}
      >
        <RefreshCw size={14} />
        Refresh research
      </button>
      {data.provider_setup?.providers.map((p) => (
        <p key={p.provider}>
          {p.provider}: {p.status}
          {p.http_status ? ` (HTTP ${p.http_status})` : ""}
          {p.message ? ` · ${p.message}` : ""}
        </p>
      ))}
      {selection && (
        <small>
          Labelled extraction F1 {percent(selection.f1)} · source-status
          accuracy {percent(selection.status_accuracy)} · small synthetic test;
          claims still require source support.
        </small>
      )}
    </section>
  );
}

export function CohortPanel({ data, busy, act }: Props) {
  const cohort = data.cohort;
  return (
    <section className="panel">
      <div className="section-title">
        <div>
          <h3>Observed manager exposure</h3>
          <p>
            {cohort
              ? `${cohort.retrieved} entries actually retrieved; ${cohort.misses} misses. GW${cohort.event}.`
              : "Sample released picks with a disclosed selection frame."}
          </p>
        </div>
        <button
          className="button secondary"
          disabled={busy}
          onClick={() => act("/cohort/refresh", {})}
        >
          Refresh sample
        </button>
      </div>
      {cohort && (
        <>
          <div className="solver-note">
            {cohort.population}. Effective ownership includes realized captain
            and substitution multipliers.
          </div>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Player</th>
                  <th>Effective ownership</th>
                  <th>Entry-resampling interval</th>
                </tr>
              </thead>
              <tbody>
                {[...cohort.effective_ownership]
                  .sort((a, b) => b.effective_ownership - a.effective_ownership)
                  .slice(0, 8)
                  .map((r) => (
                    <tr key={r.player_id}>
                      <td>
                        {data.market?.players.find((p) => p.id === r.player_id)
                          ?.web_name || r.player_id}
                      </td>
                      <td>{r.effective_ownership.toFixed(1)}%</td>
                      <td>
                        {r.resampling_interval_95
                          .map((v) => v.toFixed(1))
                          .join("–")}
                        %
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </section>
  );
}

export function ModelEvidence({ data, busy, act }: Props) {
  const suite = data.model_suite,
    policies = data.policy_evaluation,
    open = data.open_model_reproduction;
  return (
    <>
      <section className="panel">
        <div className="section-title">
          <div>
            <h3>Calibrated playing time</h3>
            <p>
              Training, calibration and test deadlines are separated
              chronologically.
            </p>
          </div>
          <span className="pill green">
            {suite?.promoted ? "Promotion gate passed" : "Prior retained"}
          </span>
        </div>
        {suite && (
          <div className="data-details">
            <div>
              <span>Held-out observations</span>
              <b>{suite.test_rows.toLocaleString()}</b>
            </div>
            <div>
              <span>Start Brier score · lower is better</span>
              <b>
                {suite.baseline.start.brier.toFixed(3)} →{" "}
                {suite.candidate.start.brier.toFixed(3)}
              </b>
            </div>
            <div>
              <span>Appearance Brier score</span>
              <b>
                {suite.baseline.appearance.brier.toFixed(3)} →{" "}
                {suite.candidate.appearance.brier.toFixed(3)}
              </b>
            </div>
            <div>
              <span>Probability calibration</span>
              <b>
                Fitted on {suite.calibration_rows.toLocaleString()} earlier
                validation rows
              </b>
            </div>
          </div>
        )}
      </section>
      <section className="panel">
        <div className="section-title">
          <div>
            <h3>Continuous policy replay</h3>
            <p>
              Equal initial squads. Evolving bank, transfers, captaincy and
              actual observed points.
            </p>
          </div>
          <button
            className="button secondary"
            disabled={busy}
            onClick={() => act("/policy-replay", {})}
          >
            Run replay
          </button>
        </div>
        {policies && (
          <>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Policy</th>
                    <th>Realized net points</th>
                    <th>Hits</th>
                    <th>Transfers</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(policies.policies).map(([name, p]) => (
                    <tr key={name}>
                      <td>{name}</td>
                      <td>
                        <b>{p.realized_net_points}</b>
                      </td>
                      <td>{p.hits}</td>
                      <td>{p.transfers}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="solver-note">
              {policies.events} historical gameweeks ·{" "}
              {policies.invalid_exported_actions} invalid exported actions.
              Chip-free comparisons isolate forecasts and transfers; archival
              prices are statistical replay inputs.
            </div>
          </>
        )}
      </section>
      <section className="panel">
        <div className="section-title">
          <div>
            <h3>Open numerical model checks</h3>
            <p>
              Version-pinned artifacts and source identities; reproduction is
              separate from generalization.
            </p>
          </div>
          <button
            className="button secondary"
            disabled={busy}
            onClick={() => act("/open-benchmark", {})}
          >
            <FlaskConical size={14} />
            Reproduce
          </button>
        </div>
        {open && (
          <div className="data-details">
            <div>
              <span>
                Dastan · {open.dastan.rows.toLocaleString()} packaging holdout
                rows
              </span>
              <b>
                RMSE {open.dastan.metrics.rmse.toFixed(3)} ·{" "}
                {open.dastan.features} features
              </b>
            </div>
            <div>
              <span>
                OpenFPL · {open.openfpl.rows.toLocaleString()} released
                historical vectors
              </span>
              <b>RMSE {open.openfpl.metrics.rmse.toFixed(3)}</b>
            </div>
          </div>
        )}
        <div className="solver-note">
          These checks reproduce disclosed historical artifacts. Prospective
          season-rank performance needs future outcomes.
        </div>
      </section>
    </>
  );
}

export function ScenarioPanel({
  data,
  busy,
  act,
  plan,
}: Props & { plan: Plan | null | undefined }) {
  const [kind, setKind] = useState("absence"),
    [player, setPlayer] = useState(0),
    [reveal, setReveal] = useState(10),
    [delta, setDelta] = useState(0.2),
    [fixtureId,setFixtureId] = useState(0),
    [destination,setDestination] = useState(0);
  if (!plan || !data.state) return null;
  const minimum = data.state.event + 1,
    maximum = plan.ledger.at(-1)?.event || 19;
  const selected = player || data.state.lots[0].player_id;
  const when = Math.max(minimum, Math.min(reveal, maximum));
  const fixtures=(data.market?.fixtures||[]).filter(f=>!f.finished && (f.event===null||f.event>=when));
  const selectedFixture=fixtures.find(f=>f.id===fixtureId)||fixtures[0];
  const teamName=(id:number)=>data.market?.teams.find(t=>t.id===id)?.short_name||String(id);
  const evaluate = () => {
    if(kind==='fixture') {
      if(!selectedFixture)return;
      act('/scenarios',{plan_id:plan.id,scenarios:[{
        name:`${teamName(selectedFixture.team_h)} v ${teamName(selectedFixture.team_a)} ${destination?`moves to GW${destination}`:'postponed, unassigned'}`,
        reveal_event:when,fixture_events:[{fixture_id:selectedFixture.id,event:destination||null}],sale_assumption:'conservative'
      }]});
      return;
    }
    const target = data.market?.players.find((p) => p.id === selected);
    const scenario = {
      name:
        kind === "absence"
          ? `${target?.web_name} unavailable from GW${when}`
          : `${target?.web_name} price ${delta >= 0 ? "+" : ""}£${delta.toFixed(1)}m in GW${when}`,
      reveal_event: when,
      sale_assumption: "conservative",
      ...(kind === "absence"
        ? {
            players: [
              {
                player_id: selected,
                from_event: when,
                through_event: Math.min(when + 1, maximum),
                availability: 0,
              },
            ],
          }
        : {
            prices: [
              {
                player_id: selected,
                event: when,
                price: Math.round((target?.price || 0) + delta * 10),
              },
            ],
          }),
    };
    act("/scenarios", { plan_id: plan.id, scenarios: [scenario] });
  };
  return (
    <section className="panel scenario-controls">
      <h3>If the picture changes</h3>
      <p>
        Test a future observation. Every earlier transfer, XI, captain and chip
        stays identical.
      </p>
      <div className="form-row">
        <label>
          Scenario
          <select value={kind} onChange={(e) => setKind(e.target.value)}>
            <option value="absence">Player unavailable</option>
            <option value="price">Price movement</option>
            <option value="fixture">Fixture rearrangement</option>
          </select>
        </label>
        {kind !== 'fixture' && <label>
          Player
          <select value={selected} onChange={(e) => setPlayer(+e.target.value)}>
            {data.market?.players.map((p) => (
              <option key={p.id} value={p.id}>
                {p.web_name}
              </option>
            ))}
          </select>
        </label>}
        <label>
          Observed in
          <select value={when} onChange={(e) => setReveal(+e.target.value)}>
            {Array.from(
              { length: Math.max(0, maximum - minimum + 1) },
              (_, i) => minimum + i,
            ).map((e) => (
              <option value={e} key={e}>
                GW {e}
              </option>
            ))}
          </select>
        </label>
        {kind === 'fixture' && <>
          <label>Fixture<select value={selectedFixture?.id||0} onChange={e=>setFixtureId(+e.target.value)}>
            {fixtures.map(f=><option key={f.id} value={f.id}>{teamName(f.team_h)} v {teamName(f.team_a)} · {f.event?`GW${f.event}`:'unassigned'}</option>)}
          </select></label>
          <label>Reassigned to<select value={destination} onChange={e=>setDestination(+e.target.value)}>
            <option value={0}>Unassigned / postponed</option>
            {(data.market?.events||[]).filter(e=>e.id>=when).map(e=><option key={e.id} value={e.id}>GW {e.id}</option>)}
          </select></label>
        </>}
        {kind === "price" && (
          <label>
            Price change (£m)
            <input
              type="number"
              min={-2}
              max={2}
              step={0.1}
              value={delta}
              onChange={(e) => setDelta(+e.target.value)}
            />
          </label>
        )}
        <button
          className="button primary"
          disabled={busy || minimum > maximum || (kind==='fixture' && !selectedFixture)}
          onClick={evaluate}
        >
          <Activity size={14} />
          Evaluate branch
        </button>
      </div>
      <small>
        No branch probability is invented. Future purchase-lot assumptions are
        shown separately from confirmed current sale quotes.
      </small>
      {data.scenario_evaluation?.branches.map((b) => (
        <div className="scenario-result" key={b.plan_id}>
          <ShieldCheck size={16} />
          <div>
            <b>{b.name}</b>
            <p>
              {b.simulation.mean.toFixed(1)} projected net points · shared
              decisions before GW{b.reveal_event}
              {b.paired_scenario_delta !== undefined &&
                ` · ${b.paired_scenario_delta.toFixed(1)} points versus the same adaptive control`}
            </p>
            <small>
              Legal, affordable ledger · conditional future sale-price
              assumptions
            </small>
            <p><a href={`/api/export/${b.plan_id}/json`} download>Download branch ledger</a></p>
          </div>
        </div>
      ))}
    </section>
  );
}

export function ChipPanel({
  data,
  plan,
  busy,
  act,
}: Props & { plan: Plan | null | undefined }) {
  if (!plan) return null;
  const comparisons =
    data.chip_counterfactual?.base_plan_id === plan.id
      ? data.chip_counterfactual.comparisons
      : [];
  return (
    <section className="panel analysis-controls">
      <div>
        <h3>Chip value across the whole plan</h3>
        <p>
          Compare preparation, transfers and recovery in identical match worlds.
          Freeze a shadow decision before the deadline to measure its later
          result.
        </p>
        {comparisons.map((c) => (
          <p key={`${c.chip}-${c.window?.[0]}`}>
            <b>
              {chipNames[c.chip] ?? c.chip}
              {c.chosen_event == null ? " · retained" : ` · GW${c.chosen_event}`}
            </b>
            : {c.whole_policy_mean_delta.toFixed(1)} points versus reoptimizing
            without it · Monte Carlo interval [
            {c.paired_mc_interval_95.map((x) => x.toFixed(1)).join(", ")}] ·
            alternative solver gap {c.counterfactual_solver_gap == null ? "bound unavailable" : percent(c.counterfactual_solver_gap)}
          </p>
        ))}
        {data.chip_counterfactual?.base_plan_id === plan.id &&
          !!data.chip_counterfactual.calendar?.length && (
            <details>
              <summary>Compare every feasible chip date ({data.chip_counterfactual.dates_evaluated})</summary>
              <p>{data.chip_counterfactual.scope}</p>
              <div className="table-scroll"><table>
                <thead><tr><th>Chip</th><th>GW</th><th>Net points</th><th>Versus selected</th><th>Solver gap</th><th>Ledger</th></tr></thead>
                <tbody>{data.chip_counterfactual.calendar.map(c => (
                  <tr key={`${c.chip}-${c.event}`}><td>{chipNames[c.chip] ?? c.chip}</td>
                    <td>{c.event}</td><td>{c.mean.toFixed(1)}</td><td>{c.delta_vs_selected.toFixed(1)}</td>
                    <td>{c.solver.gap == null ? "unavailable" : percent(c.solver.gap)}</td>
                    <td><a href={`/api/export/${c.plan_id}/json`} download>Download</a></td></tr>
                ))}</tbody>
              </table></div>
            </details>
          )}
        <small>
          Intervals measure simulation noise; model and solver uncertainty
          remain separate.
        </small>
      </div>
      <div className="form-row">
        <button
          className="button"
          disabled={busy || !!plan.stale}
          onClick={() => act("/chips/compare", {})}
        >
          Compare chip value
        </button>
        <button
          className="button"
          disabled={busy || !!plan.stale}
          onClick={() => act("/freeze", {})}
        >
          Freeze shadow decision
        </button>
      </div>
    </section>
  );
}

export function DecisionEvidence({ data, busy, act }: Props) {
  const model = data.decision_model,
    audit = data.decision_audit,
    policy = data.decision_policy,
    managers = data.manager_audit;
  if (!model)
    return (
      <section className="panel">
        <h3>Decision training required</h3>
        <p>
          Run chronological training before using the old simulator's
          recommendations.
        </p>
      </section>
    );
  return (
    <>
      <section className="panel">
        <div className="section-title">
          <div>
            <h3>The model that makes these decisions</h3>
            <p>
              The conditional points model and separately validated participation
              model supply the forecast worlds and transfer planner.
            </p>
          </div>
          <span className="pill green">
            {model.promoted ? "Held-out gate passed" : "Gate failed"}
          </span>
        </div>
        <div className="data-details">
          <div>
            <span>Historical seasons</span>
            <b>{model.seasons.join(", ")}</b>
          </div>
          <div>
            <span>Held-out player-fixtures</span>
            <b>{model.candidate.n.toLocaleString()}</b>
          </div>
          <div>
            <span>Base model points RMSE · lower is better</span>
            <b>
              {model.baseline.rmse.toFixed(3)} →{" "}
              {model.candidate.rmse.toFixed(3)}
            </b>
          </div>
          <div>
            <span>Likely starters RMSE</span>
            <b>
              {model.regular_starters.baseline.rmse.toFixed(3)} →{" "}
              {model.regular_starters.candidate.rmse.toFixed(3)}
            </b>
          </div>
          <div>
            <span>Production model fit</span>
            <b>
              {model.production_training_rows.toLocaleString()} earlier rows;{" "}
              {model.production_calibration_rows.toLocaleString()} separate
              calibration rows
            </b>
          </div>
        </div>
        <p>
          Every test block uses earlier training and calibration deadlines.
          Current-week outcomes and hindsight winners never enter its features.
        </p>
        {model.role_minutes_model_id && data.role_minutes_validation && (
          <>
            <h4>Role-aware minutes</h4>
            <p>Recent starts, archived availability and observed workload inform immediate participation. Future fixtures use their separate accepted expert. Completed labels: {data.role_minutes_model?.completed_labels_through}.</p>
            <div className="data-details">
              {Object.entries(data.role_minutes_validation.metrics).map(([period, result]) => (
                <div key={period}>
                  <span>{period === "acceptance_2024" ? "2024/25 GW20–38 acceptance" : "2025/26 replication"}</span>
                  <b>Minutes Brier {result.reference.brier.toFixed(4)} → {result.candidate.brier.toFixed(4)}; points RMSE {Math.sqrt(result.reference.points_squared).toFixed(3)} → {Math.sqrt(result.candidate.points_squared).toFixed(3)}</b>
                </div>
              ))}
            </div>
            <p>{data.role_minutes_validation.scope}</p>
          </>
        )}
        <button
          className="button secondary"
          disabled={busy}
          onClick={() => act("/train", {})}
        >
          Retrain and validate
        </button>
        <div className="form-row">
          <button
            className="button"
            disabled={busy}
            onClick={() => act("/decision-audit", {})}
          >
            Check captain and team choices
          </button>
          <button
            className="button"
            disabled={busy}
            onClick={() => act("/policy-replay", {})}
          >
            Replay transfer policy
          </button>
          <button
            className="button"
            disabled={busy}
            onClick={() => act("/uncertainty-audit", {})}
          >
            Check uncertainty
          </button>
          <button
            className="button"
            disabled={busy}
            onClick={() => act("/managers/benchmark", {})}
          >
            Compare manager decisions
          </button>
        </div>
      </section>
      {audit && (
        <section className="panel">
          <h3>Choices, scored after the deadline</h3>
          <div className="data-details">
            <div>
              <span>
                Learned captain / price baseline / rolling-points baseline
              </span>
              <b>
                {audit.captain_totals.learned_bonus} /{" "}
                {audit.captain_totals.price_baseline_bonus} /{" "}
                {audit.captain_totals.shrinkage_bonus} captain bonus points
              </b>
            </div>
            <div>
              <span>Single-week legal teams · same £100m budget</span>
              <b>
                {audit.team_totals.learned} learned /{" "}
                {audit.team_totals.shrinkage} baseline
              </b>
            </div>
            <div>
              <span>Best with perfect hindsight</span>
              <b>{audit.team_totals.oracle} · diagnostic upper bound</b>
            </div>
          </div>
          <p>
            The hindsight team knows who scored. Its gap includes unpredictable
            match variance; the model is never trained to reproduce test-week
            winners.
          </p>
          <button
            className="button"
            disabled={busy}
            onClick={() => act("/decision-audit", {})}
          >
            Recheck decision quality
          </button>
        </section>
      )}
      {policy && (
        <section className="panel">
          <h3>Thirty weeks with a real transfer ledger</h3>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Policy</th>
                  <th>Observed net points</th>
                  <th>Hit points</th>
                  <th>Transfers</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(policy.policies).map(([name, p]) => (
                  <tr key={name}>
                    <td>{name}</td>
                    <td>{p.realized_net_points}</td>
                    <td>{p.hits}</td>
                    <td>{p.transfers}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p>
            Same initial squad and bankroll. Earlier-fold predictions, actual
            points, legal transfers and autosubs. Chips are excluded to isolate
            forecast and transfer effects. Archive prices remain retrospective.
          </p>
        </section>
      )}
      {data.decision_uncertainty && (
        <section className="panel">
          <h3>Uncertainty checked on held-out outcomes</h3>
          <p>
            Likely starters:{" "}
            {percent(data.decision_uncertainty.regular_starters.coverage80)} of
            outcomes inside the central 80% interval;{" "}
            {percent(data.decision_uncertainty.regular_starters.coverage90)}{" "}
            inside the central 90% interval. Discrete points can make central
            coverage conservative. Marginal calibration does not prove the rival
            model's joint probabilities.
          </p>
        </section>
      )}
      {managers && (
        <section className="panel">
          <h3>Manager decisions as an independent comparator</h3>
          <p>
            {managers.retrieved} actual squads; {managers.prior_top10k_twice}{" "}
            with at least two prior top-10,000 finishes. The sample was selected
            by current rank, so it cannot stand in for all consistently strong
            managers.
          </p>
          <p>
            Same squad and chip, one released gameweek: model lineup difference{" "}
            {managers.mean_same_squad_lineup_delta.toFixed(1)} points;
            entry-resampling interval [
            {managers.entry_bootstrap_interval_95
              .map((x) => x.toFixed(2))
              .join(", ")}
            ]. This is a diagnostic, not a claim of manager superiority. Manager
            choices are not treated as correct training labels.
          </p>
        </section>
      )}
    </>
  );
}

export function CaptainEvidence({
  data,
  plan,
}: {
  data: Dashboard;
  plan: Plan | null | undefined;
}) {
  const row = plan?.ledger[0];
  if (!row || !data.forecast) return null;
  const candidates = data.forecast.players
    .filter(
      (p) => p.event === row.event && row.lineup.xi.includes(p.player_id),
    )
    .sort((a, b) => {
      const score=(id:number, fallback:number)=>row.captain_options?.find(r=>r.captain===id)?.expected_captain_bonus ?? fallback;
      return score(b.player_id,b.mean)-score(a.player_id,a.mean);
    })
    .slice(0, 5);
  if (!candidates.length) return null;
  return (
    <section className="panel">
      <h3>Why this captain?</h3>
      <p>
        Captaincy compares the extra points from captain and vice together.
        If the captain does not appear, the vice receives the multiplier.
        A lower individual forecast can therefore have the higher captain bonus.
      </p>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Candidate</th>
              <th>Projected points</th>
              <th>Captain bonus including backup</th>
              <th>Best backup</th>
              <th>Chance of 60+ minutes in a fixture</th>
              <th>10–90% outcome</th>
            </tr>
          </thead>
          <tbody>
            {candidates.map((p) => (
              <tr key={p.player_id}>
                <td>
                  {
                    data.market?.players.find((v) => v.id === p.player_id)
                      ?.web_name
                  }
                  {p.player_id === row.lineup.captain ? " · captain" : ""}
                </td>
                <td>{p.mean.toFixed(2)}</td>
                <td>{row.captain_options?.find(r=>r.captain===p.player_id)?.expected_captain_bonus.toFixed(2) ?? 'Recalculate plan'}</td>
                <td>{data.market?.players.find(v=>v.id===row.captain_options?.find(r=>r.captain===p.player_id)?.vice)?.web_name ?? '—'}</td>
                <td>{percent(p.p_60)}</td>
                <td>
                  {p.point_quantiles[0].toFixed(1)}–
                  {p.point_quantiles[2].toFixed(1)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p>
        These are expectations with overlapping outcome ranges. A forecast
        advantage does not guarantee a higher score this week.
        {" "}<a href="/api/forecast/diagnostics" target="_blank" rel="noreferrer">Forecast validation and independent comparison</a>
      </p>
    </section>
  );
}

export function HitEvidence({
  data,
  plan,
  busy,
  act,
  explore,
}: Props & { plan: Plan | null | undefined; explore: (id: string) => void }) {
  if (!plan) return null;
  const receipt = data.policy_selection?.selection.selected_plan_id === plan.id ? data.policy_selection : null;
  const evidence = receipt?.immediate_hit_evidence;
  const chipEvidence = receipt?.immediate_chip_evidence;
  const chipNotice = chipEvidence && !chipEvidence.supported_beyond_search_error ? <section className="panel">
      <h3>{chipNames[chipEvidence.chip] ?? chipEvidence.chip} retained</h3>
      <p>{chipEvidence.reason}. Its modeled full-horizon gain is {chipEvidence.net_mean_gain.toFixed(2)} points; the ordinary-transfer search allowance is {chipEvidence.search_allowance?.toFixed(2) ?? "unavailable"}.</p>
      <p>Later chip use remains available in the comparison. Sampling precision does not settle forecast or search uncertainty.</p>
      <button className="button secondary" onClick={() => explore(chipEvidence.candidate_plan_id)}>Inspect chip alternative</button>
    </section> : null;
  if (!plan.ledger[0].hit && receipt?.selection_version === "resource-evidence-v3" && evidence && !evidence.hit_supported_beyond_search_error) {
    return <>{chipNotice}<section className="panel">
      <h3>Paid transfer withheld</h3>
      <p>{evidence.reason}. The raw paid-hit proposal gains only {evidence.net_mean_gain.toFixed(2)} modeled points over the full horizon;
        {evidence.no_hit_search_allowance == null ? " the no-hit search has no usable global bound." : ` its additive search remains unresolved by ${evidence.no_hit_search_allowance.toFixed(2)} points.`}</p>
      <p>More simulation draws cannot remove forecast or search error. These are separate from football outcome uncertainty.</p>
      {!!evidence.source_forecast_review?.length && <details><summary>Inspect the player forecasts behind the paid proposal</summary>
        <p>Official next-week estimates are an independent sensitivity, not ground truth. Availability and starting probability are different quantities.</p>
        <div className="table-scroll"><table><thead><tr><th>Player</th><th>Fixture</th><th>Model points</th><th>Official estimate</th><th>Model minutes</th><th>Model start chance</th><th>Season starts</th></tr></thead>
          <tbody>{evidence.source_forecast_review.map(r => <tr key={r.player_id}><td>{r.name}</td><td>{r.fixtures.join(', ')}</td><td>{r.model_points.toFixed(2)}</td><td>{r.official_next_points?.toFixed(2) ?? 'Unavailable'}</td><td>{r.model_minutes.toFixed(1)}</td><td>{percent(r.model_start_probability)}</td><td>{r.season_starts ?? '?'} / {r.completed_club_fixtures}</td></tr>)}</tbody></table></div>
      </details>}
      {!!evidence.weekly_tradeoff?.length && <details><summary>See when the paid hit would repay its cost</summary>
        <div className="table-scroll"><table><thead><tr><th>GW</th><th>Paid proposal, net</th><th>No-hit policy</th><th>Cumulative advantage</th></tr></thead>
          <tbody>{evidence.weekly_tradeoff.map(r => <tr key={r.event}><td>{r.event}</td><td>{r.candidate_net.toFixed(2)}</td><td>{r.alternative_net.toFixed(2)}</td><td>{r.cumulative_net_advantage.toFixed(2)}</td></tr>)}</tbody></table></div>
      </details>}
      <a className="button secondary" href={`/api/export/${evidence.candidate_plan_id}/json`} download>Download raw paid-hit proposal</a>
    </section></>;
  }
  if (!plan.ledger[0].hit) return chipNotice;
  const comparison =
    data.hit_comparison?.base_plan_id === plan.id ? data.hit_comparison : null;
  return (
    <>{chipNotice}<section className="panel">
      <h3>Is the immediate hit worth it?</h3>
      <p>
        This plan costs {plan.ledger[0].hit} points now. Compare it with a
        full-horizon policy that takes no new hit this gameweek; chip timing and
        future transfers can change.
      </p>
      {comparison && (
        <>
          <p>
            Projected net points: {comparison.base_mean.toFixed(1)} with this
            plan; {comparison.alternative_mean.toFixed(1)} with no immediate
            hit. Difference {comparison.net_mean_advantage.toFixed(1)} points
            after all hit costs.
          </p>
          <p>
            Alternative immediate chip:{" "}
            {comparison.alternative_immediate_chip || "none"}. Solver gap{" "}
            {comparison.alternative_gap == null
              ? "not available"
              : percent(comparison.alternative_gap)}
            . The model and solver uncertainty are separate from simulation
            noise.
          </p>
          {comparison.paired_mc_interval_95[0] <= 0 && (
            <p>
              The paired simulation does not establish a clear advantage for
              taking the hit. Forecast and solver uncertainty are additional.
            </p>
          )}
          <button
            className="button"
            onClick={() => explore(comparison.alternative_plan_id)}
          >
            View no-hit alternative
          </button>
        </>
      )}
      <button
        className="button secondary"
        disabled={busy}
        onClick={() => act("/compare-hits", {})}
      >
        Recheck hit tradeoff
      </button>
    </section></>
  );
}

export function SourceCoverage({ data, busy, act }: Props) {
  const c = data.source_coverage;
  const f = data.fusion_model;
  return (
    <section className="panel">
      <div className="section-title">
        <div>
          <h3>Sources and model validation</h3>
          <p>
            Retrieved evidence, numerical influence and missing history are
            shown separately.
          </p>
        </div>
        <button
          className="button secondary"
          disabled={busy}
          onClick={() => act("/intelligence/refresh", {})}
        >
          Refresh sources
        </button>
      </div>
      <div className="data-details">
        <div>
          <span>Archived official news</span>
          <b>
            {c?.context_archive?.retrieved_deadlines ?? 0} verified deadlines
          </b>
        </div>
        <div>
          <span>Prior-qualified managers</span>
          <b>
            {c?.elite_cohort?.retrieved_managers ?? 0} managers ·{" "}
            {c?.elite_cohort?.retrieved_manager_gameweeks ?? 0} released
            gameweeks
          </b>
        </div>
        <div>
          <span>Other competitions</span>
          <b>
            {c?.workload?.accepted_player_matches ?? 0} player matches ·{" "}
            {c?.workload?.rejected_player_matches ?? 0} rejected records
          </b>
        </div>
        <div>
          <span>Public specialist evidence</span>
          <b>
            {c?.intelligence_refresh?.claims ?? 0} supported claims in latest
            refresh
          </b>
        </div>
      </div>
      <button className="button secondary" disabled={busy||!data.plan||data.plan.stale} onClick={()=>act('/evidence/analyze')}>Test report impact on this plan</button>
      {!!data.model_input_usage?.length && <details open><summary>What the active forecasts actually use</summary>
        <div className="table-scroll"><table><thead><tr><th>Component</th><th>Use</th><th>Inputs</th></tr></thead><tbody>
          {data.model_input_usage.map(row=><tr key={row.component}><td>{row.component}{row.feature_count !== null ? ` · ${row.feature_count} features` : ''}</td><td>{row.status}</td><td>{row.explanation}</td></tr>)}
        </tbody></table></div>
      </details>}
      {data.role_contract_audit && <p>Historical / live input check: {data.role_contract_audit.passed ? 'passed' : 'requires review'} across {data.role_contract_audit.rows} player-fixtures. Largest input difference: {Math.max(data.role_contract_audit.maximum_core_delta,data.role_contract_audit.maximum_role_source_delta).toFixed(6)}.</p>}
      {data.immediate_points_validation && <details><summary>Immediate scoring experiment · {data.immediate_points_validation.passed ? 'gates passed · comparison only' : 'reference retained'}</summary>
        <p>{data.immediate_points_validation.protocol}</p>
        <div className="table-scroll"><table><thead><tr><th>Period</th><th>Point RMSE · previous / candidate</th><th>Regular-starter RMSE</th><th>Captain points</th></tr></thead><tbody>
          {Object.entries(data.immediate_points_validation.metrics).map(([period,row])=><tr key={period}><td>{period}</td><td>{row.reference.rmse.toFixed(3)} / {row.candidate.rmse.toFixed(3)}</td><td>{row.reference.regular_rmse.toFixed(3)} / {row.candidate.regular_rmse.toFixed(3)}</td><td>{row.reference.captain_points} / {row.candidate.captain_points}</td></tr>)}
        </tbody></table></div>
      </details>}
      {data.evidence_impact && <p>{data.evidence_impact.links.length} report links reviewed; {data.evidence_impact.branches.length} numerical sensitivity branches. {data.evidence_impact.scope}</p>}
      <p>
        Manager choices lag one gameweek. Reddit posts supply leads to original
        evidence; raw Reddit text does not train the model. Unvalidated opinions
        receive no points uplift.
      </p>
      {c?.intelligence_refresh?.source_health?.map((source) => (
        <p key={source.source}>
          <b>{source.source}</b>: {source.status}
          {source.retrieved !== undefined ? ` · ${source.retrieved} items` : ""}
          {source.reason ? ` · ${source.reason}` : ""}
        </p>
      ))}
      {f ? (
        <>
          <p>
            <b>Multi-source boosted trees</b>: lagged match history, timestamped
            news, manager and market signals, workload and advanced statistics.
            Missing inputs retain explicit coverage indicators.
          </p>
          <p>
            {f.baseline.n.toLocaleString()} matched held-out observations.
            Reference model RMSE {f.baseline.rmse.toFixed(3)}.
          </p>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Candidate</th>
                  <th>RMSE ↓</th>
                  <th>Difference vs production · 95% interval</th>
                  <th>Result</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(f.candidates).filter(([key]) => key === "tree").map(([key, row]) => (
                  <tr key={key}>
                    <td>
                      Same-source boosted trees
                    </td>
                    <td>{row.metrics.rmse.toFixed(3)}</td>
                    <td>
                      {row.paired_vs_production_95ci
                        .map((v) => v.toFixed(4))
                        .join(" to ")}
                    </td>
                    <td>
                      {row.passed
                        ? "Forecast gate passed"
                        : "Production retained"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p>
            {f.promoted
              ? `Active multi-source model: ${f.selected}.`
              : "These candidates have not replaced the validated production model."}
          </p>
          {f.review?.reason && <p>{f.review.reason}</p>}
          {f.decision_validation && (
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Matched strategy</th>
                    <th>30-week net points</th>
                    <th>Captain bonus points</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(f.decision_validation.policies).filter(([key]) => key === "tree").map(
                    ([kind, policy]) => (
                      <tr key={kind}>
                        <td>
                          Same-source trees
                        </td>
                        <td>
                          {policy["transfer-learned"].realized_net_points} vs{" "}
                          {policy["transfer-shrinkage"].realized_net_points}{" "}
                          reference
                        </td>
                        <td>
                          {f.decision_validation?.captain_totals[kind]} vs{" "}
                          {f.decision_validation?.captain_totals.baseline}{" "}
                          reference
                        </td>
                      </tr>
                    ),
                  )}
                </tbody>
              </table>
            </div>
          )}
          {f.uncertainty && (
            <p>
              Held-out likely-starter interval coverage:{" "}
              {percent(f.uncertainty.regular_starters.coverage80)} inside the
              central 80% interval;{" "}
              {percent(f.uncertainty.regular_starters.coverage90)} inside the
              central 90% interval.
            </p>
          )}
          <details>
            <summary>Coverage and validation limits</summary>
            {f.limits.map((limit) => (
              <p key={limit}>{limit}</p>
            ))}
          </details>
        </>
      ) : (
        <p>
          The multi-source model has not completed its chronological evaluation
          yet.
        </p>
      )}
      {data.fusion_comparison && (
        <details>
          <summary>GW{data.fusion_comparison.event} model comparison</summary>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Player</th>
                  <th>Production xPts</th>
                  <th>Tree candidate</th>
                </tr>
              </thead>
              <tbody>
                {data.fusion_comparison.players.slice(0, 15).map((row) => (
                  <tr key={row.player_id}>
                    <td>{row.name}</td>
                    <td>{row.production.toFixed(2)}</td>
                    <td>{row.tree.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
    </section>
  );
}

export function ManagerSignals({ data, busy, act }: Props) {
  const model = data.manager_behavior,
    coverage = data.source_coverage?.top_manager_history;
  const name = (id: number) =>
    data.market?.players.find((p) => p.id === id)?.web_name || String(id);
  return (
    <section className="panel">
      <div className="section-title">
        <div>
          <h3>Top-1,000 manager behavior</h3>
          <p>
            {coverage?.retrieved_managers ?? 0} actual managers ·{" "}
            {coverage?.retrieved_manager_gameweeks ?? 0} released
            manager/gameweeks. Current rank is distinct from prior-season
            expertise.
          </p>
        </div>
        <button
          className="button secondary"
          disabled={busy}
          onClick={() => act("/managers/refresh", {})}
        >
          Refresh histories
        </button>
      </div>
      {model ? (
        <>
          <p>
            Conditional GW{model.event} transfer-count forecast. Earlier weeks
            choose the model mixture; GW{model.metrics.events.join("–")} tests:
            log loss {model.metrics.frequency_log_loss.toFixed(3)} →{" "}
            {model.metrics.model_log_loss.toFixed(3)}.
          </p>
          <div className="data-details">
            {model.classes.map((label, i) => (
              <div key={label}>
                <span>{label.replaceAll("_", " ")}</span>
                <b>{percent(model.aggregate_probabilities[i])}</b>
              </div>
            ))}
          </div>
          <p>
            Team exposure and past purchase/sale patterns carry{" "}
            {percent(model.rich_weight)} of the calibrated mixture. These are
            conditional scenarios, not knowledge of hidden transfers.
          </p>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Recently bought</th>
                  <th>Observed purchases</th>
                  <th>Recently sold</th>
                  <th>Observed sales</th>
                </tr>
              </thead>
              <tbody>
                {model.recent_purchases.slice(0, 6).map((row, i) => (
                  <tr key={row.player_id}>
                    <td>{name(row.player_id)}</td>
                    <td>{row.managers}</td>
                    <td>
                      {model.recent_sales[i]
                        ? name(model.recent_sales[i].player_id)
                        : "—"}
                    </td>
                    <td>{model.recent_sales[i]?.managers ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <details>
            <summary>Sampling and forecast limits</summary>
            {model.limits.map((limit) => (
              <p key={limit}>{limit}</p>
            ))}
          </details>
        </>
      ) : (
        <p>Refresh released histories to fit a disclosed behavioral prior.</p>
      )}
    </section>
  );
}

export function TransferLearning({ data, busy, act }: Props) {
  const horizon = data.horizon_model;
  const critic=data.decision_critic;
  const candidate=critic?.candidates[critic.architecture_selected_on_calibration||critic.selected||'tree'];
  const policy=candidate?{test_decisions:candidate.accounts,invalid_actions:critic?.invalid_actions||0,
      mean_heldout_gain:candidate.mean_gain,four_deadline_block_gain_95ci:candidate.block_gain_95ci,promoted:!!critic?.selected}:data.transfer_policy_model;
  const personal = data.personalized_policy_analysis;
  return (
    <section className="panel">
      <div className="section-title">
        <div>
          <h3>Learning transfer value across four weeks</h3>
          <p>
            Legal random accounts, observed outcomes, purged target windows and
            separate later tests.
          </p>
        </div>
        <button
          className="button secondary"
          disabled={busy || !policy}
          onClick={() => act("/strategy/analyze", {})}
        >
          Analyze my alternatives
        </button>
      </div>
      {horizon && (
        <p>
          Four-week point RMSE: {horizon.baseline.rmse.toFixed(3)} reference →{" "}
          {horizon.candidates[horizon.selected].metrics.rmse.toFixed(3)} direct
          value model. Extra advanced-statistics difference interval:{" "}
          {horizon.advanced_vs_context_95ci
            .map((v) => v.toFixed(4))
            .join(" to ")}
          .
        </p>
      )}
      {policy && (
        <p>
          {policy.test_decisions} held-out account decisions;{" "}
          {policy.invalid_actions} invalid actions. Average four-week gain{" "}
          {policy.mean_heldout_gain.toFixed(2)} points; 95% block interval{" "}
          {policy.four_deadline_block_gain_95ci
            .map((v) => v.toFixed(2))
            .join(" to ")}
          .{" "}
          {policy.promoted
            ? "Confidence gate passed."
            : "The interval includes zero; the transfer critic remains a challenger."}
        </p>
      )}
      {personal && (
        <>
          <p>
            GW{personal.event}: {personal.status}. These alternatives do not
            replace the selected plan.
          </p>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Candidate</th>
                  <th>Estimated four-week gain</th>
                  <th>Hit</th>
                  <th>Bank</th>
                </tr>
              </thead>
              <tbody>
                {personal.candidates.map((row, i) => (
                  <tr key={i}>
                    <td>
                      {row.transfers.length
                        ? row.transfers
                            .map((t) => `${t.out} → ${t.in}`)
                            .join("; ")
                        : "Roll"}
                    </td>
                    <td>{row.estimated_fourweek_gain.toFixed(2)}</td>
                    <td>{row.hit}</td>
                    <td>
                      {money(row.bank)}
                      {row.confirmed_absence_lineup_sensitivity !== undefined &&
                        row.confirmed_absence_lineup_sensitivity < -0.05 && (
                          <small>
                            {" "}
                            · confirmed-absence lineup stress{" "}
                            {row.confirmed_absence_lineup_sensitivity.toFixed(
                              1,
                            )}{" "}
                            pts
                          </small>
                        )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p>
            The historical interval is about average tested gains; it is not a
            confidence interval for an individual transfer. The main planner
            evaluates full chip policies separately.
          </p>
        </>
      )}
    </section>
  );
}

export function StrategyLab({ data, busy, act }: Props) {
  const critic = data.decision_critic,
    analysis = data.decision_analysis,
    value = data.state_value_model;
  const name = (p: number) =>
    data.market?.players.find((r) => r.id === p)?.web_name || String(p);
  return (
    <section className="panel">
      <div className="section-title">
        <div>
          <h3>Learning decisions, not just points</h3>
          <p>
            Legal transfer and chip trajectories carry the account forward.
            Hindsight outcomes supervise learning; they never enter a live
            decision.
          </p>
        </div>
        <button
          className="button secondary"
          disabled={busy}
          onClick={() => act("/strategy/analyze", {})}
        >
          Compare legal actions
        </button>
      </div>
      {critic && (
        <>
          <p>
            {critic.training_rows.toLocaleString()} earlier training actions ·{" "}
            {critic.test_rows.toLocaleString()} held-out actions. Decision
            model: {critic.selected || "reference retained by confidence gate"}.
          </p>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Decision challenger</th>
                  <th>Four-week gain</th>
                  <th>95% deadline-block interval</th>
                  <th>Deployment</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(critic.candidates).filter(([key]) => key === "tree").map(([name, m]) => (
                  <tr key={name}>
                    <td>
                      Tree decision critic
                    </td>
                    <td>{m.mean_gain.toFixed(2)}</td>
                    <td>
                      {m.block_gain_95ci.map((v) => v.toFixed(2)).join(" to ")}
                    </td>
                    <td>{m.promoted ? "Gate passed" : "Challenger"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
      {value && (
        <p>
          Continuation-value RMSE {value.baseline_rmse.toFixed(2)} →{" "}
          {value.rmse.toFixed(2)}.{" "}
          {value.promoted
            ? "Learned squad, bank, transfer and unused-chip values enter planning."
            : "The learned continuation model remains under its confidence gate."}
        </p>
      )}
      {data.horizon_consistency && <p>
        Horizon sensitivity: the GW{data.horizon_consistency.short_end} first action, continued through GW{data.horizon_consistency.long_end},
        scores {data.horizon_consistency.continued_short_minus_long.toFixed(2)} projected points relative to the longer search.
        This comparison includes solver-search effects and does not establish future performance.
      </p>}
      {analysis && (
        <>
          <p>{analysis.status}.</p>
          <details>
            <summary>How the components connect</summary>
            {Object.entries(analysis.connections).map(([name, text]) => (
              <p key={name}>
                <b>{name.replaceAll("_", " ")}:</b> {text}
              </p>
            ))}
          </details>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Legal action</th>
                  <th>Chip</th>
                  <th>Hit</th>
                  <th>Bank</th>
                  <th>Reference xPts</th>
                  <th>Tree xPts</th>
                </tr>
              </thead>
              <tbody>
                {analysis.candidates.slice(0, 24).map((r, i) => (
                  <tr key={i}>
                    <td>
                      {r.action.buy.length
                        ? r.action.sell.map(name).join(", ") +
                          " → " +
                          r.action.buy.map(name).join(", ")
                        : "Roll"}
                      {r.baseline_policy_first_action
                        ? " · current multiweek policy"
                        : ""}
                    </td>
                    <td>{r.action.chip || "—"}</td>
                    <td>{r.ledger.hit}</td>
                    <td>{money(r.ledger.bank_after)}</td>
                    <td>{r.reference.toFixed(2)}</td>
                    <td>{r.tree.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
      {data.manager_policy && (
        <details>
          <summary>Manager choices and chip validation</summary>
          <p>
            Separate tests on GW{data.manager_policy.test_events.join("–")}.
          </p>
          {Object.entries(data.manager_policy.metrics).map(([name, m]) => (
            <p key={name}>
              {name}: log loss {m.baseline_log_loss.toFixed(3)} →{" "}
              {(m.conditional_log_loss ?? m.log_loss ?? 0).toFixed(3)}
              {m.top5_recall != null
                ? `; top-five recall ${percent(m.top5_recall)}`
                : ""}
              . {m.promoted ? "Learned model retained." : "Baseline retained."}
            </p>
          ))}
        </details>
      )}
      {data.market_model && (
        <p>
          Pre-deadline price model: log loss{" "}
          {data.market_model.frequency_log_loss.toFixed(3)} →{" "}
          {data.market_model.price_log_loss.toFixed(3)}, tested on{" "}
          {data.market_model.test_deadlines} deadlines.
        </p>
      )}
      {data.context_ablation && (
        <details>
          <summary>Independently retrained context comparisons</summary>
          {Object.entries(data.context_ablation.metrics).map(([name, m]) => (
            <p key={name}>
              {name.replaceAll("_", " ")}: point RMSE {m.points.rmse.toFixed(3)}
              , appearance Brier {m.appearance_brier.toFixed(3)}, 60-minute
              Brier {m.sixty_brier.toFixed(3)}.
            </p>
          ))}
        </details>
      )}
      {data.coherent_model && (
        <details>
          <summary>Football-world structure and component checks</summary>
          {data.coherent_model.coherence.map((s) => (
            <p key={s}>{s}</p>
          ))}
          {Object.entries(data.coherent_model.metrics).map(([name, m]) => (
            <p key={name}>
              {name}: deviance {m.baseline_deviance.toFixed(4)} →{" "}
              {m.poisson_deviance.toFixed(4)}.{" "}
              {m.promoted
                ? "Learned head retained."
                : "Established prior retained."}
            </p>
          ))}
          {data.coherent_model.limits.map((s) => (
            <p key={s}>{s}</p>
          ))}
        </details>
      )}
      <div className="form-row">
        <button
          className="button secondary"
          disabled={busy}
          onClick={() => act("/strategy/refine", {})}
        >
          Refine multiweek frontier
        </button>
        <button
          className="button secondary"
          disabled={busy}
          onClick={() => act("/strategy/train", {})}
        >
          Retrain complete system
        </button>
      </div>
      <p>
        Complete retraining is a longer CPU job. Validated results remain
        available while it runs; only completed results can pass promotion
        gates.
      </p>
      <div className="form-row">
        <button
          className="button secondary"
          disabled={busy}
          onClick={() => act("/refresh-cycle", {})}
        >
          Refresh and recompute
        </button>
        <button
          className="button secondary"
          disabled={busy}
          onClick={() =>
            act("/refresh-schedule", {
              enabled: !data.refresh_schedule?.enabled,
              interval_minutes: 240,
            })
          }
        >
          {data.refresh_schedule?.enabled
            ? "Disable automatic refresh"
            : "Enable automatic refresh"}
        </button>
      </div>
      <p>
        Automatic refresh runs while this local application is open, with more
        frequent checks near the deadline. Account changes still require
        reconciliation.
      </p>
      <button className="button secondary" disabled={busy} onClick={()=>act('/coherent/compare',{})}>Compare football worlds</button>
      {data.coherent_comparison && <p>Selected policy: {data.coherent_comparison.production_mean.toFixed(1)} reference points / {data.coherent_comparison.football_world_mean.toFixed(1)} structural-football-world points. {data.coherent_comparison.fixtures} fixtures; {data.coherent_comparison.invalid_goal_allocations} invalid goal allocations; {data.coherent_comparison.short_lineup_worlds} incomplete starting XIs. This is a model sensitivity, not a validated future result.</p>}
    </section>
  );
}

export function TransferTiming({ data, busy, act }: Props) {
  const t = data.timing_analysis;
  const current = t?.plan_id === data.plan?.id ? t : null;
  return (
    <section className="panel">
      <div className="section-title">
        <div>
          <h3>Transfer timing and affordability</h3>
          <p>
            Compare committing now with waiting for price and availability
            information. Both branches can make later legal adjustments.
          </p>
        </div>
        <button
          className="button secondary"
          disabled={busy}
          onClick={() => act("/strategy/timing", {})}
        >
          Analyze timing
        </button>
      </div>
      {current ? (
        <>
          <p>
            Waiting minus early commitment:{" "}
            {current.wait_minus_early.toFixed(2)} projected points; paired
            simulation interval{" "}
            {current.paired_mc_interval_95
              .map((v) => v.toFixed(2))
              .join(" to ")}
            .
          </p>
          <p>
            Bundle affordability under the 48-hour price scenarios:{" "}
            {current.affordability_probability_bounds.map(percent).join(" to ")}
            .{" "}
            {current.within_trained_time_support
              ? "The deadline falls within the price model’s timing support."
              : "The wait differs from the trained 40–72-hour windows; treat this as a timing sensitivity."}
          </p>
          <details>
            <summary>Research that could change the decision</summary>
            {current.research_priority.slice(0, 6).map((r) => (
              <p key={r.player_id}>
                {
                  data.market?.players.find((p) => p.id === r.player_id)
                    ?.web_name
                }
                : expected report value {r.expected_value_of_report.toFixed(3)}{" "}
                points.
              </p>
            ))}
            {current.limits.map((s) => (
              <p key={s}>{s}</p>
            ))}
          </details>
        </>
      ) : (
        <p>
          Run the analysis for the current selected plan. Price movement and new
          news are uncertain; account purchase prices remain explicit where
          unknown.
        </p>
      )}
    </section>
  );
}

export function PopulationEvidence({ data, busy, act }: Props) {
  const population = data.population_forecast;
  return (
    <section className="panel">
      <div className="section-title">
        <div>
          <h3>Population and rank scenarios</h3>
          <p>
            Sample actual entries across rank bands, including inactive teams,
            and reuse the same player outcomes.
          </p>
        </div>
        <button
          className="button secondary"
          disabled={busy}
          onClick={() => act("/population/analyze", {})}
        >
          Refresh population analysis
        </button>
      </div>
      {population ? (
        <>
          <p>
            {population.members} rank-stratified entries · target GW
            {population.target_event}. These are conditional modeled rank
            scenarios.
          </p>
          {Object.entries(population.scenarios).map(([name, r]) => (
            <p key={name}>
              {name}: modeled top-one-percent probability{" "}
              {percent(r.top_one_percent_probability)}; entry-resampling
              interval {r.entry_resampling_95ci.map(percent).join(" to ")};
              median estimated rank{" "}
              {Math.round(r.rank_quantiles[1]).toLocaleString()}.
            </p>
          ))}
          <details>
            <summary>Sampling and model limits</summary>
            {population.limits.map((s) => (
              <p key={s}>{s}</p>
            ))}
          </details>
        </>
      ) : (
        <p>
          Refresh the rank-stratified sample to estimate a population scenario.
          A current top-1,000 cohort alone does not represent the whole game.
        </p>
      )}
    </section>
  );
}

export function FootballEvidence({data,busy,act}:Props) {
  const availability=data.availability_model, shots=data.shot_archive;
  const name=(id:number)=>data.market?.players.find(p=>p.id===id)?.web_name||String(id);
  const future=new Set(data.market?.fixtures.filter(f=>!f.finished).map(f=>f.id));
  const bans=data.discipline_report?.records.filter(r=>r.fixture_ids.some(f=>future.has(f))||r.cautions_range[1]>=4);
  return <>
    {!!data.request_metrics_24h?.length && <section className="panel"><h3>Source requests · past 24 hours</h3><p>{data.request_metrics_24h.reduce((n,r)=>n+r.calls,0).toLocaleString()} measured attempts · {(data.request_metrics_24h.reduce((n,r)=>n+r.bytes,0)/1_000_000).toFixed(1)} MB received · {data.request_metrics_24h.reduce((n,r)=>n+r.errors,0)} failed attempts. Retries respect source limits; cached valid data is retained.</p></section>}
    <section className="panel">
      <div className="section-title"><div><h3>Football models and current coverage</h3><p>Only models that pass later-deadline tests affect their production component.</p></div>
        <div><button className="button secondary" disabled={busy} onClick={()=>act('/football/shots/rejoin')}>Rejoin cached shots</button> <button className="button secondary" disabled={busy} onClick={()=>act('/football/train')}>Update football models</button></div></div>
      {availability && <div className="data-details">
        <div><span>Recovery transitions</span><b>{availability.recovery.promoted?'Accepted':'Statistical fallback'}</b></div>
        <div><span>Joint double-gameweek rotation</span><b>{availability.rotation.joint_validation?.promoted?'Accepted':'Challenger'}</b></div>
        <div><span>Joint rotation test pairs / deadlines</span><b>{availability.rotation.joint_validation?.test_pairs??0} / {availability.rotation.joint_validation?.test_deadlines??0}</b></div>
      </div>}
      {shots && <p>{shots.matched.toLocaleString()} matched shot/role observations; {shots.coverage.current_players} current-season players. {shots.refresh_mode === "verified cache only" ? "Cached match data; live source availability was not checked." : ""} {shots.errors.length>0?'Source refresh failed; last verified observations retained.':''} Unresolved identities are excluded.</p>}
      {data.attack_model && <p>Goal channels: {Object.entries(data.attack_model.metrics).map(([k,v])=>`${k.replaceAll('_',' ')} (${v.promoted?'accepted':'prior retained'})`).join(' · ')}. FPL assist attribution: {data.attack_model.assist.promoted?'accepted':'prior retained'}.</p>}
      {data.score_process_model && <p>Team strength challenger: {data.score_process_model.strength?.promoted?'accepted':'baseline retained'}. Score dependence: {data.score_process_model.score_dependence.adopted.replaceAll('_',' ')}. Extra posterior scale: {data.score_process_model.parameter_uncertainty.adopted_scale}.</p>}
      {data.temporal_model && <p>Attacking-history decay: {data.temporal_model.adopted.decay}. Finishing adjustment: {data.temporal_model.adopted.finishing_strength===null?'disabled; benefit not established':`shrinkage ${data.temporal_model.adopted.finishing_strength}`}. Goals minus xG is a diagnostic, not an identified measure of luck.</p>}
      {data.odds_refresh && <p>Optional odds: {data.odds_refresh.status}. {data.odds_model?.reason}</p>}
      {!!bans?.length && <p>Squad discipline: {bans.map(r=>`${name(r.player_id)}: ${r.cautions_range[0]===r.cautions_range[1]?r.cautions:r.cautions_range.join('–')} cautions${r.fixture_ids.some(f=>future.has(f))?', published/inferred upcoming ban':''}`).join('; ')}.</p>}
    </section>
    <section className="panel">
      <div className="section-title"><div><h3>Historical weekly teams · refitted reference</h3><p>Independent £100m teams. Each week refits using earlier completed results.</p></div>
        <button className="button secondary" disabled={busy} onClick={()=>act('/replay/weekly')}>Run weekly replay</button></div>
      {data.weekly_replay ? <>
        {data.weekly_replay.model_scope && <p>{data.weekly_replay.model_scope}</p>}
        <p>{data.weekly_replay.learning}</p>
        <div className="table-scroll"><table><thead><tr><th>GW</th><th>Forecast</th><th>Actual points</th><th>Hindsight diagnostic</th><th>Squad solve gap</th></tr></thead><tbody>
          {data.weekly_replay.weeks.map(w=><tr key={w.event}><td>{w.event}</td><td>{w.forecast_mean.toFixed(1)}</td><td>{w.actual_points}</td><td>{w.hindsight_points}</td><td>{w.solver.gap===null?'Bound unavailable':percent(w.solver.gap)}</td></tr>)}
        </tbody></table></div>
        <p>Total actual points: <b>{data.weekly_replay.total_points}</b>. Reconstructed historical experiment; these are separate weekly squads, not a continuous account with free transfers.</p>
        {data.weekly_baseline_comparison && <p>Matched controls: {Object.entries(data.weekly_baseline_comparison.comparisons).map(([k,v])=>`${k.replaceAll('_',' ')} ${v.baseline_total}`).join(' · ')} points.</p>}
        <a href="/api/replay/weekly" download="weekly-replay.json">Download teams, captaincy, bench and cutoff proofs</a>
      </>:<p>No completed weekly replay is saved yet.</p>}
    </section>
    {data.composed_policy_replay && <section className="panel">
      <h3>Combined models · continuous account</h3>
      <p>Role-aware minutes, conditional points and future fixtures, refitted before each deadline. Bank, transfers, hits and chips carry forward.</p>
      <p>{data.composed_policy_replay.protocol.season}: <b>{data.composed_policy_replay.totals.composed_multiweek}</b> net points; {data.composed_policy_replay.totals.future_reference_multiweek} without the role head; {data.composed_policy_replay.totals.composed_one_week_no_chips} with one-week planning and no chips.</p>
      <div className="table-scroll"><table><thead><tr><th>GW</th><th>Combined net</th><th>Reference net</th><th>One-week net</th><th>Hit</th><th>Chip</th></tr></thead><tbody>{data.composed_policy_replay.weeks.map(w=><tr key={w.event}><td>{w.event}</td><td>{w.outcomes.composed_multiweek.net_points}</td><td>{w.outcomes.future_reference_multiweek.net_points}</td><td>{w.outcomes.composed_one_week_no_chips.net_points}</td><td>{w.outcomes.composed_multiweek.hit}</td><td>{w.outcomes.composed_multiweek.chip||'—'}</td></tr>)}</tbody></table></div>
      <p>Historical reconstruction with bounded research search. Diagnostic only; this does not establish future superiority or certify unlimited live search.</p>
      <p>{data.composed_policy_replay.verification.independently_scored_actions} actions independently checked. {data.composed_policy_replay.protocol.production_model_id!==data.decision_model?.id?'The saved experiment predates the latest fitted model.':''}</p>
      <a href="/api/replay/composed" download="composed-policy-replay.json">Download combined-model decisions and scoring proofs</a>
    </section>}
    <section className="panel">
      <div className="section-title"><div><h3>{data.decision_model?.future_horizon_model_id?"Reference pipeline · continuous account":"Production pipeline · continuous account"}</h3><p>{data.decision_model?.future_horizon_model_id?"Historical reference comparison; the connected future-fixture model has its own continuous-account results below.":"Runs the production forecaster and full transfer/chip planner together. The account carries forward through observed prices, free transfers and hits; future actions are replanned after each deadline."}</p></div><button className="button secondary" disabled={busy} onClick={()=>act('/replay/pipeline')}>Run integrated replay</button></div>
      {data.pipeline_replay_progress && <p>Replay: {data.pipeline_replay_progress.stage}{data.pipeline_replay_progress.season?` · ${data.pipeline_replay_progress.season} GW${data.pipeline_replay_progress.event}`:''}. {data.pipeline_replay_progress.completed_deadlines} / {data.pipeline_replay_progress.total_deadlines} deadlines completed. Long runs resume from completed deadlines.</p>}
      {data.pipeline_replay ? <>
        {Object.entries(data.pipeline_replay.summaries).map(([season,s])=><div key={season}><h4>{season}</h4><p>Net points: <b>{s.primary_multiweek.net_points}</b> integrated policy · {s.reference_one_week.net_points} same-forecast one-week policy · {s.ownership_no_hits.net_points} ownership with free transfers. Integrated hits: {s.primary_multiweek.hit_points}; chips: {s.primary_multiweek.chips.map(c=>`${c.chip} GW${c.event}`).join(', ')||'retained'}.</p><div className="table-scroll"><table><thead><tr><th>GW</th><th>Integrated net</th><th>One-week net</th><th>Ownership net</th><th>Chip</th><th>Hit</th><th>Bank</th><th>Next FT</th><th>Solve gap</th></tr></thead><tbody>{data.pipeline_replay!.weeks.filter(w=>w.season===season&&w.policy==='primary_multiweek').map(w=><tr key={w.event}><td>{w.event}</td><td>{w.net_points}</td><td>{data.pipeline_replay!.weeks.find(r=>r.season===season&&r.event===w.event&&r.policy==='reference_one_week')?.net_points}</td><td>{data.pipeline_replay!.weeks.find(r=>r.season===season&&r.event===w.event&&r.policy==='ownership_no_hits')?.net_points}</td><td>{w.action.chip||'—'}</td><td>{w.ledger.hit}</td><td>{money(w.ledger.bank_after)}</td><td>{w.ledger.free_transfers_after}</td><td>{w.solver.gap===null?'Unavailable':percent(w.solver.gap)}</td></tr>)}</tbody></table></div></div>)}
        <p>{data.pipeline_replay.independent_verification.validated_actions} actions independently checked for legal finances, transfer allocation, chips and actual scoring. Full planning windows continue beyond the scored interval.</p><a href="/api/replay/pipeline" download="integrated-pipeline-replay.json">Download full integrated replay</a><details><summary>Frozen teams and input proofs</summary>{data.pipeline_replay.weeks.filter(w=>w.policy==='primary_multiweek').map(w=><p key={`${w.season}-${w.event}`}><a href={`/api/replay/pipeline/decision/${w.season.replace('/','-')}/${w.event}`} download>{w.season} GW{w.event}: teams, captaincy, forecasts and cutoff proofs</a></p>)}</details><details><summary>Evaluation scope</summary>{data.pipeline_replay.limits.map(v=><p key={v}>{v}</p>)}</details>
      </>:<p>No completed integrated replay has been saved. The separate weekly and ordinary-transfer experiments below do not substitute for this test.</p>}
    </section>
    <section className="panel">
      <div className="section-title"><div><h3>Prospective continuous accounts</h3><p>Equal £100m hypothetical accounts with known purchase prices. Decisions are frozen before the deadline, then scored against finalized results and carried into the following week.</p></div><button className="button secondary" disabled={busy} onClick={()=>act('/prospective/accounts')}>Freeze synchronized accounts</button></div>
      {data.shadow_registry && <p>{data.shadow_registry.seed_scope}. Started GW{data.shadow_registry.seed_event}, {data.shadow_registry.season}. {data.shadow_registry.missed_capture_policy}.</p>}
      {data.shadow_capture_status && <p>{data.shadow_capture_status.status}{data.shadow_capture_status.reason?`: ${data.shadow_capture_status.reason}`:''}.</p>}
      {data.shadow_cohort ? <><p>Latest capture: GW{data.shadow_cohort.event}, {new Date(data.shadow_cohort.frozen_at).toLocaleString()}. {data.shadow_cohort.track}.</p><p>{Object.entries(data.shadow_cohort.actions).map(([policy,row])=>`${policy.replaceAll('_',' ')}: captain ${name(row.action.lineup.captain)}, vice ${name(row.action.lineup.vice)}${row.action.chip?`, ${row.action.chip}`:''}`).join(' · ')}.</p></>:<p>No synchronized decisions have been frozen yet.</p>}
      {data.shadow_evaluation && <><p>{data.shadow_evaluation.finalized_deadlines} finalized deadlines. Net points: {Object.entries(data.shadow_evaluation.total_net_points).map(([policy,total])=>`${policy.replaceAll('_',' ')} ${total}`).join(' · ')}.</p>{data.shadow_evaluation.outcomes.length>0 && <div className="table-scroll"><table><thead><tr><th>GW</th><th>Policy</th><th>Actual net</th><th>Hit</th><th>Chip</th><th>Decision</th></tr></thead><tbody>{data.shadow_evaluation.outcomes.map(row=><tr key={`${row.event}-${row.policy}`}><td>{row.event}</td><td>{row.policy.replaceAll('_',' ')}</td><td>{row.net_points}</td><td>{row.hit}</td><td>{row.chip||'—'}</td><td>{row.track}</td></tr>)}</tbody></table></div>}</>}
      <p>Automatic refresh captures each new forecast and scores finalized weeks. Future match results appear only after they happen; these accounts do not submit transfers to your FPL team.</p><a href="/api/prospective/accounts" download="prospective-accounts.json">Download frozen decisions and results</a>
    </section>
    {data.horizon_refinement && <section className="panel">
      <h3>Future-fixture forecasting</h3>
      {data.future_horizon_model && <p>Latest fit: {new Date(data.future_horizon_model.fit_as_of).toLocaleString()}. {data.future_horizon_model.feature_count} features, {data.future_horizon_model.training_evidence.training_rows.toLocaleString()} fitting rows and {data.future_horizon_model.training_evidence.calibration_rows.toLocaleString()} reserved calibration rows. Completed labels through {data.future_horizon_model.training_evidence.cal_max_label}. {data.future_horizon_model.learning}.</p>}
      <p>The model learns minutes, starts and conditional points from dated player statistics, fixtures, prices, ownership and availability. Lead time is a training input; published next-week estimates are excluded from distant weeks.</p>
      <p>Forecast validation: {data.horizon_refinement.forecast_gate_passed?'passed':'reference retained'}. Primary integration: {data.decision_model?.future_horizon_model_id?'active for future weeks through an 18-week lead':'reference retained'}. Current-week forecasts stay unchanged.</p>
      <div className="table-scroll"><table><thead><tr><th>Earlier season</th><th>Reference error</th><th>Future model error</th><th>Relevant-player error</th></tr></thead><tbody>{Object.entries(data.horizon_refinement.metrics).map(([season,m])=><tr key={season}><td>{season}</td><td>{m.reference.rmse.toFixed(3)}</td><td>{m.candidate.rmse.toFixed(3)}</td><td>{m.relevant_reference.rmse.toFixed(3)} → {m.relevant_candidate.rmse.toFixed(3)}</td></tr>)}</tbody></table></div>
      {data.future_policy_progress && data.future_policy_progress.completed<data.future_policy_progress.total && <p>Continuous-policy replay: {data.future_policy_progress.completed} / {data.future_policy_progress.total} completed deadlines.</p>}
      {[data.future_policy_replay,data.future_policy_diagnostic].filter(p=>p!==null&&p!==undefined).map(p=><div key={p.id}><h4>{p.season} continuous account</h4><p>Candidate {p.candidate_points} net points; matched reference {p.baseline_points}. {p===data.future_policy_diagnostic?'Current-season diagnostic; never selects configuration.':`Full-policy acceptance: ${p.policy_gate_passed?'passed':'reference retained'}.`} {p.verification.sealed_deadlines} sealed decisions independently checked.</p><div className="table-scroll"><table><thead><tr><th>GW</th><th>Candidate net</th><th>Reference net</th><th>Hit</th><th>Chip</th></tr></thead><tbody>{p.weeks.map(w=><tr key={w.event}><td>{w.event}</td><td>{w.net_points}</td><td>{w.baseline_points}</td><td>{w.hit}</td><td>{w.chip||'—'}</td></tr>)}</tbody></table></div></div>)}
      {data.future_distribution_validation && <p>Marginal probability and interval validation: {data.future_distribution_validation.passed?'passed':'reference retained'}. Relevant-player distribution error {data.future_distribution_validation.metrics.relevant.reference.crps.toFixed(3)} → {data.future_distribution_validation.metrics.relevant.candidate.crps.toFixed(3)}; 80% interval coverage {percent(data.future_distribution_validation.metrics.relevant.candidate.coverage80)}. {data.future_distribution_validation.rows.toLocaleString()} sampled future forecasts across {data.future_distribution_validation.deadlines} earlier deadlines.</p>}
      <p>{data.horizon_refinement.protocol.limits}</p><a href="/api/horizon/expert" download="future-fixture-evidence.json">Download models and validation evidence</a>
    </section>}
    {data.consensus_model && <section className="panel">
      <div className="section-title"><div><h3>Published-signal fusion · weekly experiment</h3><p>Football history, ownership, official forecast, form, advanced statistics and dated source features. Weights chosen on {data.consensus_model.protocol.weight_selection}; checked on {data.consensus_model.protocol.acceptance}.</p></div><button className="button secondary" disabled={busy} onClick={()=>act('/consensus/compare')}>Compare current squad</button></div>
      <p>Selected crowd-model weight: {percent(data.consensus_model.weight)}. Weekly validation: {data.consensus_model.acceptance.passed?'passed':'reference retained'}. Published-signal ensemble in primary policy: {data.consensus_model.promoted?'active':'comparison only; ordinary-policy acceptance failed'}.</p>
      <p>GW1–GW9 actual points: <b>{data.consensus_model.current_totals[data.consensus_model.selected]}</b>; matched frozen reference {data.consensus_model.current_totals.mix_0}; ownership {data.consensus_model.current_totals.ownership}. Independent weekly squads, no chips.</p>
      <div className="table-scroll"><table><thead><tr><th>GW</th><th>Selected ensemble</th><th>Frozen reference</th><th>Ownership</th><th>Ensemble captain</th></tr></thead><tbody>{data.consensus_model.weeks.filter(w=>w.policy===data.consensus_model!.selected).map(w=><tr key={w.event}><td>{w.event}</td><td>{w.actual_points}</td><td>{data.consensus_model!.weeks.find(v=>v.event===w.event&&v.policy==='mix_0')?.actual_points}</td><td>{data.consensus_model!.weeks.find(v=>v.event===w.event&&v.policy==='ownership')?.actual_points}</td><td>{w.players.find(p=>p.id===w.lineup.captain)?.name}</td></tr>)}</tbody></table></div>
      <p>Earlier-season return gain: {data.consensus_model.acceptance.return_delta} points; block interval for the weekly gain {data.consensus_model.acceptance.weekly_return_delta_95ci.map(n=>n.toFixed(2)).join(' to ')}. A short historical result cannot guarantee future scores.</p>
      {data.consensus_policy_validation && <><p>{data.consensus_policy_validation.scope}</p><p>Continuous-account GW1–GW9 net points: {Object.entries(data.consensus_policy_validation.current_totals).map(([p,n])=>`${p===data.consensus_model!.selected?'ensemble':p.replaceAll('_',' ')} ${n}`).join(' · ')}. Ordinary-policy validation: {data.consensus_policy_validation.acceptance.passed?'passed':'benefit not established; primary retained'}.</p></>}
      {data.consensus_comparison && <><p>{data.consensus_comparison.scope}</p><div className="table-scroll"><table><thead><tr><th>Current GW{data.consensus_comparison.event} comparison</th><th>Transfers</th><th>Hit</th><th>Captain</th><th>Expected net</th><th>Additive solve gap</th></tr></thead><tbody>{data.consensus_comparison.decisions.map(d=><tr key={d.name}><td>{d.name.replaceAll('_',' ')}</td><td>{d.action.sell.map(name).join(', ')||'Hold'}{d.action.buy.length>0?` → ${d.action.buy.map(name).join(', ')}`:''}</td><td>{d.ledger.hit}</td><td>{name(d.action.lineup.captain)}</td><td>{d.additive_expected_net.toFixed(1)}</td><td>{d.solver.gap===null?'Unavailable':percent(d.solver.gap)}</td></tr>)}</tbody></table></div></>}
      <a href="/api/consensus/model" download="published-fusion.json">Download full experiment and frozen teams</a>
      <details><summary>Scope and limits</summary>{data.consensus_model.limits.map(v=><p key={v}>{v}</p>)}</details>
    </section>}
  </>;
}
