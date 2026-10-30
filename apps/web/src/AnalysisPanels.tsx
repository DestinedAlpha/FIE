import { useState } from "react";
import { Activity, FlaskConical, RefreshCw, ShieldCheck } from "lucide-react";
import { Dashboard, Plan, money, percent } from "./types";

type Props = {
  data: Dashboard;
  busy: boolean;
  act: (path: string, body?: unknown) => Promise<unknown>;
};

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
    [delta, setDelta] = useState(0.2);
  if (!plan || !data.state) return null;
  const minimum = data.state.event + 1,
    maximum = plan.ledger.at(-1)?.event || 19;
  const selected = player || data.state.lots[0].player_id;
  const when = Math.max(minimum, Math.min(reveal, maximum));
  const evaluate = () => {
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
          </select>
        </label>
        <label>
          Player
          <select value={selected} onChange={(e) => setPlayer(+e.target.value)}>
            {data.market?.players.map((p) => (
              <option key={p.id} value={p.id}>
                {p.web_name}
              </option>
            ))}
          </select>
        </label>
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
          disabled={busy || minimum > maximum}
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
          <p key={c.chip}>
            <b>
              {c.chip === "freehit" ? "Free Hit" : "Bench Boost"} · GW
              {c.chosen_event}
            </b>
            : {c.whole_policy_mean_delta.toFixed(1)} points versus reoptimizing
            without it · Monte Carlo interval [
            {c.paired_mc_interval_95.map((x) => x.toFixed(1)).join(", ")}] ·
            alternative solver gap {c.counterfactual_solver_gap == null ? "bound unavailable" : percent(c.counterfactual_solver_gap)}
          </p>
        ))}
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
              The points model tested here supplies the forecast worlds and the
              transfer planner.
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
            <span>Points RMSE · lower is better</span>
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
        {model.neural && !data.fusion_model && (
          <p>
            Neural challenger RMSE {model.neural.metrics.rmse.toFixed(3)}. Its
            paired difference interval [
            {model.neural.paired_vs_gbm_95ci
              .map((x) => x.toFixed(4))
              .join(", ")}
            ]{" "}
            {model.neural.paired_vs_gbm_95ci[0] <= 0 &&
            model.neural.paired_vs_gbm_95ci[1] >= 0
              ? "includes zero; this run does not establish an advantage over the tree model."
              : "must be reviewed together with decision returns before a production replacement."}
          </p>
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
      (p) => p.event === row.event && row.playing_squad.includes(p.player_id),
    )
    .sort((a, b) => b.mean - a.mean)
    .slice(0, 5);
  if (!candidates.length) return null;
  return (
    <section className="panel">
      <h3>Why this captain?</h3>
      <p>
        The highest projected return in the planned squad, with appearances and
        captain fallback considered. Player reputation and ownership do not
        force the choice.
      </p>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Candidate</th>
              <th>Projected points</th>
              <th>Chance of 60+ minutes</th>
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
  if (!plan || !plan.ledger[0].hit) return null;
  const comparison =
    data.hit_comparison?.base_plan_id === plan.id ? data.hit_comparison : null;
  return (
    <section className="panel">
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
    </section>
  );
}

export function SourceCoverage({ data, busy, act }: Props) {
  const c = data.source_coverage;
  const f = data.fusion_model;
  return (
    <section className="panel">
      <div className="section-title">
        <div>
          <h3>Sources and neural validation</h3>
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
            <b>Purpose-built temporal fusion network</b>: match-history
            attention, official-news branch, manager/market and workload
            branches, masked source attention, and minutes/points/component
            training heads. {f.folds[0]?.neural.parameters.toLocaleString()}{" "}
            trainable parameters.
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
                {Object.entries(f.candidates).map(([key, row]) => (
                  <tr key={key}>
                    <td>
                      {key === "network"
                        ? "Temporal neural fusion"
                        : "Same-source boosted trees"}
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
                  {Object.entries(f.decision_validation.policies).map(
                    ([kind, policy]) => (
                      <tr key={kind}>
                        <td>
                          {kind === "network"
                            ? "Neural fusion"
                            : "Same-source trees"}
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
                  <th>Neural candidate</th>
                </tr>
              </thead>
              <tbody>
                {data.fusion_comparison.players.slice(0, 15).map((row) => (
                  <tr key={row.player_id}>
                    <td>{row.name}</td>
                    <td>{row.production.toFixed(2)}</td>
                    <td>{row.tree.toFixed(2)}</td>
                    <td>{row.network.toFixed(2)}</td>
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
  const horizon = data.horizon_model,
    policy = data.transfer_policy_model,
    personal = data.personalized_policy_analysis;
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
          onClick={() => act("/policy/analyze", {})}
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
                {Object.entries(critic.candidates).map(([name, m]) => (
                  <tr key={name}>
                    <td>
                      {name === "network"
                        ? "Neural decision critic"
                        : "Tree decision critic"}
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
                  <th>Neural xPts</th>
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
                    <td>{r.network.toFixed(2)}</td>
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
