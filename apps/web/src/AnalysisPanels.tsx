import { useState } from "react";
import { Activity, FlaskConical, RefreshCw, ShieldCheck } from "lucide-react";
import { Dashboard, Plan, percent } from "./types";

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
            ? `${selection.model} selected from measured extraction tests; eligible free fallback available.`
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
            alternative solver gap {percent(c.counterfactual_solver_gap)}
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
