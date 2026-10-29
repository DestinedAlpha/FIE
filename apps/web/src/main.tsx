import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  ArrowDownToLine,
  ArrowRight,
  CalendarDays,
  ChartNoAxesCombined,
  Check,
  ChevronRight,
  CircleHelp,
  Database,
  FlaskConical,
  LayoutDashboard,
  LoaderCircle,
  Menu,
  RefreshCw,
  Search,
  Settings2,
  ShieldCheck,
  Sparkles,
  Users,
  X,
} from "lucide-react";
import {
  api,
  Dashboard,
  Ledger,
  money,
  percent,
  Plan,
  position,
} from "./types";
import "./style.css";
import {
  ResearchControls,
  SourceCoverage,
  ManagerSignals,
  TransferLearning,
  ModelEvidence,
  DecisionEvidence,
  CaptainEvidence,
  HitEvidence,
  ScenarioPanel,
  ChipPanel,
  CohortPanel,
} from "./AnalysisPanels";

const tabs = [
  ["Overview", LayoutDashboard],
  ["My squad", Users],
  ["Player market", Search],
  ["Season planner", CalendarDays],
  ["Rivals", ChartNoAxesCombined],
  ["Research", FlaskConical],
  ["Model & data", Database],
] as const;

function App() {
  const [tab, setTab] = useState("Overview");
  const [data, setData] = useState<Dashboard | null>(null);
  const [event, setEvent] = useState<number | undefined>();
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [job, setJob] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState(0);
  const [sort, setSort] = useState<"points" | "value" | "minutes">("points");
  const [end, setEnd] = useState(19);
  const [objective, setObjective] = useState("expected_points");
  const [budget, setBudget] = useState(5);
  const [editor, setEditor] = useState(false);
  const [stateText, setStateText] = useState("");
  const [mobile, setMobile] = useState(false);
  const [beliefPlayer, setBeliefPlayer] = useState(0);
  const [beliefValue, setBeliefValue] = useState(0.75);
  const [beliefReason, setBeliefReason] = useState("");
  const [beliefDays, setBeliefDays] = useState(3);
  const [sell, setSell] = useState(0);
  const [buy, setBuy] = useState(0);
  const [transferResult, setTransferResult] = useState("");
  const [viewPlan, setViewPlan] = useState<Plan | null>(null);

  async function reload() {
    const result = await api<Dashboard>(
      "/dashboard" + (event ? `?event=${event}` : ""),
    );
    setData(result);
  }
  useEffect(() => {
    reload().catch((e) => setError(e.message));
  }, [event]);
  useEffect(() => {
    if (!job) return;
    let stopped = false;
    const poll = async () => {
      try {
        const value = await api<{
          status: string;
          error?: string;
          result?: { stale?: boolean };
        }>(`/jobs/${job}`);
        if (["completed", "failed", "interrupted"].includes(value.status)) {
          if (!stopped) {
            setJob(null);
            setBusy(false);
            if (value.status !== "completed")
              setError(value.error || "Job interrupted");
            else {
              setNotice(
                value.result?.stale
                  ? "Completed for an older account version. Recompute for your current team."
                  : "Calculation complete. Your workspace is up to date.",
              );
              await reload();
            }
          }
        }
      } catch (e) {
        if (!stopped) {
          setError((e as Error).message);
          setBusy(false);
          setJob(null);
        }
      }
    };
    const timer = setInterval(poll, 1500);
    poll();
    return () => {
      stopped = true;
      clearInterval(timer);
    };
  }, [job]);

  const players = useMemo(
    () => new Map(data?.market?.players.map((p) => [p.id, p]) || []),
    [data?.market],
  );
  const projections = useMemo(
    () => new Map(data?.forecast?.players.map((p) => [p.player_id, p]) || []),
    [data?.forecast],
  );
  const teams = useMemo(
    () => new Map(data?.market?.teams.map((t) => [t.id, t.short_name]) || []),
    [data?.market],
  );
  const currentEvent =
    event || data?.market?.events.find((e) => e.is_next)?.id || 9;
  const plan = viewPlan || data?.plan;
  const first = plan?.ledger[0];
  const deadline = data?.market?.events.find(
    (e) => e.id === currentEvent,
  )?.deadline_time;
  const name = (id: number) => players.get(id)?.web_name || String(id);
  const activeRow = plan?.ledger.find((r) => r.event === currentEvent) || first;
  const market = useMemo(
    () =>
      [...players.values()]
        .filter(
          (p) =>
            (!filter || p.position === filter) &&
            `${p.name} ${teams.get(p.team)}`
              .toLowerCase()
              .includes(query.toLowerCase()),
        )
        .sort((a, b) => {
          const x = projections.get(a.id),
            y = projections.get(b.id);
          const score = (p: typeof a, f: typeof x) =>
            sort === "minutes"
              ? f?.minutes_mean || 0
              : sort === "value"
                ? (f?.mean || 0) / (p.price / 10)
                : f?.mean || 0;
          return score(b, y) - score(a, x);
        }),
    [players, projections, filter, query, sort],
  );

  async function action(path: string, body: unknown = {}) {
    setBusy(true);
    setError("");
    setNotice("");
    setViewPlan(null);
    try {
      const result = await api<{ job_id?: string }>(path, body);
      if (result.job_id) setJob(result.job_id);
      else {
        await reload();
        setBusy(false);
        setNotice("Saved.");
      }
      return true;
    } catch (e) {
      setError((e as Error).message);
      setBusy(false);
      return false;
    }
  }
  function solve() {
    if (data?.state && data?.forecast)
      action("/optimize", {
        state_id: data.state.id,
        forecast_id: data.forecast.id,
        end_event: end,
        time_limit: 60,
        objective,
        sacrifice_budget: budget,
      });
  }
  function openEditor() {
    if (!data?.state) return;
    const { id, ...state } = data.state;
    setStateText(JSON.stringify(state, null, 2));
    setEditor(true);
  }
  async function saveState() {
    try {
      const state = JSON.parse(stateText);
      state.known_at = new Date().toISOString();
      if (await action("/state", { state, expected_state_id: data?.state?.id }))
        setEditor(false);
    } catch (e) {
      setError((e as Error).message);
    }
  }
  function stat(
    label: string,
    value: React.ReactNode,
    caption: string,
    icon: React.ReactNode,
  ) {
    return (
      <div className="stat">
        <div className="stat-label">
          {label}
          {icon}
        </div>
        <strong>{value}</strong>
        <small>{caption}</small>
      </div>
    );
  }
  function pill(text: string, tone = "") {
    return <span className={"pill " + tone}>{text}</span>;
  }
  function playerChip(id: number) {
    const p = players.get(id),
      r = projections.get(id);
    return (
      <div className="pitch-player" key={id}>
        <div className={"shirt club-" + ((p?.team || 0) % 5)}>
          {(teams.get(p?.team || 0) || "PL").slice(0, 3)}
        </div>
        <div className="player-name">
          {name(id)}{" "}
          {activeRow?.lineup.captain === id ? (
            <b className="captain">C</b>
          ) : activeRow?.lineup.vice === id ? (
            <b className="captain vice">V</b>
          ) : null}
        </div>
        <span className="player-points">
          {r?.mean.toFixed(1) || "—"} <small>xPts</small>
        </span>
      </div>
    );
  }
  function pitch(row: Ledger | undefined) {
    return (
      <div className="pitch">
        <div className="pitch-circle" />
        <div className="pitch-box top" />
        <div className="pitch-box bottom" />
        {row ? (
          [1, 2, 3, 4].map((pos) => (
            <div className="pitch-row" key={pos}>
              {row.lineup.xi
                .filter((id) => players.get(id)?.position === pos)
                .map(playerChip)}
            </div>
          ))
        ) : (
          <div className="pitch-empty">
            Import your squad and calculate a plan to see your recommended XI.
          </div>
        )}
      </div>
    );
  }
  function table(ids?: number[], sellQuotes = false) {
    return (
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Player</th>
              <th>Pos</th>
              <th>{sellQuotes ? "Sell value" : "Price"}</th>
              <th>Fixture</th>
              <th>xMins</th>
              <th>P(start)</th>
              <th>Ownership</th>
              <th className="numeric">xPts ↓</th>
            </tr>
          </thead>
          <tbody>
            {(ids
              ? ids.map((id) => players.get(id)!).filter(Boolean)
              : market
            ).map((p) => {
              const f = projections.get(p.id);
              const fixtures = data?.market?.fixtures.filter(
                (f) =>
                  f.event === currentEvent &&
                  (f.team_h === p.team || f.team_a === p.team),
              );
              return (
                <tr key={p.id}>
                  <td>
                    <div className="player-cell">
                      <div className={"mini-shirt club-" + (p.team % 5)} />
                      <div>
                        <strong>{p.web_name}</strong>
                        <small>
                          {teams.get(p.team)}{" "}
                          {p.status !== "a"
                            ? " · " + p.status.toUpperCase()
                            : ""}
                        </small>
                      </div>
                    </div>
                  </td>
                  <td>
                    <span className="position">{position(p.position)}</span>
                  </td>
                  <td>
                    {money(
                      sellQuotes
                        ? (data?.state?.lots.find((l) => l.player_id === p.id)
                            ?.sell_price ?? p.price)
                        : p.price,
                    )}
                  </td>
                  <td>
                    {fixtures
                      ?.map(
                        (f) =>
                          `${teams.get(f.team_h === p.team ? f.team_a : f.team_h)} (${f.team_h === p.team ? "H" : "A"})`,
                      )
                      .join(", ") || "Blank"}
                  </td>
                  <td>{f ? Math.round(f.minutes_mean) : "—"}</td>
                  <td>{f ? percent(f.p_start) : "—"}</td>
                  <td>{p.ownership.toFixed(1)}%</td>
                  <td className="numeric">
                    <b className="point-value">{f?.mean.toFixed(2) || "—"}</b>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <aside className={mobile ? "sidebar open" : "sidebar"}>
        <a className="brand" href="#" onClick={() => setTab("Overview")}>
          <div className="brand-mark">
            <ChartNoAxesCombined size={22} />
          </div>
          <span>
            FPL<span className="brand-light"> Intelligence</span>
          </span>
        </a>
        <div className="workspace-label">
          YOUR WORKSPACE <span>{data?.market?.season || "2026/27"}</span>
        </div>
        <nav>
          {tabs.map(([label, Icon]) => (
            <button
              key={label}
              className={tab === label ? "nav-item active" : "nav-item"}
              onClick={() => {
                setTab(label);
                setMobile(false);
              }}
            >
              <Icon size={18} />
              {label}
              {label === "Research" && data?.evidence.length ? (
                <span className="nav-count">{data.evidence.length}</span>
              ) : null}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="local-card">
            <div className="status-dot" />
            <div>
              <b>Private by default</b>
              <small>Local compute · free services</small>
            </div>
            <ShieldCheck size={17} />
          </div>
          <button className="profile" onClick={openEditor}>
            <div className="avatar">
              {data?.state?.name.slice(0, 2) || "FI"}
            </div>
            <div>
              <b>{data?.state?.name || "Your team"}</b>
              <small>Personal workspace</small>
            </div>
            <Settings2 size={16} />
          </button>
        </div>
      </aside>
      <main>
        <header className="topbar">
          <div className="breadcrumb">
            <button
              className="icon-button mobile-menu"
              onClick={() => setMobile(!mobile)}
              aria-label="Toggle navigation"
            >
              <Menu size={20} />
            </button>
            <span>Workspace</span>
            <ChevronRight size={14} />
            <b>{tab}</b>
          </div>
          <div className="topbar-right">
            <span className="live-tag">
              <span className="status-dot" />
              Local engine
            </span>
            <button
              className="icon-button"
              aria-label="About the model"
              onClick={() => setTab("Model & data")}
            >
              <CircleHelp size={18} />
            </button>
            <div className="avatar small">FI</div>
          </div>
        </header>
        <div className="page">
          <div className="page-heading">
            <div>
              <div className="eyebrow">YOUR NEXT MOVE, WITH THE EVIDENCE.</div>
              <h1>{tab === "Overview" ? "The gameweek, in focus." : tab}</h1>
              <p>
                {tab === "Overview"
                  ? "A clearer view of your team. A considered plan for what comes next."
                  : (
                      {
                        "My squad":
                          "Every player, every assumption, every pound accounted for.",
                        "Player market":
                          "Independent projections across the entire player market.",
                        "Season planner":
                          "A season of options. Recalculate as the picture changes.",
                        Rivals:
                          "One set of match worlds. Multiple credible opponent policies.",
                        Research:
                          "Original sources, explicit beliefs, and a traceable effect.",
                        "Model & data":
                          "Measured performance and visible data provenance.",
                      } as Record<string, string>
                    )[tab]}
              </p>
            </div>
            <div className="heading-actions">
              <label className="gw-select">
                <CalendarDays size={16} />
                <select
                  aria-label="Gameweek"
                  value={currentEvent}
                  onChange={(e) => setEvent(Number(e.target.value))}
                >
                  {(
                    data?.market?.events ||
                    Array.from({ length: 38 }, (_, i) => ({ id: i + 1 }))
                  ).map((e) => (
                    <option key={e.id} value={e.id}>
                      Gameweek {e.id}
                    </option>
                  ))}
                </select>
              </label>
              <button
                className="button secondary"
                disabled={busy}
                onClick={() => action("/refresh")}
              >
                <RefreshCw size={15} className={busy ? "spin" : ""} />
                Refresh data
              </button>
            </div>
          </div>
          {error && !editor && (
            <div role="alert" className="message error">
              <span>{error}</span>
              <button onClick={() => setError("")} aria-label="Dismiss error">
                <X size={16} />
              </button>
            </div>
          )}
          {notice && (
            <div role="status" className="message success">
              <span>{notice}</span>
              <button
                onClick={() => setNotice("")}
                aria-label="Dismiss notification"
              >
                <X size={16} />
              </button>
            </div>
          )}
          {busy && (
            <div role="status" className="message">
              <LoaderCircle size={16} className="spin" />
              Calculating in the local engine. You can keep exploring.
            </div>
          )}
          {plan?.stale && (
            <div className="message">
              This plan uses an older account, forecast or market version.
              Recalculate before acting.
            </div>
          )}
          {!data?.market ? (
            <section className="panel onboarding">
              <Database size={36} />
              <h2>Your intelligence starts here.</h2>
              <p>
                Refresh the official feeds to build the player registry,
                fixtures and historical archive.
              </p>
              <button
                className="button primary"
                disabled={busy}
                onClick={() => action("/refresh")}
              >
                Load official data <ArrowRight size={16} />
              </button>
            </section>
          ) : (
            <>
              {!data.state && (
                <div className="message">
                  The market is ready.{" "}
                  <button
                    className="text-button"
                    onClick={() => action("/state/seed")}
                  >
                    Import the supplied dated squad
                  </button>
                </div>
              )}
              {tab === "Overview" && (
                <>
                  <div className="stats-grid">
                    {stat(
                      "Projected horizon",
                      plan ? `${plan.simulation.mean.toFixed(1)}` : "—",
                      plan
                        ? `${plan.ledger.length} gameweeks · ${plan.simulation.worlds.toLocaleString()} shared worlds`
                        : "Calculate a plan to see your projection",
                      <ChartNoAxesCombined size={16} />,
                    )}
                    {stat(
                      "In the bank",
                      money(first?.bank_after ?? data.state?.bank ?? 0),
                      first
                        ? "After the recommended moves"
                        : "From the imported account snapshot",
                      <span className="pound">£</span>,
                    )}
                    {stat(
                      "Free transfers",
                      data.state?.free_transfers ?? "—",
                      "Available before this gameweek",
                      <ArrowRight size={16} />,
                    )}
                    {stat(
                      "Next deadline",
                      deadline
                        ? new Date(deadline).toLocaleDateString("en-GB", {
                            timeZone: "Asia/Dubai",
                            day: "numeric",
                            month: "short",
                          })
                        : "—",
                      deadline
                        ? new Date(deadline).toLocaleTimeString("en-GB", {
                            timeZone: "Asia/Dubai",
                            hour: "2-digit",
                            minute: "2-digit",
                          }) + " · Dubai time"
                        : "From the official events feed",
                      <CalendarDays size={16} />,
                    )}
                  </div>
                  <div className="overview-grid">
                    <section className="panel recommendation">
                      <div className="section-top">
                        <span className="section-kicker">
                          <Sparkles size={15} /> THE RECOMMENDATION
                        </span>
                        {pill("Independent model", "green")}
                      </div>
                      <h2>
                        {first
                          ? first.chip === "freehit"
                            ? `Play your Free Hit for GW${first.event}.`
                            : first.buy.length
                              ? `${first.buy.length} move${first.buy.length > 1 ? "s" : ""}. A stronger starting point.`
                              : first.chip
                                ? `Play your ${first.chip === "bboost" ? "Bench Boost" : first.chip === "freehit" ? "Free Hit" : first.chip}.`
                                : "Hold your transfers. Keep your options."
                          : "Turn the evidence into a plan."}
                      </h2>
                      <p className="recommendation-copy">
                        {first
                          ? `GW${first.event}–${plan?.ledger.at(-1)?.event} · Independently checked for squad legality and affordability.`
                          : "Freeze a forecast, then calculate legal transfers, your XI, captain and chip timing."}
                      </p>
                      {first ? (
                        <div className="transfer-list">
                          {first.buy.length ? (
                            first.buy.map((b, i) => (
                              <div className="transfer-pair" key={b.player_id}>
                                <div>
                                  <small>
                                    {first.chip === "freehit"
                                      ? "TEMPORARY OUT"
                                      : "TRANSFER OUT"}
                                  </small>
                                  <strong>
                                    {name(first.sell[i].player_id)}
                                  </strong>
                                  <span>{money(first.sell[i].units)}</span>
                                </div>
                                <ArrowRight size={20} />
                                <div>
                                  <small>
                                    {first.chip === "freehit"
                                      ? "TEMPORARY IN"
                                      : "TRANSFER IN"}
                                  </small>
                                  <strong>{name(b.player_id)}</strong>
                                  <span>{money(b.units)}</span>
                                </div>
                              </div>
                            ))
                          ) : (
                            <div className="roll">
                              <Check size={19} />
                              No permanent transfers recommended this week.
                            </div>
                          )}
                        </div>
                      ) : null}
                      <div className="recommendation-footer">
                        <div>
                          <ShieldCheck size={17} />
                          <span>
                            {first?.hit
                              ? `${first.hit}-point hit`
                              : "No points hit"}{" "}
                            <i /> {first?.chip || "No chip selected"}
                          </span>
                        </div>
                        <button
                          className="button primary"
                          disabled={busy || !data.state || !data.forecast}
                          onClick={solve}
                        >
                          {first ? "Recalculate plan" : "Calculate plan"}
                          <ArrowRight size={16} />
                        </button>
                      </div>
                      {!data.forecast && (
                        <button
                          className="text-button forecast-link"
                          disabled={busy}
                          onClick={() =>
                            action("/forecast", { end_event: 38, worlds: 4000 })
                          }
                        >
                          Generate full-season forecasts first →
                        </button>
                      )}
                    </section>
                    <section className="panel validity">
                      <div className="section-top">
                        <h3>The decision context</h3>
                        <Activity size={16} />
                      </div>
                      <div className="context-row">
                        <span className="context-icon green">
                          <Database size={17} />
                        </span>
                        <div>
                          <b>Official data</b>
                          <small>
                            {data.market.players.length} players ·{" "}
                            {data.market.teams.length} clubs
                          </small>
                        </div>
                        {pill(
                          data.market.age_hours < 6 ? "Fresh" : "Refresh due",
                          data.market.age_hours < 6 ? "green" : "amber",
                        )}
                      </div>
                      <div className="context-row">
                        <span className="context-icon sand">
                          <ShieldCheck size={17} />
                        </span>
                        <div>
                          <b>Account state</b>
                          <small>
                            {data.state?.status === "confirmed"
                              ? "Current values confirmed"
                              : "Dated snapshot · confirm before acting"}
                          </small>
                        </div>
                        <button
                          className="icon-button"
                          onClick={openEditor}
                          aria-label="Edit account state"
                        >
                          <ChevronRight size={16} />
                        </button>
                      </div>
                      <div className="context-row">
                        <span className="context-icon blue">
                          <FlaskConical size={17} />
                        </span>
                        <div>
                          <b>Forecast confidence</b>
                          <small>
                            Learned points and minutes · prospective evidence
                            accruing
                          </small>
                        </div>
                      </div>
                      <div className="context-note">
                        <span className="status-dot" />
                        <p>
                          Every calculation has a source, a version and a
                          timestamp. Future moves are conditional on new
                          information.
                        </p>
                      </div>
                    </section>
                  </div>
                  <div className="lineup-grid">
                    <section className="panel lineup-panel">
                      <div className="section-title">
                        <div>
                          <h3>Your recommended XI</h3>
                          <p>Captain, vice and a legal formation.</p>
                        </div>
                        {pill(`GW ${activeRow?.event || currentEvent}`)}
                      </div>
                      {pitch(activeRow)}
                      {activeRow && (
                        <div className="bench">
                          <span>BENCH ORDER</span>
                          <div>
                            {activeRow.lineup.bench.map((id, i) => (
                              <span key={id}>
                                <b>{i === 0 ? "GK" : i}</b>
                                {name(id)}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </section>
                    <section className="panel horizon-panel">
                      <div className="section-title">
                        <div>
                          <h3>The road ahead</h3>
                          <p>Your plan, with room to adapt.</p>
                        </div>
                        <CalendarDays size={17} />
                      </div>
                      <div className="timeline">
                        {plan?.ledger.slice(0, 5).map((r, i) => (
                          <div className="timeline-item" key={r.event}>
                            <div
                              className={
                                i === 0
                                  ? "timeline-dot current"
                                  : "timeline-dot"
                              }
                            />
                            <div className="timeline-title">
                              <b>Gameweek {r.event}</b>
                              <span>
                                {r.simulated_net_mean?.toFixed(1)} xPts
                              </span>
                            </div>
                            <p>
                              {r.chip ? `${r.chip} · ` : ""}
                              {r.buy.length
                                ? r.buy
                                    .map((p) => name(p.player_id))
                                    .join(" + ")
                                : "Roll · retain flexibility"}
                            </p>
                            <small>
                              {money(r.bank_after)} bank ·{" "}
                              {r.free_transfers_after} FT next
                            </small>
                          </div>
                        )) || (
                          <p className="empty-copy">
                            Calculate a plan to explore the weeks ahead.
                          </p>
                        )}
                      </div>
                      <button
                        className="full-link"
                        onClick={() => setTab("Season planner")}
                      >
                        Open season planner <ArrowRight size={16} />
                      </button>
                    </section>
                  </div>
                  <section className="panel">
                    <div className="section-title">
                      <div>
                        <h3>On the radar</h3>
                        <p>
                          Full-market projections. Explore the reasoning behind
                          the numbers.
                        </p>
                      </div>
                      <button
                        className="text-button"
                        onClick={() => setTab("Player market")}
                      >
                        Explore all players <ArrowRight size={15} />
                      </button>
                    </div>
                    {table(market.slice(0, 6).map((p) => p.id))}
                  </section>
                </>
              )}
              {tab === "My squad" && (
                <>
                  <section className="panel">
                    <div className="section-title">
                      <div>
                        <h3>{data.state?.name || "Import your squad"}</h3>
                        <p>
                          Sell quotes and purchase lots are distinct. Unverified
                          account inputs remain dated.
                        </p>
                      </div>
                      <button className="button secondary" onClick={openEditor}>
                        Edit account state
                      </button>
                    </div>
                    {table(
                      data.state?.lots.map((l) => l.player_id) || [],
                      true,
                    )}
                  </section>
                  <section className="panel transfer-builder">
                    <h3>Transfer package check</h3>
                    <p>
                      Check the exact buy/sell ledger before comparing a move.
                    </p>
                    <div className="form-row">
                      <label>
                        Sell
                        <select
                          value={sell}
                          onChange={(e) => setSell(+e.target.value)}
                        >
                          <option value={0}>Choose a player</option>
                          {data.state?.lots.map((l) => (
                            <option key={l.player_id} value={l.player_id}>
                              {name(l.player_id)} · {money(l.sell_price)}
                            </option>
                          ))}
                        </select>
                      </label>
                      <ArrowRight size={18} />
                      <label>
                        Buy
                        <select
                          value={buy}
                          onChange={(e) => setBuy(+e.target.value)}
                        >
                          <option value={0}>Choose a player</option>
                          {market
                            .filter(
                              (p) =>
                                !data.state?.lots.some(
                                  (l) => l.player_id === p.id,
                                ),
                            )
                            .map((p) => (
                              <option key={p.id} value={p.id}>
                                {p.web_name} · {money(p.price)}
                              </option>
                            ))}
                        </select>
                      </label>
                      <button
                        className="button primary"
                        disabled={!sell || !buy}
                        onClick={async () => {
                          try {
                            const r = await api<{
                              ledger: { bank_after: number; hit: number };
                            }>("/transfers/check", {
                              state_id: data.state?.id,
                              action: { sell: [sell], buy: [buy] },
                            });
                            setTransferResult(
                              `Legal and affordable · ${money(r.ledger.bank_after)} remaining · ${r.ledger.hit} hit points`,
                            );
                          } catch (e) {
                            setTransferResult((e as Error).message);
                          }
                        }}
                      >
                        Check move
                      </button>
                    </div>
                    {transferResult && (
                      <div className="message" role="status">
                        {transferResult}
                      </div>
                    )}
                  </section>
                </>
              )}
              {tab === "Player market" && (
                <section className="panel">
                  <div className="market-controls">
                    <label className="search">
                      <Search size={17} />
                      <input
                        placeholder="Search players or clubs…"
                        aria-label="Search players"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                      />
                    </label>
                    <div className="filter-group">
                      {[0, 1, 2, 3, 4].map((p) => (
                        <button
                          key={p}
                          className={filter === p ? "chosen" : ""}
                          onClick={() => setFilter(p)}
                        >
                          {p ? position(p) : "All"}
                        </button>
                      ))}
                    </div>
                    <select
                      aria-label="Sort players"
                      value={sort}
                      onChange={(e) => setSort(e.target.value as typeof sort)}
                    >
                      <option value="points">Expected points</option>
                      <option value="value">Points per £m</option>
                      <option value="minutes">Expected minutes</option>
                    </select>
                  </div>
                  <p className="table-caption">
                    {market.length} players · GW{currentEvent} · missing
                    predictions shown as —
                  </p>
                  {table()}
                </section>
              )}
              {tab === "Season planner" && (
                <>
                  <CaptainEvidence data={data} plan={plan} />
                  <HitEvidence
                    data={data}
                    plan={plan}
                    busy={busy}
                    act={action}
                    explore={async (id) => {
                      setViewPlan(await api<Plan>(`/plans/${id}`));
                      setTab("Season planner");
                    }}
                  />
                  <ChipPanel data={data} plan={plan} busy={busy} act={action} />
                  <ScenarioPanel
                    data={data}
                    plan={plan}
                    busy={busy}
                    act={action}
                  />
                  <section className="panel planner-controls">
                    <div>
                      <h3>A plan that can change with the evidence.</h3>
                      <p>
                        The immediate move is actionable after account
                        reconciliation. Later moves are conditional.
                      </p>
                    </div>
                    <div className="form-row">
                      <label>
                        Plan through
                        <select
                          value={end}
                          onChange={(e) => setEnd(+e.target.value)}
                        >
                          {data.market.events
                            .filter((e) => e.id >= currentEvent)
                            .map((e) => (
                              <option key={e.id} value={e.id}>
                                Gameweek {e.id}
                              </option>
                            ))}
                        </select>
                      </label>
                      <label>
                        Objective
                        <select
                          value={objective}
                          onChange={(e) => setObjective(e.target.value)}
                        >
                          <option value="expected_points">
                            Expected net points
                          </option>
                          <option value="beat_both">Beat both rivals</option>
                        </select>
                      </label>
                      {objective === "beat_both" && (
                        <label>
                          Points sacrifice budget
                          <input
                            type="number"
                            value={budget}
                            min={0}
                            max={100}
                            onChange={(e) => setBudget(+e.target.value)}
                          />
                        </label>
                      )}
                      <button
                        className="button primary"
                        disabled={busy || !data.forecast || !data.state}
                        onClick={solve}
                      >
                        Optimize plan <Sparkles size={16} />
                      </button>
                    </div>
                  </section>
                  {plan && (
                    <section className="panel">
                      <div className="section-title">
                        <div>
                          <h3>The full gameweek ledger</h3>
                          <p>
                            Every transfer, bank balance and free-transfer
                            transition is independently replayed.
                          </p>
                        </div>
                        <div className="export-links">
                          {["json", "csv", "md", "ics"].map((f) => (
                            <a
                              className="button secondary compact"
                              key={f}
                              href={`/api/export/${plan.id}/${f}`}
                              download
                            >
                              <ArrowDownToLine size={13} />
                              {f.toUpperCase()}
                            </a>
                          ))}
                        </div>
                      </div>
                      <div className="table-scroll">
                        <table>
                          <thead>
                            <tr>
                              <th>GW</th>
                              <th>FT</th>
                              <th>Transfers</th>
                              <th>Hit</th>
                              <th>Bank</th>
                              <th>Captain / vice</th>
                              <th>Chip</th>
                              <th>xPts</th>
                              <th>10–90%</th>
                            </tr>
                          </thead>
                          <tbody>
                            {plan.ledger.map((r) => (
                              <tr key={r.event}>
                                <td>
                                  <strong>{r.event}</strong>
                                </td>
                                <td>
                                  {r.free_transfers_before} →{" "}
                                  {r.free_transfers_after}
                                </td>
                                <td>
                                  {r.buy.length
                                    ? r.buy.map((b, i) => (
                                        <div key={b.player_id}>
                                          {name(r.sell[i].player_id)} →{" "}
                                          <b>{name(b.player_id)}</b>
                                        </div>
                                      ))
                                    : "Roll"}
                                </td>
                                <td>{r.hit}</td>
                                <td>{money(r.bank_after)}</td>
                                <td>
                                  {name(r.lineup.captain)} /{" "}
                                  {name(r.lineup.vice)}
                                </td>
                                <td>{r.chip || "—"}</td>
                                <td className="point-value">
                                  {r.simulated_net_mean.toFixed(1)}
                                </td>
                                <td>
                                  {r.net_quantiles[0].toFixed(1)}–
                                  {r.net_quantiles[2].toFixed(1)}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      {plan.solver && (
                        <div className="solver-note">
                          {plan.solver.status} · gap {percent(plan.solver.gap)}{" "}
                          · {plan.solver.runtime_seconds.toFixed(1)}s.
                          Optimality applies to the specified additive
                          formulation; shared worlds rescore autosubs and
                          captain fallback.
                        </div>
                      )}
                    </section>
                  )}
                  <section className="panel">
                    <div className="section-title">
                      <div>
                        <h3>Season calendar</h3>
                        <p>
                          Deadline times in Dubai. Fixture assignment can
                          change.
                        </p>
                      </div>
                    </div>
                    <div className="calendar-grid">
                      {data.market.events.map((e) => (
                        <button
                          className={
                            e.id === currentEvent
                              ? "calendar-cell selected"
                              : "calendar-cell"
                          }
                          key={e.id}
                          onClick={() => setEvent(e.id)}
                        >
                          <b>GW {e.id}</b>
                          <span>
                            {new Date(e.deadline_time).toLocaleDateString(
                              "en-GB",
                              {
                                timeZone: "Asia/Dubai",
                                day: "numeric",
                                month: "short",
                              },
                            )}
                          </span>
                          <small>
                            {e.finished
                              ? "Completed"
                              : plan?.ledger.find((r) => r.event === e.id)
                                  ?.chip || "Scheduled"}
                          </small>
                        </button>
                      ))}
                    </div>
                  </section>
                  {plan?.contingencies && (
                    <section className="panel">
                      <div className="section-title">
                        <h3>When the picture changes</h3>
                      </div>
                      <div className="contingencies">
                        {plan.contingencies.map((c) => (
                          <div key={c.trigger}>
                            <Activity size={18} />
                            <div>
                              <b>{c.trigger}</b>
                              <p>{c.action}</p>
                            </div>
                          </div>
                        ))}
                      </div>
                    </section>
                  )}
                </>
              )}
              {tab === "Rivals" && (
                <>
                  <ManagerSignals data={data} busy={busy} act={action} />
                  <CohortPanel data={data} busy={busy} act={action} />
                  <div className="rival-controls">
                    <button
                      className="button secondary"
                      disabled={busy}
                      onClick={() =>
                        action("/rivals", {
                          league_id: 1873551,
                          entry_id: data.state?.entry_id || 9375678,
                        })
                      }
                    >
                      Refresh league
                    </button>
                    <button
                      className="button primary"
                      disabled={busy || !data.forecast || !data.state}
                      onClick={() =>
                        action("/rivals", {
                          league_id: 1873551,
                          entry_id: data.state?.entry_id || 9375678,
                          compare_case: true,
                        })
                      }
                    >
                      Compare original strategies
                    </button>
                  </div>
                  <div className="stats-grid rivals-stats">
                    {data.league?.members.map((m) =>
                      stat(
                        m.team,
                        m.points,
                        `${m.name} · ${m.points - (data.state?.points || 0) > 0 ? "+" : ""}${m.points - (data.state?.points || 0)} vs your supplied points`,
                        <Users size={16} />,
                      ),
                    )}
                  </div>
                  <section className="panel">
                    <div className="section-title">
                      <div>
                        <h3>The comeback, under different assumptions</h3>
                        <p>
                          Strict probability of finishing above both rivals in
                          the same match worlds.
                        </p>
                      </div>
                      {pill("Scenario analysis", "amber")}
                    </div>
                    {data.rivals ? (
                      <>
                        <div className="table-scroll">
                          <table>
                            <thead>
                              <tr>
                                <th>Policy</th>
                                <th>Net xPts</th>
                                <th>Robust joint win</th>
                                <th>Learned-count scenario</th>
                                <th>Explore</th>
                              </tr>
                            </thead>
                            <tbody>
                              {data.rivals.candidate_frontier.map((r) => (
                                <tr key={r.plan_id}>
                                  <td>
                                    <b>{r.label}</b>
                                  </td>
                                  <td>{r.expected_net_points.toFixed(1)}</td>
                                  <td>
                                    <div className="probability">
                                      <span
                                        style={{
                                          width: percent(r.robust_joint_win),
                                        }}
                                      />
                                      <b>{percent(r.robust_joint_win)}</b>
                                    </div>
                                  </td>
                                  <td>
                                    {r.behavior_conditioned_joint_win == null
                                      ? "—"
                                      : percent(
                                          r.behavior_conditioned_joint_win,
                                        )}
                                  </td>
                                  <td>
                                    <button
                                      className="text-button"
                                      onClick={async () => {
                                        setViewPlan(
                                          await api<Plan>(
                                            `/plans/${r.plan_id}`,
                                          ),
                                        );
                                        setTab("Season planner");
                                      }}
                                    >
                                      View policy <ArrowRight size={14} />
                                    </button>
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                        <div className="solver-note">
                          {data.rivals.optimization_scope}. {data.rivals.ties}{" "}
                          Rival hold and greedy-transfer assumptions do not
                          reveal their private future moves.
                          {data.rivals.behavior_assumptions?.map((note) => (
                            <p key={note}>{note}</p>
                          ))}
                        </div>
                      </>
                    ) : (
                      <div className="empty-copy">
                        Run the mini-league analysis with{" "}
                        <code>uv run fpl rivals --case</code>. The comparison
                        retrieves actual released squads and evaluates multiple
                        opponent policies.
                      </div>
                    )}
                  </section>
                </>
              )}
              {tab === "Research" && (
                <>
                  <ResearchControls data={data} busy={busy} act={action} />
                  <SourceCoverage data={data} busy={busy} act={action} />
                  <section className="panel">
                    <div className="section-title">
                      <div>
                        <h3>Evidence inbox</h3>
                        <p>
                          Verbatim support, original sources and an expiry for
                          every claim.
                        </p>
                      </div>
                      {pill(`${data.evidence.length} claims`)}
                    </div>
                    {data.evidence.length ? (
                      data.evidence.map((c) => (
                        <article className="claim" key={c.id}>
                          <div>
                            {pill(c.status)}
                            <small>
                              {name(c.player_id)} ·{" "}
                              {new Date(c.known_at).toLocaleDateString()}
                            </small>
                          </div>
                          <h4>{c.statement}</h4>
                          <blockquote>{c.evidence_span}</blockquote>
                          <a
                            href={c.source_url}
                            target="_blank"
                            rel="noreferrer"
                          >
                            Original source ↗
                          </a>
                          <small>
                            Expires {new Date(c.expires_at).toLocaleString()}
                          </small>
                        </article>
                      ))
                    ) : (
                      <div className="empty-copy">
                        <FlaskConical size={25} />
                        <h4>No supported claims imported yet.</h4>
                        <p>
                          Import structured source evidence or configure a
                          verified free hosted extractor. Official player
                          availability flags already enter the baseline;
                          articles cannot silently override it.
                        </p>
                      </div>
                    )}
                  </section>
                  <section className="panel belief-editor">
                    <h3>Test an explicit belief</h3>
                    <p>
                      A separate forecast scenario keeps the original intact.
                      Use a reason and short expiry.
                    </p>
                    <div className="form-row">
                      <label>
                        Player
                        <select
                          value={beliefPlayer}
                          onChange={(e) => setBeliefPlayer(+e.target.value)}
                        >
                          <option value={0}>Choose a player</option>
                          {market.map((p) => (
                            <option key={p.id} value={p.id}>
                              {p.name}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label>
                        Start probability
                        <input
                          type="number"
                          min={0}
                          max={1}
                          step={0.05}
                          value={beliefValue}
                          onChange={(e) => setBeliefValue(+e.target.value)}
                        />
                      </label>
                      <label>
                        Expires in days
                        <input
                          type="number"
                          min={1}
                          max={30}
                          value={beliefDays}
                          onChange={(e) => setBeliefDays(+e.target.value)}
                        />
                      </label>
                    </div>
                    <label>
                      Reason
                      <textarea
                        value={beliefReason}
                        onChange={(e) => setBeliefReason(e.target.value)}
                        placeholder="What evidence changes your view?"
                      />
                    </label>
                    <button
                      className="button primary"
                      disabled={
                        !beliefPlayer || beliefReason.length < 8 || busy
                      }
                      onClick={() =>
                        action("/beliefs", {
                          player_id: beliefPlayer,
                          parameter: "p_start",
                          value: beliefValue,
                          reason: beliefReason,
                          known_at: new Date().toISOString(),
                          expires_at: new Date(
                            Date.now() + beliefDays * 86400000,
                          ).toISOString(),
                          evidence_ids: [],
                        })
                      }
                    >
                      Save belief scenario
                    </button>
                    <button
                      className="button secondary"
                      disabled={busy || !data.beliefs.length}
                      onClick={() =>
                        action("/forecast", {
                          end_event: 38,
                          worlds: 4000,
                          use_beliefs: true,
                        })
                      }
                    >
                      Forecast with beliefs
                    </button>
                    {data.beliefs.map((b) => (
                      <div className="belief-row" key={b.id}>
                        <b>{name(b.player_id)}</b>
                        <span>
                          {b.parameter} = {b.value}
                        </span>
                        <small>
                          {b.reason} · expires{" "}
                          {new Date(b.expires_at).toLocaleDateString()}
                        </small>
                      </div>
                    ))}
                  </section>
                </>
              )}
              {tab === "Model & data" && (
                <>
                  <DecisionEvidence data={data} busy={busy} act={action} />
                  <TransferLearning data={data} busy={busy} act={action} />
                  <ManagerSignals data={data} busy={busy} act={action} />
                  <SourceCoverage data={data} busy={busy} act={action} />
                  <details className="panel">
                    <summary>
                      Earlier model experiments and open-model reproduction
                    </summary>
                    <ModelEvidence data={data} busy={busy} act={action} />
                  </details>
                  <div className="stats-grid">
                    {stat(
                      "Market coverage",
                      data.market.players.length,
                      "Every registered player; typed missingness",
                      <Database size={16} />,
                    )}
                    {stat(
                      "Baseline RMSE",
                      data.decision_model?.baseline.rmse.toFixed(3) || "—",
                      "Held-out completed-match replay",
                      <ChartNoAxesCombined size={16} />,
                    )}
                    {stat(
                      "Production points RMSE",
                      data.decision_model?.candidate.rmse.toFixed(3) || "—",
                      `${data.decision_model?.candidate.n || 0} held-out fixture predictions`,
                      <FlaskConical size={16} />,
                    )}
                    {stat(
                      "Provider spending",
                      "£0 required",
                      "Paid hosted calls disabled by policy",
                      <ShieldCheck size={16} />,
                    )}
                  </div>
                  <section className="panel">
                    <div className="section-title">
                      <div>
                        <h3>Models earn their place.</h3>
                        <p>
                          Historical statistical replay and frozen prospective
                          evaluation are separate tracks.
                        </p>
                      </div>
                      <button
                        className="button secondary"
                        disabled={busy}
                        onClick={() => action("/train")}
                      >
                        Run evaluation
                      </button>
                    </div>
                    <p>
                      Completed-match statistical replay uses earlier deadlines
                      for every fold. The production points model, held-out
                      captain choices and evolving transfer accounts are
                      evaluated above. The earlier component simulator is
                      available only as an explicit research baseline.
                    </p>
                  </section>
                  <section className="panel">
                    <div className="section-title">
                      <div>
                        <h3>Source & model provenance</h3>
                        <p>
                          Immutable raw snapshots, content hashes and a local
                          transactional state database.
                        </p>
                      </div>
                      <button
                        className="button primary"
                        disabled={busy}
                        onClick={() =>
                          action("/forecast", { end_event: 38, worlds: 4000 })
                        }
                      >
                        Rebuild forecasts <RefreshCw size={15} />
                      </button>
                    </div>
                    <div className="data-details">
                      <div>
                        <span>Official registry captured</span>
                        <b>{new Date(data.market.known_at).toLocaleString()}</b>
                      </div>
                      <div>
                        <span>Forecast version</span>
                        <b>{data.forecast?.model_version || "Not generated"}</b>
                      </div>
                      <div>
                        <span>Shared worlds / seed</span>
                        <b>
                          {data.forecast
                            ? `${data.forecast.worlds.toLocaleString()} / ${data.forecast.seed}`
                            : "—"}
                        </b>
                      </div>
                      <div>
                        <span>Forecast ID</span>
                        <code>{data.forecast?.id.slice(0, 24) || "—"}</code>
                      </div>
                      <div>
                        <span>Hosted language inference</span>
                        <b>Optional · verified free capability required</b>
                      </div>
                      <div>
                        <span>Local language models</span>
                        <b>None</b>
                      </div>
                    </div>
                    {data.forecast?.assumptions.map((a) => (
                      <p className="assumption" key={a}>
                        {a}
                      </p>
                    ))}
                  </section>
                </>
              )}
            </>
          )}
          <footer className="page-footer">
            <span>
              <ShieldCheck size={13} />
              Evidence before confidence.
            </span>
            <span>FPL Intelligence · Private local workspace</span>
          </footer>
        </div>
      </main>
      {editor && (
        <div className="modal-overlay" onClick={() => setEditor(false)}>
          <section
            className="modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="editor-title"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="section-title">
              <h2 id="editor-title">Reconcile account state</h2>
              <button
                className="icon-button"
                onClick={() => setEditor(false)}
                aria-label="Close editor"
              >
                <X size={20} />
              </button>
            </div>
            <p>
              Update all 15 players, bank, free transfers and exact sell quotes
              from your current account. Set status to confirmed only after
              checking these values. Purchase prices may remain null.
            </p>
            {error && (
              <div role="alert" className="message error">
                <span>{error}</span>
                <button onClick={() => setError("")} aria-label="Dismiss error">
                  <X size={16} />
                </button>
              </div>
            )}
            <textarea
              className="json-editor"
              aria-label="Account state JSON"
              value={stateText}
              onChange={(e) => setStateText(e.target.value)}
            />
            <button className="button primary" onClick={saveState}>
              Validate & save account
            </button>
          </section>
        </div>
      )}
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
