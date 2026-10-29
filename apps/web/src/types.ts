export type Player = {
  id: number;
  name: string;
  web_name: string;
  team: number;
  position: number;
  price: number;
  ownership: number;
  status: string;
  news: string;
};
export type Projection = {
  player_id: number;
  event: number;
  mean: number;
  sd: number;
  minutes_mean: number;
  p_start: number;
  p_zero: number;
  p_60: number;
  point_quantiles: number[];
  components: Record<string, number>;
  fixture_count: number;
  action_coverage: number;
};
export type Lineup = {
  xi: number[];
  bench: number[];
  captain: number;
  vice: number;
};
export type Ledger = {
  event: number;
  sell: { player_id: number; units: number }[];
  buy: { player_id: number; units: number }[];
  hit: number;
  bank_after: number;
  free_transfers_before: number;
  free_transfers_after: number;
  lineup: Lineup;
  chip: string | null;
  playing_squad: number[];
  permanent_squad: number[];
  simulated_net_mean: number;
  net_quantiles: number[];
};
export type Plan = {
  id: string;
  state_id: string;
  forecast_id: string;
  execution_ready?: boolean;
  stale?: boolean;
  ledger: Ledger[];
  solver?: {
    status: string;
    gap: number;
    runtime_seconds: number;
    best_bound: number;
    objective: number;
  };
  simulation: { mean: number; sd: number; quantiles: number[]; worlds: number };
  contingencies?: { trigger: string; action: string }[];
  limitations?: string[];
};
export type State = {
  id: string;
  season: string;
  event: number;
  name: string;
  entry_id: number | null;
  status: string;
  bank: number;
  points: number;
  free_transfers: number;
  lots: {
    player_id: number;
    sell_price: number;
    purchase_price: number | null;
    quote_market_price: number;
    provenance: string;
  }[];
  used_chips: { chip: string; event: number }[];
  known_at: string;
  transfers_made: number;
  hit_paid: number;
  initial: boolean;
  grandfathered_clubs: number[];
  deadline_lots: unknown;
  deadline_bank: number | null;
};
export type Dashboard = {
  at: string;
  market: {
    id: string;
    season: string;
    known_at: string;
    age_hours: number;
    players: Player[];
    teams: { id: number; name: string; short_name: string }[];
    events: {
      id: number;
      deadline_time: string;
      is_next: boolean;
      finished: boolean;
    }[];
    fixtures: {
      id: number;
      event: number | null;
      team_h: number;
      team_a: number;
      kickoff_time: string | null;
      finished: boolean;
    }[];
  } | null;
  state: State | null;
  forecast: {
    id: string;
    issued_at: string;
    start_event: number;
    end_event: number;
    model_version: string;
    worlds: number;
    seed: number;
    players: Projection[];
    assumptions: string[];
    belief_scenario: string[];
  } | null;
  plan: Plan | null;
  model_suite?: {
    promoted: boolean;
    test_rows: number;
    calibration_rows: number;
    baseline: { start: { brier: number }; appearance: { brier: number } };
    candidate: { start: { brier: number }; appearance: { brier: number } };
  } | null;
  policy_evaluation?: {
    events: number;
    invalid_exported_actions: number;
    policies: Record<
      string,
      { realized_net_points: number; hits: number; transfers: number }
    >;
  } | null;
  open_model_reproduction?: {
    dastan: { rows: number; features: number; metrics: { rmse: number } };
    openfpl: { rows: number; metrics: { rmse: number } };
  } | null;
  extraction_benchmark?: {
    selected: { model: string; f1: number; status_accuracy: number } | null;
  } | null;
  chip_counterfactual?: {
    base_plan_id: string;
    comparisons: {
      chip: string;
      chosen_event: number;
      whole_policy_mean_delta: number;
      paired_mc_interval_95: number[];
      counterfactual_solver_gap: number;
    }[];
  } | null;
  scenario_evaluation?: {
    branches: {
      name: string;
      plan_id: string;
      reveal_event: number;
      paired_scenario_delta?: number;
      paired_mc_interval_95?: number[];
      simulation: { mean: number };
    }[];
  } | null;
  cohort?: {
    retrieved: number;
    misses: number;
    event: number;
    population: string;
    effective_ownership: {
      player_id: number;
      effective_ownership: number;
      resampling_interval_95: number[];
    }[];
  } | null;
  evaluation: {
    baseline: { rmse: number; mae: number; n: number };
    challenger: { rmse: number; mae: number };
    promoted: boolean;
    start_brier: number;
    paired_rmse_difference_95ci: number[];
    track: string;
    events: number[];
    limitations: string[];
  } | null;
  league: {
    members: { entry: number; name: string; team: string; points: number }[];
    population: string;
  } | null;
  rivals: {
    sacrifice_budget: number;
    starting_points: number;
    candidate_frontier: {
      plan_id: string;
      label: string;
      expected_net_points: number;
      robust_joint_win: number;
      policies: {
        rival_policies: string[];
        joint_strict_win: number;
        mc_standard_error: number;
      }[];
    }[];
    optimization_scope: string;
    ties: string;
  } | null;
  evidence: {
    id: string;
    player_id: number;
    statement: string;
    evidence_span: string;
    source_url: string;
    status: string;
    known_at: string;
    expires_at: string;
  }[];
  beliefs: {
    id: string;
    player_id: number;
    parameter: string;
    value: number;
    reason: string;
    expires_at: string;
  }[];
};

export async function api<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(
    "/api" + path,
    body === undefined
      ? undefined
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  const result = await response.json();
  if (!response.ok)
    throw new Error(
      typeof result.detail === "string"
        ? result.detail
        : JSON.stringify(result.detail),
    );
  return result as T;
}

export const money = (units: number) => `£${(units / 10).toFixed(1)}m`;
export const percent = (value: number) => `${(value * 100).toFixed(0)}%`;
export const position = (value: number) =>
  ["", "GK", "DEF", "MID", "FWD"][value];
