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
    gap: number | null;
    runtime_seconds: number;
    best_bound: number | null;
    objective: number;
  };
  simulation: { mean: number; sd: number; quantiles: number[]; worlds: number };
  contingencies?: { trigger: string; action: string }[];
  limitations?: string[];
};
export type State = {
  active_chip?: string | null;
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
  refresh_schedule?: { enabled: boolean; interval_minutes: number } | null;
  decision_critic?: {
    selected: string | null;
    training_rows: number;
    test_rows: number;
    candidates: Record<
      string,
      {
        mean_gain: number;
        block_gain_95ci: number[];
        promoted: boolean;
        accounts: number;
      }
    >;
  } | null;
  horizon_consistency?: { short_end: number; long_end: number; continued_short_minus_long: number; scope: string } | null;
  state_value_model?: {
    promoted: boolean;
    rmse: number;
    baseline_rmse: number;
    coefficients: number[];
    features: string[];
    paired_rmse_95ci: number[];
  } | null;
  manager_policy?: {
    metrics: Record<
      string,
      {
        conditional_log_loss?: number;
        log_loss?: number;
        baseline_log_loss: number;
        top5_recall?: number;
        promoted: boolean;
      }
    >;
    test_events: number[];
  } | null;
  market_model?: {
    price_log_loss: number;
    frequency_log_loss: number;
    promoted: boolean;
    news_promoted: boolean;
    test_deadlines: number;
    train_rows: number;
    test_rows: number;
  } | null;
  coherent_comparison?: {plan_id:string;production_mean:number;football_world_mean:number;fixtures:number;invalid_goal_allocations:number;short_lineup_worlds:number;scope:string}|null;
  coherent_model?: {
    metrics: Record<
      string,
      { promoted: boolean; poisson_deviance: number; baseline_deviance: number }
    >;
    coherence: string[];
    limits: string[];
  } | null;
  context_ablation?: {
    rows: number;
    metrics: Record<
      string,
      {
        points: { rmse: number };
        appearance_brier: number;
        sixty_brier: number;
      }
    >;
    captain_bonus: Record<string, number>;
  } | null;
  decision_analysis?: {
    status: string;
    promotion_gate_passed: boolean;
    selected_architecture: string | null;
    selected_candidate: number;
    connections: Record<string, string>;
    candidates: {
      reference: number;
      tree: number;
      network: number;
      continuation: number;
      critic_scores: Record<string, number>;
      action: { buy: number[]; sell: number[]; chip: string | null };
      ledger: { hit: number; bank_after: number };
      baseline_policy_first_action: boolean;
    }[];
  } | null;
  timing_analysis?: {
    plan_id: string;
    early_mean: number;
    wait_mean: number;
    wait_minus_early: number;
    paired_mc_interval_95: number[];
    hours_to_deadline: number;
    within_trained_time_support: boolean;
    affordability_probability_bounds: number[];
    research_priority: {
      player_id: number;
      expected_value_of_report: number;
      report_unavailable_probability: number;
    }[];
    limits: string[];
  } | null;
  population_forecast?: {
    members: number;
    target_event: number;
    scenarios: Record<
      string,
      {
        top_one_percent_probability: number;
        entry_resampling_95ci: number[];
        rank_quantiles: number[];
      }
    >;
    limits: string[];
  } | null;
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
  manager_behavior?: {
    event: number;
    aggregate_probabilities: number[];
    classes: string[];
    rich_weight: number;
    metrics: {
      rows: number;
      events: number[];
      model_log_loss: number;
      frequency_log_loss: number;
      model_accuracy: number;
    };
    recent_purchases: { player_id: number; managers: number }[];
    recent_sales: { player_id: number; managers: number }[];
    limits: string[];
  } | null;
  horizon_model?: {
    horizon: number;
    baseline: { rmse: number };
    selected: string;
    candidates: Record<string, { metrics: { rmse: number } }>;
    advanced_vs_context_95ci: number[];
  } | null;
  transfer_policy_model?: {
    test_decisions: number;
    mean_heldout_gain: number;
    four_deadline_block_gain_95ci: number[];
    promoted: boolean;
    invalid_actions: number;
  } | null;
  personalized_policy_analysis?: {
    event: number;
    status: string;
    heldout_gain_95ci: number[];
    candidates: {
      source: string;
      transfers: { out: string; in: string }[];
      estimated_fourweek_gain: number;
      hit: number;
      bank: number;
      confirmed_absence_lineup_sensitivity?: number;
      evidence?: {
        player_id: number;
        claim_type: string;
        statement: string;
        source_url: string;
        status: string;
      }[];
    }[];
  } | null;
  source_coverage?: {
    top_manager_history?: {
      retrieved_managers?: number;
      retrieved_manager_gameweeks?: number;
      selection_event?: number;
    };
    context_archive?: {
      retrieved_deadlines?: number;
      requested_deadlines?: number;
    };
    elite_cohort?: {
      retrieved_managers?: number;
      retrieved_manager_gameweeks?: number;
    };
    workload?: {
      accepted_player_matches?: number;
      rejected_player_matches?: number;
    };
    intelligence_refresh?: {
      claims?: number;
      community_leads?: number;
      source_health?: {
        source: string;
        status: string;
        retrieved?: number;
        reason?: string;
      }[];
    };
    fusion_features?: {
      rows?: number;
      coverage?: { news: number; expert_lag: number; workload: number };
    };
  };
  fusion_model?: {
    id: string;
    promoted: boolean;
    selected: string | null;
    baseline: { n: number; rmse: number };
    candidates: Record<
      string,
      {
        metrics: { rmse: number };
        paired_vs_production_95ci: number[];
        passed: boolean;
      }
    >;
    folds: {
      neural: { parameters: number; epochs: number; selected_epoch: number };
    }[];
    limits: string[];
    decision_validation?: {
      captain_totals: Record<string, number>;
      policies: Record<string, Record<string, { realized_net_points: number }>>;
      promotable: boolean;
    };
    review?: { reason?: string; promoted: boolean };
    uncertainty?: {
      regular_starters: { coverage80: number; coverage90: number };
    };
  } | null;
  fusion_comparison?: {
    event: number;
    players: {
      player_id: number;
      name: string;
      production: number;
      tree: number;
      network: number;
    }[];
  } | null;
  decision_model?: {
    promoted: boolean;
    seasons: string[];
    candidate: { n: number; rmse: number };
    baseline: { rmse: number };
    regular_starters: {
      baseline: { rmse: number };
      candidate: { rmse: number };
    };
    production_training_rows: number;
    production_calibration_rows: number;
    neural?: { metrics: { rmse: number }; paired_vs_gbm_95ci: number[] };
  } | null;
  decision_audit?: {
    captain_totals: {
      learned_bonus: number;
      price_baseline_bonus: number;
      shrinkage_bonus: number;
    };
    team_totals: { learned: number; shrinkage: number; oracle: number };
  } | null;
  decision_policy?: {
    policies: Record<
      string,
      { realized_net_points: number; hits: number; transfers: number }
    >;
  } | null;
  decision_uncertainty?: {
    regular_starters: { coverage80: number; coverage90: number };
  } | null;
  manager_audit?: {
    retrieved: number;
    prior_top10k_twice: number;
    mean_same_squad_lineup_delta: number;
    entry_bootstrap_interval_95: number[];
  } | null;
  provider_setup?: {
    providers: {
      provider: string;
      status: string;
      http_status?: number;
      message?: string;
    }[];
  } | null;
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
  hit_comparison?: {
    base_plan_id: string;
    alternative_plan_id: string;
    base_mean: number;
    alternative_mean: number;
    net_mean_advantage: number;
    paired_mc_interval_95: number[];
    alternative_immediate_chip: string | null;
    alternative_gap: number | null;
  } | null;
  chip_counterfactual?: {
    base_plan_id: string;
    comparisons: {
      chip: string;
      chosen_event: number;
      whole_policy_mean_delta: number;
      paired_mc_interval_95: number[];
      counterfactual_solver_gap: number | null;
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
      behavior_conditioned_joint_win?: number | null;
      learned_policy_joint_win?: number | null;
      policies: {
        rival_policies: string[];
        joint_strict_win: number;
        mc_standard_error: number;
      }[];
    }[];
    optimization_scope: string;
    behavior_assumptions?: string[];
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
