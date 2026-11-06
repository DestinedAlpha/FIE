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
    quote_confirmed?: boolean;
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
export type Dashboard = FootballEvidenceData & {
  role_minutes_model?: { id: string; completed_labels_through: string; training_rows: number; calibration_rows: number } | null;
  role_minutes_validation?: { passed: boolean; scope: string; metrics: Record<string, { rows: number; deadlines: number; reference: { brier: number; points_squared: number }; candidate: { brier: number; points_squared: number } }> } | null;
  refresh_schedule?: { enabled: boolean; interval_minutes: number } | null;
  decision_critic?: {
    invalid_actions:number;
    architecture_selected_on_calibration?:string;
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

    }[];
  } | null;
  decision_model?: {
    role_minutes_model_id?: string;
    future_horizon_model_id?: string;
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
  policy_selection?: {
    immediate_chip_evidence?: { chip: string; candidate_plan_id: string; alternative_plan_id: string; net_mean_gain: number; search_allowance: number | null; supported_beyond_search_error: boolean; reason: string } | null;
    selection_version?: string;
    selection: { selected_plan_id: string; reason?: string; actual_ev_sacrifice: number };
    immediate_hit_evidence?: {
      candidate_plan_id: string;
      no_hit_plan_id: string;
      net_mean_gain: number;
      no_hit_search_allowance: number | null;
      hit_supported_beyond_search_error: boolean;
      paired_mc_interval_95: number[];
      reason: string;
      weekly_tradeoff: {event: number; candidate_net: number; alternative_net: number; cumulative_net_advantage: number}[];
      source_forecast_review?: {player_id:number; name:string; fixtures:string[]; model_points:number; model_minutes:number; model_start_probability:number; official_next_points:number|null; official_status:string; official_availability:number|null; season_starts:number|null; completed_club_fixtures:number}[];
    } | null;
  } | null;
  chip_counterfactual?: {
    base_plan_id: string;
    dates_evaluated?: number;
    scope?: string;
    calendar?: { chip: string; event: number; plan_id: string; mean: number; delta_vs_selected: number; solver: { gap: number | null } }[];
    comparisons: {
      chip: string;
      chosen_event: number | null;
      window?: number[];
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

export type ValidationMetric = {
  promoted: boolean;
  log_loss?: number;
  baseline_log_loss?: number;
  block_difference_95ci?: number[];
};
export type FuturePolicyReport = {
  id:string;season:string;candidate_points:number;baseline_points:number;policy_gate_passed:boolean;
  net_return_delta_95ci:number[];future_weighted_loss_delta_95ci:number[];
  weeks:{event:number;net_points:number;baseline_points:number;hit:number;chip:string|null}[];
  verification:{sealed_deadlines:number;current_week_world_identity:string;continuous_accounts_and_actual_scores:string};
};

export type FootballEvidenceData = {
  future_horizon_model?: {id:string;fit_before:string;fit_as_of:string;training_evidence:{training_rows:number;calibration_rows:number;fit_max_label:string;cal_max_label:string};weight:number;max_lead:number;learning:string;feature_count:number}|null;
  horizon_refinement?: {selected:string;forecast_gate_passed:boolean;protocol:{scope:string;limits:string};metrics:Record<string,{reference:{rmse:number};candidate:{rmse:number};relevant_reference:{rmse:number};relevant_candidate:{rmse:number};weighted_loss_delta_95ci:number[]}>}|null;
  future_policy_replay?: FuturePolicyReport|null;
  future_policy_diagnostic?: FuturePolicyReport|null;
  future_policy_progress?: {season:string;event:number;completed:number;total:number}|null;
  future_distribution_validation?: {passed:boolean;rows:number;deadlines:number;metrics:Record<string,{reference:{crps:number;coverage80:number;appearance_brier:number;sixty_brier:number};candidate:{crps:number;coverage80:number;appearance_brier:number;sixty_brier:number};crps_delta_95ci:number[]}>}|null;
  shadow_capture_status?: {at:string;status:string;reason?:string}|null;
  shadow_registry?: {seed_event:number;season:string;seed_scope:string;missed_capture_policy:string}|null;
  shadow_cohort?: {event:number;frozen_at:string;track:string;real_account_unchanged?:boolean;actions:Record<string,{action:{chip:string|null;lineup:{captain:number;vice:number}}}>}|null;
  shadow_evaluation?: {scope:string;finalized_deadlines:number;pending_deadlines:number[];total_net_points:Record<string,number>;outcomes:{event:number;policy:string;net_points:number;hit:number;chip:string|null;track:string}[]}|null;
  pipeline_replay_progress?: {stage:string;season:string|null;event:number|null;completed_deadlines:number;total_deadlines:number;at:string}|null;
  pipeline_replay?: {id:string;protocol:{worlds:number;seconds:number;horizon:string;track:string};production_budget:boolean;live_pointers_unchanged:boolean;independent_verification:{sealed_deadlines:number;validated_actions:number;ledger_and_actual_scoring:string};summaries:Record<string,{primary_multiweek:{net_points:number;hit_points:number;chips:{event:number;chip:string}[]};reference_one_week:{net_points:number;hit_points:number;chips:{event:number;chip:string}[]};ownership_no_hits:{net_points:number;hit_points:number;chips:{event:number;chip:string}[]};paired_primary_minus_ownership_95ci:number[]|null}>;weeks:{season:string;event:number;policy:string;net_points:number;action:{chip:string|null;lineup:{captain:number}};ledger:{hit:number;bank_after:number;free_transfers_after:number};solver:{status:string;gap:number|null}}[];limits:string[]}|null;
  consensus_policy_validation?: {scope:string;acceptance:{passed:boolean;ordinary_return_delta_95ci:number[]};current_totals:Record<string,number>;previous_season_totals:Record<string,number>}|null;
  weekly_baseline_comparison?: {reference_total:number;comparisons:Record<string,{baseline_total:number}>}|null;
  consensus_model?: {id:string;selected:string;weight:number;promoted:boolean;current_totals:Record<string,number>;acceptance:{passed:boolean;return_delta:number;weekly_return_delta_95ci:number[];deadline_mse_delta_95ci:number[]};protocol:{weight_selection:string;acceptance:string};weeks:{event:number;policy:string;actual_points:number;lineup:{captain:number};players:{id:number;name:string;price:number}[]}[];limits:string[]}|null;
  consensus_comparison?: {event:number;scope:string;production_active:boolean;historical_gate_passed:boolean;players:{player_id:number;name:string;consensus:number;frozen_reference:number}[];decisions:{name:string;additive_expected_net:number;ledger:{hit:number};action:{buy:number[];sell:number[];lineup:{captain:number;vice:number}};solver:{gap:number|null}}[]}|null;
  request_metrics_24h?: {host:string;calls:number;bytes:number;mean_latency_seconds:number;errors:number}[];
  evidence_impact?: {scope:string;links:{claim_id:string;use:string;assumption?:string}[];branches:{name:string;paired_scenario_delta?:number}[]}|null;
  availability_model?: { recovery: ValidationMetric; rotation: ValidationMetric & {joint_validation?: ValidationMetric & {test_pairs:number;test_deadlines:number}} } | null;
  attack_model?: { metrics: Record<string, {promoted:boolean;weight:number}>; assist: {promoted:boolean} } | null;
  score_process_model?: { strength?:{promoted:boolean};decay:{adopted:number};score_dependence:{adopted:string};parameter_uncertainty:{adopted_scale:number} } | null;
  temporal_model?: { adopted:{decay:number;finishing_strength:number|null};promoted:boolean;scope:string } | null;
  odds_refresh?: {status:string;premier_league_rows:number} | null;
  odds_model?: {promoted:boolean;observations:number;reason:string} | null;
  shot_archive?: {matched:number;unresolved:number;errors:unknown[];coverage:{current_players:number};temporal_track:string} | null;
  discipline_report?: {records:{player_id:number;cautions:number;cautions_range:number[];fixture_ids:number[];red_reason_required:boolean}[]} | null;
  account_sync?: {status:string;revealed_event:number;planning_event:number;limitation:string;at:string} | null;
  weekly_replay?: {id:string;model_scope?:string;total_points:number;learning:string;limits:string[];weeks:{event:number;actual_points:number;forecast_mean:number;hindsight_points:number;solver:{gap:number|null};training:{calibration_max_group:string}}[]} | null;
};
