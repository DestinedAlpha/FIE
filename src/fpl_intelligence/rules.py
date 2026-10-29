"""Independent deterministic scoring and account transition oracle.

Money is always integer tenths of a million. Unknown purchase lots never become
invented acquisition prices. Narrative thresholds are a versioned rules overlay.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import permutations

from .domain import Action, Lineup, Lot, Player, TeamState
from .storage import digest


@dataclass
class Rules:
    season: str = "2026/27"
    composition: dict[int, int] = field(default_factory=lambda: {1: 2, 2: 5, 3: 5, 4: 3})
    minimum: dict[int, int] = field(default_factory=lambda: {1: 1, 2: 3, 3: 2, 4: 1})
    maximum: dict[int, int] = field(default_factory=lambda: {1: 1, 2: 5, 3: 5, 4: 3})
    club_limit: int = 3
    ft_cap: int = 5
    transfer_cap: int = 20
    hit: int = 4
    scoring: dict = field(default_factory=dict)
    chips: list[dict] = field(default_factory=list)
    event_overrides: dict = field(default_factory=dict)
    # Official PL defensive-contribution explanation, rule overlay v1.
    dc_threshold: dict[int, int] = field(default_factory=lambda: {1: 9999, 2: 10, 3: 12, 4: 12})
    version: str = "official-overlay-v1"

    @classmethod
    def from_bootstrap(cls, bootstrap: dict):
        from .ingestion import season_of

        settings = bootstrap["game_settings"]
        config = bootstrap.get("game_config", {})
        scoring = config.get("scoring", {})
        types = bootstrap["element_types"]
        return cls(
            season=season_of(bootstrap),
            composition={t["id"]: t["squad_select"] for t in types},
            minimum={t["id"]: t["squad_min_play"] for t in types},
            maximum={t["id"]: t["squad_max_play"] for t in types},
            club_limit=settings["squad_team_limit"],
            ft_cap=1 + settings["max_extra_free_transfers"],
            transfer_cap=settings["transfers_cap"],
            scoring=scoring,
            chips=bootstrap["chips"],
            event_overrides={e["id"]: e.get("overrides", {}) for e in bootstrap["events"]},
        )

    @property
    def hash(self):
        from dataclasses import asdict

        return digest(asdict(self))

    def reward(self, key: str, position: int, default):
        labels = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
        value = self.scoring.get(key, default)
        if isinstance(value, dict):
            return value[labels[position]]
        return value

    def chip_window(self, chip: str, event: int) -> tuple[int, int]:
        windows = [
            c for c in self.chips if c["name"] == chip and c["start_event"] <= event <= c["stop_event"]
        ]
        if not windows:
            raise ValueError(f"{chip} unavailable at GW{event}")
        return windows[0]["start_event"], windows[0]["stop_event"]


def selling_price(purchase: int, market: int) -> int:
    return market if market <= purchase else purchase + (market - purchase) // 2


def quote(lot: Lot, market_price: int) -> int:
    if market_price == lot.quote_market_price:
        return lot.sell_price
    if lot.purchase_price is not None:
        return selling_price(lot.purchase_price, market_price)
    raise ValueError(
        f"Player {lot.player_id}: unknown purchase lot; changed price requires a fresh sell quote"
    )


def validate_squad(
    state: TeamState, players: dict[int, Player], rules: Rules, allow_grandfathered: bool = True
):
    ids = [lot.player_id for lot in state.lots]
    if set(ids) - players.keys():
        raise ValueError("Squad contains unknown season-scoped player IDs")
    if len(ids) != 15 or len(set(ids)) != 15:
        raise ValueError("Squad requires 15 distinct players")
    if state.season != rules.season or any(players[p].season != state.season for p in ids):
        raise ValueError("Season mismatch")
    counts: Counter[int] = Counter(players[p].position for p in ids)
    if dict(counts) != rules.composition:
        raise ValueError(f"Illegal composition: {dict(counts)}")
    for club, count in Counter(players[p].team for p in ids).items():
        if count > rules.club_limit and not (allow_grandfathered and club in state.grandfathered_clubs):
            raise ValueError(f"More than {rules.club_limit} players from club {club}")
    if state.bank < 0:
        raise ValueError("Negative bank")
    return True


def validate_lineup(lineup: Lineup, squad: list[int], players: dict[int, Player], rules: Rules):
    if len(lineup.xi) != 11 or len(lineup.bench) != 4:
        raise ValueError("Lineup requires 11 starters and four ordered substitutes")
    if len(set(lineup.xi + lineup.bench)) != 15 or set(lineup.xi + lineup.bench) != set(squad):
        raise ValueError("Lineup must partition the entire squad")
    if lineup.captain == lineup.vice or not {lineup.captain, lineup.vice} <= set(lineup.xi):
        raise ValueError("Captain and vice must be distinct starters")
    counts: Counter[int] = Counter(players[p].position for p in lineup.xi)
    if any(not rules.minimum[pos] <= counts[pos] <= rules.maximum[pos] for pos in rules.composition):
        raise ValueError("Illegal starting formation")
    if players[lineup.bench[0]].position != 1 or any(players[p].position == 1 for p in lineup.bench[1:]):
        raise ValueError("Reserve goalkeeper must occupy first bench slot")
    return True


def bonus_awards(bps: dict[int, float]) -> dict[int, int]:
    """Competition ranking: 1,1,3 -> 3,3,1; 1,2,2 -> 3,2,2."""
    return {p: max(0, 3 - sum(v > value for v in bps.values())) for p, value in bps.items()}


def score_fixture(events: dict, position: int, rules: Rules) -> int:
    minutes = events.get("minutes", 0)
    if not minutes:
        return 0
    reward = rules.reward
    points = reward("long_play", position, 2) if minutes >= 60 else reward("short_play", position, 1)
    points += events.get("goals_scored", 0) * reward(
        "goals_scored", position, {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4}
    )
    points += events.get("assists", 0) * reward("assists", position, 3)
    if minutes >= 60 and events.get("goals_conceded", 0) == 0:
        points += reward("clean_sheets", position, {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0})
    points += (events.get("goals_conceded", 0) // 2) * reward(
        "goals_conceded", position, {"GKP": -1, "DEF": -1, "MID": 0, "FWD": 0}
    )
    if position == 1:
        points += (events.get("saves", 0) // 3) * reward("saves", position, 1)
    for key, default in [
        ("penalties_saved", 5),
        ("penalties_missed", -2),
        ("yellow_cards", -1),
        ("red_cards", -3),
        ("own_goals", -2),
        ("bonus", 1),
    ]:
        points += events.get(key, 0) * reward(key, position, default)
    if "defensive_actions" in events:
        actions = events["defensive_actions"]
    else:
        actions = events.get("clearances_blocks_interceptions", 0) + events.get("tackles", 0)
        if position in (3, 4):
            actions += events.get("recoveries", 0)
    if actions >= rules.dc_threshold[position]:
        points += reward("defensive_contribution", position, {"GKP": 0, "DEF": 2, "MID": 2, "FWD": 2})
    return int(points)


def scoring_ids(
    lineup: Lineup,
    appeared: dict[int, bool],
    players: dict[int, Player],
    rules: Rules,
    bench_boost: bool = False,
) -> list[int]:
    if bench_boost:
        return lineup.xi + lineup.bench
    active = list(lineup.xi)
    keeper = next(p for p in active if players[p].position == 1)
    if not appeared.get(keeper, False) and appeared.get(lineup.bench[0], False):
        active[active.index(keeper)] = lineup.bench[0]
    missing = [p for p in active if players[p].position != 1 and not appeared.get(p, False)]
    available = [p for p in lineup.bench[1:] if appeared.get(p, False)]
    # Find the lexicographically earliest feasible subset of bench slots. Formation
    # counts include nonappearing unreplaced XI, matching official substitution rules.
    best_key = None
    best = active
    from itertools import combinations

    for size in range(min(len(missing), len(available)) + 1):
        for entering in combinations(available, size):
            for leaving in combinations(missing, size):
                candidate = [p for p in active if p not in leaving] + list(entering)
                counts: Counter[int] = Counter(players[p].position for p in candidate)
                if all(rules.minimum[pos] <= counts[pos] <= rules.maximum[pos] for pos in rules.composition):
                    key = (-size, tuple(lineup.bench.index(p) for p in entering))
                    if best_key is None or key < best_key:
                        best_key, best = key, candidate
    return best


def score_gameweek(
    points: dict[int, int | float],
    minutes: dict[int, int],
    lineup: Lineup,
    players: dict[int, Player],
    rules: Rules,
    chip: str | None = None,
) -> float:
    ids = scoring_ids(lineup, {p: m > 0 for p, m in minutes.items()}, players, rules, chip == "bboost")
    captain = lineup.captain if minutes.get(lineup.captain, 0) > 0 else lineup.vice
    extra = points.get(captain, 0) if minutes.get(captain, 0) > 0 else 0
    return sum(points.get(p, 0) for p in ids) + extra * (2 if chip == "3xc" else 1)


def transition(
    state: TeamState,
    action: Action,
    players: dict[int, Player],
    rules: Rules,
    prices: dict[int, int] | None = None,
) -> tuple[TeamState, dict]:
    """Apply one entire event bundle and return the underlying NEXT deadline state."""
    validate_squad(state, players, rules)
    prices = prices or {p: player.price for p, player in players.items()}
    owned = {lot.player_id: lot for lot in state.lots}
    if set(action.buy + action.sell) - players.keys():
        raise ValueError("Transfer bundle contains unknown season-scoped player IDs")
    if len(action.buy) != len(action.sell):
        raise ValueError("Equal numbers of transfers in and out are required")
    if not set(action.sell) <= owned.keys() or set(action.buy) & owned.keys():
        raise ValueError("Cannot sell unowned players or buy owned players")
    if any(not players[p].selectable for p in action.buy):
        raise ValueError("Unselectable incoming player")
    chip = action.chip
    if chip:
        start, stop = rules.chip_window(chip, state.event)
        if any(c.chip == chip and start <= c.event <= stop for c in state.used_chips):
            raise ValueError("Chip already used in this period")
        if chip == "freehit" and any(c.chip == chip and c.event == state.event - 1 for c in state.used_chips):
            raise ValueError("Consecutive Free Hits are prohibited")
        if (
            chip == "freehit"
            and state.transfers_made
            and (state.deadline_lots is None or state.deadline_bank is None)
        ):
            raise ValueError("Free Hit after prior transfers requires the prior deadline squad and bank")
    transfers = state.transfers_made + len(action.buy)
    if transfers > rules.transfer_cap and chip not in ("freehit", "wildcard") and not state.initial:
        raise ValueError("Event transfer cap exceeded")
    sales = [{"player_id": p, "units": quote(owned[p], prices[p])} for p in action.sell]
    purchases = [{"player_id": p, "units": prices[p]} for p in action.buy]
    bank = state.bank + sum(x["units"] for x in sales) - sum(x["units"] for x in purchases)
    if bank < 0:
        raise ValueError("Transfer bundle is unaffordable")
    lots = [lot for lot in state.lots if lot.player_id not in action.sell]
    lots.extend(
        Lot(
            player_id=p,
            purchase_price=prices[p],
            sell_price=prices[p],
            quote_market_price=prices[p],
            provenance=f"plan_gw{state.event}",
        )
        for p in action.buy
    )
    playing_state = state.model_copy(update={"lots": lots, "bank": bank})
    validate_squad(playing_state, players, rules, allow_grandfathered=not action.buy)
    if action.lineup:
        validate_lineup(action.lineup, [lot.player_id for lot in lots], players, rules)
    total_hit = (
        0
        if chip in ("wildcard", "freehit") or state.initial
        else max(0, transfers - state.free_transfers) * rules.hit
    )
    incremental_hit = total_hit - state.hit_paid
    if chip in ("wildcard", "freehit"):
        ft = state.free_transfers
    elif state.initial:
        ft = 1
    else:
        ft = min(rules.ft_cap, max(0, state.free_transfers - transfers) + 1)
    # Exceptional allocation is a narrative overlay explicitly supplied in the
    # authoritative event rules, never inferred from a calendar date.
    allocation = rules.event_overrides.get(state.event + 1, {}).get("rules", {}).get("free_transfers")
    if allocation is not None:
        ft = int(allocation)
    if chip == "freehit":
        next_lots = state.deadline_lots if state.deadline_lots is not None else state.lots
        next_bank = state.deadline_bank if state.deadline_bank is not None else state.bank
    else:
        next_lots, next_bank = lots, bank
    from .domain import ChipUse

    used = state.used_chips + ([ChipUse(chip=chip, event=state.event)] if chip else [])
    next_state = state.model_copy(
        update={
            "event": min(38, state.event + 1),
            "lots": next_lots,
            "bank": next_bank,
            "free_transfers": ft,
            "used_chips": used,
            "transfers_made": 0,
            "hit_paid": 0,
            "deadline_lots": None,
            "deadline_bank": None,
            "initial": False,
        }
    )
    validate_squad(next_state, players, rules)
    ledger = {
        "event": state.event,
        "sell": sales,
        "buy": purchases,
        "bank_before": state.bank,
        "bank_after": bank,
        "restored_bank": next_bank if chip == "freehit" else None,
        "free_transfers_before": state.free_transfers,
        "free_transfers_after": ft,
        "hit": incremental_hit,
        "event_total_hit": total_hit,
        "chip": chip,
        "playing_squad": [lot.player_id for lot in lots],
        "permanent_squad": [lot.player_id for lot in next_lots],
        "lineup": action.lineup.model_dump() if action.lineup else None,
    }
    return next_state, ledger


def best_lineup(
    squad: list[int], means: dict[int, float], players: dict[int, Player], rules: Rules
) -> Lineup:
    """Exact additive XI/captain selection over all legal formations."""
    ordered = {
        pos: sorted((p for p in squad if players[p].position == pos), key=lambda p: (-means.get(p, 0), p))
        for pos in rules.composition
    }
    candidates = []
    for defenders in range(3, 6):
        for midfielders in range(2, 6):
            forwards = 10 - defenders - midfielders
            if not 1 <= forwards <= 3:
                continue
            xi = ordered[1][:1] + ordered[2][:defenders] + ordered[3][:midfielders] + ordered[4][:forwards]
            if len(xi) == 11:
                ranked = sorted(xi, key=lambda p: (-means.get(p, 0), p))
                candidates.append((sum(means.get(p, 0) for p in xi) + means.get(ranked[0], 0), xi, ranked))
    if not candidates:
        raise ValueError("No legal XI")
    _, xi, ranked = max(candidates, key=lambda x: x[0])
    bench = [p for p in squad if p not in xi]
    bench = sorted(bench, key=lambda p: (players[p].position != 1, -means.get(p, 0), p))
    result = Lineup(xi=xi, bench=bench, captain=ranked[0], vice=ranked[1])
    validate_lineup(result, squad, players, rules)
    return result


def optimize_bench(lineup: Lineup, points, minutes, ids: list[int], players: dict[int, Player], rules: Rules):
    """Rescore six outfield orders in identical worlds; no additive bench weights."""
    import numpy as np

    index = {p: i for i, p in enumerate(ids)}
    relevant = lineup.xi + lineup.bench
    patterns, inverse = np.unique(minutes[:, [index[p] for p in relevant]] > 0, axis=0, return_inverse=True)
    best = None
    for order in permutations(lineup.bench[1:]):
        candidate = lineup.model_copy(update={"bench": [lineup.bench[0], *order]})
        score = np.zeros(len(points))
        for group, pattern in enumerate(patterns):
            selected = scoring_ids(candidate, dict(zip(relevant, pattern)), players, rules)
            mask = inverse == group
            score[mask] = points[mask][:, [index[p] for p in selected]].sum(axis=1)
        if best is None or score.mean() > best[0]:
            best = (score.mean(), candidate)
    assert best is not None
    return best[1]
