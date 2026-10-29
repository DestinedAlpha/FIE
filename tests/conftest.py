import pytest

from fpl_intelligence.domain import Lot, Player, TeamState
from fpl_intelligence.rules import Rules


@pytest.fixture
def market():
    players = {}
    identity = 1
    # Two complete independent squads plus cheap alternatives; each club gets
    # at most two players in the reference squad.
    for group in range(3):
        for position, count in [(1, 2), (2, 5), (3, 5), (4, 3)]:
            for _ in range(count):
                p = Player(
                    id=identity,
                    code=identity,
                    season="2026/27",
                    name=f"Player {identity}",
                    web_name=f"P{identity}",
                    team=(identity - 1) % 10 + 1,
                    position=position,
                    price=50 + group,
                    ownership=1,
                    status="a",
                    news="",
                    raw={},
                )
                players[identity] = p
                identity += 1
    return players


@pytest.fixture
def rules():
    return Rules(
        chips=[
            {"name": name, "start_event": start, "stop_event": stop}
            for name in ("wildcard", "freehit", "bboost", "3xc")
            for start, stop in ((1, 19), (20, 38))
        ]
    )


@pytest.fixture
def state(market):
    return TeamState(
        season="2026/27",
        event=9,
        bank=30,
        free_transfers=2,
        lots=[
            Lot(player_id=p, purchase_price=48, sell_price=49, quote_market_price=50) for p in range(1, 16)
        ],
        known_at="2026-10-29T07:00:00+00:00",
    )
