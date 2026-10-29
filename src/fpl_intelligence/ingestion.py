"""Bounded official adapters. All transformations retain raw content hashes."""

from __future__ import annotations

import asyncio
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any

import httpx
import polars as pl

from .domain import ChipUse, Lot, Player, TeamState
from .storage import Store, digest, now

BASE = "https://fantasy.premierleague.com/api/"
PARSER = "official-v1"


class Official:
    def __init__(self, store: Store, concurrency: int = 4):
        self.store = store
        self.semaphore = asyncio.Semaphore(concurrency)
        self.errors: list[dict] = []

    async def fetch(self, client: httpx.AsyncClient, path: str, source: str | None = None):
        source = source or path.strip("/").replace("/", "_")
        async with self.semaphore:
            for attempt in range(3):
                try:
                    response = await client.get(BASE + path)
                    if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                        retry = response.headers.get("Retry-After", "1")
                        await asyncio.sleep(min(5, float(retry) if retry.isdigit() else 2**attempt))
                        continue
                    response.raise_for_status()
                    body = response.json()
                    self.store.snapshot(source, BASE + path, body)
                    return body
                except (httpx.HTTPError, ValueError) as exc:
                    if attempt == 2:
                        self.errors.append({"source": source, "error": str(exc)[:180], "at": now()})
                        return None
                    await asyncio.sleep(2**attempt)

    async def ingest(
        self, entry: int | None = None, league: int | None = None, histories: bool = True
    ) -> dict:
        async with httpx.AsyncClient(
            timeout=30,
            follow_redirects=False,
            headers={"User-Agent": "FPL-Intelligence/0.1 personal research"},
        ) as client:
            bootstrap, fixtures = await asyncio.gather(
                self.fetch(client, "bootstrap-static/", "bootstrap"),
                self.fetch(client, "fixtures/", "fixtures"),
            )
            if not isinstance(bootstrap, dict) or not isinstance(fixtures, list):
                raise ValueError("Core refresh failed; retained prior snapshots are unchanged")
            normalize(self.store, bootstrap, fixtures)
            current = next((e["id"] for e in bootstrap["events"] if e.get("is_current")), None)
            tasks: list[Any] = []
            if histories:
                tasks.extend(
                    self.fetch(client, f"element-summary/{p['id']}/", f"history_{p['id']}")
                    for p in bootstrap["elements"]
                )
            if entry:
                tasks.extend(
                    [
                        self.fetch(client, f"entry/{entry}/", f"entry_{entry}"),
                        self.fetch(client, f"entry/{entry}/history/", f"entry_history_{entry}"),
                        self.fetch(client, f"entry/{entry}/transfers/", f"transfers_{entry}"),
                    ]
                )
                if current:
                    tasks.append(
                        self.fetch(
                            client, f"entry/{entry}/event/{current}/picks/", f"picks_{entry}_{current}"
                        )
                    )
            if current:
                tasks.append(self.fetch(client, f"event/{current}/live/", f"live_{current}"))
            await asyncio.gather(*tasks)
            if league:
                for page in range(1, 101):
                    data = await self.fetch(
                        client,
                        f"leagues-classic/{league}/standings/?page_standings={page}",
                        f"league_{league}_{page}",
                    )
                    if not data or not data.get("standings", {}).get("has_next"):
                        break
            try:
                response = await client.get("https://fpl.solioanalytics.com/api/data/latest.json")
                response.raise_for_status()
                body = response.json()
                self.store.snapshot("solio", str(response.url), body, body.get("generatedAt"))
            except (httpx.HTTPError, ValueError) as exc:
                self.errors.append({"source": "solio", "error": str(exc)[:180]})
        report = {
            "captured_at": now(),
            "players": len(bootstrap["elements"]),
            "teams": len(bootstrap["teams"]),
            "events": len(bootstrap["events"]),
            "histories_requested": histories,
            "errors": self.errors,
            "parser": PARSER,
            "core_hash": digest(bootstrap),
        }
        report["id"] = self.store.put("ingestion", report)
        return report


def season_of(bootstrap: dict) -> str:
    # Event dates describe this source's competition, independent of the host calendar.
    first = min(e["deadline_time"] for e in bootstrap["events"])
    year = datetime.fromisoformat(first.replace("Z", "+00:00")).year
    return f"{year}/{str(year + 1)[2:]}"


def normalize(store: Store, bootstrap: dict, fixtures: list) -> None:
    if not bootstrap.get("elements") or not bootstrap.get("element_types") or not bootstrap.get("events"):
        raise ValueError("Official bootstrap schema missing required registry fields")
    season = season_of(bootstrap)
    players = []
    for p in bootstrap["elements"]:
        player = Player(
            id=p["id"],
            code=p["code"],
            season=season,
            name=f"{p['first_name']} {p['second_name']}",
            web_name=p["web_name"],
            team=p["team"],
            position=p["element_type"],
            price=p["now_cost"],
            selectable=p.get("can_select", True) and not p.get("removed", False),
            ownership=float(p["selected_by_percent"]),
            status=p["status"],
            news=p.get("news", ""),
            chance=None
            if p.get("chance_of_playing_next_round") is None
            else p["chance_of_playing_next_round"] / 100,
            raw=p,
        )
        players.append(player.model_dump())
    directory = store.root / "normalized"
    directory.mkdir(exist_ok=True)
    pl.DataFrame([{k: v for k, v in p.items() if k != "raw"} for p in players]).write_parquet(
        directory / "players.parquet"
    )
    pl.DataFrame(fixtures, infer_schema_length=None).write_parquet(directory / "fixtures.parquet")
    store.put(
        "market",
        {
            "season": season,
            "players": players,
            "teams": bootstrap["teams"],
            "events": bootstrap["events"],
            "fixtures": fixtures,
            "source_hash": digest(bootstrap),
            "fixture_hash": digest(fixtures),
            "known_at": now(),
        },
    )


def load_market(store: Store, identity: str | None = None) -> tuple[dict[int, Player], dict]:
    market = store.get("market", identity)
    return {p["id"]: Player.model_validate(p) for p in market["players"]}, market


def alias(value: str) -> str:
    return re.sub(
        r"[^a-z0-9]", "", unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().lower()
    )


def resolve_name(name: str, position: int, players: dict[int, Player]) -> int:
    target = alias(name)
    compatible = [p for p in players.values() if p.position == position]
    exact = [p for p in compatible if target in (alias(p.name), alias(p.web_name))]
    # Full given/family tokens can match a longer registered name (e.g. Raya Martín).
    tokens = [alias(x) for x in name.split()]
    candidates = exact or [p for p in compatible if all(t in alias(p.name) for t in tokens)]
    if len(candidates) != 1:
        raise ValueError(f"Identity unresolved or ambiguous: {name} ({len(candidates)} matches)")
    return candidates[0].id


def reconcile_seed(store: Store, context: dict) -> dict:
    existing = store.list("state", 1)
    if existing and store.get("state")["status"] == "confirmed":
        raise ValueError("A confirmed account is already saved; a dated research seed cannot replace it")
    players, market = load_market(store)
    event = next((e["id"] for e in market["events"] if e.get("is_next")), 38)
    team = context["team"]
    lots = []
    for item in team["squad"]:
        position = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}[item["position"]]
        identity = item.get("fpl_id") or resolve_name(item["name"], position, players)
        p = players[identity]
        lots.append(
            Lot(
                player_id=identity,
                sell_price=item["selling_price_units"],
                quote_market_price=p.price,
                purchase_price=item.get("purchase_price_units"),
                provenance="user_supplied_dated_snapshot",
            )
        )
    state = TeamState(
        season=market["season"],
        event=event,
        name=team["name"],
        entry_id=team["entry_id"],
        bank=team["bank_units"],
        free_transfers=team["available_ft_before_gw9"],
        lots=lots,
        used_chips=[
            ChipUse(chip={"triple_captain": "3xc"}.get(c["name"], c["name"]), event=c["gw"])
            for c in team["chips_used"]
        ],
        points=team["supplied_points"],
        status="dated_seed",
        known_at=now(),
    )
    from .rules import Rules, validate_squad

    validate_squad(state, players, Rules.from_bootstrap(store.latest("bootstrap")[0]))
    state_id = store.put("state", state.model_dump())
    store.event("seed_import", state_id, {"source": "04_USER_CONTEXT.json", "status": state.status})
    return state.model_dump() | {"id": state_id}


def freshness(meta: dict) -> float:
    return (datetime.now(timezone.utc) - datetime.fromisoformat(meta["captured_at"])).total_seconds() / 3600
