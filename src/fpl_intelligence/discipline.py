"""Competition-specific suspensions, publication cutoffs and appeal/service state."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .domain import Strict
from .storage import now

RULES_PATH = Path(__file__).resolve().parents[2] / "configs/discipline-2026-27.json"


class BanDecision(Strict):
    player_code: int = Field(ge=1)
    offence: Literal[
        "second_caution",
        "denying_goal",
        "offensive_language",
        "serious_foul_play",
        "violent_conduct",
        "spitting",
        "caution_accumulation",
        "other",
    ]
    imposed_at: str
    known_at: str
    source_url: str
    matches: int | None = Field(default=None, ge=1, le=100)
    competition: str = "PL"
    service_competitions: list[str] = Field(default_factory=list)
    appeal: Literal["none", "pending", "upheld", "rescinded", "stayed"] = "none"
    previous_dismissals: int = Field(default=0, ge=0)
    supersedes: str | None = None
    additional_fixtures: list[dict] = Field(default_factory=list)

    @model_validator(mode="after")
    def dated(self):
        for value in (self.imposed_at, self.known_at):
            if datetime.fromisoformat(value).tzinfo is None:
                raise ValueError("Disciplinary decisions require timezone-aware publication dates")
        if not self.source_url.startswith("https://"):
            raise ValueError("A disciplinary decision requires an original HTTPS source")
        if datetime.fromisoformat(self.imposed_at) > datetime.fromisoformat(self.known_at):
            raise ValueError("A sanction cannot be known before it is imposed")
        for fixture in self.additional_fixtures:
            if (
                fixture.get("competition") not in ("PL", "FA_CUP", "LEAGUE_CUP")
                or not fixture.get("kickoff_time")
                or "id" not in fixture
            ):
                raise ValueError("Cup-service fixtures need an explicit competition, ID and kickoff")
        if self.offence in ("other", "caution_accumulation") and self.matches is None:
            raise ValueError("This sanction requires an explicit published length")
        return self


def ban_service(decision: BanDecision, fixtures: list[dict], cutoff: str, config=None):
    config = config or json.loads(RULES_PATH.read_text())
    if datetime.fromisoformat(decision.known_at) > datetime.fromisoformat(cutoff):
        return {"active": False, "reason": "Not yet known at cutoff", "fixture_ids": []}
    if decision.appeal in ("rescinded", "stayed"):
        return {"active": False, "reason": decision.appeal, "fixture_ids": []}
    length = decision.matches or config["red_offences"][decision.offence] + decision.previous_dismissals
    competitions = decision.service_competitions or (
        [decision.competition]
        if decision.offence == "caution_accumulation"
        else config["red_service_competitions"]
    )
    eligible = sorted(
        (
            f
            for f in fixtures
            if f.get("competition", "PL") in competitions
            and f.get("kickoff_time")
            and datetime.fromisoformat(f["kickoff_time"].replace("Z", "+00:00"))
            > datetime.fromisoformat(decision.imposed_at)
            and not f.get("postponed")
            and f.get("status") not in ("postponed", "cancelled", "conditional", "proposal", "unconfirmed")
            and f.get("confirmed", True)
        ),
        key=lambda f: (f["kickoff_time"], str(f["id"])),
    )
    served = eligible[:length]
    return {
        "active": True,
        "matches": length,
        "fixture_ids": [f["id"] for f in served],
        "remaining_unassigned": max(0, length - len(served)),
        "appeal": decision.appeal,
        "competition_specific": decision.offence == "caution_accumulation",
        "source_url": decision.source_url,
    }


def infer_caution_bans(rows, fixtures, thresholds=None):
    thresholds = thresholds or json.loads(RULES_PATH.read_text())["caution_thresholds"]
    club_matches = defaultdict(list)
    for fixture in fixtures:
        if (
            fixture.get("competition", "PL") != "PL"
            or not fixture.get("kickoff_time")
            or fixture.get("postponed")
        ):
            continue
        for club in (fixture["home"], fixture["away"]):
            club_matches[club].append(fixture)
    for matches in club_matches.values():
        matches.sort(key=lambda f: (f["kickoff_time"], str(f["id"])))
    count = 0
    crossed = set()
    result = []
    for row in sorted(rows, key=lambda r: (r.get("kickoff_time") or "", r["fixture"])):
        if row.get("cautions_count_toward_accumulation") is False:
            continue
        before = count
        count += int(row.get("yellow_cards") or 0)
        schedule = club_matches.get(row["team"], [])
        ordinal = next((i + 1 for i, f in enumerate(schedule) if f["id"] == row["fixture"]), None)
        for threshold in thresholds:
            n = threshold["count"]
            if (
                ordinal
                and n not in crossed
                and before < n <= count
                and ordinal <= threshold["by_club_league_match"]
            ):
                crossed.add(n)
                next_matches = schedule[ordinal : ordinal + threshold["matches"]]
                result.append(
                    {
                        "threshold": n,
                        "trigger_fixture": row["fixture"],
                        "club_match": ordinal,
                        "fixture_ids": [f["id"] for f in next_matches],
                        "matches": threshold["matches"],
                        "reason": "Competition-specific accumulated PL cautions",
                    }
                )
    return result, count


def build_discipline(store):
    from .decision import canonical_rows
    from .ingestion import load_market
    from .observations import information_time

    players, market = load_market(store)
    rows, _, _ = canonical_rows(store)
    names = {t["id"]: t["name"] for t in market["teams"]}
    fixtures = [
        {**f, "competition": "PL", "home": names[f["team_h"]], "away": names[f["team_a"]]}
        for f in market["fixtures"]
    ]
    by_code = defaultdict(list)
    for row in rows:
        if row["season"] == market["season"] and row.get("code"):
            by_code[int(row["code"])].append(row)
    decisions = [store.get("ban_decision", r["id"]) for r in store.list("ban_decision", 1000)]
    cutoff = information_time(store)
    at = datetime.fromisoformat(cutoff)
    decisions = [r for r in decisions if datetime.fromisoformat(r["known_at"]) <= at]
    windows = [store.get("ban_window", r["id"]) for r in store.list("ban_window", 1000)]
    if store.list("discipline_refresh", 1):
        latest = store.get("discipline_refresh")
        if latest.get("authoritative_roster"):
            windows = [
                w
                for w in windows
                if w.get("source_url") != latest["source_url"] or w["id"] in latest["ban_windows"]
            ]
    published_cautions: dict[int, dict] = {}
    for item in store.list("official_cautions", 1000):
        r = store.get("official_cautions", item["id"])
        if r["season"] == market["season"] and datetime.fromisoformat(r["known_at"]) <= at:
            published_cautions.setdefault(r["player_code"], r)
    superseded = {d["supersedes"] for d in decisions if d.get("supersedes")}
    records = []
    for player in players.values():
        history = by_code[player.code]
        lower = [
            {**r, "cautions_count_toward_accumulation": False}
            if int(r.get("red_cards") or 0) > 0 and int(r.get("yellow_cards") or 0) > 0
            else r
            for r in history
        ]
        bans, cautions = infer_caution_bans(history, fixtures)
        lower_bans, minimum = infer_caution_bans(lower, fixtures)
        upper_ids = set(p for ban in bans for p in ban["fixture_ids"])
        lower_ids = set(p for ban in lower_bans for p in ban["fixture_ids"])
        unavailable = upper_ids & lower_ids
        for window in windows:
            if (
                window["player_code"] == player.code
                and window["season"] == market["season"]
                and datetime.fromisoformat(window["known_at"]) <= at
            ):
                unavailable.update(
                    f["id"]
                    for f in fixtures
                    if f.get("event") in window["events"] and player.team in (f["team_h"], f["team_a"])
                )
        explicit = []
        for saved in decisions:
            if saved["id"] in superseded or saved["player_code"] != player.code:
                continue
            decision = BanDecision.model_validate({k: v for k, v in saved.items() if k != "id"})
            schedule = [f for f in fixtures if player.team in (f["team_h"], f["team_a"])]
            schedule += saved.get("additional_fixtures", [])
            service = ban_service(decision, schedule, cutoff)
            unavailable.update(service["fixture_ids"])
            explicit.append(service)
        relevant = [f for f in fixtures if player.team in (f["team_h"], f["team_a"])]
        played = sum(bool(f.get("finished")) for f in relevant)
        future_threshold = next(
            (
                t
                for t in json.loads(RULES_PATH.read_text())["caution_thresholds"]
                if cautions < t["count"] and played < t["by_club_league_match"]
            ),
            None,
        )
        records.append(
            {
                "player_id": player.id,
                "player_code": player.code,
                "cautions": cautions,
                "cautions_range": [minimum, cautions],
                "published_cautions": published_cautions.get(player.code),
                "caution_source_conflict": bool(
                    player.code in published_cautions
                    and not minimum <= published_cautions[player.code]["count"] <= cautions
                ),
                "uncertain_fixture_ids": sorted(upper_ids ^ lower_ids),
                "next_threshold": future_threshold,
                "league_matches_played": played,
                "fixture_ids": sorted(unavailable),
                "inferred": bans,
                "published_decisions": explicit,
                "official_status": player.status,
                "red_reason_required": player.status == "s"
                and any(int(r.get("red_cards") or 0) > 0 for r in by_code[player.code])
                and not explicit
                and not any(w["player_code"] == player.code for w in windows),
            }
        )
    report = {
        "at": now(),
        "market_id": market["id"],
        "records": records,
        "rules": json.loads(RULES_PATH.read_text()),
        "limits": [
            "Unknown dismissal reasons and unresolved appeals require a published decision; current official unavailability remains authoritative",
            "Transfer histories and competition schedules determine service; missing domestic cup dates are not silently treated as league matches",
        ],
    }
    report["id"] = store.put("discipline_report", report)
    from .observations import evaluation_report

    evaluation_report(store, "discipline.json", {k: v for k, v in report.items() if k != "records"})
    return report


async def collect_discipline_news(store):
    import re

    import httpx
    from bs4 import BeautifulSoup

    from .ingestion import alias, load_market, resolve_name
    from .transport import PublicTransport

    url = "https://www.premierleague.com/en/news/4425344/which-players-are-suspended-or-close-to-a-ban-in-fantasy"
    async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
        text = await PublicTransport(store).get(
            client, url, "discipline_official", json_body=False, max_bytes=2_000_000
        )
    store.snapshot("discipline_official", url, {"text": text})
    players, market = load_market(store)
    teams = {t["id"]: t for t in market["teams"]}
    soup = BeautifulSoup(text, "html.parser")
    if not any("Suspended for" in table.get_text(" ", strip=True) for table in soup.find_all("table")):
        raise ValueError("Official suspension table schema changed; retained last verified publication")
    captured = now()
    matched = []
    rejected = []
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in row.find_all(("td", "th"))]
            if len(cells) != 4:
                continue
            position_id = {"GK": 1, "GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}.get(cells[2])
            if position_id is None:
                continue
            club_players = {
                p.id: p
                for p in players.values()
                if p.position == position_id
                and cells[1].casefold()
                in (teams[p.team]["short_name"].casefold(), teams[p.team]["name"].casefold())
            }
            try:
                found = [players[resolve_name(cells[0], position_id, club_players)]]
            except ValueError:
                # Published short given names (Matty/Matthew) can differ; a
                # unique surname within the same club and role resolves them.
                surname = alias(cells[0].split()[-1])
                found = [
                    p
                    for p in club_players.values()
                    if surname in (alias(p.name.split()[-1]), alias(p.web_name))
                    and alias(p.name).startswith(alias(cells[0])[0])
                ]
            if len(found) != 1:
                rejected.append({"name": cells[0], "reason": "No unique name/club registration"})
                continue
            player = found[0]
            position = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}[player.position]
            if cells[2] not in (position, "GKP" if position == "GK" else position):
                continue
            window = re.fullmatch(r"GW\s*(\d+)(?:\s*[-–]\s*(\d+))?", cells[3])
            if window:
                first, last = int(window[1]), int(window[2] or window[1])
                if not 1 <= first <= last <= 38:
                    continue
                record = {
                    "player_code": player.code,
                    "player_id": player.id,
                    "events": list(range(first, last + 1)),
                    "season": market["season"],
                    "known_at": captured,
                    "source_url": url,
                    "source_text": " | ".join(cells),
                }
                record["id"] = store.put("ban_window", record)
                matched.append(record["id"])
            elif cells[3].isdigit():
                store.put(
                    "official_cautions",
                    {
                        "player_code": player.code,
                        "season": market["season"],
                        "count": int(cells[3]),
                        "known_at": captured,
                        "source_url": url,
                    },
                )
    result = {
        "at": captured,
        "source_url": url,
        "ban_windows": matched,
        "rejected": rejected,
        "authoritative_roster": not rejected,
    }
    result["id"] = store.put("discipline_refresh", result)
    return result
