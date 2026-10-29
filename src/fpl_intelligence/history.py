"""Season-scoped historical archive import with stable official-code joins."""

from __future__ import annotations

import csv
import io

import httpx

from .storage import Store, now


def import_season(store: Store, season: str) -> dict:
    if len(season) != 7 or not season[:4].isdigit() or season[4] != "-" or not season[5:].isdigit():
        raise ValueError("Archive season format is YYYY-YY")
    base = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/"
    with httpx.Client(timeout=45, follow_redirects=False) as client:
        response = client.get(base + "gws/merged_gw.csv")
        response.raise_for_status()
        raw = list(csv.DictReader(io.StringIO(response.text)))
        registration = client.get(base + "players_raw.csv")
        registration.raise_for_status()
        identities = list(csv.DictReader(io.StringIO(registration.text)))
    codes = {int(r["id"]): int(r["code"]) for r in identities if r.get("code")}
    positions = {"GK": 1, "GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}
    rows = []
    numeric = {
        "element",
        "fixture",
        "minutes",
        "total_points",
        "starts",
        "round",
        "assists",
        "goals_scored",
        "saves",
        "yellow_cards",
        "red_cards",
        "bonus",
        "bps",
        "expected_goals",
        "expected_assists",
        "expected_goals_conceded",
        "clearances_blocks_interceptions",
        "recoveries",
        "tackles",
    }
    for record in raw:
        row = dict(record)
        for key in numeric:
            value = row.get(key)
            row[key] = None if value in (None, "") else float(value)
        if row["element"] is None or row["round"] is None or record.get("position") not in positions:
            continue
        row["element"] = int(row["element"])
        row["fixture"] = int(row["fixture"])
        row["round"] = int(row["round"])
        row["position"] = positions[record["position"]]
        row["code"] = codes.get(row["element"])
        row["was_home"] = record["was_home"].lower() in ("true", "1")
        row["season"] = season.replace("-", "/")
        rows.append(row)
    meta = store.snapshot(
        f"archive_{season}", base + "gws/merged_gw.csv", {"rows": rows, "identity_codes": codes}
    )
    report = {
        "season": season,
        "rows": len(rows),
        "known_at": now(),
        "hash": meta["hash"],
        "identity_mapping": "season-scoped element ID to official stable player code",
        "track": "retrospective_statistical_archive; not a pre-deadline news snapshot",
    }
    report["id"] = store.put("archive_import", report)
    return report


def archived_rows(store: Store, cutoff: str | None = None) -> list[dict]:
    with store.db() as db:
        sources = [
            r[0]
            for r in db.execute(
                "SELECT DISTINCT source FROM snapshots WHERE source LIKE 'archive_%' AND captured_at<=?",
                (cutoff or now(),),
            )
        ]
    return [row for source in sorted(sources) for row in store.latest(source, cutoff=cutoff)[0]["rows"]]
