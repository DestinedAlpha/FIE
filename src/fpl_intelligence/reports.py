"""Reports are rendered from independently validated numerical artifacts."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path

from .ingestion import load_market
from .storage import Store, atomic_json, now


def report_forecast(store: Store, plan: dict):
    try:
        return store.get("forecast", plan["forecast_id"])
    except ValueError:
        return store.get("scenario_forecast", plan["forecast_id"])


def calendar(store: Store) -> str:
    _, market = load_market(store)
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//FPL Intelligence//EN", "CALSCALE:GREGORIAN"]
    for event in market["events"]:
        deadline = datetime.fromisoformat(event["deadline_time"])
        lines.extend(
            [
                "BEGIN:VEVENT",
                f"UID:fpl-{market['season'].replace('/', '-')}-{event['id']}@localhost",
                f"DTSTAMP:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
                f"DTSTART:{deadline.strftime('%Y%m%dT%H%M%SZ')}",
                f"SUMMARY:FPL {event['name']} deadline",
                "END:VEVENT",
            ]
        )
    return "\r\n".join(lines + ["END:VCALENDAR", ""])


def render(store: Store, plan: dict) -> str:
    players, _ = load_market(store, plan.get("market_id"))

    def name(p):
        return players[p].web_name

    rows = plan["ledger"]
    first = rows[0]
    checklist = []
    for sale, buy in zip(first["sell"], first["buy"]):
        checklist.append(
            f"- Sell {name(sale['player_id'])} (£{sale['units'] / 10:.1f}m); "
            f"buy {name(buy['player_id'])} (£{buy['units'] / 10:.1f}m)."
        )
    if not checklist:
        checklist = ["- Roll the transfer; keep the squad."]
    checklist += [
        f"- Chip: {first['chip'] or 'none'}. Hit: {first['hit']} points. Bank £{first['bank_after'] / 10:.1f}m.",
        f"- Captain {name(first['lineup']['captain'])}; vice {name(first['lineup']['vice'])}.",
        "- Bench: " + ", ".join(name(p) for p in first["lineup"]["bench"]) + ".",
    ]
    lines = [
        "# FPL Intelligence · plan",
        "",
        f"Generated {plan.get('created_at', now())}.",
        "",
        f"State `{plan['state_id']}` · forecast `{plan['forecast_id']}`.",
        "",
        "Execution status: "
        + (
            "confirmed account input"
            if plan.get("execution_ready")
            else "dated/public scenario; reconcile current account and sell quotes before acting"
        ),
        "",
        "## Immediate action",
        "",
        *checklist,
        "",
        "## Multiweek ledger",
        "",
        "|GW|FT|Transfers|Hit|Bank|Captain / vice|Chip|Shared-world net xPts|",
        "|---|---:|---|---:|---:|---|---|---:|",
    ]
    for row in rows:
        changes = (
            ", ".join(
                f"{name(s['player_id'])} → {name(b['player_id'])}" for s, b in zip(row["sell"], row["buy"])
            )
            or "Roll"
        )
        lines.append(
            f"|{row['event']}|{row['free_transfers_before']}|{changes}|{row['hit']}|"
            f"£{row['bank_after'] / 10:.1f}m|{name(row['lineup']['captain'])} / {name(row['lineup']['vice'])}|"
            f"{row['chip'] or '—'}|{row.get('simulated_net_mean', 0):.2f}|"
        )
    lines += [
        "",
        "## Squad audit",
        "",
        "|Player|Position|Sell quote|xMins|P(start)|P(60+)|xPts|DefCon xPts|",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    from .domain import POSITIONS

    forecast = report_forecast(store, plan)
    state = store.get("state", plan["state_id"])
    projections = {r["player_id"]: r for r in forecast["players"] if r["event"] == first["event"]}
    for lot in state["lots"]:
        p = players[lot["player_id"]]
        r = projections[p.id]
        lines.append(
            f"|{p.name}|{POSITIONS[p.position]}|£{lot['sell_price'] / 10:.1f}m|{r['minutes_mean']:.1f}|"
            f"{r['p_start']:.0%}|{r['p_60']:.0%}|{r['mean']:.2f}|{r['components'].get('defcon', 0):.2f}|"
        )
    lines += [
        "",
        "## Captain comparison",
        "",
        "|Player|Expected points|P(60+)|10–90% points|",
        "|---|---:|---:|---:|",
    ]
    for candidate in sorted(
        (r for r in projections.values() if r["player_id"] in first["playing_squad"]),
        key=lambda r: r["mean"],
        reverse=True,
    )[:5]:
        lines.append(
            f"|{name(candidate['player_id'])}|{candidate['mean']:.2f}|{candidate['p_60']:.0%}|{candidate['point_quantiles'][0]:.1f}–{candidate['point_quantiles'][2]:.1f}|"
        )
    lines += [
        "",
        f"Production forecast: `{forecast['model_version']}`. The tested points model supplies these SAME outcome worlds. Historical improvement does not guarantee future points.",
    ]
    lines += [
        "",
        "## Execution audit",
        "",
        "```json",
        json.dumps(plan.get("solver", {}), indent=2),
        "```",
        "",
        "Shared-world outcome: " + json.dumps(plan.get("simulation", {})) + ".",
        "",
        "Future moves require refreshed availability, fixtures, prices and exact sell quotes. "
        "Unknown purchase lots remain unknown. Opponent policies are scenarios; no rank is guaranteed.",
    ]
    for condition in plan.get("contingencies", []):
        lines.append(f"- {condition['trigger']}: {condition['action']}")
    return "\n".join(lines) + "\n"


def export(store: Store, plan: dict, destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    atomic_json(destination / "plan.json", plan)
    (destination / "report.md").write_text(render(store, plan))
    (destination / "deadlines.ics").write_text(calendar(store), newline="")
    output = io.StringIO()
    fields = [
        "event",
        "free_transfers_before",
        "free_transfers_after",
        "hit",
        "bank_before",
        "settlement_bank_before",
        "settlement_sales",
        "settlement_purchases",
        "bank_after",
        "chip",
        "sell",
        "buy",
        "playing_squad",
        "permanent_squad",
        "lineup",
        "simulated_net_mean",
        "net_quantiles",
    ]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in plan["ledger"]:
        writer.writerow(
            {k: json.dumps(row[k]) if isinstance(row.get(k), (dict, list)) else row.get(k) for k in fields}
        )
    (destination / "ledger.csv").write_text(output.getvalue())
    forecast = report_forecast(store, plan)
    projections = io.StringIO()
    columns = [
        "player_id",
        "event",
        "mean",
        "sd",
        "minutes_mean",
        "p_start",
        "p_zero",
        "p_60",
        "point_quantiles",
        "components",
        "fixture_count",
        "action_coverage",
    ]
    writer = csv.DictWriter(projections, fieldnames=columns)
    writer.writeheader()
    for row in forecast["players"]:
        writer.writerow(
            {
                k: json.dumps(row.get(k)) if isinstance(row.get(k), (dict, list)) else row.get(k)
                for k in columns
            }
        )
    (destination / "market-forecasts.csv").write_text(projections.getvalue(), encoding="utf-8")
    alternatives = {}
    for kind in (
        "case_comparison",
        "chip_counterfactual",
        "scenario_evaluation",
        "decision_analysis",
        "timing_analysis",
        "price_bridge",
        "population_forecast",
    ):
        try:
            value = store.get(kind)
            if (
                value.get("forecast_id") == plan["forecast_id"]
                or value.get("base_plan_id") == plan.get("id")
                or value.get("plan_id") == plan.get("id")
            ):
                alternatives[kind] = value
        except ValueError:
            pass
    atomic_json(destination / "alternatives.json", alternatives)
    manifest = {
        "exported_at": now(),
        "plan_id": plan.get("id"),
        "files": [
            "plan.json",
            "report.md",
            "deadlines.ics",
            "ledger.csv",
            "market-forecasts.csv",
            "alternatives.json",
        ],
    }
    atomic_json(destination / "manifest.json", manifest)
    return manifest
