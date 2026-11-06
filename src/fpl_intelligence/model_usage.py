"""Describe actual serving connections, rather than equating collection with use."""

from __future__ import annotations

from .decision import NAMES


def model_input_usage(store):
    if not store.list("decision_model", 1):
        return []
    active = store.get("decision_model")
    fused = bool(active.get("fusion_model_id"))
    usage = [
        {
            "component": "Immediate points",
            "status": "Active",
            "feature_count": None if fused else len(NAMES),
            "explanation": (
                "The accepted multi-source tree uses its pinned feature contract."
                if fused
                else "Lagged xG/xA, points, minutes, actions, price and team/opponent results. "
                "The larger advanced-statistics candidate has not replaced this head."
            ),
        },
        {
            "component": "Immediate participation",
            "status": "Active",
            "feature_count": 57 if active.get("role_minutes_model_id") else len(NAMES),
            "explanation": "Dated starts, minutes, official availability and other-competition workload."
            if active.get("role_minutes_model_id")
            else "Calibrated historical minute bands and official availability.",
        },
    ]
    future_id = active.get("future_horizon_model_id")
    if future_id:
        future = store.get("future_horizon_model", future_id)
        usage.append(
            {
                "component": "Future fixtures",
                "status": "Active",
                "feature_count": len(future["features"]),
                "explanation": "Full archived FPL statistics including xG/xA, threat, creativity, ICT, starts, "
                "defensive actions, fixture strength and lead time. Separate accepted future expert.",
            }
        )
    usage += [
        {
            "component": "Understat shots and tactical roles",
            "status": "Active scoring" if fused else "Research / sensitivity",
            "feature_count": None,
            "explanation": "Open-play, set-piece and penalty channels retain their provider definitions and missingness. "
            "Collected shot data is not automatically an adjustment to the primary immediate forecast.",
        },
        {
            "component": "Top managers",
            "status": "Manager behavior / rival scenarios",
            "feature_count": None,
            "explanation": "Released picks and transfers inform manager-choice models and opponent scenarios. "
            "Ownership is not treated as extra player points.",
        },
        {
            "component": "Scout and community reports",
            "status": "Evidence / named sensitivities",
            "feature_count": None,
            "explanation": "Source-linked claims can change explicit scenarios. Reddit opinions are not numerical training labels.",
        },
    ]
    return usage
