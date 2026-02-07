import json
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

import streamlit as st

# Ensure project root is on sys.path so Streamlit can import local packages
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.maintenance_optimizer import (
    MaintenanceOptimizer,
    MaintenanceType,
    DecisionPriority,
    MaintenanceActivity,
    OptimizationConstraint,
)


def normalize_activity(act, default_asset_id="VESSEL_001"):
    activity_type = act.get("type", "preventive")
    prio = act.get("priority", "medium")
    cost_breakdown = act.get("cost_breakdown") or {
        "labor": act.get("labor", 0),
        "materials": act.get("materials", 0),
        "downtime": act.get("downtime", act.get("cost_usd", 0)),
    }

    return MaintenanceActivity(
        activity_id=act.get("activity_id"),
        asset_id=act.get("asset_id", default_asset_id),
        activity_type=(
            MaintenanceType.PREVENTIVE
            if activity_type == "preventive"
            else MaintenanceType.CORRECTIVE
        ),
        description=act.get("description", ""),
        priority=(
            DecisionPriority.HIGH
            if prio == "high"
            else DecisionPriority.MEDIUM
            if prio == "medium"
            else DecisionPriority.LOW
        ),
        estimated_duration_hours=act.get("duration_hours", 24),
        required_resources=act.get("required_resources", []),
        required_skills=act.get("required_skills", []),
        cost_breakdown=cost_breakdown,
        safety_requirements=act.get("safety_requirements", []),
    )


st.set_page_config(page_title="Vibe Guard — Maintenance Dashboard", layout="wide")

st.title("Vibe Guard — Maintenance Optimizer")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Input")
    uploaded = st.file_uploader("Upload input JSON", type=["json"])
    use_example = st.button("Load example input")

    if uploaded is not None:
        raw = uploaded.read().decode("utf-8")
        data = json.loads(raw)
    elif use_example:
        with open("examples/minimal_input.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = None

    if data:
        st.subheader("Loaded input JSON")
        st.json(data)
    else:
        st.info("No input loaded — load an example or upload a JSON file.")

with col2:
    st.header("Actions & Results")

    if data:
        weight = data.get("preferences", {}).get("weight_risk_vs_cost", 0.7)
        st.metric("Preference: weight risk vs cost", f"{weight}")

        if st.button("Run Optimization"):
            optimizer = MaintenanceOptimizer()

            activities = [normalize_activity(a, default_asset_id=data.get("asset_id")) for a in data.get("activities", [])]

            cons = data.get("constraints", {})
            time_start = cons.get("time_window_start")
            time_end = cons.get("time_window_end")
            time_start_dt = datetime.fromisoformat(time_start) if time_start else datetime.now()
            time_end_dt = datetime.fromisoformat(time_end) if time_end else (datetime.now() + timedelta(days=90))

            constraint_obj = OptimizationConstraint(
                max_budget=cons.get("max_budget_usd", cons.get("max_budget", 150000)),
                max_downtime_hours=cons.get("max_downtime_hours", 200),
                available_crew_size=cons.get("available_crew", 10),
                available_skills=set(cons.get("available_skills", [])),
                time_window_start=time_start_dt,
                time_window_end=time_end_dt,
                regulatory_requirements=cons.get("regulatory_requirements", []),
                safety_constraints=cons.get("safety_constraints", []),
            )

            try:
                result = optimizer.optimize_maintenance_schedule([
                    data.get("asset_id")
                ], activities, constraint_obj)

                st.success("Optimization completed")

                if isinstance(result, dict):
                    st.write("Optimal schedule:")
                    st.json(result)
                else:
                    st.write("Selected activities:" )
                    rows = []
                    for a in getattr(result, "optimal_schedule", []):
                        rows.append({
                            "activity_id": getattr(a, "activity_id", str(a)),
                            "type": getattr(a, "activity_type", ""),
                            "priority": getattr(a, "priority", ""),
                        })
                    st.table(rows)

                    st.write("Summary:")
                    try:
                        st.json(result.get_schedule_summary())
                    except Exception:
                        st.write({
                            "total_cost": getattr(result, "total_cost", None),
                            "risk_reduction": getattr(result, "risk_reduction", None),
                        })

            except Exception as e:
                st.error(f"Optimization failed: {e}")

    else:
        st.write("Load input to enable actions")


st.sidebar.header("Run")
st.sidebar.write("Use the main panel to load JSON and run the optimizer.")
