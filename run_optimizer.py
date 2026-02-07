#!/usr/bin/env python3
"""Run optimizer with minimal input JSON or interactive prompt.

Usage:
  python run_optimizer.py --input examples/minimal_input.json

If no input provided the script will offer to use defaults.
"""

import argparse
import json
from datetime import datetime, timedelta

from analytics.maintenance_optimizer import MaintenanceOptimizer, MaintenanceType, DecisionPriority, MaintenanceActivity


def get_defaults():
    now = datetime.now()
    return {
        "asset_id": "VESSEL_001",
        "component": {"type":"pipeline","material":"carbon_steel","age_years":10},
        "risk": {"probability":5.0,"consequence":5.0},
        "impacts": {"financial_usd":50000,"safety":5.0,"environmental":2.0,"production":4.0},
        "operational": {"pressure_mpa":10.0,"temperature_c":60.0,"cycles_per_year":2000},
        "activities": [
            {"activity_id":"ACT_001","type":"preventive","duration_hours":24,"cost_usd":12000,"priority":"high"},
            {"activity_id":"ACT_002","type":"corrective","duration_hours":48,"cost_usd":25000,"priority":"high"}
        ],
        "constraints": {"max_budget_usd":150000,"max_downtime_hours":200,"available_crew":10,
                        "time_window_start": now.isoformat(),
                        "time_window_end": (now + timedelta(days=90)).isoformat()},
        "preferences": {"weight_risk_vs_cost":0.7,"prefer_min_downtime":True}
    }


def load_input(path):
    with open(path, 'r') as f:
        return json.load(f)


def normalize_activity(act):
    # Convert minimal activity structure to MaintenanceActivity-like input used by the optimizer
    # Map simple fields into the MaintenanceActivity dataclass
    activity_type = act.get("type", "preventive")
    # Convert to enums expected by MaintenanceActivity if necessary
    prio = act.get("priority", "medium")
    cost_breakdown = act.get("cost_breakdown") or {
        "labor": act.get("labor", 0),
        "materials": act.get("materials", 0),
        "downtime": act.get("downtime", act.get("cost_usd", 0))
    }

    return MaintenanceActivity(
        activity_id=act.get("activity_id"),
        asset_id=act.get("asset_id", "VESSEL_001"),
        activity_type=MaintenanceType.PREVENTIVE if activity_type == "preventive" else (
            MaintenanceType.CORRECTIVE if activity_type == "corrective" else MaintenanceType.PREVENTIVE
        ),
        description=act.get("description", ""),
        priority=DecisionPriority.HIGH if prio == "high" else (
            DecisionPriority.MEDIUM if prio == "medium" else DecisionPriority.LOW
        ),
        estimated_duration_hours=act.get("duration_hours", 24),
        required_resources=act.get("required_resources", []),
        required_skills=act.get("required_skills", []),
        cost_breakdown=cost_breakdown,
        safety_requirements=act.get("safety_requirements", [])
    )


def run_with_input(data):
    optimizer = MaintenanceOptimizer()

    # Risk score
    risk = data.get("risk", {})
    impacts = data.get("impacts", {})
    rs = optimizer.calculate_risk_score(
        asset_id=data.get("asset_id"),
        probability=float(risk.get("probability", 5.0)),
        consequence=float(risk.get("consequence", 5.0)),
        financial_impact=float(impacts.get("financial_usd", 50000)),
        safety_impact=float(impacts.get("safety", 5.0)),
        environmental_impact=float(impacts.get("environmental", 2.0)),
        production_impact=float(impacts.get("production", 4.0))
    )

    print(f"Risk weighted: {rs.weighted_risk:.2f} category: {rs.get_category().value}")

    # Create activities for optimizer
    activities = []
    for act in data.get("activities", []):
        a = normalize_activity(act)
        # The optimizer in repo expects MaintenanceActivity dataclass; we'll call optimizer.add_activity if exists
        try:
            # If optimizer exposes a method to create/add from dict, use it; else pass dicts to optimize_maintenance_schedule which accepts minimal activity objects in tests
            activities.append(a)
        except Exception:
            activities.append(a)

    # Build constraints object similar to tests
    cons = data.get("constraints", {})
    from analytics.maintenance_optimizer import OptimizationConstraint
    time_start = cons.get("time_window_start")
    time_end = cons.get("time_window_end")
    time_start_dt = datetime.fromisoformat(time_start) if time_start else datetime.now()
    time_end_dt = datetime.fromisoformat(time_end) if time_end else datetime.now() + timedelta(days=90)

    constraint_obj = OptimizationConstraint(
        max_budget=cons.get("max_budget_usd", cons.get("max_budget", 150000)),
        max_downtime_hours=cons.get("max_downtime_hours", 200),
        available_crew_size=cons.get("available_crew", 10),
        available_skills=set(cons.get("available_skills", [])),
        time_window_start=time_start_dt,
        time_window_end=time_end_dt,
        regulatory_requirements=cons.get("regulatory_requirements", []),
        safety_constraints=cons.get("safety_constraints", [])
    )

    # Call optimizer (tests use optimize_maintenance_schedule with candidate_activities as MaintenanceActivity objects or dicts)
    try:
        result = optimizer.optimize_maintenance_schedule(
            asset_list=[data.get("asset_id")],
            candidate_activities=activities,
            constraints=constraint_obj
        )

        print("Optimization results:")
        # Support result as object or dict
        if isinstance(result, dict):
            optimal = result.get('optimal_schedule', [])
            total_cost = result.get('total_cost', result.get('totalCost', None))
            risk_reduction = result.get('risk_reduction', result.get('riskReduction', None))
            summary = result.get('schedule_summary', None)
            print(f"  selected: {len(optimal)}")
            print(f"  total_cost: {total_cost}")
            print(f"  risk_reduction: {risk_reduction}")
            if summary:
                print(summary)
        else:
            print(f"  selected: {len(result.optimal_schedule)}")
            print(f"  total_cost: {result.total_cost}")
            print(f"  risk_reduction: {result.risk_reduction}")
            try:
                print(result.get_schedule_summary())
            except Exception:
                pass

    except Exception as e:
        import traceback
        print("Optimization failed:", e)
        print(traceback.format_exc())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help='Path to JSON input file (optional)')
    parser.add_argument('--interactive', action='store_true', help='Prompt for input values interactively')
    args = parser.parse_args()

    def prompt_value(prompt: str, default, cast=str):
        raw = input(f"{prompt} [{default}]: ")
        if raw.strip() == "":
            return default
        try:
            return cast(raw)
        except Exception:
            return default

    def prompt_activities():
        acts = []
        n = prompt_value("Number of activities to enter", 1, int)
        for i in range(int(n)):
            print(f"\nEntering activity {i+1}:")
            aid = prompt_value("  activity_id", f"ACT_{i+1}")
            atype = prompt_value("  type (preventive/corrective)", "preventive")
            dur = prompt_value("  duration_hours", 24, int)
            cost = prompt_value("  cost_usd", 12000, float)
            prio = prompt_value("  priority (high/medium/low)", "high")
            acts.append({
                "activity_id": aid,
                "type": atype,
                "duration_hours": int(dur),
                "cost_usd": float(cost),
                "priority": prio
            })
        return acts

    if args.interactive:
        print("Interactive mode — press Enter to accept defaults.")
        asset_id = prompt_value("Asset ID", "VESSEL_001")
        prob = prompt_value("Risk probability (0-10)", 5.0, float)
        cof = prompt_value("Risk consequence (0-10)", 5.0, float)
        fin = prompt_value("Financial impact (USD)", 50000, float)
        safety = prompt_value("Safety impact (0-10)", 5.0, float)
        env = prompt_value("Environmental impact (0-10)", 2.0, float)
        prod = prompt_value("Production impact (0-10)", 4.0, float)

        pressure = prompt_value("Operating pressure (MPa)", 10.0, float)
        temp = prompt_value("Operating temperature (C)", 60.0, float)
        cycles = prompt_value("Cycles per year", 2000, int)

        activities = prompt_activities()

        max_budget = prompt_value("Max budget (USD)", 150000, float)
        max_downtime = prompt_value("Max downtime hours", 200, float)
        crew = prompt_value("Available crew size", 10, int)

        prefs_weight = prompt_value("Weight risk vs cost (0-1)", 0.7, float)
        prefer_min = prompt_value("Prefer min downtime (True/False)", True, lambda v: v.lower() not in ("false", "0", "no"))

        data = {
            "asset_id": asset_id,
            "component": {"type": "pipeline", "material": "carbon_steel", "age_years": 10},
            "risk": {"probability": float(prob), "consequence": float(cof)},
            "impacts": {"financial_usd": float(fin), "safety": float(safety), "environmental": float(env), "production": float(prod)},
            "operational": {"pressure_mpa": float(pressure), "temperature_c": float(temp), "cycles_per_year": int(cycles)},
            "activities": activities,
            "constraints": {"max_budget_usd": float(max_budget), "max_downtime_hours": float(max_downtime), "available_crew": int(crew), "time_window_start": datetime.now().isoformat(), "time_window_end": (datetime.now() + timedelta(days=90)).isoformat()},
            "preferences": {"weight_risk_vs_cost": float(prefs_weight), "prefer_min_downtime": bool(prefer_min)}
        }

    else:
        if args.input:
            data = load_input(args.input)
        else:
            # No input; use defaults
            data = get_defaults()

    # Run main optimizer
    run_with_input(data)

    # Optional: run full analysis pipeline (fingerprinting, cross-asset, corrosion, FFS)
    if args.interactive or True:
        # allow user to specify full-analysis via interactive prompt when interactive; default: do it if interactive
        pass

    # If user wants full analysis, check flag (we add separate --full-analysis)
    # Note: handle missing modules/methods gracefully
    if hasattr(args, 'full_analysis') and args.full_analysis:
        print('\n🔬 Running full analysis pipeline...')
        try:
            from core.data_collector import DataCollector
            dc = DataCollector()
            sensor_readings = dc.simulate_sensor_data(data.get('asset_id'), duration_seconds=1)
            # Convert readings to simple dict: sensor_type -> last value
            sensor_dict = {}
            for r in sensor_readings:
                sensor_dict[r.sensor_type] = r.value
        except Exception:
            sensor_dict = {}

        asset_type = data.get('component', {}).get('type', 'pressure_vessel')

        # Failure fingerprinting
        try:
            from core.failure_fingerprinting import FailureFingerprinter
            ff = FailureFingerprinter()
            if sensor_dict:
                match = ff.detect_failure(sensor_dict, asset_type, asset_id=data.get('asset_id'))
                if match:
                    print('🧾 Failure fingerprint detected:', match.failure_type, f'confidence={match.confidence:.2f}')
                else:
                    print('🧾 No failure fingerprint detected')
            else:
                print('🧾 No sensor data available for fingerprinting')
        except Exception as e:
            print('🧾 Failure fingerprinting skipped:', e)

        # Cross-asset learning
        try:
            from core.cross_asset_learning import CrossAssetLearner
            cal = CrossAssetLearner()
            alerts = cal.monitor_asset(data.get('asset_id'), sensor_dict)
            print(f'🔗 Cross-asset alerts generated: {len(alerts)}')
        except Exception as e:
            print('🔗 Cross-asset analysis skipped:', e)

        # Corrosion checks
        try:
            from core.corrosion_detector import CorrosionDetector
            cd = CorrosionDetector()
            # Use operational temps, provide defaults for insulation/humidity
            op = data.get('operational', {})
            internal_temp = op.get('temperature_c', 60.0)
            external_temp = 25.0
            insulation = 'fair'
            humidity = 70.0
            cui = cd.detect_cui_risk(data.get('asset_id'), insulation, external_temp, internal_temp, humidity)
            print('🧪 CUI assessment:', cui.get('cui_risk_level'), f"score={cui.get('cui_risk_score'):.2f}")
        except Exception as e:
            print('🧪 Corrosion analysis skipped:', e)

        # FFS statistics
        try:
            from core.ffs_fit_analyzer import FFSFitAnalyzer
            fa = FFSFitAnalyzer()
            stats = fa.get_ffs_statistics()
            print('🔎 FFS statistics:', stats.get('total_assessments', 0))
            rep = fa.export_ffs_report(data.get('asset_id'))
            if rep:
                print('🔧 FFS report available for asset')
            else:
                print('🔧 No FFS report for asset')
        except Exception as e:
            print('🔎 FFS analysis skipped:', e)
