#!/usr/bin/env python3
"""
🧪 Test script for Maintenance Optimization Module
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from analytics.maintenance_optimizer import (
    MaintenanceOptimizer,
    RiskScore,
    MaintenanceActivity,
    InspectionPlan,
    OptimizationConstraint,
    DecisionPriority,
    MaintenanceType,
    InspectionMethod
)

def run_tests():
    """Run comprehensive maintenance optimization tests"""
    print("🧪 MAINTENANCE OPTIMIZATION TEST SUITE")
    print("="*50)
    
    # Initialize optimizer
    optimizer = MaintenanceOptimizer()
    
    test_results = []
    
    # Test 1: Risk Score Calculation
    print("\n1️⃣ TEST: Risk Score Calculation")
    print("-"*30)
    
    risk_score = optimizer.calculate_risk_score(
        asset_id="test_vessel_01",
        probability=8.5,
        consequence=7.2,
        financial_impact=750000,
        safety_impact=6.5,
        environmental_impact=5.0,
        production_impact=8.0
    )
    
    print(f"Total Risk: {risk_score.total_risk:.2f}")
    print(f"Weighted Risk: {risk_score.weighted_risk:.2f}")
    print(f"Category: {risk_score.get_category().value}")
    
    test_passed = risk_score.weighted_risk > 0
    test_results.append(("Risk Calculation", test_passed, True))
    
    # Test 2: RBI Analysis
    print("\n2️⃣ TEST: RBI Analysis")
    print("-"*30)
    
    component_data = {
        "age_years": 12,
        "fluid_type": "oil",
        "location": "offshore",
        "type": "pipeline",
        "replacement_cost_usd": 300000,
        "criticality": "high"
    }
    
    rbi_result = optimizer.perform_rbi_analysis(
        asset_id="test_vessel_02",
        component_data=component_data,
        inspection_history=[{"date": "2023-06-01", "findings": []}],
        failure_history=[],
        operational_conditions={
            "pressure_ratio": 0.8,
            "temperature_ratio": 0.7,
            "pressure_mpa": 10.0,
            "inventory_tonnes": 1000
        }
    )
    
    print(f"PoF: {rbi_result['risk_score']['probability_of_failure']:.2f}")
    print(f"CoF: {rbi_result['risk_score']['consequence_of_failure']:.2f}")
    print(f"Risk Category: {rbi_result['risk_score']['risk_category']}")
    print(f"Inspection Method: {rbi_result['inspection_plan']['method']}")
    
    test_passed = rbi_result['risk_score']['probability_of_failure'] > 0
    test_results.append(("RBI Analysis", test_passed, True))
    
    # Test 3: Activity Creation and Management
    print("\n3️⃣ TEST: Activity Creation")
    print("-"*30)
    
    activity = MaintenanceActivity(
        activity_id="TEST_ACT_001",
        asset_id="test_vessel_03",
        activity_type=MaintenanceType.PREVENTIVE,
        description="Comprehensive ultrasonic testing",
        priority=DecisionPriority.HIGH,
        estimated_duration_hours=48,
        required_resources=["UT scanner", "data logger", "safety equipment"],
        required_skills=["UT Level II", "data analysis", "offshore safety"],
        cost_breakdown={
            "labor": 12000,
            "materials": 3000,
            "downtime": 40000,
            "equipment": 5000
        },
        safety_requirements=["offshore safety cert", "gas testing", "permit to work"]
    )
    
    print(f"Activity: {activity.description}")
    print(f"Total Cost: ${activity.total_cost:,.2f}")
    print(f"Complexity: {activity.complexity_score:.2f}/10")
    
    test_passed = activity.total_cost > 0
    test_results.append(("Activity Creation", test_passed, True))
    
    # Test 4: Schedule Optimization
    print("\n4️⃣ TEST: Schedule Optimization")
    print("-"*30)
    
    # Create multiple activities
    activities = []
    for i in range(6):
        act = MaintenanceActivity(
            activity_id=f"TEST_ACT_{100+i}",
            asset_id=f"test_asset_{i+1}",
            activity_type=MaintenanceType.PREVENTIVE if i % 3 == 0 else MaintenanceType.CORRECTIVE,
            description=f"Test activity {i+1}",
            priority=DecisionPriority.HIGH if i < 3 else DecisionPriority.MEDIUM,
            estimated_duration_hours=12 * (i + 1),
            required_resources=[f"resource_{i}"],
            required_skills=[f"skill_{i}"],
            cost_breakdown={
                "labor": 2000 * (i + 1),
                "materials": 1000 * (i + 1),
                "downtime": 5000 * (i + 1)
            },
            safety_requirements=["basic"]
        )
        activities.append(act)
    
    # Define constraints
    constraints = OptimizationConstraint(
        max_budget=100000,
        max_downtime_hours=200,
        available_crew_size=15,
        available_skills={f"skill_{i}" for i in range(6)},
        time_window_start=datetime.now(),
        time_window_end=datetime.now() + timedelta(days=60),
        regulatory_requirements=["test_reg_1", "test_reg_2"],
        safety_constraints=["safety_1", "safety_2"]
    )
    
    # Optimize
    optimization_result = optimizer.optimize_maintenance_schedule(
        asset_list=[f"test_asset_{i+1}" for i in range(6)],
        candidate_activities=activities,
        constraints=constraints
    )
    
    print(f"Selected Activities: {len(optimization_result.optimal_schedule)}")
    print(f"Total Cost: ${optimization_result.total_cost:,.2f}")
    print(f"Risk Reduction: {optimization_result.risk_reduction:.1f}%")
    print(f"ROI: {optimization_result.roi:.1f}%")
    
    schedule_summary = optimization_result.get_schedule_summary()
    print(f"Schedule Summary: {schedule_summary}")
    
    test_passed = len(optimization_result.optimal_schedule) > 0
    test_results.append(("Schedule Optimization", test_passed, True))
    
    # Test 5: Decision Support
    print("\n5️⃣ TEST: Decision Support")
    print("-"*30)
    
    options = [
        {
            "description": "Full replacement with upgraded material",
            "estimated_cost": 120000,
            "estimated_duration": 21,
            "estimated_risk_reduction": 0.95,
            "resources": ["crane", "welding", "new materials"],
            "dependencies": ["shutdown", "engineering"]
        },
        {
            "description": "Major repair with reinforcement",
            "estimated_cost": 45000,
            "estimated_duration": 10,
            "estimated_risk_reduction": 0.8,
            "resources": ["welding", "reinforcement plates"],
            "dependencies": ["hot work permit"]
        },
        {
            "description": "Minor repair with increased monitoring",
            "estimated_cost": 15000,
            "estimated_duration": 3,
            "estimated_risk_reduction": 0.5,
            "resources": ["welding", "monitoring equipment"],
            "dependencies": []
        }
    ]
    
    decision = optimizer.generate_decision_support(
        asset_id="test_vessel_04",
        problem_statement="Severe corrosion in critical pipeline section",
        options=options,
        constraints={"max_cost": 80000, "max_duration": 14}
    )
    
    print(f"Decision ID: {decision.recommendation_id}")
    print(f"Recommended: {decision.recommended_option['description']}")
    print(f"Justification: {decision.justification}")
    print(f"Confidence: {decision.confidence:.2%}")
    
    test_passed = decision.confidence > 0
    test_results.append(("Decision Support", test_passed, True))
    
    # Test 6: Statistics and Reporting
    print("\n6️⃣ TEST: Statistics and Reporting")
    print("-"*30)
    
    stats = optimizer.get_maintenance_statistics()
    print(f"Total Activities: {stats['total_activities_scheduled']}")
    print(f"Assets with Plans: {stats['assets_with_plans']}")
    print(f"Estimated Savings: ${stats['cost_savings_estimated']:,.2f}")
    
    report = optimizer.export_maintenance_report(time_period_days=30)
    if report:
        print(f"System Report Generated")
        print(f"Overdue Inspections: {len(report['overdue_inspections'])}")
        print(f"Upcoming Inspections: {len(report['upcoming_inspections'])}")
        test_passed = True
    else:
        test_passed = False
        print("No report generated")
    
    test_results.append(("Reporting", test_passed, True))
    
    # Test 7: Data Persistence
    print("\n7️⃣ TEST: Data Persistence")
    print("-"*30)
    
    # Save data
    optimizer._save_data()
    
    # Create new optimizer instance to test loading
    optimizer2 = MaintenanceOptimizer()
    
    # Check if data was loaded
    loaded_plans = len(optimizer2.inspection_plans)
    loaded_decisions = len(optimizer2.decision_log)
    
    print(f"Loaded Inspection Plans: {loaded_plans}")
    print(f"Loaded Decisions: {loaded_decisions}")
    
    test_passed = loaded_plans > 0 or loaded_decisions > 0
    test_results.append(("Data Persistence", test_passed, True))
    
    # Summary
    print(f"\n📊 TEST SUMMARY")
    print("="*50)
    
    passed_count = sum(1 for _, passed, _ in test_results if passed)
    total = len(test_results)
    
    print(f"Tests Passed: {passed_count}/{total} ({passed_count/total*100:.0f}%)")
    
    print(f"\n📈 Detailed Results:")
    for test_name, passed, condition_met in test_results:
        status = "✅" if passed and condition_met else "❌"
        print(f"  {status} {test_name}")
    
    # Cleanup test files
    import os
    try:
        for file in ["maintenance_history.pkl", "inspection_plans.pkl", "decision_log.pkl"]:
            filepath = f"data/models/{file}"
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass
    except:
        pass
    
    print(f"\n✅ Maintenance Optimization tests complete!")
    
    return passed_count == total

if __name__ == "__main__":
    success = run_tests()
    if success:
        print(f"\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"\n❌ Some tests failed")
        sys.exit(1)