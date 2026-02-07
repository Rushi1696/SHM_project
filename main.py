# Add to main.py after FFS analysis
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

# Initialize maintenance optimizer
maintenance_optimizer = MaintenanceOptimizer()

# In your main loop, for assets with FFS results:
for asset_id, ffs_result in ffs_results.items():
    # Get risk score from FFS
    risk_score = maintenance_optimizer.calculate_risk_score(
        asset_id=asset_id,
        probability=calculate_probability_from_ffs(ffs_result),
        consequence=calculate_consequence_from_asset(asset_info),
        financial_impact=calculate_financial_impact(asset_info),
        safety_impact=calculate_safety_impact(asset_info),
        environmental_impact=calculate_environmental_impact(asset_info),
        production_impact=calculate_production_impact(asset_info)
    )
    
    print(f"📊 Risk Assessment for {asset_id}:")
    print(f"   Risk Score: {risk_score.weighted_risk:.2f}")
    print(f"   Category: {risk_score.get_category().value}")
    
    # Perform RBI analysis if needed
    if risk_score.get_category().value in ["high", "very_high", "critical"]:
        rbi_result = maintenance_optimizer.perform_rbi_analysis(
            asset_id=asset_id,
            component_data=get_component_data(asset_info),
            inspection_history=get_inspection_history(asset_id),
            failure_history=get_failure_history(asset_id),
            operational_conditions=get_operational_conditions(asset_id)
        )
        
        print(f"   🔍 RBI Analysis Complete:")
        print(f"   Inspection Method: {rbi_result['inspection_plan']['method']}")
        print(f"   Frequency: {rbi_result['inspection_plan']['frequency_months']} months")
        
        # Schedule maintenance if needed
        if needs_maintenance(ffs_result, risk_score):
            activity = MaintenanceActivity(
                activity_id=f"MAINT_{asset_id}_{datetime.now().strftime('%Y%m%d')}",
                asset_id=asset_id,
                activity_type=MaintenanceType.CORRECTIVE if ffs_result.remaining_strength_factor < 0.7 
                         else MaintenanceType.PREVENTIVE,
                description=f"Address {ffs_result.flaw_type.value} - RSF: {ffs_result.remaining_strength_factor:.3f}",
                priority=DecisionPriority.IMMEDIATE if ffs_result.remaining_strength_factor < 0.7
                        else DecisionPriority.HIGH,
                estimated_duration_hours=estimate_duration(ffs_result),
                required_resources=get_required_resources(ffs_result),
                required_skills=get_required_skills(ffs_result),
                cost_breakdown=estimate_costs(ffs_result, asset_info),
                safety_requirements=get_safety_requirements(ffs_result)
            )
            
            print(f"   🛠️ Maintenance Activity Created:")
            print(f"   Description: {activity.description}")
            print(f"   Priority: {activity.priority.value}")
            print(f"   Estimated Cost: ${activity.total_cost:,.2f}")
            print(f"   Duration: {activity.estimated_duration_hours} hours")

# Periodically optimize maintenance schedule
if datetime.now().day == 1:  # Monthly optimization
    print("\n📅 Monthly Maintenance Schedule Optimization")
    print("="*50)
    
    # Get all pending maintenance activities
    pending_activities = get_all_pending_activities(maintenance_optimizer)
    
    # Define optimization constraints
    constraints = OptimizationConstraint(
        max_budget=monthly_maintenance_budget,
        max_downtime_hours=available_downtime_hours,
        available_crew_size=available_crew,
        available_skills=available_skill_set,
        time_window_start=datetime.now(),
        time_window_end=datetime.now() + timedelta(days=30),
        regulatory_requirements=current_regulations,
        safety_constraints=safety_procedures
    )
    
    # Optimize schedule
    optimization_result = maintenance_optimizer.optimize_maintenance_schedule(
        asset_list=critical_assets_list,
        candidate_activities=pending_activities,
        constraints=constraints,
        optimization_horizon_days=90
    )
    
    print(f"📋 Optimization Results:")
    print(f"   Selected Activities: {len(optimization_result.optimal_schedule)}")
    print(f"   Total Cost: ${optimization_result.total_cost:,.2f}")
    print(f"   Risk Reduction: {optimization_result.risk_reduction:.1f}%")
    print(f"   ROI: {optimization_result.roi:.1f}%")
    
    # Generate schedule
    schedule_summary = optimization_result.get_schedule_summary()
    print(f"\n📅 Schedule Summary:")
    print(f"   By Priority: {json.dumps(schedule_summary['by_priority'], indent=2)}")
    print(f"   By Type: {json.dumps(schedule_summary['by_type'], indent=2)}")
    
    # For complex decisions, use decision support
    if has_complex_decision_needed():
        decision = maintenance_optimizer.generate_decision_support(
            asset_id="critical_asset_001",
            problem_statement="Multiple repair options with different cost/benefit profiles",
            options=get_decision_options(),
            constraints={"max_cost": 100000, "max_duration": 14}
        )
        
        print(f"\n🤔 Decision Support:")
        print(f"   Recommended: {decision.recommended_option['description']}")
        print(f"   Confidence: {decision.confidence:.1%}")
        print(f"   Expected Benefits: ${decision.expected_benefits['financial_usd']:,.2f}")

# Generate monthly report
if datetime.now().day == 28:  # End of month
    monthly_report = maintenance_optimizer.export_maintenance_report(time_period_days=30)
    
    if monthly_report:
        print(f"\n📊 Monthly Maintenance Report:")
        print(f"   Overdue Inspections: {len(monthly_report['overdue_inspections'])}")
        print(f"   Upcoming Inspections: {len(monthly_report['upcoming_inspections'])}")
        print(f"   Critical Decisions: {len(monthly_report['critical_decisions'])}")
        
        # Email or save report
        save_report(monthly_report, f"maintenance_report_{datetime.now().strftime('%Y%m')}.json")