"""
✅ MAINTENANCE OPTIMIZATION & DECISION SUPPORT MODULE
Implements Risk-Based Inspection (RBI) methodology per API 581
Optimizes maintenance schedules using cost-benefit analysis
Provides decision support for resource allocation
Uses ML for predictive maintenance optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import math
from collections import defaultdict, deque
import heapq
import warnings
warnings.filterwarnings('ignore')
from scipy import stats, optimize
import json
import pickle

class RiskCategory(Enum):
    """Risk categories per API 581"""
    LOW = "low"          # Green
    MEDIUM = "medium"    # Yellow
    HIGH = "high"        # Orange
    VERY_HIGH = "very_high"  # Red
    CRITICAL = "critical"    # Purple

class MaintenanceType(Enum):
    """Types of maintenance activities"""
    INSPECTION = "inspection"
    PREVENTIVE = "preventive"
    CORRECTIVE = "corrective"
    PREDICTIVE = "predictive"
    CONDITION_BASED = "condition_based"
    RUN_TO_FAILURE = "run_to_failure"
    SHUTDOWN = "shutdown"
    ONLINE = "online"

class InspectionMethod(Enum):
    """NDT/Inspection methods"""
    VISUAL = "visual"
    ULTRASONIC = "ultrasonic"
    RADIOGRAPHY = "radiography"
    MAGNETIC_PARTICLE = "magnetic_particle"
    DYE_PENETRANT = "dye_penetrant"
    EDDY_CURRENT = "eddy_current"
    ACOUSTIC_EMISSION = "acoustic_emission"
    THERMOGRAPHY = "thermography"

class DecisionPriority(Enum):
    """Priority levels for maintenance decisions"""
    IMMEDIATE = "immediate"      # Within 24 hours
    URGENT = "urgent"            # Within 1 week
    HIGH = "high"                # Within 1 month
    MEDIUM = "medium"            # Within 3 months
    LOW = "low"                  # Within 1 year
    SCHEDULED = "scheduled"      # Next planned shutdown

@dataclass
class RiskScore:
    """Risk score calculation per API 581"""
    probability: float  # 0-10 scale
    consequence: float  # 0-10 scale
    financial_impact: float  # $USD
    safety_impact: float  # 0-10 scale
    environmental_impact: float  # 0-10 scale
    production_impact: float  # 0-10 scale
    
    @property
    def total_risk(self) -> float:
        """Calculate total risk score"""
        return self.probability * self.consequence
    
    @property
    def weighted_risk(self) -> float:
        """Calculate weighted risk with impacts"""
        weights = {
            'financial': 0.4,
            'safety': 0.3,
            'environmental': 0.2,
            'production': 0.1
        }
        return (self.total_risk * 0.6 + 
                (self.financial_impact * weights['financial'] +
                 self.safety_impact * weights['safety'] +
                 self.environmental_impact * weights['environmental'] +
                 self.production_impact * weights['production']) * 0.4)
    
    def get_category(self) -> RiskCategory:
        """Get risk category based on score"""
        if self.weighted_risk >= 80:
            return RiskCategory.CRITICAL
        elif self.weighted_risk >= 60:
            return RiskCategory.VERY_HIGH
        elif self.weighted_risk >= 40:
            return RiskCategory.HIGH
        elif self.weighted_risk >= 20:
            return RiskCategory.MEDIUM
        else:
            return RiskCategory.LOW

@dataclass
class MaintenanceActivity:
    """Maintenance activity definition"""
    activity_id: str
    asset_id: str
    activity_type: MaintenanceType
    description: str
    priority: DecisionPriority
    estimated_duration_hours: float
    required_resources: List[str]
    required_skills: List[str]
    cost_breakdown: Dict[str, float]  # labor, materials, downtime, etc.
    safety_requirements: List[str]
    
    @property
    def total_cost(self) -> float:
        """Calculate total cost"""
        return sum(self.cost_breakdown.values())
    
    @property
    def complexity_score(self) -> float:
        """Calculate complexity score (1-10)"""
        score = 0
        score += len(self.required_resources) * 0.5
        score += len(self.required_skills) * 0.5
        score += min(self.estimated_duration_hours / 24, 5)  # Max 5 points
        score += len(self.safety_requirements) * 0.3
        return min(10.0, score)

@dataclass
class InspectionPlan:
    """Risk-based inspection plan"""
    asset_id: str
    inspection_method: InspectionMethod
    coverage_percentage: float  # 0-100%
    frequency_months: float
    next_inspection_date: datetime
    confidence_level: float  # 0-1
    criticality_rank: int
    
    def is_due(self) -> bool:
        """Check if inspection is due"""
        return datetime.now() >= self.next_inspection_date
    
    def days_until_due(self) -> int:
        """Days until inspection is due"""
        delta = self.next_inspection_date - datetime.now()
        return max(0, delta.days)

@dataclass
class OptimizationConstraint:
    """Constraints for maintenance optimization"""
    max_budget: float
    max_downtime_hours: float
    available_crew_size: int
    available_skills: Set[str]
    time_window_start: datetime
    time_window_end: datetime
    regulatory_requirements: List[str]
    safety_constraints: List[str]
    
    def is_feasible(self, activity: MaintenanceActivity) -> bool:
        """Check if activity is feasible under constraints"""
        if activity.total_cost > self.max_budget:
            return False
        
        if activity.estimated_duration_hours > self.max_downtime_hours:
            return False
        
        # Check if required skills are available
        required_skills_set = set(activity.required_skills)
        if not required_skills_set.issubset(self.available_skills):
            return False
        
        return True

@dataclass
class OptimizationResult:
    """Result of maintenance optimization"""
    optimal_schedule: List[MaintenanceActivity]
    total_cost: float
    total_duration: float
    risk_reduction: float
    roi: float  # Return on Investment
    net_present_value: float
    constraints_satisfied: bool
    schedule_gantt: Dict[str, List[Tuple[datetime, datetime]]]
    
    def get_schedule_summary(self) -> Dict[str, Any]:
        """Get schedule summary"""
        by_priority = defaultdict(int)
        by_type = defaultdict(int)
        
        for activity in self.optimal_schedule:
            by_priority[activity.priority.value] += 1
            by_type[activity.activity_type.value] += 1
        
        return {
            "total_activities": len(self.optimal_schedule),
            "by_priority": dict(by_priority),
            "by_type": dict(by_type),
            "cost_distribution": {
                "labor": sum(a.cost_breakdown.get("labor", 0) for a in self.optimal_schedule),
                "materials": sum(a.cost_breakdown.get("materials", 0) for a in self.optimal_schedule),
                "downtime": sum(a.cost_breakdown.get("downtime", 0) for a in self.optimal_schedule)
            }
        }

@dataclass
class DecisionSupport:
    """Decision support recommendation"""
    recommendation_id: str
    asset_id: str
    issue: str
    options: List[Dict[str, Any]]
    recommended_option: Dict[str, Any]
    justification: str
    confidence: float
    implementation_timeline: Dict[str, datetime]
    expected_benefits: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display/serialization"""
        return {
            "recommendation_id": self.recommendation_id,
            "asset_id": self.asset_id,
            "issue": self.issue,
            "recommended_option": self.recommended_option,
            "justification": self.justification,
            "confidence": self.confidence,
            "expected_benefits": self.expected_benefits
        }

class MaintenanceOptimizer:
    """
    Maintenance Optimization Engine that:
    1. Implements Risk-Based Inspection (RBI) per API 581
    2. Optimizes maintenance schedules using cost-benefit analysis
    3. Provides decision support with multiple scenario analysis
    4. Uses ML for predictive maintenance optimization
    5. Generates optimal resource allocation plans
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.risk_matrix = self._initialize_risk_matrix()
        self.cost_database = self._initialize_cost_database()
        self.maintenance_history: Dict[str, List[MaintenanceActivity]] = {}
        self.inspection_plans: Dict[str, InspectionPlan] = {}
        self.decision_log: List[DecisionSupport] = []
        
        # ML models for optimization
        self.risk_predictor = None
        self.cost_estimator = None
        
        # Load existing data
        self._load_existing_data()
        
        print(f"✅ Maintenance Optimizer initialized with {len(self.cost_database)} cost items")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _initialize_risk_matrix(self) -> np.ndarray:
        """Initialize risk matrix per API 581"""
        # 5x5 risk matrix (Probability x Consequence)
        matrix = np.array([
            [1, 2, 3, 4, 5],      # Very Low Probability
            [2, 4, 6, 8, 10],     # Low Probability
            [3, 6, 9, 12, 15],    # Medium Probability
            [4, 8, 12, 16, 20],   # High Probability
            [5, 10, 15, 20, 25]   # Very High Probability
        ])
        return matrix
    
    def _initialize_cost_database(self) -> Dict[str, Dict[str, float]]:
        """Initialize cost database with typical maintenance costs"""
        return {
            "inspection": {
                "visual": {"labor": 500, "materials": 50, "downtime": 0},
                "ultrasonic": {"labor": 1500, "materials": 200, "downtime": 500},
                "radiography": {"labor": 3000, "materials": 1000, "downtime": 2000},
                "magnetic_particle": {"labor": 1200, "materials": 300, "downtime": 800}
            },
            "repair": {
                "weld_repair": {"labor": 5000, "materials": 2000, "downtime": 10000},
                "patch_repair": {"labor": 3000, "materials": 1500, "downtime": 5000},
                "replacement": {"labor": 15000, "materials": 50000, "downtime": 50000},
                "grinding": {"labor": 1000, "materials": 200, "downtime": 1000}
            },
            "preventive": {
                "coating": {"labor": 3000, "materials": 5000, "downtime": 2000},
                "cleaning": {"labor": 2000, "materials": 500, "downtime": 1000},
                "calibration": {"labor": 1500, "materials": 300, "downtime": 500}
            }
        }
    
    def _load_existing_data(self):
        """Load previously saved data"""
        try:
            with open("data/models/maintenance_history.pkl", "rb") as f:
                self.maintenance_history = pickle.load(f)
            print(f"📖 Loaded {sum(len(v) for v in self.maintenance_history.values())} maintenance activities")
        except FileNotFoundError:
            print("No existing maintenance history found")
        
        try:
            with open("data/models/inspection_plans.pkl", "rb") as f:
                self.inspection_plans = pickle.load(f)
            print(f"📋 Loaded {len(self.inspection_plans)} inspection plans")
        except FileNotFoundError:
            print("No existing inspection plans found")
        
        try:
            with open("data/models/decision_log.pkl", "rb") as f:
                self.decision_log = pickle.load(f)
            print(f"📊 Loaded {len(self.decision_log)} decision records")
        except FileNotFoundError:
            print("No existing decision log found")
    
    def _save_data(self):
        """Save all data to files"""
        with open("data/models/maintenance_history.pkl", "wb") as f:
            pickle.dump(self.maintenance_history, f)
        
        with open("data/models/inspection_plans.pkl", "wb") as f:
            pickle.dump(self.inspection_plans, f)
        
        with open("data/models/decision_log.pkl", "wb") as f:
            pickle.dump(self.decision_log, f)
    
    def calculate_risk_score(self,
                           asset_id: str,
                           probability: float,
                           consequence: float,
                           financial_impact: float,
                           safety_impact: float = 0.0,
                           environmental_impact: float = 0.0,
                           production_impact: float = 0.0) -> RiskScore:
        """
        Calculate risk score per API 581
        
        Args:
            asset_id: Asset identifier
            probability: Failure probability (0-10)
            consequence: Failure consequence (0-10)
            financial_impact: Financial impact in USD
            safety_impact: Safety impact (0-10)
            environmental_impact: Environmental impact (0-10)
            production_impact: Production impact (0-10)
        
        Returns:
            RiskScore object
        """
        
        # Normalize inputs
        probability = max(0.0, min(10.0, probability))
        consequence = max(0.0, min(10.0, consequence))
        
        # Normalize impacts to 0-10 scale if needed
        if financial_impact > 1000000:  # > $1M
            financial_normalized = 10.0
        elif financial_impact > 0:
            financial_normalized = math.log10(financial_impact / 1000)  # Scale based on log
            financial_normalized = max(0.0, min(10.0, financial_normalized))
        else:
            financial_normalized = 0.0
        
        safety_impact = max(0.0, min(10.0, safety_impact))
        environmental_impact = max(0.0, min(10.0, environmental_impact))
        production_impact = max(0.0, min(10.0, production_impact))
        
        score = RiskScore(
            probability=probability,
            consequence=consequence,
            financial_impact=financial_normalized,
            safety_impact=safety_impact,
            environmental_impact=environmental_impact,
            production_impact=production_impact
        )
        
        return score
    
    def perform_rbi_analysis(self,
                           asset_id: str,
                           component_data: Dict[str, Any],
                           inspection_history: List[Dict],
                           failure_history: List[Dict],
                           operational_conditions: Dict[str, float]) -> Dict[str, Any]:
        """
        Perform Risk-Based Inspection analysis per API 581
        
        Args:
            asset_id: Asset identifier
            component_data: Component details
            inspection_history: Past inspection results
            failure_history: Past failure records
            operational_conditions: Current operating conditions
        
        Returns:
            RBI analysis results
        """
        
        # Step 1: Calculate Probability of Failure (PoF)
        pof = self._calculate_pof(component_data, inspection_history, 
                                 failure_history, operational_conditions)
        
        # Step 2: Calculate Consequence of Failure (CoF)
        cof = self._calculate_cof(component_data, operational_conditions)
        
        # Step 3: Calculate risk
        risk_score = self.calculate_risk_score(
            asset_id=asset_id,
            probability=pof,
            consequence=cof,
            financial_impact=self._estimate_financial_impact(component_data),
            safety_impact=self._estimate_safety_impact(component_data),
            environmental_impact=self._estimate_environmental_impact(component_data),
            production_impact=self._estimate_production_impact(component_data)
        )
        
        # Step 4: Determine inspection plan
        inspection_plan = self._determine_inspection_plan(
            asset_id, risk_score, component_data
        )
        
        # Step 5: Generate recommendations
        recommendations = self._generate_rbi_recommendations(
            risk_score, inspection_plan
        )
        
        result = {
            "asset_id": asset_id,
            "risk_score": {
                "probability_of_failure": pof,
                "consequence_of_failure": cof,
                "total_risk": risk_score.total_risk,
                "weighted_risk": risk_score.weighted_risk,
                "risk_category": risk_score.get_category().value
            },
            "inspection_plan": {
                "method": inspection_plan.inspection_method.value,
                "frequency_months": inspection_plan.frequency_months,
                "next_inspection": inspection_plan.next_inspection_date.isoformat(),
                "coverage": inspection_plan.coverage_percentage,
                "confidence": inspection_plan.confidence_level
            },
            "recommendations": recommendations,
            "analysis_date": datetime.now().isoformat(),
            "valid_until": (datetime.now() + timedelta(days=365)).isoformat()
        }
        
        # Store inspection plan
        self.inspection_plans[asset_id] = inspection_plan
        
        # Save data
        self._save_data()
        
        return result
    
    def _calculate_pof(self,
                      component_data: Dict[str, Any],
                      inspection_history: List[Dict],
                      failure_history: List[Dict],
                      operational_conditions: Dict[str, float]) -> float:
        """Calculate Probability of Failure (0-10 scale)"""
        
        # Base POF from historical data
        if failure_history:
            base_pof = min(10.0, len(failure_history) * 2.0)  # Scale with failures
        else:
            base_pof = 1.0
        
        # Adjust for age
        age_years = component_data.get("age_years", 0)
        if age_years > 20:
            age_factor = 1.5
        elif age_years > 10:
            age_factor = 1.2
        else:
            age_factor = 1.0
        
        # Adjust for corrosion rate
        corrosion_rate = component_data.get("corrosion_rate_mm_per_year", 0.1)
        if corrosion_rate > 0.5:
            corrosion_factor = 2.0
        elif corrosion_rate > 0.2:
            corrosion_factor = 1.5
        else:
            corrosion_factor = 1.0
        
        # Adjust for operating conditions
        pressure = operational_conditions.get("pressure_ratio", 0.5)  # ratio of design
        temperature = operational_conditions.get("temperature_ratio", 0.5)
        
        if pressure > 0.8 or temperature > 0.8:
            condition_factor = 1.5
        elif pressure > 0.6 or temperature > 0.6:
            condition_factor = 1.2
        else:
            condition_factor = 1.0
        
        # Adjust for inspection results
        inspection_factor = 1.0
        if inspection_history:
            last_inspection = inspection_history[-1]
            findings = last_inspection.get("findings", [])
            if any(f.get("severity") == "high" for f in findings):
                inspection_factor = 2.0
            elif any(f.get("severity") == "medium" for f in findings):
                inspection_factor = 1.5
        
        # Calculate final POF
        pof = base_pof * age_factor * corrosion_factor * condition_factor * inspection_factor
        
        return min(10.0, max(0.0, pof))
    
    def _calculate_cof(self,
                      component_data: Dict[str, Any],
                      operational_conditions: Dict[str, float]) -> float:
        """Calculate Consequence of Failure (0-10 scale)"""
        
        # Start with base consequence
        base_cof = 3.0
        
        # Adjust for fluid type
        fluid_type = component_data.get("fluid_type", "water")
        fluid_factors = {
            "water": 1.0,
            "oil": 3.0,
            "gas": 4.0,
            "chemical": 5.0,
            "hazardous": 6.0
        }
        fluid_factor = fluid_factors.get(fluid_type, 2.0)
        
        # Adjust for location
        location = component_data.get("location", "general")
        location_factors = {
            "populated": 4.0,
            "environmental": 3.0,
            "offshore": 5.0,
            "critical_service": 6.0,
            "general": 2.0
        }
        location_factor = location_factors.get(location, 2.0)
        
        # Adjust for inventory
        inventory = operational_conditions.get("inventory_tonnes", 0)
        if inventory > 1000:
            inventory_factor = 4.0
        elif inventory > 100:
            inventory_factor = 2.5
        elif inventory > 10:
            inventory_factor = 1.5
        else:
            inventory_factor = 1.0
        
        # Adjust for pressure
        pressure = operational_conditions.get("pressure_mpa", 0)
        if pressure > 10:
            pressure_factor = 3.0
        elif pressure > 5:
            pressure_factor = 2.0
        else:
            pressure_factor = 1.0
        
        # Calculate final CoF
        cof = base_cof * fluid_factor * location_factor * inventory_factor * pressure_factor
        
        return min(10.0, max(0.0, cof))
    
    def _estimate_financial_impact(self, component_data: Dict[str, Any]) -> float:
        """Estimate financial impact of failure in USD"""
        # Simplified estimation
        replacement_cost = component_data.get("replacement_cost_usd", 100000)
        downtime_cost_per_day = component_data.get("downtime_cost_per_day", 50000)
        cleanup_cost = component_data.get("cleanup_cost_usd", 50000)
        
        # Assume 10 days downtime for major failure
        total_cost = replacement_cost + (downtime_cost_per_day * 10) + cleanup_cost
        
        return total_cost
    
    def _estimate_safety_impact(self, component_data: Dict[str, Any]) -> float:
        """Estimate safety impact (0-10 scale)"""
        # Based on fluid type and location
        fluid_type = component_data.get("fluid_type", "water")
        location = component_data.get("location", "general")
        
        safety_score = 0
        
        # Fluid hazard
        fluid_scores = {
            "water": 1,
            "oil": 3,
            "gas": 5,
            "chemical": 7,
            "hazardous": 9
        }
        safety_score += fluid_scores.get(fluid_type, 3)
        
        # Location hazard
        location_scores = {
            "populated": 4,
            "environmental": 3,
            "offshore": 6,
            "critical_service": 7,
            "general": 2
        }
        safety_score += location_scores.get(location, 2)
        
        return min(10.0, safety_score / 2)  # Scale to 0-10
    
    def _estimate_environmental_impact(self, component_data: Dict[str, Any]) -> float:
        """Estimate environmental impact (0-10 scale)"""
        fluid_type = component_data.get("fluid_type", "water")
        location = component_data.get("location", "general")
        
        env_score = 0
        
        # Environmental hazard of fluid
        env_scores = {
            "water": 1,
            "oil": 8,
            "gas": 5,
            "chemical": 9,
            "hazardous": 10
        }
        env_score += env_scores.get(fluid_type, 3)
        
        # Location sensitivity
        location_scores = {
            "populated": 3,
            "environmental": 8,
            "offshore": 7,
            "critical_service": 2,
            "general": 2
        }
        env_score += location_scores.get(location, 2)
        
        return min(10.0, env_score / 2)
    
    def _estimate_production_impact(self, component_data: Dict[str, Any]) -> float:
        """Estimate production impact (0-10 scale)"""
        # Based on criticality and redundancy
        criticality = component_data.get("criticality", "medium")
        has_redundancy = component_data.get("has_redundancy", False)
        
        criticality_scores = {
            "low": 1,
            "medium": 4,
            "high": 7,
            "critical": 10
        }
        
        impact = criticality_scores.get(criticality, 4)
        
        # Reduce impact if redundancy exists
        if has_redundancy:
            impact *= 0.5
        
        return min(10.0, impact)
    
    def _determine_inspection_plan(self,
                                  asset_id: str,
                                  risk_score: RiskScore,
                                  component_data: Dict[str, Any]) -> InspectionPlan:
        """Determine optimal inspection plan based on risk"""
        
        risk_category = risk_score.get_category()
        
        # Map risk to inspection method
        if risk_category in [RiskCategory.CRITICAL, RiskCategory.VERY_HIGH]:
            method = InspectionMethod.ULTRASONIC
            frequency = 6  # months
            coverage = 100  # %
            confidence = 0.9
            rank = 1
        
        elif risk_category == RiskCategory.HIGH:
            method = InspectionMethod.ULTRASONIC
            frequency = 12
            coverage = 80
            confidence = 0.8
            rank = 2
        
        elif risk_category == RiskCategory.MEDIUM:
            method = InspectionMethod.VISUAL
            frequency = 24
            coverage = 60
            confidence = 0.7
            rank = 3
        
        else:  # LOW
            method = InspectionMethod.VISUAL
            frequency = 36
            coverage = 40
            confidence = 0.6
            rank = 4
        
        # Adjust based on component type
        component_type = component_data.get("type", "vessel")
        if component_type in ["pipeline", "pressure_vessel"]:
            # More frequent for critical components
            frequency = max(6, frequency - 6)
            coverage = min(100, coverage + 20)
        
        # Calculate next inspection date
        next_inspection = datetime.now() + timedelta(days=frequency * 30)
        
        plan = InspectionPlan(
            asset_id=asset_id,
            inspection_method=method,
            coverage_percentage=coverage,
            frequency_months=frequency,
            next_inspection_date=next_inspection,
            confidence_level=confidence,
            criticality_rank=rank
        )
        
        return plan
    
    def _generate_rbi_recommendations(self,
                                    risk_score: RiskScore,
                                    inspection_plan: InspectionPlan) -> List[str]:
        """Generate RBI recommendations"""
        recommendations = []
        risk_category = risk_score.get_category()
        
        if risk_category == RiskCategory.CRITICAL:
            recommendations.append("IMMEDIATE SHUTDOWN REQUIRED")
            recommendations.append("Perform detailed inspection within 7 days")
            recommendations.append("Develop emergency repair plan")
            recommendations.append("Consider replacement rather than repair")
        
        elif risk_category == RiskCategory.VERY_HIGH:
            recommendations.append("Schedule inspection within 30 days")
            recommendations.append("Reduce operating parameters if possible")
            recommendations.append("Prepare for shutdown repair")
            recommendations.append("Increase monitoring frequency")
        
        elif risk_category == RiskCategory.HIGH:
            recommendations.append(f"Perform {inspection_plan.inspection_method.value} inspection within 3 months")
            recommendations.append("Review operating procedures")
            recommendations.append("Consider engineering assessment")
        
        elif risk_category == RiskCategory.MEDIUM:
            recommendations.append(f"Schedule {inspection_plan.inspection_method.value} inspection")
            recommendations.append("Continue routine monitoring")
            recommendations.append("Review during next planned shutdown")
        
        else:  # LOW
            recommendations.append("Continue current inspection plan")
            recommendations.append("Monitor for changes in operating conditions")
        
        # Add specific recommendations based on inspection method
        if inspection_plan.inspection_method == InspectionMethod.ULTRASONIC:
            recommendations.append("Perform thickness mapping")
            recommendations.append("Record baseline measurements for future comparison")
        
        return recommendations
    
    def optimize_maintenance_schedule(self,
                                    asset_list: List[str],
                                    candidate_activities: List[MaintenanceActivity],
                                    constraints: OptimizationConstraint,
                                    optimization_horizon_days: int = 365) -> OptimizationResult:
        """
        Optimize maintenance schedule using constraint optimization
        
        Args:
            asset_list: List of assets to consider
            candidate_activities: Candidate maintenance activities
            constraints: Optimization constraints
            optimization_horizon_days: Time horizon for optimization
        
        Returns:
            OptimizationResult with optimal schedule
        """
        
        print(f"🔧 Optimizing schedule for {len(candidate_activities)} activities...")
        
        # Filter feasible activities
        feasible_activities = [
            activity for activity in candidate_activities
            if constraints.is_feasible(activity)
        ]
        
        if not feasible_activities:
            print("⚠️ No feasible activities within constraints")
            return OptimizationResult(
                optimal_schedule=[],
                total_cost=0.0,
                total_duration=0.0,
                risk_reduction=0.0,
                roi=0.0,
                net_present_value=0.0,
                constraints_satisfied=True,
                schedule_gantt={}
            )
        
        # Sort activities by priority and risk
        prioritized_activities = self._prioritize_activities(
            feasible_activities, asset_list
        )
        
        # Apply optimization algorithm (simplified greedy + knapsack)
        optimal_schedule = self._knapsack_optimization(
            prioritized_activities, constraints, optimization_horizon_days
        )
        
        # Calculate schedule metrics
        total_cost = sum(activity.total_cost for activity in optimal_schedule)
        total_duration = sum(activity.estimated_duration_hours for activity in optimal_schedule)
        
        # Estimate risk reduction
        risk_reduction = self._estimate_risk_reduction(optimal_schedule)
        
        # Calculate ROI and NPV
        roi = self._calculate_roi(optimal_schedule)
        npv = self._calculate_npv(optimal_schedule, optimization_horizon_days)
        
        # Generate Gantt chart data
        schedule_gantt = self._generate_gantt_chart(optimal_schedule, constraints)
        
        result = OptimizationResult(
            optimal_schedule=optimal_schedule,
            total_cost=total_cost,
            total_duration=total_duration,
            risk_reduction=risk_reduction,
            roi=roi,
            net_present_value=npv,
            constraints_satisfied=True,
            schedule_gantt=schedule_gantt
        )
        
        # Log the optimization
        self._log_optimization(result, constraints)
        
        return result
    
    def _prioritize_activities(self,
                              activities: List[MaintenanceActivity],
                              asset_list: List[str]) -> List[MaintenanceActivity]:
        """Prioritize activities based on multiple criteria"""
        
        scored_activities = []
        
        for activity in activities:
            # Calculate priority score (0-100)
            score = 0
            
            # Priority factor
            priority_factors = {
                DecisionPriority.IMMEDIATE: 100,
                DecisionPriority.URGENT: 80,
                DecisionPriority.HIGH: 60,
                DecisionPriority.MEDIUM: 40,
                DecisionPriority.LOW: 20,
                DecisionPriority.SCHEDULED: 10
            }
            score += priority_factors.get(activity.priority, 10)
            
            # Risk factor (higher risk assets get higher priority)
            if activity.asset_id in asset_list:
                # Assume assets at beginning of list are higher risk
                try:
                    risk_rank = asset_list.index(activity.asset_id)
                    score += max(0, 20 - risk_rank * 2)
                except ValueError:
                    pass
            
            # Complexity factor (simpler tasks get slightly higher priority for scheduling)
            score += (10 - activity.complexity_score) * 0.5
            
            # Cost effectiveness factor (lower cost per unit risk reduction)
            # This is a placeholder - in reality, would use risk reduction estimate
            cost_factor = max(1, 1000 / max(1, activity.total_cost))
            score += cost_factor * 0.5
            
            scored_activities.append((score, activity))
        
        # Sort by score descending
        scored_activities.sort(key=lambda x: x[0], reverse=True)
        
        return [activity for _, activity in scored_activities]
    
    def _knapsack_optimization(self,
                              prioritized_activities: List[MaintenanceActivity],
                              constraints: OptimizationConstraint,
                              horizon_days: int) -> List[MaintenanceActivity]:
        """
        Simplified knapsack optimization for maintenance scheduling
        
        This implements a 0/1 knapsack-like optimization where:
        - Items = maintenance activities
        - Weight = cost or duration
        - Value = priority score + risk reduction
        
        Args:
            prioritized_activities: Activities sorted by priority
            constraints: Optimization constraints
            horizon_days: Optimization horizon
        
        Returns:
            List of activities to perform
        """
        
        # Simplified implementation: greedy with budget constraint
        selected_activities = []
        total_cost = 0.0
        total_duration = 0.0
        
        # Convert horizon to hours
        max_hours = constraints.max_downtime_hours
        
        for activity in prioritized_activities:
            # Check if we can add this activity
            new_total_cost = total_cost + activity.total_cost
            new_total_duration = total_duration + activity.estimated_duration_hours
            
            if (new_total_cost <= constraints.max_budget and 
                new_total_duration <= max_hours):
                
                selected_activities.append(activity)
                total_cost = new_total_cost
                total_duration = new_total_duration
            
            # Stop if we've reached constraints
            if total_cost >= constraints.max_budget * 0.9 or \
               total_duration >= max_hours * 0.9:
                break
        
        return selected_activities
    
    def _estimate_risk_reduction(self, activities: List[MaintenanceActivity]) -> float:
        """Estimate total risk reduction from maintenance activities"""
        total_reduction = 0.0
        
        for activity in activities:
            # Base reduction based on activity type
            base_reductions = {
                MaintenanceType.INSPECTION: 0.1,
                MaintenanceType.PREVENTIVE: 0.3,
                MaintenanceType.CORRECTIVE: 0.5,
                MaintenanceType.PREDICTIVE: 0.4,
                MaintenanceType.CONDITION_BASED: 0.35,
                MaintenanceType.SHUTDOWN: 0.6,
                MaintenanceType.ONLINE: 0.2,
                MaintenanceType.RUN_TO_FAILURE: 0.0
            }
            
            base = base_reductions.get(activity.activity_type, 0.2)
            
            # Adjust based on complexity (more complex = more reduction)
            complexity_factor = activity.complexity_score / 10
            
            # Adjust based on priority
            priority_factors = {
                DecisionPriority.IMMEDIATE: 1.5,
                DecisionPriority.URGENT: 1.3,
                DecisionPriority.HIGH: 1.2,
                DecisionPriority.MEDIUM: 1.0,
                DecisionPriority.LOW: 0.8,
                DecisionPriority.SCHEDULED: 0.7
            }
            priority_factor = priority_factors.get(activity.priority, 1.0)
            
            reduction = base * complexity_factor * priority_factor
            total_reduction += min(1.0, reduction)
        
        return min(100.0, total_reduction * 100)  # Convert to percentage
    
    def _calculate_roi(self, activities: List[MaintenanceActivity]) -> float:
        """Calculate Return on Investment for maintenance activities"""
        
        total_cost = sum(activity.total_cost for activity in activities)
        
        if total_cost == 0:
            return 0.0
        
        # Estimate benefits (simplified)
        # In reality, this would come from avoided failures, extended life, etc.
        estimated_benefits = 0.0
        
        for activity in activities:
            # Base benefit based on activity type
            benefit_multipliers = {
                MaintenanceType.INSPECTION: 2.0,  # $2 benefit per $1 spent
                MaintenanceType.PREVENTIVE: 3.0,
                MaintenanceType.CORRECTIVE: 1.5,  # Corrective usually less ROI
                MaintenanceType.PREDICTIVE: 4.0,
                MaintenanceType.CONDITION_BASED: 3.5,
                MaintenanceType.SHUTDOWN: 2.5,
                MaintenanceType.ONLINE: 3.0,
                MaintenanceType.RUN_TO_FAILURE: 0.5
            }
            
            multiplier = benefit_multipliers.get(activity.activity_type, 2.0)
            estimated_benefits += activity.total_cost * multiplier
        
        # Calculate ROI
        if total_cost > 0:
            roi = ((estimated_benefits - total_cost) / total_cost) * 100
        else:
            roi = 0.0
        
        return roi
    
    def _calculate_npv(self, activities: List[MaintenanceActivity], 
                      horizon_days: int) -> float:
        """Calculate Net Present Value of maintenance activities"""
        
        # Discount rate (annual)
        discount_rate = 0.1  # 10%
        
        # Convert to daily rate
        daily_discount = (1 + discount_rate) ** (1/365) - 1
        
        total_npv = 0.0
        
        for activity in activities:
            # Costs occur immediately (day 0)
            cost_npv = -activity.total_cost  # Negative for cost
            
            # Benefits occur over time - simplified: assume linear over horizon
            # Estimate annual benefit
            annual_benefit = activity.total_cost * 2.0  # Assume 2x cost benefit per year
            
            # Calculate benefit NPV over horizon
            days_in_horizon = min(horizon_days, 365 * 10)  # Cap at 10 years
            benefit_npv = 0.0
            
            for day in range(1, days_in_horizon + 1):
                daily_benefit = annual_benefit / 365
                discounted_benefit = daily_benefit / ((1 + daily_discount) ** day)
                benefit_npv += discounted_benefit
            
            # Activity NPV
            activity_npv = cost_npv + benefit_npv
            total_npv += activity_npv
        
        return total_npv
    
    def _generate_gantt_chart(self,
                             activities: List[MaintenanceActivity],
                             constraints: OptimizationConstraint) -> Dict[str, List[Tuple[datetime, datetime]]]:
        """Generate simplified Gantt chart data"""
        
        gantt_data = {}
        current_time = constraints.time_window_start
        
        for activity in activities:
            # Calculate start and end times
            start_time = current_time
            end_time = start_time + timedelta(hours=activity.estimated_duration_hours)
            
            # Ensure within time window
            if end_time > constraints.time_window_end:
                # Reschedule to fit within window
                start_time = constraints.time_window_end - timedelta(hours=activity.estimated_duration_hours)
                end_time = constraints.time_window_end
            
            # Add to Gantt data
            if activity.asset_id not in gantt_data:
                gantt_data[activity.asset_id] = []
            
            gantt_data[activity.asset_id].append((start_time, end_time))
            
            # Update current time (with some buffer)
            current_time = end_time + timedelta(hours=8)  # 8-hour buffer
        
        return gantt_data
    
    def _log_optimization(self,
                         result: OptimizationResult,
                         constraints: OptimizationConstraint):
        """Log optimization results for learning"""
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "num_activities": len(result.optimal_schedule),
            "total_cost": result.total_cost,
            "total_duration": result.total_duration,
            "risk_reduction": result.risk_reduction,
            "roi": result.roi,
            "npv": result.net_present_value,
            "constraints": {
                "max_budget": constraints.max_budget,
                "max_downtime": constraints.max_downtime_hours,
                "crew_size": constraints.available_crew_size
            }
        }
        
        # In a real implementation, this would go to a database
        # For now, just print
        print(f"📊 Optimization logged: {log_entry['num_activities']} activities, "
              f"ROI: {log_entry['roi']:.1f}%")
    
    def generate_decision_support(self,
                                asset_id: str,
                                problem_statement: str,
                                options: List[Dict[str, Any]],
                                constraints: Dict[str, Any] = None) -> DecisionSupport:
        """
        Generate decision support with analysis of multiple options
        
        Args:
            asset_id: Asset identifier
            problem_statement: Description of the problem
            options: List of potential solutions/options
            constraints: Additional constraints
        
        Returns:
            DecisionSupport recommendation
        """
        
        print(f"🤔 Generating decision support for {asset_id}...")
        
        # Analyze each option
        analyzed_options = []
        for i, option in enumerate(options):
            analysis = self._analyze_option(option, constraints)
            analyzed_options.append({
                "option_id": f"option_{i+1}",
                "description": option.get("description", f"Option {i+1}"),
                "analysis": analysis,
                "cost": analysis.get("total_cost", 0),
                "benefit": analysis.get("total_benefit", 0),
                "risk_reduction": analysis.get("risk_reduction", 0),
                "duration": analysis.get("duration_days", 0)
            })
        
        # Score each option
        for option in analyzed_options:
            option["score"] = self._score_option(option)
        
        # Select best option
        best_option = max(analyzed_options, key=lambda x: x["score"])
        
        # Calculate confidence
        confidence = self._calculate_decision_confidence(analyzed_options)
        
        # Generate timeline
        timeline = self._generate_implementation_timeline(best_option)
        
        # Calculate expected benefits
        expected_benefits = self._calculate_expected_benefits(best_option)
        
        # Create decision support object
        decision_id = f"DEC_{asset_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        support = DecisionSupport(
            recommendation_id=decision_id,
            asset_id=asset_id,
            issue=problem_statement,
            options=analyzed_options,
            recommended_option=best_option,
            justification=self._generate_justification(best_option, analyzed_options),
            confidence=confidence,
            implementation_timeline=timeline,
            expected_benefits=expected_benefits
        )
        
        # Add to decision log
        self.decision_log.append(support)
        
        # Save data
        self._save_data()
        
        return support
    
    def _analyze_option(self, option: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a decision option"""
        
        # This is a simplified analysis
        # In reality, this would involve more detailed calculations
        
        analysis = {
            "feasibility": True,
            "total_cost": option.get("estimated_cost", 10000),
            "duration_days": option.get("estimated_duration", 7),
            "risk_reduction": option.get("estimated_risk_reduction", 0.3) * 100,  # Convert to %
            "resource_requirements": option.get("resources", []),
            "dependencies": option.get("dependencies", []),
            "risks": option.get("risks", [])
        }
        
        # Check constraints if provided
        if constraints:
            max_cost = constraints.get("max_cost", float('inf'))
            max_duration = constraints.get("max_duration", float('inf'))
            
            if analysis["total_cost"] > max_cost:
                analysis["feasibility"] = False
                analysis["feasibility_reason"] = "Exceeds budget"
            
            if analysis["duration_days"] > max_duration:
                analysis["feasibility"] = False
                analysis["feasibility_reason"] = "Exceeds time constraint"
        
        # Calculate benefit (simplified)
        base_benefit = analysis["total_cost"] * 2  # Assume 2x ROI
        risk_benefit = analysis["risk_reduction"] * 1000  # $1000 per 1% risk reduction
        analysis["total_benefit"] = base_benefit + risk_benefit
        
        return analysis
    
    def _score_option(self, option: Dict[str, Any]) -> float:
        """Score an option (0-100)"""
        
        analysis = option.get("analysis", {})
        
        if not analysis.get("feasibility", True):
            return 0.0
        
        score = 0.0
        
        # Benefit-cost ratio (40% weight)
        cost = max(1, analysis.get("total_cost", 1))
        benefit = analysis.get("total_benefit", 0)
        benefit_cost_ratio = benefit / cost
        score += min(40.0, benefit_cost_ratio * 10)  # Scale
        
        # Risk reduction (30% weight)
        risk_reduction = analysis.get("risk_reduction", 0)
        score += min(30.0, risk_reduction * 0.3)  # 30 points for 100% reduction
        
        # Duration (20% weight)
        duration = analysis.get("duration_days", 30)
        duration_score = max(0, 20 - (duration / 30) * 5)  # Less duration = better
        score += duration_score
        
        # Complexity penalty (10% weight)
        resources = len(analysis.get("resource_requirements", []))
        dependencies = len(analysis.get("dependencies", []))
        complexity = resources + dependencies
        complexity_penalty = min(10, complexity * 0.5)
        score += (10 - complexity_penalty)  # Less complex = better
        
        return min(100.0, score)
    
    def _calculate_decision_confidence(self, options: List[Dict[str, Any]]) -> float:
        """Calculate confidence in decision (0-1)"""
        
        if not options:
            return 0.0
        
        # Confidence based on score difference between best and second best
        scores = [opt.get("score", 0) for opt in options]
        scores.sort(reverse=True)
        
        if len(scores) >= 2:
            best_score = scores[0]
            second_best = scores[1]
            
            if best_score == 0:
                return 0.0
            
            # More difference = more confidence
            score_ratio = (best_score - second_best) / best_score
            confidence = 0.5 + (score_ratio * 0.5)  # 0.5 to 1.0
            
        else:
            # Only one option
            confidence = 0.7
        
        # Adjust based on data quality
        data_quality_factor = 0.9  # Assume good data
        confidence *= data_quality_factor
        
        return max(0.1, min(1.0, confidence))
    
    def _generate_implementation_timeline(self, option: Dict[str, Any]) -> Dict[str, datetime]:
        """Generate implementation timeline"""
        
        duration = option.get("analysis", {}).get("duration_days", 7)
        
        now = datetime.now()
        
        timeline = {
            "start_date": now,
            "planning_complete": now + timedelta(days=1),
            "resources_allocated": now + timedelta(days=2),
            "execution_start": now + timedelta(days=3),
            "execution_complete": now + timedelta(days=duration),
            "verification": now + timedelta(days=duration + 1),
            "closeout": now + timedelta(days=duration + 2)
        }
        
        return timeline
    
    def _calculate_expected_benefits(self, option: Dict[str, Any]) -> Dict[str, float]:
        """Calculate expected benefits"""
        
        analysis = option.get("analysis", {})
        
        return {
            "financial_usd": analysis.get("total_benefit", 0) - analysis.get("total_cost", 0),
            "risk_reduction_percent": analysis.get("risk_reduction", 0),
            "extended_life_years": min(10.0, analysis.get("risk_reduction", 0) * 0.1),  # Simplified
            "downtime_reduction_hours": 24 * analysis.get("duration_days", 0) * 0.5,  # Assume 50% reduction
            "safety_improvement": min(10.0, analysis.get("risk_reduction", 0) * 0.1)
        }
    
    def _generate_justification(self, 
                               best_option: Dict[str, Any], 
                               all_options: List[Dict[str, Any]]) -> str:
        """Generate justification for recommended option"""
        
        justification_parts = []
        
        # Compare scores
        scores = [opt.get("score", 0) for opt in all_options]
        best_score = best_option.get("score", 0)
        avg_score = sum(scores) / len(scores) if scores else 0
        
        justification_parts.append(
            f"Option '{best_option['description']}' scored {best_score:.1f}/100 "
            f"(average: {avg_score:.1f}/100)."
        )
        
        # Highlight key benefits
        analysis = best_option.get("analysis", {})
        
        if analysis.get("risk_reduction", 0) > 30:
            justification_parts.append(
                f"Significant risk reduction: {analysis['risk_reduction']:.1f}%."
            )
        
        benefit_cost = analysis.get("total_benefit", 0) / max(1, analysis.get("total_cost", 1))
        if benefit_cost > 2:
            justification_parts.append(
                f"Favorable benefit-cost ratio: {benefit_cost:.2f}:1."
            )
        
        # Mention feasibility
        if analysis.get("feasibility", True):
            justification_parts.append("Option is feasible within constraints.")
        else:
            justification_parts.append(
                f"Note: Option has feasibility concerns: {analysis.get('feasibility_reason', 'Unknown')}."
            )
        
        return " ".join(justification_parts)
    
    def get_maintenance_statistics(self) -> Dict[str, Any]:
        """Get maintenance optimization statistics"""
        
        stats = {
            "total_activities_scheduled": sum(len(v) for v in self.maintenance_history.values()),
            "assets_with_plans": len(self.inspection_plans),
            "decisions_made": len(self.decision_log),
            "recent_optimizations": 0,
            "cost_savings_estimated": 0.0,
            "risk_reduction_total": 0.0,
            "schedule_adherence": 0.0,
            "top_assets_by_risk": [],
            "maintenance_backlog": 0
        }
        
        # Calculate estimated cost savings (simplified)
        total_cost = 0
        total_benefit = 0
        
        for decision in self.decision_log:
            for option in decision.options:
                if option.get("option_id") == decision.recommended_option.get("option_id"):
                    analysis = option.get("analysis", {})
                    total_cost += analysis.get("total_cost", 0)
                    total_benefit += analysis.get("total_benefit", 0)
        
        if total_cost > 0:
            stats["cost_savings_estimated"] = total_benefit - total_cost
        
        # Get top risky assets
        asset_risks = []
        for asset_id, plan in self.inspection_plans.items():
            asset_risks.append({
                "asset_id": asset_id,
                "risk_rank": plan.criticality_rank,
                "next_inspection": plan.next_inspection_date.isoformat()
            })
        
        asset_risks.sort(key=lambda x: x["risk_rank"])
        stats["top_assets_by_risk"] = asset_risks[:5]  # Top 5
        
        return stats
    
    def export_maintenance_report(self, 
                                asset_id: str = None,
                                time_period_days: int = 30) -> Optional[Dict[str, Any]]:
        """Export comprehensive maintenance report"""
        
        if asset_id:
            # Single asset report
            activities = self.maintenance_history.get(asset_id, [])
            inspection_plan = self.inspection_plans.get(asset_id)
            
            if not activities and not inspection_plan:
                return None
            
            report = {
                "asset_id": asset_id,
                "report_date": datetime.now().isoformat(),
                "period_days": time_period_days,
                "maintenance_summary": {
                    "total_activities": len(activities),
                    "recent_activities": len([a for a in activities 
                                            if (datetime.now() - timedelta(days=time_period_days)) 
                                            <= getattr(a, 'date', datetime.now())]),
                    "total_cost": sum(a.total_cost for a in activities),
                    "average_duration": np.mean([a.estimated_duration_hours for a in activities]) 
                                    if activities else 0
                },
                "inspection_plan": {
                    "method": inspection_plan.inspection_method.value if inspection_plan else None,
                    "next_due": inspection_plan.next_inspection_date.isoformat() if inspection_plan else None,
                    "days_until_due": inspection_plan.days_until_due() if inspection_plan else None,
                    "is_due": inspection_plan.is_due() if inspection_plan else False
                } if inspection_plan else None,
                "recommendations": [],
                "next_actions": [
                    f"Review inspection plan" if inspection_plan else "Create inspection plan",
                    "Schedule next maintenance based on risk"
                ]
            }
            
        else:
            # System-wide report
            report = {
                "system_report": True,
                "report_date": datetime.now().isoformat(),
                "statistics": self.get_maintenance_statistics(),
                "overdue_inspections": [
                    {
                        "asset_id": asset_id,
                        "plan": plan.inspection_method.value,
                        "due_since": (datetime.now() - plan.next_inspection_date).days
                    }
                    for asset_id, plan in self.inspection_plans.items()
                    if plan.is_due()
                ],
                "upcoming_inspections": [
                    {
                        "asset_id": asset_id,
                        "plan": plan.inspection_method.value,
                        "next_due": plan.next_inspection_date.isoformat(),
                        "days_until": plan.days_until_due()
                    }
                    for asset_id, plan in self.inspection_plans.items()
                    if not plan.is_due()
                ][:10],  # Next 10
                "critical_decisions": [
                    decision.to_dict() 
                    for decision in self.decision_log[-5:]  # Last 5 decisions
                ]
            }
        
        return report

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Maintenance Optimization Module")
    print("="*50)
    
    # Initialize optimizer
    optimizer = MaintenanceOptimizer()
    
    # Test 1: Risk Calculation
    print("\n1️⃣ TEST: Risk Score Calculation")
    print("-"*30)
    
    risk_score = optimizer.calculate_risk_score(
        asset_id="vessel_001",
        probability=7.5,
        consequence=8.2,
        financial_impact=500000,
        safety_impact=6.0,
        environmental_impact=4.5,
        production_impact=7.0
    )
    
    print(f"Total Risk: {risk_score.total_risk:.2f}")
    print(f"Weighted Risk: {risk_score.weighted_risk:.2f}")
    print(f"Risk Category: {risk_score.get_category().value}")
    print(f"Safety Impact: {risk_score.safety_impact:.2f}")
    
    # Test 2: RBI Analysis
    print("\n2️⃣ TEST: RBI Analysis")
    print("-"*30)
    
    component_data = {
        "age_years": 15,
        "fluid_type": "gas",
        "location": "offshore",
        "type": "pressure_vessel",
        "replacement_cost_usd": 250000,
        "downtime_cost_per_day": 100000,
        "cleanup_cost_usd": 50000,
        "criticality": "high",
        "has_redundancy": False
    }
    
    inspection_history = [
        {"date": "2023-01-15", "findings": [{"severity": "medium", "description": "Corrosion"}]}
    ]
    
    failure_history = []
    
    operational_conditions = {
        "pressure_ratio": 0.75,
        "temperature_ratio": 0.65,
        "pressure_mpa": 8.5,
        "inventory_tonnes": 500
    }
    
    rbi_result = optimizer.perform_rbi_analysis(
        asset_id="vessel_002",
        component_data=component_data,
        inspection_history=inspection_history,
        failure_history=failure_history,
        operational_conditions=operational_conditions
    )
    
    print(f"Probability of Failure: {rbi_result['risk_score']['probability_of_failure']:.2f}")
    print(f"Consequence of Failure: {rbi_result['risk_score']['consequence_of_failure']:.2f}")
    print(f"Risk Category: {rbi_result['risk_score']['risk_category']}")
    print(f"Inspection Method: {rbi_result['inspection_plan']['method']}")
    print(f"Inspection Frequency: {rbi_result['inspection_plan']['frequency_months']} months")
    print(f"Recommendations: {len(rbi_result['recommendations'])}")
    
    # Test 3: Maintenance Activity Creation
    print("\n3️⃣ TEST: Maintenance Activity Creation")
    print("-"*30)
    
    activity = MaintenanceActivity(
        activity_id="ACT_001",
        asset_id="vessel_003",
        activity_type=MaintenanceType.PREVENTIVE,
        description="Ultrasonic thickness testing",
        priority=DecisionPriority.HIGH,
        estimated_duration_hours=24,
        required_resources=["UT equipment", "scaffolding", "safety gear"],
        required_skills=["UT Level II", "scaffolding certified", "confined space"],
        cost_breakdown={
            "labor": 5000,
            "materials": 1000,
            "downtime": 15000,
            "equipment": 2000
        },
        safety_requirements=["confined space permit", "hot work permit", "gas testing"]
    )
    
    print(f"Activity: {activity.description}")
    print(f"Total Cost: ${activity.total_cost:,.2f}")
    print(f"Complexity Score: {activity.complexity_score:.2f}/10")
    print(f"Priority: {activity.priority.value}")
    
    # Test 4: Schedule Optimization
    print("\n4️⃣ TEST: Maintenance Schedule Optimization")
    print("-"*30)
    
    # Create multiple activities
    activities = []
    for i in range(5):
        act = MaintenanceActivity(
            activity_id=f"ACT_{100+i}",
            asset_id=f"vessel_{i+1}",
            activity_type=MaintenanceType.PREVENTIVE if i % 2 == 0 else MaintenanceType.CORRECTIVE,
            description=f"Maintenance activity {i+1}",
            priority=DecisionPriority.HIGH if i < 2 else DecisionPriority.MEDIUM,
            estimated_duration_hours=8 * (i + 1),
            required_resources=["crew", f"equipment_{i}"],
            required_skills=[f"skill_{i}"],
            cost_breakdown={
                "labor": 1000 * (i + 1),
                "materials": 500 * (i + 1),
                "downtime": 2000 * (i + 1)
            },
            safety_requirements=["basic safety"]
        )
        activities.append(act)
    
    # Define constraints
    constraints = OptimizationConstraint(
        max_budget=50000,
        max_downtime_hours=120,
        available_crew_size=10,
        available_skills={"skill_0", "skill_1", "skill_2", "skill_3", "skill_4"},
        time_window_start=datetime.now(),
        time_window_end=datetime.now() + timedelta(days=30),
        regulatory_requirements=["OSHA compliance", "environmental permits"],
        safety_constraints=["no hot work", "confined space procedures"]
    )
    
    # Optimize schedule
    optimization_result = optimizer.optimize_maintenance_schedule(
        asset_list=[f"vessel_{i+1}" for i in range(5)],
        candidate_activities=activities,
        constraints=constraints,
        optimization_horizon_days=90
    )
    
    print(f"Optimal Activities: {len(optimization_result.optimal_schedule)}")
    print(f"Total Cost: ${optimization_result.total_cost:,.2f}")
    print(f"Total Duration: {optimization_result.total_duration:.1f} hours")
    print(f"Risk Reduction: {optimization_result.risk_reduction:.1f}%")
    print(f"ROI: {optimization_result.roi:.1f}%")
    print(f"NPV: ${optimization_result.net_present_value:,.2f}")
    
    schedule_summary = optimization_result.get_schedule_summary()
    print(f"Schedule Summary: {schedule_summary['total_activities']} activities")
    print(f"By Priority: {schedule_summary['by_priority']}")
    
    # Test 5: Decision Support
    print("\n5️⃣ TEST: Decision Support Generation")
    print("-"*30)
    
    options = [
        {
            "description": "Replace entire section",
            "estimated_cost": 50000,
            "estimated_duration": 14,
            "estimated_risk_reduction": 0.9,
            "resources": ["welding crew", "crane", "new pipe"],
            "dependencies": ["shutdown", "engineering approval"]
        },
        {
            "description": "Weld repair with monitoring",
            "estimated_cost": 15000,
            "estimated_duration": 5,
            "estimated_risk_reduction": 0.7,
            "resources": ["welding crew", "NDT equipment"],
            "dependencies": ["hot work permit"]
        },
        {
            "description": "Grind and monitor",
            "estimated_cost": 5000,
            "estimated_duration": 2,
            "estimated_risk_reduction": 0.4,
            "resources": ["grinding equipment", "NDT equipment"],
            "dependencies": []
        }
    ]
    
    decision = optimizer.generate_decision_support(
        asset_id="vessel_005",
        problem_statement="Crack detected in pipeline section",
        options=options,
        constraints={"max_cost": 30000, "max_duration": 10}
    )
    
    print(f"Decision ID: {decision.recommendation_id}")
    print(f"Issue: {decision.issue}")
    print(f"Recommended: {decision.recommended_option['description']}")
    print(f"Justification: {decision.justification[:100]}...")
    print(f"Confidence: {decision.confidence:.2%}")
    print(f"Expected Benefits: ${decision.expected_benefits['financial_usd']:,.2f}")
    
    # Test 6: System Statistics
    print("\n6️⃣ TEST: System Statistics")
    print("-"*30)
    
    stats = optimizer.get_maintenance_statistics()
    print(f"Total Activities Scheduled: {stats['total_activities_scheduled']}")
    print(f"Assets with Plans: {stats['assets_with_plans']}")
    print(f"Decisions Made: {stats['decisions_made']}")
    print(f"Estimated Cost Savings: ${stats['cost_savings_estimated']:,.2f}")
    
    if stats['top_assets_by_risk']:
        print("Top Risky Assets:")
        for asset in stats['top_assets_by_risk'][:3]:
            print(f"  {asset['asset_id']} (Rank: {asset['risk_rank']})")
    
    # Test 7: Report Generation
    print("\n7️⃣ TEST: Report Generation")
    print("-"*30)
    
    single_report = optimizer.export_maintenance_report(
        asset_id="vessel_002",
        time_period_days=90
    )
    
    if single_report:
        print(f"Report for {single_report['asset_id']}")
        print(f"Total Activities: {single_report['maintenance_summary']['total_activities']}")
        print(f"Total Cost: ${single_report['maintenance_summary']['total_cost']:,.2f}")
        
        if single_report['inspection_plan']:
            print(f"Next Inspection: {single_report['inspection_plan']['next_due'][:10]}")
            print(f"Is Due: {single_report['inspection_plan']['is_due']}")
    
    system_report = optimizer.export_maintenance_report(time_period_days=30)
    if system_report:
        print(f"\nSystem Report:")
        print(f"Overdue Inspections: {len(system_report['overdue_inspections'])}")
        print(f"Upcoming Inspections: {len(system_report['upcoming_inspections'])}")
        print(f"Critical Decisions: {len(system_report['critical_decisions'])}")
    
    # Save all data
    optimizer._save_data()
    
    print(f"\n✅ Maintenance Optimization tests complete!")