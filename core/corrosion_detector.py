"""
🛡️ CORROSION DETECTION & MONITORING MODULE
Monitors corrosion rates, predicts remaining life, and detects corrosion-related failures
Implements industry standards (API, NACE, ASME) for corrosion assessment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import pickle
from scipy import stats, interpolate
import warnings
warnings.filterwarnings('ignore')

class CorrosionType(Enum):
    """Types of corrosion"""
    UNIFORM = "uniform"
    PITTING = "pitting"
    CREVICE = "crevice"
    GALVANIC = "galvanic"
    STRESS_CORROSION = "stress_corrosion"
    CORROSION_FATIGUE = "corrosion_fatigue"
    EROSION_CORROSION = "erosion_corrosion"
    CUI = "corrosion_under_insulation"
    MIC = "microbiologically_influenced"

class MaterialType(Enum):
    """Common industrial materials"""
    CARBON_STEEL = "carbon_steel"
    STAINLESS_316 = "stainless_steel_316"
    STAINLESS_304 = "stainless_steel_304"
    DUPLEX = "duplex_stainless"
    INCONEL = "inconel"
    ALUMINUM = "aluminum"
    BRASS = "brass"
    COPPER = "copper"

class EnvironmentType(Enum):
    """Corrosive environments"""
    SEAWATER = "seawater"
    FRESHWATER = "freshwater"
    BRACKISH = "brackish_water"
    ATMOSPHERIC = "atmospheric"
    SOIL = "soil"
    HIGH_TEMP = "high_temperature"
    ACIDIC = "acidic"
    ALKALINE = "alkaline"
    CHLORIDE = "chloride_containing"

@dataclass
class MaterialProperties:
    """Material properties for corrosion calculations"""
    material: MaterialType
    density: float  # g/cm³
    yield_strength: float  # MPa
    tensile_strength: float  # MPa
    corrosion_resistance: float  # 0-1 scale
    typical_corrosion_rate: float  # mm/year in mild environment
    galvanic_series_position: float  # -1 to +1
    
    # Industry standard corrosion allowances (mm/year)
    def get_corrosion_allowance(self, environment: EnvironmentType) -> float:
        """Get corrosion allowance based on material and environment"""
        allowances = {
            (MaterialType.CARBON_STEEL, EnvironmentType.SEAWATER): 0.5,
            (MaterialType.CARBON_STEEL, EnvironmentType.ATMOSPHERIC): 0.1,
            (MaterialType.CARBON_STEEL, EnvironmentType.HIGH_TEMP): 0.3,
            (MaterialType.STAINLESS_316, EnvironmentType.SEAWATER): 0.1,
            (MaterialType.STAINLESS_316, EnvironmentType.CHLORIDE): 0.05,
            (MaterialType.DUPLEX, EnvironmentType.SEAWATER): 0.03,
            (MaterialType.INCONEL, EnvironmentType.HIGH_TEMP): 0.02,
        }
        return allowances.get((self.material, environment), 0.1)

@dataclass
class CorrosionMeasurement:
    """Single corrosion measurement"""
    timestamp: datetime
    location: str  # e.g., "bottom_6oclock", "top_12oclock"
    measurement_type: str  # "ultrasonic", "visual", "radiographic"
    thickness: float  # mm
    is_pitting: bool = False
    pit_depth: Optional[float] = None  # mm
    pit_diameter: Optional[float] = None  # mm
    confidence: float = 0.95  # 0-1
    inspector: Optional[str] = None
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "location": self.location,
            "measurement_type": self.measurement_type,
            "thickness": self.thickness,
            "is_pitting": self.is_pitting,
            "pit_depth": self.pit_depth,
            "pit_diameter": self.pit_diameter,
            "confidence": self.confidence,
            "inspector": self.inspector,
            "notes": self.notes
        }

@dataclass
class CorrosionRate:
    """Calculated corrosion rate"""
    rate_mm_per_year: float
    rate_mils_per_year: Optional[float] = None
    confidence: float = 0.8
    calculation_method: str = "linear_regression"
    data_points: int = 0
    r_squared: Optional[float] = None
    
    def is_critical(self, threshold: float = 0.5) -> bool:
        """Check if corrosion rate exceeds critical threshold"""
        return self.rate_mm_per_year > threshold
    
    def to_mpy(self) -> float:
        """Convert mm/year to mils per year"""
        return self.rate_mm_per_year * 39.37 if self.rate_mm_per_year else 0.0

@dataclass
class RemainingLife:
    """Remaining life calculation"""
    years: float
    months: float
    days: float
    confidence: float
    failure_probability: float
    based_on: str  # "corrosion_rate", "fitness_for_service", "api_579"
    next_inspection_date: Optional[datetime] = None
    criticality: str = "low"  # low, medium, high, critical
    
    def is_critical(self) -> bool:
        """Check if remaining life is critical"""
        return self.criticality in ["high", "critical"] or self.years < 1

@dataclass
class CorrosionAlert:
    """Corrosion-related alert"""
    asset_id: str
    alert_type: str  # "high_rate", "critical_thickness", "pitting", "cui_detected"
    severity: str  # low, medium, high, critical
    message: str
    location: str
    current_value: float
    threshold: float
    recommended_action: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "asset_id": self.asset_id,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "location": self.location,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "recommended_action": self.recommended_action,
            "timestamp": self.timestamp.isoformat()
        }

class CorrosionDetector:
    """
    Main corrosion detection engine that:
    1. Tracks corrosion measurements over time
    2. Calculates corrosion rates using multiple methods
    3. Predicts remaining life based on industry standards
    4. Detects pitting and localized corrosion
    5. Implements fitness-for-service (FFS) calculations
    6. Generates maintenance and inspection plans
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.measurements: Dict[str, List[CorrosionMeasurement]] = {}  # asset_id -> measurements
        self.corrosion_rates: Dict[str, CorrosionRate] = {}
        self.material_db = self._initialize_materials()
        self.alerts_history: List[CorrosionAlert] = []
        self.inspection_plans: Dict[str, Dict] = {}
        
        # Industry standards thresholds
        self.thresholds = {
            "critical_corrosion_rate": 0.5,  # mm/year
            "minimum_thickness": 3.0,  # mm
            "pitting_depth_critical": 0.8,  # 80% of wall
            "pitting_ratio_critical": 0.4,  # API 579
            "cui_risk_temperature": 50.0,  # °C
        }
        
        # Load existing data
        self._load_existing_data()
        
        print(f"🛡️ Corrosion Detection initialized with {len(self.measurements)} assets")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _initialize_materials(self) -> Dict[MaterialType, MaterialProperties]:
        """Initialize material properties database"""
        return {
            MaterialType.CARBON_STEEL: MaterialProperties(
                material=MaterialType.CARBON_STEEL,
                density=7.85,
                yield_strength=250.0,
                tensile_strength=400.0,
                corrosion_resistance=0.3,
                typical_corrosion_rate=0.1,
                galvanic_series_position=-0.7
            ),
            MaterialType.STAINLESS_316: MaterialProperties(
                material=MaterialType.STAINLESS_316,
                density=8.0,
                yield_strength=205.0,
                tensile_strength=515.0,
                corrosion_resistance=0.85,
                typical_corrosion_rate=0.02,
                galvanic_series_position=0.2
            ),
            MaterialType.STAINLESS_304: MaterialProperties(
                material=MaterialType.STAINLESS_304,
                density=8.0,
                yield_strength=205.0,
                tensile_strength=515.0,
                corrosion_resistance=0.7,
                typical_corrosion_rate=0.03,
                galvanic_series_position=0.1
            ),
            MaterialType.DUPLEX: MaterialProperties(
                material=MaterialType.DUPLEX,
                density=7.8,
                yield_strength=450.0,
                tensile_strength=620.0,
                corrosion_resistance=0.9,
                typical_corrosion_rate=0.01,
                galvanic_series_position=0.3
            ),
            MaterialType.INCONEL: MaterialProperties(
                material=MaterialType.INCONEL,
                density=8.5,
                yield_strength=240.0,
                tensile_strength=550.0,
                corrosion_resistance=0.95,
                typical_corrosion_rate=0.005,
                galvanic_series_position=0.5
            )
        }
    
    def _load_existing_data(self):
        """Load previously saved data"""
        try:
            with open("data/models/corrosion_measurements.pkl", "rb") as f:
                self.measurements = pickle.load(f)
            print(f"📖 Loaded corrosion data for {len(self.measurements)} assets")
        except FileNotFoundError:
            print("No existing corrosion data found")
        
        try:
            with open("data/models/corrosion_rates.pkl", "rb") as f:
                self.corrosion_rates = pickle.load(f)
            print(f"📊 Loaded {len(self.corrosion_rates)} corrosion rates")
        except FileNotFoundError:
            print("No existing corrosion rates found")
    
    def _save_data(self):
        """Save all data to files"""
        with open("data/models/corrosion_measurements.pkl", "wb") as f:
            pickle.dump(self.measurements, f)
        
        with open("data/models/corrosion_rates.pkl", "wb") as f:
            pickle.dump(self.corrosion_rates, f)
        
        # Save alerts
        alerts_data = [alert.to_dict() for alert in self.alerts_history[-1000:]]
        with open("data/models/corrosion_alerts.json", "w") as f:
            json.dump(alerts_data, f, indent=2, default=str)
    
    def add_measurement(self, asset_id: str, measurement: CorrosionMeasurement):
        """Add a new corrosion measurement"""
        if asset_id not in self.measurements:
            self.measurements[asset_id] = []
        
        self.measurements[asset_id].append(measurement)
        
        # Sort by timestamp
        self.measurements[asset_id].sort(key=lambda x: x.timestamp)
        
        print(f"📏 Added corrosion measurement for {asset_id} at {measurement.location}")
        
        # Recalculate corrosion rate
        self._calculate_corrosion_rate(asset_id)
        
        # Check for alerts
        self._check_alerts(asset_id, measurement)
        
        # Save data
        self._save_data()
    
    def _calculate_corrosion_rate(self, asset_id: str) -> Optional[CorrosionRate]:
        """Calculate corrosion rate from measurements"""
        if asset_id not in self.measurements or len(self.measurements[asset_id]) < 2:
            return None
        
        measurements = self.measurements[asset_id]
        
        # Prepare data for regression
        timestamps = [(m.timestamp - measurements[0].timestamp).days / 365.25 
                     for m in measurements]
        thicknesses = [m.thickness for m in measurements]
        confidences = [m.confidence for m in measurements]
        
        # Weighted linear regression
        weights = np.array(confidences)
        x = np.array(timestamps)
        y = np.array(thicknesses)
        
        # Calculate slope (corrosion rate, negative because thickness decreases)
        if len(x) > 1:
            # Simple linear regression for now
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            rate = CorrosionRate(
                rate_mm_per_year=abs(slope),  # Convert to positive rate
                confidence=max(0.5, r_value**2),  # R-squared as confidence
                calculation_method="linear_regression",
                data_points=len(x),
                r_squared=r_value**2
            )
            
            self.corrosion_rates[asset_id] = rate
            
            print(f"📈 Calculated corrosion rate for {asset_id}: "
                  f"{rate.rate_mm_per_year:.3f} mm/year (R²={rate.r_squared:.3f})")
            
            return rate
        
        return None
    
    def _check_alerts(self, asset_id: str, measurement: CorrosionMeasurement):
        """Check for corrosion-related alerts"""
        alerts = []
        
        # Check for critical thickness
        if measurement.thickness < self.thresholds["minimum_thickness"]:
            alert = CorrosionAlert(
                asset_id=asset_id,
                alert_type="critical_thickness",
                severity="critical",
                message=f"Critical wall thickness: {measurement.thickness:.2f} mm",
                location=measurement.location,
                current_value=measurement.thickness,
                threshold=self.thresholds["minimum_thickness"],
                recommended_action=[
                    "Immediate inspection required",
                    "Consider pressure reduction",
                    "Prepare for repair/replacement"
                ]
            )
            alerts.append(alert)
        
        # Check for pitting
        if measurement.is_pitting and measurement.pit_depth:
            pit_ratio = measurement.pit_depth / measurement.thickness
            
            if pit_ratio > self.thresholds["pitting_ratio_critical"]:
                alert = CorrosionAlert(
                    asset_id=asset_id,
                    alert_type="critical_pitting",
                    severity="critical",
                    message=f"Critical pitting: {pit_ratio:.1%} of wall thickness",
                    location=measurement.location,
                    current_value=pit_ratio,
                    threshold=self.thresholds["pitting_ratio_critical"],
                    recommended_action=[
                        "Detailed pitting assessment required",
                        "Consider repair by grinding/welding",
                        "Increase inspection frequency"
                    ]
                )
                alerts.append(alert)
        
        # Check corrosion rate
        if asset_id in self.corrosion_rates:
            rate = self.corrosion_rates[asset_id]
            
            if rate.is_critical(self.thresholds["critical_corrosion_rate"]):
                alert = CorrosionAlert(
                    asset_id=asset_id,
                    alert_type="high_corrosion_rate",
                    severity="high",
                    message=f"High corrosion rate: {rate.rate_mm_per_year:.3f} mm/year",
                    location="overall",
                    current_value=rate.rate_mm_per_year,
                    threshold=self.thresholds["critical_corrosion_rate"],
                    recommended_action=[
                        "Review corrosion protection system",
                        "Consider chemical treatment",
                        "Increase monitoring frequency"
                    ]
                )
                alerts.append(alert)
        
        # Add alerts to history
        for alert in alerts:
            self.alerts_history.append(alert)
            print(f"🚨 Corrosion alert for {asset_id}: {alert.message}")
    
    def predict_remaining_life(self, asset_id: str, 
                             design_thickness: float,
                             material: MaterialType,
                             environment: EnvironmentType,
                             operating_pressure: float = 0.0,
                             temperature: float = 25.0) -> RemainingLife:
        """
        Predict remaining life based on corrosion data
        
        Args:
            asset_id: Asset identifier
            design_thickness: Original design thickness (mm)
            material: Material type
            environment: Operating environment
            operating_pressure: Operating pressure (bar)
            temperature: Operating temperature (°C)
        
        Returns:
            RemainingLife object
        """
        
        if asset_id not in self.measurements or not self.measurements[asset_id]:
            # No measurements, return conservative estimate
            return RemainingLife(
                years=20.0,
                months=0.0,
                days=0.0,
                confidence=0.5,
                failure_probability=0.1,
                based_on="industry_average",
                criticality="low"
            )
        
        # Get latest measurement
        latest = self.measurements[asset_id][-1]
        current_thickness = latest.thickness
        
        # Get corrosion rate
        if asset_id in self.corrosion_rates:
            corrosion_rate = self.corrosion_rates[asset_id].rate_mm_per_year
            rate_confidence = self.corrosion_rates[asset_id].confidence
        else:
            # Use material-based estimate
            material_props = self.material_db.get(material)
            if material_props:
                corrosion_rate = material_props.get_corrosion_allowance(environment)
                rate_confidence = 0.7
            else:
                corrosion_rate = 0.1  # Conservative default
                rate_confidence = 0.5
        
        # Calculate remaining life (simple linear projection)
        if corrosion_rate > 0:
            remaining_thickness = current_thickness - self.thresholds["minimum_thickness"]
            years_remaining = remaining_thickness / corrosion_rate
            
            # Apply safety factor
            years_remaining *= 0.7  # 30% safety margin
            
            # Calculate failure probability based on corrosion rate
            failure_prob = min(0.95, corrosion_rate * 2)  # Higher rate = higher probability
            
            # Determine criticality
            if years_remaining < 1:
                criticality = "critical"
            elif years_remaining < 3:
                criticality = "high"
            elif years_remaining < 10:
                criticality = "medium"
            else:
                criticality = "low"
            
            # Calculate next inspection date (API 510 guideline: 1/2 remaining life or 5 years max)
            inspection_interval = min(years_remaining / 2, 5)
            next_inspection = datetime.now() + timedelta(days=inspection_interval * 365.25)
            
            return RemainingLife(
                years=max(0, years_remaining),
                months=(years_remaining % 1) * 12,
                days=((years_remaining % 1) * 12 % 1) * 30,
                confidence=rate_confidence * 0.9,  # Reduce confidence for projection
                failure_probability=failure_prob,
                based_on="corrosion_rate_projection",
                next_inspection_date=next_inspection,
                criticality=criticality
            )
        
        # If no corrosion rate, return high remaining life
        return RemainingLife(
            years=50.0,
            months=0.0,
            days=0.0,
            confidence=0.8,
            failure_probability=0.01,
            based_on="no_corrosion_detected",
            next_inspection_date=datetime.now() + timedelta(days=365 * 5),
            criticality="low"
        )
    
    def calculate_fitness_for_service(self, asset_id: str,
                                    design_pressure: float,
                                    design_temperature: float,
                                    material: MaterialType) -> Dict[str, Any]:
        """
        Perform Fitness-for-Service (FFS) assessment per API 579
        
        Args:
            asset_id: Asset identifier
            design_pressure: Design pressure (bar)
            design_temperature: Design temperature (°C)
            material: Material type
        
        Returns:
            FFS assessment results
        """
        
        if asset_id not in self.measurements or not self.measurements[asset_id]:
            return {"status": "insufficient_data", "message": "No corrosion measurements"}
        
        latest = self.measurements[asset_id][-1]
        current_thickness = latest.thickness
        
        # Get material properties
        material_props = self.material_db.get(material)
        if not material_props:
            return {"status": "error", "message": "Material not found in database"}
        
        # Simplified FFS calculations (real implementation would follow API 579)
        
        # 1. Thickness assessment
        required_thickness = self._calculate_required_thickness(
            design_pressure, design_temperature, material_props
        )
        
        thickness_ratio = current_thickness / required_thickness
        
        # 2. MAWP (Maximum Allowable Working Pressure) calculation
        mawp = design_pressure * (current_thickness / required_thickness)
        
        # 3. Remaining strength factor (RSF)
        # Simplified: RSF = (t_actual - CA) / (t_nominal - CA)
        corrosion_allowance = 3.0  # mm (typical)
        rsf = (current_thickness - corrosion_allowance) / (required_thickness - corrosion_allowance)
        
        # 4. Assessment level
        if thickness_ratio >= 1.0 and rsf >= 0.9:
            ffs_level = 1  # Acceptable
            status = "fit_for_service"
        elif thickness_ratio >= 0.8 and rsf >= 0.7:
            ffs_level = 2  # Monitor
            status = "monitor_required"
        else:
            ffs_level = 3  # Repair
            status = "repair_required"
        
        return {
            "status": status,
            "ffs_level": ffs_level,
            "current_thickness_mm": current_thickness,
            "required_thickness_mm": required_thickness,
            "thickness_ratio": thickness_ratio,
            "mawp_bar": mawp,
            "remaining_strength_factor": rsf,
            "material": material.value,
            "assessment_date": datetime.now().isoformat(),
            "next_assessment": (datetime.now() + timedelta(days=365)).isoformat()
        }
    
    def _calculate_required_thickness(self, pressure: float, 
                                    temperature: float, 
                                    material: MaterialProperties) -> float:
        """
        Calculate required thickness using Barlow's formula (simplified)
        t = (P * D) / (2 * S * E) + CA
        
        Where:
        P = Internal pressure
        D = Diameter (assumed 1000mm for calculations)
        S = Allowable stress
        E = Joint efficiency (0.85 for welded)
        CA = Corrosion allowance
        """
        # Simplified constants
        diameter = 1000.0  # mm
        joint_efficiency = 0.85
        corrosion_allowance = 3.0  # mm
        
        # Allowable stress (simplified: 1/3 of yield at temperature)
        # Real implementation would use ASME B31.3 tables
        allowable_stress = material.yield_strength / 3.0
        
        # Barlow's formula
        required_thickness = (pressure * diameter) / (2 * allowable_stress * joint_efficiency)
        required_thickness += corrosion_allowance
        
        return max(6.0, required_thickness)  # Minimum 6mm
    
    def detect_cui_risk(self, asset_id: str, 
                       insulation_condition: str,
                       external_temperature: float,
                       internal_temperature: float,
                       humidity: float) -> Dict[str, Any]:
        """
        Detect Corrosion Under Insulation (CUI) risk
        
        Args:
            asset_id: Asset identifier
            insulation_condition: "good", "fair", "poor", "damaged"
            external_temperature: External temperature (°C)
            internal_temperature: Internal temperature (°C)
            humidity: Relative humidity (%)
        
        Returns:
            CUI risk assessment
        """
        
        # CUI risk factors (NACE SP0198)
        risk_score = 0.0
        
        # Temperature risk (50-120°C is highest risk for CUI)
        if 50 <= external_temperature <= 120:
            risk_score += 0.4
        elif 120 < external_temperature <= 175:
            risk_score += 0.3
        elif external_temperature > 175:
            risk_score += 0.2
        else:
            risk_score += 0.1
        
        # Insulation condition
        condition_weights = {
            "good": 0.1,
            "fair": 0.3,
            "poor": 0.6,
            "damaged": 0.9
        }
        risk_score += condition_weights.get(insulation_condition.lower(), 0.5)
        
        # Humidity risk
        if humidity > 80:
            risk_score += 0.4
        elif humidity > 60:
            risk_score += 0.2
        else:
            risk_score += 0.1
        
        # Normalize to 0-1
        risk_score = min(1.0, risk_score / 3.0)
        
        # Determine risk level
        if risk_score > 0.7:
            risk_level = "high"
            action = "Immediate insulation inspection required"
        elif risk_score > 0.4:
            risk_level = "medium"
            action = "Schedule insulation inspection"
        else:
            risk_level = "low"
            action = "Continue routine monitoring"
        
        return {
            "asset_id": asset_id,
            "cui_risk_score": risk_score,
            "cui_risk_level": risk_level,
            "recommended_action": action,
            "factors": {
                "temperature_risk": external_temperature,
                "insulation_condition": insulation_condition,
                "humidity_risk": humidity,
                "temperature_delta": internal_temperature - external_temperature
            },
            "assessment_date": datetime.now().isoformat()
        }
    
    def generate_inspection_plan(self, asset_id: str,
                               remaining_life: RemainingLife,
                               criticality: str = "medium") -> Dict[str, Any]:
        """
        Generate inspection plan based on corrosion risk
        
        Args:
            asset_id: Asset identifier
            remaining_life: Remaining life calculation
            criticality: Asset criticality
        
        Returns:
            Inspection plan
        """
        
        # Determine inspection interval based on remaining life and criticality
        if remaining_life.is_critical():
            interval_days = 30  # Monthly for critical
            methods = ["ultrasonic", "visual", "radiographic"]
            coverage = "100%"
        elif remaining_life.criticality == "high":
            interval_days = 90  # Quarterly
            methods = ["ultrasonic", "visual"]
            coverage = "75%"
        elif remaining_life.criticality == "medium":
            interval_days = 180  # Semi-annual
            methods = ["ultrasonic"]
            coverage = "50%"
        else:
            interval_days = 365  # Annual
            methods = ["ultrasonic"]
            coverage = "25%"
        
        next_inspection = datetime.now() + timedelta(days=interval_days)
        
        # Generate inspection locations based on historical data
        locations = []
        if asset_id in self.measurements:
            # Get locations with highest corrosion rates
            location_data = {}
            for measurement in self.measurements[asset_id]:
                loc = measurement.location
                if loc not in location_data:
                    location_data[loc] = []
                location_data[loc].append(measurement.thickness)
            
            # Calculate thickness loss per location
            for loc, thicknesses in location_data.items():
                if len(thicknesses) > 1:
                    loss = thicknesses[0] - thicknesses[-1]
                    locations.append({
                        "location": loc,
                        "priority": "high" if loss > 1.0 else "medium" if loss > 0.5 else "low",
                        "last_inspection": self.measurements[asset_id][-1].timestamp.isoformat()
                    })
        
        plan = {
            "asset_id": asset_id,
            "inspection_interval_days": interval_days,
            "next_inspection_date": next_inspection.isoformat(),
            "inspection_methods": methods,
            "coverage_requirement": coverage,
            "priority_locations": locations[:10],  # Top 10 locations
            "estimated_duration_hours": len(methods) * 4,  # 4 hours per method
            "special_requirements": ["safety_permit", "scaffolding"] if criticality == "high" else ["safety_permit"],
            "generated_date": datetime.now().isoformat()
        }
        
        self.inspection_plans[asset_id] = plan
        return plan
    
    def get_corrosion_trend(self, asset_id: str, 
                           location: Optional[str] = None,
                           days_back: int = 365) -> Dict[str, Any]:
        """
        Get corrosion trend data for analysis
        
        Args:
            asset_id: Asset identifier
            location: Specific location (optional)
            days_back: Number of days to look back
        
        Returns:
            Trend analysis
        """
        
        if asset_id not in self.measurements:
            return {"error": "No corrosion data for asset"}
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Filter measurements
        measurements = [
            m for m in self.measurements[asset_id]
            if m.timestamp >= cutoff_date
        ]
        
        if location:
            measurements = [m for m in measurements if m.location == location]
        
        if not measurements:
            return {"error": "No measurements in specified period"}
        
        # Prepare trend data
        trend_data = {
            "timestamps": [m.timestamp.isoformat() for m in measurements],
            "thicknesses": [m.thickness for m in measurements],
            "locations": [m.location for m in measurements],
            "measurement_types": [m.measurement_type for m in measurements]
        }
        
        # Calculate statistics
        thicknesses = np.array(trend_data["thicknesses"])
        
        stats_dict = {
            "current_thickness": thicknesses[-1],
            "initial_thickness": thicknesses[0],
            "total_loss": thicknesses[0] - thicknesses[-1],
            "average_thickness": float(np.mean(thicknesses)),
            "min_thickness": float(np.min(thicknesses)),
            "max_thickness": float(np.max(thicknesses)),
            "std_dev": float(np.std(thicknesses)),
            "measurement_count": len(thicknesses)
        }
        
        # Calculate rate if enough data
        if len(thicknesses) > 1:
            days = [(datetime.fromisoformat(t) - cutoff_date).days 
                   for t in trend_data["timestamps"]]
            slope, intercept, r_value, p_value, std_err = stats.linregress(days, thicknesses)
            
            stats_dict.update({
                "corrosion_rate_mm_per_year": abs(slope * 365.25),
                "correlation_r_squared": r_value**2,
                "predicted_thickness_30_days": intercept + slope * (days[-1] + 30),
                "predicted_thickness_90_days": intercept + slope * (days[-1] + 90)
            })
        
        return {
            "trend_data": trend_data,
            "statistics": stats_dict,
            "period_days": days_back,
            "location_filter": location
        }
    
    def export_corrosion_report(self, asset_id: str, 
                               format: str = "json") -> Dict[str, Any]:
        """
        Export comprehensive corrosion report
        
        Args:
            asset_id: Asset identifier
            format: Output format ("json" or "dict")
        
        Returns:
            Corrosion report
        """
        
        if asset_id not in self.measurements:
            return {"error": "Asset not found"}
        
        # Gather all data
        measurements = [m.to_dict() for m in self.measurements[asset_id]]
        
        # Get corrosion rate
        corrosion_rate = self.corrosion_rates.get(asset_id)
        rate_data = {
            "rate_mm_per_year": corrosion_rate.rate_mm_per_year if corrosion_rate else None,
            "confidence": corrosion_rate.confidence if corrosion_rate else None,
            "calculation_method": corrosion_rate.calculation_method if corrosion_rate else None
        }
        
        # Get recent alerts
        recent_alerts = [
            alert.to_dict() for alert in self.alerts_history[-10:]
            if alert.asset_id == asset_id
        ]
        
        # Get inspection plan
        inspection_plan = self.inspection_plans.get(asset_id, {})
        
        # Calculate statistics
        thicknesses = [m.thickness for m in self.measurements[asset_id]]
        
        report = {
            "asset_id": asset_id,
            "report_date": datetime.now().isoformat(),
            "summary": {
                "total_measurements": len(measurements),
                "first_measurement": measurements[0]["timestamp"] if measurements else None,
                "last_measurement": measurements[-1]["timestamp"] if measurements else None,
                "current_thickness": thicknesses[-1] if thicknesses else None,
                "minimum_thickness": min(thicknesses) if thicknesses else None,
                "maximum_thickness": max(thicknesses) if thicknesses else None,
                "average_corrosion_rate": rate_data["rate_mm_per_year"]
            },
            "corrosion_rate": rate_data,
            "measurements": measurements[-100:],  # Last 100 measurements
            "recent_alerts": recent_alerts,
            "inspection_plan": inspection_plan,
            "recommendations": self._generate_recommendations(asset_id)
        }
        
        if format == "json":
            return json.dumps(report, indent=2, default=str)
        
        return report
    
    def _generate_recommendations(self, asset_id: str) -> List[Dict[str, Any]]:
        """Generate recommendations based on corrosion data"""
        recommendations = []
        
        if asset_id not in self.measurements:
            return recommendations
        
        # Check corrosion rate
        if asset_id in self.corrosion_rates:
            rate = self.corrosion_rates[asset_id]
            
            if rate.is_critical(0.3):
                recommendations.append({
                    "type": "urgent",
                    "title": "High Corrosion Rate Detected",
                    "description": f"Corrosion rate ({rate.rate_mm_per_year:.3f} mm/year) exceeds acceptable limits",
                    "actions": [
                        "Implement corrosion inhibitor",
                        "Increase monitoring frequency",
                        "Consider cathodic protection"
                    ],
                    "priority": "high"
                })
        
        # Check for pitting
        measurements = self.measurements[asset_id]
        pitting_measurements = [m for m in measurements if m.is_pitting]
        
        if pitting_measurements:
            max_pit_depth = max(m.pit_depth for m in pitting_measurements if m.pit_depth)
            recommendations.append({
                "type": "warning",
                "title": "Pitting Corrosion Detected",
                "description": f"Maximum pit depth: {max_pit_depth:.2f} mm",
                "actions": [
                    "Perform detailed pitting assessment",
                    "Consider repair by grinding",
                    "Increase inspection frequency in affected areas"
                ],
                "priority": "medium"
            })
        
        # Check measurement frequency
        if len(measurements) > 1:
            time_diff = (measurements[-1].timestamp - measurements[-2].timestamp).days
            
            if time_diff > 365:
                recommendations.append({
                    "type": "info",
                    "title": "Increase Inspection Frequency",
                    "description": f"Last inspection was {time_diff} days ago",
                    "actions": [
                        "Schedule more frequent inspections",
                        "Consider continuous monitoring",
                        "Review inspection schedule"
                    ],
                    "priority": "low"
                })
        
        return recommendations
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get overall corrosion system statistics"""
        stats = {
            "total_assets": len(self.measurements),
            "total_measurements": sum(len(m) for m in self.measurements.values()),
            "active_alerts": len([a for a in self.alerts_history 
                                if (datetime.now() - a.timestamp).days < 30]),
            "high_corrosion_assets": sum(
                1 for rate in self.corrosion_rates.values()
                if rate.is_critical(0.3)
            ),
            "asset_with_most_measurements": max(
                self.measurements.items(),
                key=lambda x: len(x[1]),
                default=(None, 0)
            )[0],
            "recent_activity": {
                "last_24h": sum(
                    1 for asset_measurements in self.measurements.values()
                    for m in asset_measurements
                    if (datetime.now() - m.timestamp).days < 1
                ),
                "last_7d": sum(
                    1 for asset_measurements in self.measurements.values()
                    for m in asset_measurements
                    if (datetime.now() - m.timestamp).days < 7
                ),
                "last_30d": sum(
                    1 for asset_measurements in self.measurements.values()
                    for m in asset_measurements
                    if (datetime.now() - m.timestamp).days < 30
                )
            }
        }
        
        return stats

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Corrosion Detection Module")
    print("="*50)
    
    # Initialize detector
    detector = CorrosionDetector()
    
    # Test 1: Add measurements
    print("\n1️⃣ TEST: Adding Corrosion Measurements")
    print("-"*30)
    
    # Create test measurements for an asset
    asset_id = "vessel_001"
    
    # Historical measurements (simulating 2 years of data)
    base_date = datetime.now() - timedelta(days=730)
    
    measurements = [
        CorrosionMeasurement(
            timestamp=base_date + timedelta(days=i*180),
            location=f"location_{i%4}",
            measurement_type="ultrasonic",
            thickness=25.0 - (i * 0.2),  # Simulating corrosion
            confidence=0.95
        )
        for i in range(5)
    ]
    
    # Add a pitting measurement
    measurements.append(
        CorrosionMeasurement(
            timestamp=datetime.now(),
            location="bottom_6oclock",
            measurement_type="ultrasonic",
            thickness=22.5,
            is_pitting=True,
            pit_depth=2.5,
            pit_diameter=10.0,
            confidence=0.90,
            inspector="inspector_001",
            notes="Localized pitting detected"
        )
    )
    
    # Add all measurements
    for measurement in measurements:
        detector.add_measurement(asset_id, measurement)
    
    print(f"✅ Added {len(measurements)} measurements for {asset_id}")
    
    # Test 2: Corrosion Rate Calculation
    print("\n2️⃣ TEST: Corrosion Rate Calculation")
    print("-"*30)
    
    if asset_id in detector.corrosion_rates:
        rate = detector.corrosion_rates[asset_id]
        print(f"Corrosion rate: {rate.rate_mm_per_year:.3f} mm/year")
        print(f"Confidence: {rate.confidence:.2%}")
        print(f"Calculation method: {rate.calculation_method}")
        print(f"Critical: {"Yes" if rate.is_critical(0.3) else "No"}")
    else:
        print("❌ No corrosion rate calculated")
    
    # Test 3: Remaining Life Prediction
    print("\n3️⃣ TEST: Remaining Life Prediction")
    print("-"*30)
    
    remaining_life = detector.predict_remaining_life(
        asset_id=asset_id,
        design_thickness=25.0,
        material=MaterialType.CARBON_STEEL,
        environment=EnvironmentType.SEAWATER,
        operating_pressure=15.0,
        temperature=85.0
    )
    
    print(f"Remaining life: {remaining_life.years:.1f} years")
    print(f"Criticality: {remaining_life.criticality}")
    print(f"Confidence: {remaining_life.confidence:.2%}")
    print(f"Next inspection: {remaining_life.next_inspection_date}")
    
    # Test 4: Fitness-for-Service Assessment
    print("\n4️⃣ TEST: Fitness-for-Service Assessment")
    print("-"*30)
    
    ffs_result = detector.calculate_fitness_for_service(
        asset_id=asset_id,
        design_pressure=15.0,
        design_temperature=150.0,
        material=MaterialType.CARBON_STEEL
    )
    
    print(f"FFS Status: {ffs_result.get('status', 'N/A')}")
    print(f"FFS Level: {ffs_result.get('ffs_level', 'N/A')}")
    print(f"Current thickness: {ffs_result.get('current_thickness_mm', 'N/A')} mm")
    print(f"Required thickness: {ffs_result.get('required_thickness_mm', 'N/A')} mm")
    print(f"Thickness ratio: {ffs_result.get('thickness_ratio', 'N/A'):.2f}")
    
    # Test 5: CUI Risk Detection
    print("\n5️⃣ TEST: CUI Risk Detection")
    print("-"*30)
    
    cui_risk = detector.detect_cui_risk(
        asset_id=asset_id,
        insulation_condition="poor",
        external_temperature=65.0,
        internal_temperature=85.0,
        humidity=75.0
    )
    
    print(f"CUI Risk Score: {cui_risk.get('cui_risk_score', 'N/A'):.2f}")
    print(f"CUI Risk Level: {cui_risk.get('cui_risk_level', 'N/A')}")
    print(f"Recommended Action: {cui_risk.get('recommended_action', 'N/A')}")
    
    # Test 6: Inspection Plan Generation
    print("\n6️⃣ TEST: Inspection Plan Generation")
    print("-"*30)
    
    inspection_plan = detector.generate_inspection_plan(
        asset_id=asset_id,
        remaining_life=remaining_life,
        criticality="high"
    )
    
    print(f"Inspection interval: {inspection_plan.get('inspection_interval_days', 'N/A')} days")
    print(f"Next inspection: {inspection_plan.get('next_inspection_date', 'N/A')}")
    print(f"Methods: {', '.join(inspection_plan.get('inspection_methods', []))}")
    print(f"Coverage: {inspection_plan.get('coverage_requirement', 'N/A')}")
    
    # Test 7: Trend Analysis
    print("\n7️⃣ TEST: Trend Analysis")
    print("-"*30)
    
    trend = detector.get_corrosion_trend(asset_id, days_back=730)
    
    if "trend_data" in trend:
        print(f"Measurements in period: {len(trend['trend_data']['timestamps'])}")
        
        stats = trend.get('statistics', {})
        print(f"Current thickness: {stats.get('current_thickness', 'N/A')} mm")
        print(f"Total loss: {stats.get('total_loss', 'N/A'):.2f} mm")
        print(f"Corrosion rate: {stats.get('corrosion_rate_mm_per_year', 'N/A'):.3f} mm/year")
    
    # Test 8: System Statistics
    print("\n8️⃣ TEST: System Statistics")
    print("-"*30)
    
    stats = detector.get_system_statistics()
    print(f"Total assets monitored: {stats.get('total_assets', 0)}")
    print(f"Total measurements: {stats.get('total_measurements', 0)}")
    print(f"Active alerts: {stats.get('active_alerts', 0)}")
    print(f"High corrosion assets: {stats.get('high_corrosion_assets', 0)}")
    
    # Test 9: Export Report
    print("\n9️⃣ TEST: Export Corrosion Report")
    print("-"*30)
    
    report = detector.export_corrosion_report(asset_id)
    if isinstance(report, dict):
        print(f"Report generated for {asset_id}")
        print(f"Summary: {report.get('summary', {})}")
    else:
        print("✅ Report exported")
    
    # Save data
    detector._save_data()
    
    print(f"\n✅ Corrosion detection tests complete!")