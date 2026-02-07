"""
✅ FFS/FIT ANALYSIS & CRACK GROWTH PREDICTION MODULE
Performs Fitness-for-Service (FFS) assessments per API 579/580 standards
Predicts crack growth using fracture mechanics (Paris' Law)
Provides remaining life calculations for flawed components
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import math
from scipy import stats, integrate, optimize
import warnings
warnings.filterwarnings('ignore')

class FFSLevel(Enum):
    """FFS Assessment Levels per API 579"""
    LEVEL_1 = "Level 1"  # Screening assessment
    LEVEL_2 = "Level 2"  # Detailed assessment
    LEVEL_3 = "Level 3"  # Advanced assessment (FEA, etc.)

class FlawType(Enum):
    """Types of flaws for FFS assessment"""
    GENERAL_CORROSION = "general_corrosion"
    LOCAL_THINNING = "local_thinning"
    PITTING = "pitting"
    CRACK_LIKE = "crack_like"
    LAMINATION = "lamination"
    DENT = "dent"
    GOUGE = "gouge"

class CrackType(Enum):
    """Types of cracks for fracture mechanics analysis"""
    SURFACE_CRACK = "surface_crack"
    THROUGH_WALL_CRACK = "through_wall_crack"
    EMBEDDED_CRACK = "embedded_crack"
    CORNER_CRACK = "corner_crack"

class MaterialCategory(Enum):
    """Material categories for fracture toughness"""
    CARBON_STEEL = "carbon_steel"
    LOW_ALLOW_STEEL = "low_alloy_steel"
    STAINLESS_STEEL = "stainless_steel"
    DUPLEX = "duplex"
    NICKEL_ALLOW = "nickel_alloy"
    ALUMINUM = "aluminum"

@dataclass
class MaterialPropertiesFFS:
    """Material properties for FFS analysis"""
    category: MaterialCategory
    yield_strength: float  # Sy, MPa
    tensile_strength: float  # Su, MPa
    youngs_modulus: float  # E, GPa
    poissons_ratio: float
    fracture_toughness: float  # KIC, MPa√m
    fatigue_crack_growth_C: float  # Paris' Law C coefficient
    fatigue_crack_growth_m: float  # Paris' Law m exponent
    charpy_impact: Optional[float] = None  # Joules
    allowable_stress: Optional[float] = None  # S, MPa
    
    def get_allowable_stress(self, temperature: float = 38) -> float:
        """Get allowable stress per ASME B31.3"""
        if self.allowable_stress:
            return self.allowable_stress
        
        # Default based on yield strength (2/3 rule)
        return min(self.yield_strength * 2/3, self.tensile_strength * 1/3)

@dataclass
class FlawGeometry:
    """Geometry of a flaw for FFS assessment"""
    flaw_type: FlawType
    length: float  # mm
    depth: float  # mm
    width: Optional[float] = None  # mm (for pits)
    orientation: float = 0.0  # degrees from longitudinal
    location: str = "unknown"
    
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio (a/2c for cracks)"""
        if self.length > 0:
            return self.depth / self.length
        return 0.0

@dataclass
class CrackGeometry:
    """Geometry of a crack for fracture mechanics"""
    crack_type: CrackType
    surface_length: float  # 2c, mm
    depth: float  # a, mm
    angle: float = 0.0  # degrees from normal
    location: Optional[Dict[str, float]] = None
    
    def get_shape_factor(self) -> float:
        """Get shape factor (Y) based on crack type and geometry"""
        if self.crack_type == CrackType.SURFACE_CRACK:
            # Simplified shape factor for surface crack
            aspect = self.depth / (self.surface_length / 2)
            if aspect <= 0.1:
                return 1.12
            elif aspect <= 0.5:
                return 1.12 + 0.23 * aspect
            else:
                return 1.12 + 0.23 * 0.5  # Maximum
            
        elif self.crack_type == CrackType.THROUGH_WALL_CRACK:
            return 1.0  # Through-wall crack
            
        elif self.crack_type == CrackType.EMBEDDED_CRACK:
            return 1.0  # Conservative
            
        elif self.crack_type == CrackType.CORNER_CRACK:
            return 1.2  # Corner crack stress concentration
        
        return 1.0  # Default conservative value

@dataclass
class LoadingConditions:
    """Loading conditions for FFS assessment"""
    internal_pressure: float  # P, MPa
    external_pressure: float = 0.0  # MPa
    axial_stress: float = 0.0  # σ_ax, MPa
    bending_stress: float = 0.0  # σ_b, MPa
    torsional_stress: float = 0.0  # τ, MPa
    temperature: float = 20.0  # °C
    cycles_per_year: int = 0
    stress_ratio_R: float = 0.0  # R = σ_min/σ_max
    
    def get_primary_stress(self) -> float:
        """Calculate primary membrane stress"""
        return max(self.internal_pressure, self.external_pressure) + self.axial_stress
    
    def get_secondary_stress(self) -> float:
        """Calculate secondary (bending) stress"""
        return self.bending_stress
    
    def get_peak_stress(self) -> float:
        """Calculate peak stress (simplified)"""
        return self.get_primary_stress() + self.get_secondary_stress()

@dataclass
class ComponentGeometry:
    """Component geometry for FFS assessment"""
    diameter: float  # D, mm
    thickness: float  # t, mm
    length: float = 0.0  # L, mm
    corrosion_allowance: float = 3.0  # CA, mm
    joint_efficiency: float = 0.85
    
    def get_radius(self) -> float:
        """Get mean radius"""
        return self.diameter / 2
    
    def required_thickness(self, pressure: float, 
                          allowable_stress: float) -> float:
        """Calculate required thickness per Barlow's formula"""
        # t = (P * D) / (2 * S * E) + CA
        return (pressure * self.diameter) / (2 * allowable_stress * self.joint_efficiency) + self.corrosion_allowance

@dataclass
class FFSAssessmentResult:
    """Results of FFS assessment"""
    assessment_level: FFSLevel
    flaw_type: FlawType
    is_acceptable: bool
    remaining_strength_factor: float  # RSF
    allowable_rsf: float  # RSFa
    mawp_reduction: float  # % reduction in MAWP
    assessment_date: datetime
    next_assessment_date: datetime
    recommendations: List[str]
    
    def get_criticality(self) -> str:
        """Get criticality based on RSF"""
        if self.remaining_strength_factor < 0.7:
            return "critical"
        elif self.remaining_strength_factor < 0.9:
            return "high"
        elif self.remaining_strength_factor < 1.0:
            return "medium"
        else:
            return "low"

@dataclass
class CrackGrowthResult:
    """Results of crack growth prediction"""
    initial_size: float  # mm
    final_size: float  # mm
    cycles_to_failure: int
    years_to_failure: float
    crack_growth_rate: float  # mm/cycle
    critical_crack_size: float  # mm
    failure_mode: str  # "leak_before_break", "rupture", "plastic_collapse"
    confidence: float
    
    def is_critical(self) -> bool:
        """Check if crack is critical"""
        return self.years_to_failure < 1.0

@dataclass
class RemainingLifeFFS:
    """Remaining life from FFS assessment"""
    years: float
    months: float
    days: float
    confidence: float
    limiting_factor: str  # "corrosion", "crack_growth", "fatigue", "creep"
    next_inspection: datetime
    repair_recommended: bool
    repair_urgency: str  # "immediate", "1_year", "3_years", "5_years"

class FFSFitAnalyzer:
    """
    Main FFS/FIT analysis engine that:
    1. Performs Fitness-for-Service assessments per API 579
    2. Predicts crack growth using fracture mechanics
    3. Calculates remaining life for flawed components
    4. Provides repair recommendations
    5. Implements industry standards (API, ASME, BS)
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.material_db = self._initialize_materials()
        self.assessment_history: Dict[str, List[FFSAssessmentResult]] = {}
        self.crack_growth_history: Dict[str, List[CrackGrowthResult]] = {}
        
        # Industry standard factors
        self.safety_factors = {
            "yield": 1.5,
            "tensile": 3.0,
            "fatigue": 2.0,
            "fracture": 2.0
        }
        
        # Load existing data
        self._load_existing_data()
        
        print(f"✅ FFS/FIT Analyzer initialized with {len(self.material_db)} materials")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _initialize_materials(self) -> Dict[MaterialCategory, MaterialPropertiesFFS]:
        """Initialize material properties database"""
        return {
            MaterialCategory.CARBON_STEEL: MaterialPropertiesFFS(
                category=MaterialCategory.CARBON_STEEL,
                yield_strength=250.0,
                tensile_strength=400.0,
                youngs_modulus=200.0,
                poissons_ratio=0.3,
                fracture_toughness=100.0,  # MPa√m
                fatigue_crack_growth_C=6.9e-12,  # Paris' Law C
                fatigue_crack_growth_m=3.0,  # Paris' Law m
                charpy_impact=40.0,
                allowable_stress=138.0  # MPa at 38°C
            ),
            MaterialCategory.LOW_ALLOW_STEEL: MaterialPropertiesFFS(
                category=MaterialCategory.LOW_ALLOW_STEEL,
                yield_strength=415.0,
                tensile_strength=585.0,
                youngs_modulus=200.0,
                poissons_ratio=0.3,
                fracture_toughness=120.0,
                fatigue_crack_growth_C=4.8e-12,
                fatigue_crack_growth_m=3.0,
                charpy_impact=54.0,
                allowable_stress=207.0
            ),
            MaterialCategory.STAINLESS_STEEL: MaterialPropertiesFFS(
                category=MaterialCategory.STAINLESS_STEEL,
                yield_strength=205.0,
                tensile_strength=515.0,
                youngs_modulus=193.0,
                poissons_ratio=0.3,
                fracture_toughness=150.0,
                fatigue_crack_growth_C=1.7e-11,
                fatigue_crack_growth_m=3.25,
                charpy_impact=100.0,
                allowable_stress=115.0
            ),
            MaterialCategory.DUPLEX: MaterialPropertiesFFS(
                category=MaterialCategory.DUPLEX,
                yield_strength=450.0,
                tensile_strength=620.0,
                youngs_modulus=200.0,
                poissons_ratio=0.3,
                fracture_toughness=200.0,
                fatigue_crack_growth_C=1.0e-11,
                fatigue_crack_growth_m=3.1,
                charpy_impact=150.0,
                allowable_stress=172.0
            )
        }
    
    def _load_existing_data(self):
        """Load previously saved data"""
        try:
            with open("data/models/ffs_assessments.pkl", "rb") as f:
                self.assessment_history = pickle.load(f)
            print(f"📖 Loaded {sum(len(v) for v in self.assessment_history.values())} FFS assessments")
        except FileNotFoundError:
            print("No existing FFS assessments found")
        
        try:
            with open("data/models/crack_growth_history.pkl", "rb") as f:
                self.crack_growth_history = pickle.load(f)
            print(f"📊 Loaded {sum(len(v) for v in self.crack_growth_history.values())} crack growth predictions")
        except FileNotFoundError:
            print("No existing crack growth data found")
    
    def _save_data(self):
        """Save all data to files"""
        with open("data/models/ffs_assessments.pkl", "wb") as f:
            pickle.dump(self.assessment_history, f)
        
        with open("data/models/crack_growth_history.pkl", "wb") as f:
            pickle.dump(self.crack_growth_history, f)
    
    def perform_ffs_assessment(self,
                             asset_id: str,
                             component: ComponentGeometry,
                             material: MaterialCategory,
                             flaw: FlawGeometry,
                             loading: LoadingConditions,
                             assessment_level: FFSLevel = FFSLevel.LEVEL_2) -> FFSAssessmentResult:
        """
        Perform Fitness-for-Service assessment per API 579
        
        Args:
            asset_id: Asset identifier
            component: Component geometry
            material: Material category
            flaw: Flaw geometry
            loading: Loading conditions
            assessment_level: FFS assessment level
        
        Returns:
            FFSAssessmentResult
        """
        
        # Get material properties
        material_props = self.material_db.get(material)
        if not material_props:
            raise ValueError(f"Material {material} not found in database")
        
        # Calculate RSF based on flaw type
        rsf = self._calculate_rsf(component, material_props, flaw, loading, assessment_level)
        
        # Determine acceptability
        allowable_rsf = 0.9  # API 579 default
        is_acceptable = rsf >= allowable_rsf
        
        # Calculate MAWP reduction
        mawp_reduction = (1 - rsf) * 100 if rsf < 1.0 else 0.0
        
        # Generate recommendations
        recommendations = self._generate_ffs_recommendations(
            rsf, flaw.flaw_type, mawp_reduction, assessment_level
        )
        
        # Determine next assessment date
        if rsf < 0.7:
            next_assessment_days = 30  # Critical: monthly
        elif rsf < 0.9:
            next_assessment_days = 90  # High risk: quarterly
        elif rsf < 1.0:
            next_assessment_days = 180  # Medium risk: semi-annual
        else:
            next_assessment_days = 365  # Low risk: annual
        
        next_assessment = datetime.now() + timedelta(days=next_assessment_days)
        
        result = FFSAssessmentResult(
            assessment_level=assessment_level,
            flaw_type=flaw.flaw_type,
            is_acceptable=is_acceptable,
            remaining_strength_factor=rsf,
            allowable_rsf=allowable_rsf,
            mawp_reduction=mawp_reduction,
            assessment_date=datetime.now(),
            next_assessment_date=next_assessment,
            recommendations=recommendations
        )
        
        # Store in history
        if asset_id not in self.assessment_history:
            self.assessment_history[asset_id] = []
        self.assessment_history[asset_id].append(result)
        
        # Save data
        self._save_data()
        
        return result
    
    def _calculate_rsf(self,
                      component: ComponentGeometry,
                      material: MaterialPropertiesFFS,
                      flaw: FlawGeometry,
                      loading: LoadingConditions,
                      level: FFSLevel) -> float:
        """
        Calculate Remaining Strength Factor (RSF)
        
        Implementation per API 579 Part 4-6
        """
        
        if flaw.flaw_type == FlawType.GENERAL_CORROSION:
            # API 579 Part 4: General Metal Loss
            return self._calculate_rsf_general_corrosion(component, flaw)
        
        elif flaw.flaw_type == FlawType.LOCAL_THINNING:
            # API 579 Part 5: Local Metal Loss
            return self._calculate_rsf_local_thinning(component, flaw, loading)
        
        elif flaw.flaw_type == FlawType.PITTING:
            # API 579 Part 6: Pitting Corrosion
            return self._calculate_rsf_pitting(component, flaw)
        
        elif flaw.flaw_type == FlawType.CRACK_LIKE:
            # API 579 Part 9: Crack-like Flaws
            return self._calculate_rsf_crack_like(component, material, flaw, loading)
        
        else:
            # Conservative default
            return 0.8
    
    def _calculate_rsf_general_corrosion(self,
                                        component: ComponentGeometry,
                                        flaw: FlawGeometry) -> float:
        """Calculate RSF for general corrosion (API 579 Part 4)"""
        # t_min = minimum measured thickness
        # t_nom = nominal thickness
        
        t_min = component.thickness - flaw.depth
        t_nom = component.thickness
        
        if t_min <= 0 or t_nom <= 0:
            return 0.0
        
        # RSF = (t_min - CA) / (t_nom - CA)
        ca = component.corrosion_allowance
        rsf = (t_min - ca) / (t_nom - ca)
        
        return max(0.0, min(1.0, rsf))
    
    def _calculate_rsf_local_thinning(self,
                                     component: ComponentGeometry,
                                     flaw: FlawGeometry,
                                     loading: LoadingConditions) -> float:
        """Calculate RSF for local thinning (API 579 Part 5)"""
        # Simplified approach using area replacement method
        
        t_avg = component.thickness - flaw.depth / 2
        t_nom = component.thickness
        
        # Calculate membrane stress in thinned area
        s_m = loading.get_primary_stress()
        
        # Calculate bending stress magnification
        # M_t = Folias factor for longitudinal extent
        s = flaw.length
        D = component.diameter
        t = t_avg
        
        # Folias factor (simplified)
        if s / math.sqrt(D * t) <= 1.0:
            M_t = 1.0
        else:
            M_t = 1.0 + 0.48 * (s / math.sqrt(D * t))
        
        # RSF using area replacement
        A_loss = flaw.depth * flaw.length
        A_total = t_nom * flaw.length
        
        if A_total > 0:
            rsf_area = 1.0 - (A_loss / A_total)
        else:
            rsf_area = 1.0
        
        # Apply Folias factor
        rsf = rsf_area / M_t
        
        return max(0.0, min(1.0, rsf))
    
    def _calculate_rsf_pitting(self,
                              component: ComponentGeometry,
                              flaw: FlawGeometry) -> float:
        """Calculate RSF for pitting corrosion (API 579 Part 6)"""
        # Pit depth ratio method
        
        if not flaw.width:
            # Assume circular pit
            pit_area = math.pi * (flaw.width/2)**2 if flaw.width else math.pi * (flaw.depth)**2
        else:
            pit_area = flaw.depth * flaw.width
        
        # Remaining ligament thickness
        t_lig = component.thickness - flaw.depth
        
        if t_lig <= 0:
            return 0.0
        
        # Calculate remaining cross-sectional area
        # Assuming pitting affects a representative area
        rep_area = 100.0  # mm² (representative area for assessment)
        pit_density = pit_area / rep_area
        
        # RSF based on remaining ligament and pit density
        rsf = (t_lig / component.thickness) * (1 - pit_density)
        
        return max(0.1, min(1.0, rsf))
    
    def _calculate_rsf_crack_like(self,
                                 component: ComponentGeometry,
                                 material: MaterialPropertiesFFS,
                                 flaw: FlawGeometry,
                                 loading: LoadingConditions) -> float:
        """Calculate RSF for crack-like flaws (API 579 Part 9)"""
        # Simplified fracture mechanics approach
        
        # Stress intensity factor
        K_I = self._calculate_stress_intensity(component, flaw, loading)
        
        # Material fracture toughness
        K_IC = material.fracture_toughness
        
        if K_IC <= 0:
            return 0.8  # Conservative default
        
        # Safety factor on fracture
        safety_factor = self.safety_factors["fracture"]
        
        # RSF based on ratio of applied K to critical K
        rsf = 1.0 / (1.0 + (K_I * safety_factor / K_IC)**2)
        
        return max(0.1, min(1.0, rsf))
    
    def _calculate_stress_intensity(self,
                                   component: ComponentGeometry,
                                   flaw: FlawGeometry,
                                   loading: LoadingConditions) -> float:
        """Calculate stress intensity factor K_I"""
        # Simplified K calculation
        # K_I = Y * σ * sqrt(π * a)
        
        # Total stress at flaw location
        sigma = loading.get_peak_stress()
        
        # Crack depth
        a = flaw.depth
        
        # Geometric factor (simplified)
        if flaw.flaw_type == FlawType.CRACK_LIKE:
            # Surface crack
            Y = 1.12  # Surface correction factor
        else:
            Y = 1.0
        
        # Calculate K
        K_I = Y * sigma * math.sqrt(math.pi * a)
        
        return K_I
    
    def _generate_ffs_recommendations(self,
                                     rsf: float,
                                     flaw_type: FlawType,
                                     mawp_reduction: float,
                                     level: FFSLevel) -> List[str]:
        """Generate FFS recommendations based on assessment results"""
        recommendations = []
        
        # Base recommendations on RSF
        if rsf < 0.7:
            recommendations.append("IMMEDIATE REPAIR REQUIRED")
            recommendations.append("Reduce operating pressure immediately")
            recommendations.append("Schedule for replacement")
            
        elif rsf < 0.9:
            recommendations.append("Schedule repair within 3 months")
            recommendations.append(f"Reduce MAWP by {mawp_reduction:.1f}%")
            recommendations.append("Increase inspection frequency to quarterly")
            
        elif rsf < 1.0:
            recommendations.append("Monitor condition closely")
            recommendations.append("Next assessment within 6 months")
            recommendations.append("Consider repair during next shutdown")
            
        else:
            recommendations.append("Component is fit for service")
            recommendations.append("Continue routine monitoring")
        
        # Flaw-specific recommendations
        if flaw_type == FlawType.PITTING:
            recommendations.append("Perform detailed pitting assessment")
            recommendations.append("Consider corrosion inhibitor addition")
        
        elif flaw_type == FlawType.CRACK_LIKE:
            recommendations.append("Perform detailed fracture mechanics analysis")
            recommendations.append("Consider crack growth monitoring")
        
        # Level-specific recommendations
        if level == FFSLevel.LEVEL_1:
            recommendations.append("Consider Level 2 assessment for more accuracy")
        elif level == FFSLevel.LEVEL_2:
            recommendations.append("Level 2 assessment complete - results valid")
        
        return recommendations
    
    def predict_crack_growth(self,
                           asset_id: str,
                           component: ComponentGeometry,
                           material: MaterialCategory,
                           crack: CrackGeometry,
                           loading: LoadingConditions,
                           inspection_interval_years: float = 1.0) -> CrackGrowthResult:
        """
        Predict crack growth using fracture mechanics (Paris' Law)
        
        Args:
            asset_id: Asset identifier
            component: Component geometry
            material: Material category
            crack: Crack geometry
            loading: Loading conditions
            inspection_interval_years: Time between inspections
        
        Returns:
            CrackGrowthResult with growth prediction
        """
        
        # Get material properties
        material_props = self.material_db.get(material)
        if not material_props:
            raise ValueError(f"Material {material} not found in database")
        
        # Calculate stress intensity factor range
        delta_K = self._calculate_stress_intensity_range(component, crack, loading)
        
        # Get Paris' Law coefficients
        C = material_props.fatigue_crack_growth_C
        m = material_props.fatigue_crack_growth_m
        
        # Calculate critical crack size
        a_critical = self._calculate_critical_crack_size(
            component, material_props, crack, loading
        )
        
        # Predict crack growth using Paris' Law
        # da/dN = C * (ΔK)^m
        
        a_initial = crack.depth
        cycles_per_year = loading.cycles_per_year
        
        if cycles_per_year == 0:
            # No cyclic loading, only static analysis
            cycles_to_failure = 10**9  # Very high (effectively infinite)
            crack_growth_rate = 0.0
        else:
            # Calculate growth rate at initial size
            crack_growth_rate = C * (delta_K ** m)
            
            # Integrate Paris' Law to predict cycles to failure
            cycles_to_failure = self._integrate_paris_law(
                a_initial, a_critical, C, m, delta_K, crack
            )
        
        # Calculate time to failure
        if cycles_per_year > 0:
            years_to_failure = cycles_to_failure / cycles_per_year
        else:
            # Static loading - use time-based approach
            years_to_failure = 20.0  # Conservative default
        
        # Determine failure mode
        failure_mode = self._determine_failure_mode(
            component, crack, a_critical, loading
        )
        
        # Calculate confidence based on input quality
        confidence = self._calculate_prediction_confidence(
            crack, loading, material_props
        )
        
        result = CrackGrowthResult(
            initial_size=a_initial,
            final_size=a_critical,
            cycles_to_failure=int(cycles_to_failure),
            years_to_failure=years_to_failure,
            crack_growth_rate=crack_growth_rate,
            critical_crack_size=a_critical,
            failure_mode=failure_mode,
            confidence=confidence
        )
        
        # Store in history
        if asset_id not in self.crack_growth_history:
            self.crack_growth_history[asset_id] = []
        self.crack_growth_history[asset_id].append(result)
        
        # Save data
        self._save_data()
        
        return result
    
    def _calculate_stress_intensity_range(self,
                                         component: ComponentGeometry,
                                         crack: CrackGeometry,
                                         loading: LoadingConditions) -> float:
        """Calculate stress intensity factor range ΔK"""
        # ΔK = Y * Δσ * sqrt(π * a)
        
        # Stress range (simplified)
        sigma_max = loading.get_peak_stress()
        sigma_min = sigma_max * loading.stress_ratio_R
        delta_sigma = sigma_max - sigma_min
        
        # Shape factor
        Y = crack.get_shape_factor()
        
        # Crack depth
        a = crack.depth
        
        # Calculate ΔK
        delta_K = Y * delta_sigma * math.sqrt(math.pi * a)
        
        return delta_K
    
    def _calculate_critical_crack_size(self,
                                      component: ComponentGeometry,
                                      material: MaterialPropertiesFFS,
                                      crack: CrackGeometry,
                                      loading: LoadingConditions) -> float:
        """Calculate critical crack size for failure"""
        
        # Failure criteria:
        # 1. Fracture: K_I >= K_IC
        # 2. Plastic collapse: Net section stress >= flow stress
        # 3. Leak-before-break: Through-wall crack
        
        # Method 1: Fracture toughness based
        K_IC = material.fracture_toughness
        sigma = loading.get_peak_stress()
        Y = crack.get_shape_factor()
        
        # Critical size from K_IC = Y * σ * sqrt(π * a_c)
        if Y > 0 and sigma > 0:
            a_critical_fracture = (K_IC / (Y * sigma))**2 / math.pi
        else:
            a_critical_fracture = component.thickness
        
        # Method 2: Plastic collapse (simplified)
        # Based on remaining ligament
        flow_stress = (material.yield_strength + material.tensile_strength) / 2
        a_critical_collapse = component.thickness * (1 - material.yield_strength / flow_stress)
        
        # Method 3: Through-wall (leak-before-break)
        a_critical_through_wall = component.thickness
        
        # Take minimum of all criteria
        a_critical = min(
            a_critical_fracture,
            a_critical_collapse,
            a_critical_through_wall
        )
        
        # Ensure it's not less than current size
        a_critical = max(crack.depth * 1.1, a_critical)
        
        # Limit to component thickness
        a_critical = min(a_critical, component.thickness)
        
        return a_critical
    
    def _integrate_paris_law(self,
                            a_initial: float,
                            a_critical: float,
                            C: float,
                            m: float,
                            delta_K_ref: float,
                            crack: CrackGeometry) -> float:
        """Integrate Paris' Law to predict cycles to failure"""
        
        # Simplified integration assuming constant ΔK
        # In reality, ΔK changes with crack growth
        
        if m == 2:
            # Special case for m=2
            Nf = (1 / (C * delta_K_ref**2)) * math.log(a_critical / a_initial)
        else:
            # General case
            Nf = (1 / (C * (2-m) * delta_K_ref**m)) * \
                 (a_critical**(1-m/2) - a_initial**(1-m/2))
        
        # Ensure positive result
        return max(1, abs(Nf))
    
    def _determine_failure_mode(self,
                               component: ComponentGeometry,
                               crack: CrackGeometry,
                               a_critical: float,
                               loading: LoadingConditions) -> str:
        """Determine most likely failure mode"""
        
        # Check for leak-before-break
        if a_critical >= component.thickness * 0.8:
            return "leak_before_break"
        
        # Check for brittle fracture
        if loading.temperature < 0:  # Low temperature
            return "brittle_fracture"
        
        # Check for plastic collapse
        sigma_primary = loading.get_primary_stress()
        if sigma_primary > 0.7 * component.thickness:  # High stress
            return "plastic_collapse"
        
        # Default
        return "fatigue_crack_growth"
    
    def _calculate_prediction_confidence(self,
                                        crack: CrackGeometry,
                                        loading: LoadingConditions,
                                        material: MaterialPropertiesFFS) -> float:
        """Calculate confidence in crack growth prediction"""
        confidence = 1.0
        
        # Reduce confidence for uncertainties
        if crack.depth < 1.0:  # Small cracks hard to measure
            confidence *= 0.8
        
        if loading.cycles_per_year == 0:  # No cyclic loading data
            confidence *= 0.7
        
        if material.fatigue_crack_growth_C <= 0:  # No material data
            confidence *= 0.6
        
        # Increase confidence for good data
        if crack.location is not None:  # Known location
            confidence *= 1.1
        
        return max(0.1, min(1.0, confidence))
    
    def calculate_remaining_life_ffs(self,
                                   asset_id: str,
                                   component: ComponentGeometry,
                                   material: MaterialCategory,
                                   loading: LoadingConditions,
                                   flaw: Optional[FlawGeometry] = None,
                                   crack: Optional[CrackGeometry] = None) -> RemainingLifeFFS:
        """
        Calculate remaining life considering all failure modes
        
        Args:
            asset_id: Asset identifier
            component: Component geometry
            material: Material category
            loading: Loading conditions
            flaw: Optional flaw geometry
            crack: Optional crack geometry
        
        Returns:
            RemainingLifeFFS object
        """
        
        remaining_lives = []
        limiting_factors = []
        
        # 1. Corrosion-based life
        if flaw and flaw.flaw_type in [FlawType.GENERAL_CORROSION, 
                                       FlawType.LOCAL_THINNING, 
                                       FlawType.PITTING]:
            corrosion_life = self._calculate_corrosion_life(
                component, flaw, loading
            )
            remaining_lives.append(corrosion_life)
            limiting_factors.append("corrosion")
        
        # 2. Crack growth life
        if crack:
            crack_growth_result = self.predict_crack_growth(
                asset_id, component, material, crack, loading
            )
            remaining_lives.append(crack_growth_result.years_to_failure)
            limiting_factors.append("crack_growth")
        
        # 3. Fatigue life (if cyclic loading)
        if loading.cycles_per_year > 0:
            fatigue_life = self._calculate_fatigue_life(
                component, material, loading
            )
            remaining_lives.append(fatigue_life)
            limiting_factors.append("fatigue")
        
        # 4. Creep life (if high temperature)
        if loading.temperature > 400:  # °C
            creep_life = self._calculate_creep_life(
                component, material, loading
            )
            remaining_lives.append(creep_life)
            limiting_factors.append("creep")
        
        # Take minimum remaining life
        if remaining_lives:
            min_life = min(remaining_lives)
            limiting_factor = limiting_factors[remaining_lives.index(min_life)]
        else:
            min_life = 20.0  # Conservative default
            limiting_factor = "general_aging"
        
        # Calculate confidence
        confidence = self._calculate_life_confidence(
            component, flaw, crack, loading
        )
        
        # Determine if repair is needed
        repair_recommended = min_life < 5.0
        
        # Determine repair urgency
        if min_life < 1.0:
            repair_urgency = "immediate"
        elif min_life < 3.0:
            repair_urgency = "1_year"
        elif min_life < 5.0:
            repair_urgency = "3_years"
        else:
            repair_urgency = "5_years"
        
        # Calculate next inspection (half of remaining life or 5 years max)
        next_inspection_years = min(min_life / 2, 5.0)
        next_inspection = datetime.now() + timedelta(days=next_inspection_years * 365.25)
        
        # Convert years to months and days
        years = math.floor(min_life)
        months = math.floor((min_life - years) * 12)
        days = math.floor(((min_life - years) * 12 - months) * 30)
        
        return RemainingLifeFFS(
            years=years,
            months=months,
            days=days,
            confidence=confidence,
            limiting_factor=limiting_factor,
            next_inspection=next_inspection,
            repair_recommended=repair_recommended,
            repair_urgency=repair_urgency
        )
    
    def _calculate_corrosion_life(self,
                                 component: ComponentGeometry,
                                 flaw: FlawGeometry,
                                 loading: LoadingConditions) -> float:
        """Calculate remaining life based on corrosion"""
        # Simplified corrosion life calculation
        
        current_thickness = component.thickness - flaw.depth
        required_thickness = self._calculate_required_thickness(
            loading.get_primary_stress(), component, 1.5  # Safety factor
        )
        
        if current_thickness <= required_thickness:
            return 0.1  # Already at minimum
        
        # Assume corrosion rate of 0.1 mm/year (conservative)
        corrosion_rate = 0.1  # mm/year
        
        remaining_thickness = current_thickness - required_thickness
        remaining_years = remaining_thickness / corrosion_rate
        
        return max(0.1, remaining_years)
    
    def _calculate_fatigue_life(self,
                               component: ComponentGeometry,
                               material: MaterialCategory,
                               loading: LoadingConditions) -> float:
        """Calculate fatigue life using S-N curves"""
        # Simplified fatigue life calculation
        
        stress_range = loading.get_peak_stress() * (1 - loading.stress_ratio_R)
        
        # Get material properties
        material_props = self.material_db.get(material)
        if not material_props:
            return 20.0  # Default
        
        # Estimate fatigue strength at given cycles
        # Using simplified approach
        fatigue_strength = material_props.tensile_strength * 0.5
        
        if stress_range <= 0:
            return 10**6  # Infinite life
        
        # Calculate life using Basquin's equation (simplified)
        # N = (σ_a / σ_f')^(1/b)
        # Where σ_f' ~ 0.9*UTS and b ~ -0.1
        
        sigma_f = 0.9 * material_props.tensile_strength
        b = -0.1
        
        cycles_to_failure = (stress_range / sigma_f) ** (1/b)
        
        # Convert to years
        if loading.cycles_per_year > 0:
            years_to_failure = cycles_to_failure / loading.cycles_per_year
        else:
            years_to_failure = 10**6  # Very long
        
        return max(1.0, years_to_failure)
    
    def _calculate_creep_life(self,
                             component: ComponentGeometry,
                             material: MaterialCategory,
                             loading: LoadingConditions) -> float:
        """Calculate creep life using Larson-Miller parameter"""
        # Simplified creep life calculation
        
        # Get material properties
        material_props = self.material_db.get(material)
        if not material_props:
            return 10.0  # Default
        
        # Calculate stress
        sigma = loading.get_primary_stress()
        
        # Calculate Larson-Miller parameter (simplified)
        # P = T * (C + log(t))
        # Where T in Kelvin, C ~ 20, t in hours
        
        T_kelvin = loading.temperature + 273.15
        
        # For carbon steel at 100MPa and 500°C, LMP ~ 34
        if material == MaterialCategory.CARBON_STEEL:
            base_lmp = 34.0
        elif material == MaterialCategory.LOW_ALLOW_STEEL:
            base_lmp = 36.0
        elif material == MaterialCategory.STAINLESS_STEEL:
            base_lmp = 38.0
        else:
            base_lmp = 35.0
        
        # Adjust for stress
        stress_factor = 1.0 - (sigma / material_props.tensile_strength)
        lmp = base_lmp * stress_factor
        
        # Solve for time: t = 10^(P/T - C)
        C = 20  # Material constant
        log_t = lmp / T_kelvin - C
        t_hours = 10 ** log_t
        
        # Convert to years
        years_to_failure = t_hours / (24 * 365.25)
        
        return max(0.1, years_to_failure)
    
    def _calculate_required_thickness(self,
                                     pressure: float,
                                     component: ComponentGeometry,
                                     safety_factor: float = 1.5) -> float:
        """Calculate required thickness with safety factor"""
        # Simplified Barlow's formula with safety factor
        
        S = 138.0  # Allowable stress, MPa (carbon steel at 38°C)
        E = component.joint_efficiency
        D = component.diameter
        CA = component.corrosion_allowance
        
        required = (pressure * D * safety_factor) / (2 * S * E) + CA
        
        return max(3.0, required)  # Minimum 3mm
    
    def _calculate_life_confidence(self,
                                  component: ComponentGeometry,
                                  flaw: Optional[FlawGeometry],
                                  crack: Optional[CrackGeometry],
                                  loading: LoadingConditions) -> float:
        """Calculate confidence in remaining life prediction"""
        confidence = 1.0
        
        # Reduce confidence based on uncertainties
        if component.thickness <= 0:
            confidence *= 0.5
        
        if loading.cycles_per_year == 0 and loading.temperature > 400:
            confidence *= 0.7  # High temp but no creep data
        
        if flaw and flaw.depth <= 0.1:  # Very small flaw
            confidence *= 0.9
        
        if crack and crack.depth <= 0.5:  # Small crack
            confidence *= 0.8
        
        # Increase confidence for good data
        if loading.cycles_per_year > 0:
            confidence *= 1.1  # Have cyclic loading data
        
        if loading.temperature > 0:
            confidence *= 1.05  # Have temperature data
        
        return max(0.1, min(1.0, confidence))
    
    def generate_repair_plan(self,
                           asset_id: str,
                           component: ComponentGeometry,
                           flaw: FlawGeometry,
                           crack: Optional[CrackGeometry] = None) -> Dict[str, Any]:
        """
        Generate repair plan based on FFS assessment
        
        Args:
            asset_id: Asset identifier
            component: Component geometry
            flaw: Primary flaw
            crack: Optional crack
        
        Returns:
            Repair plan dictionary
        """
        
        repair_methods = []
        estimated_costs = []
        durations = []
        
        # Determine repair method based on flaw type
        if flaw.flaw_type == FlawType.PITTING:
            if flaw.depth < component.thickness * 0.2:
                repair_methods.append("Grind and blend")
                estimated_costs.append(5000)  # USD
                durations.append(2)  # days
            else:
                repair_methods.append("Weld repair")
                estimated_costs.append(15000)
                durations.append(5)
        
        elif flaw.flaw_type == FlawType.LOCAL_THINNING:
            if flaw.depth < component.thickness * 0.3:
                repair_methods.append("Weld overlay")
                estimated_costs.append(20000)
                durations.append(7)
            else:
                repair_methods.append("Section replacement")
                estimated_costs.append(50000)
                durations.append(14)
        
        elif flaw.flaw_type == FlawType.CRACK_LIKE or crack:
            repair_methods.append("Crack removal by grinding")
            estimated_costs.append(10000)
            durations.append(3)
            
            repair_methods.append("Weld repair with post-weld heat treatment")
            estimated_costs.append(25000)
            durations.append(10)
        
        else:  # General corrosion
            repair_methods.append("Corrosion inhibitor application")
            estimated_costs.append(3000)
            durations.append(1)
            
            repair_methods.append("Cathodic protection installation")
            estimated_costs.append(20000)
            durations.append(5)
        
        # Select optimal repair method (lowest cost for acceptable result)
        optimal_index = 0  # Default to first method
        
        plan = {
            "asset_id": asset_id,
            "flaw_type": flaw.flaw_type.value,
            "repair_method": repair_methods[optimal_index],
            "estimated_cost_usd": estimated_costs[optimal_index],
            "estimated_duration_days": durations[optimal_index],
            "alternatives": [
                {"method": m, "cost": c, "duration": d}
                for m, c, d in zip(repair_methods[1:], estimated_costs[1:], durations[1:])
            ],
            "prerequisites": ["Safety permit", "Isolation", "Cleaning"],
            "post_repair_actions": ["NDT inspection", "Hydrotest", "Documentation"],
            "generated_date": datetime.now().isoformat()
        }
        
        return plan
    
    def get_ffs_statistics(self) -> Dict[str, Any]:
        """Get FFS system statistics"""
        stats = {
            "total_assessments": sum(len(v) for v in self.assessment_history.values()),
            "assets_assessed": len(self.assessment_history),
            "crack_growth_predictions": sum(len(v) for v in self.crack_growth_history.values()),
            "unacceptable_assessments": sum(
                1 for assessments in self.assessment_history.values()
                for a in assessments
                if not a.is_acceptable
            ),
            "critical_cracks": sum(
                1 for predictions in self.crack_growth_history.values()
                for p in predictions
                if p.is_critical()
            ),
            "common_flaw_types": {},
            "recent_activity": {
                "last_24h": 0,
                "last_7d": 0,
                "last_30d": 0
            }
        }
        
        # Count flaw types
        for assessments in self.assessment_history.values():
            for assessment in assessments:
                flaw_type = assessment.flaw_type.value
                stats["common_flaw_types"][flaw_type] = \
                    stats["common_flaw_types"].get(flaw_type, 0) + 1
        
        return stats
    
    def export_ffs_report(self, asset_id: str) -> Optional[Dict[str, Any]]:
        """Export comprehensive FFS report for an asset"""
        if asset_id not in self.assessment_history:
            return None
        
        assessments = self.assessment_history[asset_id]
        crack_predictions = self.crack_growth_history.get(asset_id, [])
        
        # Get latest assessment
        latest_assessment = assessments[-1] if assessments else None
        latest_crack = crack_predictions[-1] if crack_predictions else None
        
        report = {
            "asset_id": asset_id,
            "report_date": datetime.now().isoformat(),
            "assessment_summary": {
                "total_assessments": len(assessments),
                "latest_assessment": latest_assessment.assessment_date.isoformat() if latest_assessment else None,
                "latest_rsf": latest_assessment.remaining_strength_factor if latest_assessment else None,
                "status": "ACCEPTABLE" if latest_assessment and latest_assessment.is_acceptable else "UNACCEPTABLE",
                "criticality": latest_assessment.get_criticality() if latest_assessment else "unknown"
            },
            "crack_growth_summary": {
                "total_predictions": len(crack_predictions),
                "latest_prediction": latest_crack.years_to_failure if latest_crack else None,
                "latest_critical": latest_crack.is_critical() if latest_crack else False
            } if crack_predictions else None,
            "recommendations": latest_assessment.recommendations if latest_assessment else [],
            "next_actions": [
                f"Next assessment: {latest_assessment.next_assessment_date.isoformat()}" if latest_assessment else "Schedule initial assessment",
                "Review corrosion protection" if latest_assessment and latest_assessment.remaining_strength_factor < 0.9 else "Continue routine monitoring"
            ]
        }
        
        return report

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing FFS/FIT Analysis Module")
    print("="*50)
    
    # Initialize analyzer
    analyzer = FFSFitAnalyzer()
    
    # Test 1: FFS Assessment for General Corrosion
    print("\n1️⃣ TEST: FFS Assessment - General Corrosion")
    print("-"*30)
    
    component = ComponentGeometry(
        diameter=1000.0,
        thickness=25.0,
        length=5000.0,
        corrosion_allowance=3.0
    )
    
    flaw = FlawGeometry(
        flaw_type=FlawType.GENERAL_CORROSION,
        length=500.0,
        depth=5.0  # 5mm general corrosion
    )
    
    loading = LoadingConditions(
        internal_pressure=2.0,  # 20 bar = 2 MPa
        temperature=85.0,
        cycles_per_year=1000
    )
    
    result = analyzer.perform_ffs_assessment(
        asset_id="vessel_001",
        component=component,
        material=MaterialCategory.CARBON_STEEL,
        flaw=flaw,
        loading=loading,
        assessment_level=FFSLevel.LEVEL_2
    )
    
    print(f"RSF: {result.remaining_strength_factor:.3f}")
    print(f"Acceptable: {result.is_acceptable}")
    print(f"Criticality: {result.get_criticality()}")
    print(f"MAWP Reduction: {result.mawp_reduction:.1f}%")
    print(f"Next Assessment: {result.next_assessment_date.date()}")
    print(f"Recommendations: {result.recommendations[0]}")
    
    # Test 2: FFS Assessment for Local Thinning
    print("\n2️⃣ TEST: FFS Assessment - Local Thinning")
    print("-"*30)
    
    flaw2 = FlawGeometry(
        flaw_type=FlawType.LOCAL_THINNING,
        length=200.0,
        depth=8.0  # 8mm local thinning
    )
    
    result2 = analyzer.perform_ffs_assessment(
        asset_id="vessel_002",
        component=component,
        material=MaterialCategory.CARBON_STEEL,
        flaw=flaw2,
        loading=loading,
        assessment_level=FFSLevel.LEVEL_2
    )
    
    print(f"RSF: {result2.remaining_strength_factor:.3f}")
    print(f"Acceptable: {result2.is_acceptable}")
    
    # Test 3: Crack Growth Prediction
    print("\n3️⃣ TEST: Crack Growth Prediction")
    print("-"*30)
    
    crack = CrackGeometry(
        crack_type=CrackType.SURFACE_CRACK,
        surface_length=50.0,  # 2c = 50mm
        depth=5.0  # a = 5mm
    )
    
    loading_crack = LoadingConditions(
        internal_pressure=2.0,
        temperature=85.0,
        cycles_per_year=10000,
        stress_ratio_R=0.1
    )
    
    growth_result = analyzer.predict_crack_growth(
        asset_id="vessel_003",
        component=component,
        material=MaterialCategory.CARBON_STEEL,
        crack=crack,
        loading=loading_crack
    )
    
    print(f"Initial crack size: {growth_result.initial_size:.2f} mm")
    print(f"Critical crack size: {growth_result.critical_crack_size:.2f} mm")
    print(f"Cycles to failure: {growth_result.cycles_to_failure:,}")
    print(f"Years to failure: {growth_result.years_to_failure:.1f}")
    print(f"Failure mode: {growth_result.failure_mode}")
    print(f"Confidence: {growth_result.confidence:.2%}")
    
    # Test 4: Remaining Life Calculation
    print("\n4️⃣ TEST: Remaining Life Calculation")
    print("-"*30)
    
    remaining_life = analyzer.calculate_remaining_life_ffs(
        asset_id="vessel_004",
        component=component,
        material=MaterialCategory.CARBON_STEEL,
        loading=loading_crack,
        flaw=flaw,
        crack=crack
    )
    
    print(f"Remaining life: {remaining_life.years} years, "
          f"{remaining_life.months} months, {remaining_life.days} days")
    print(f"Limiting factor: {remaining_life.limiting_factor}")
    print(f"Confidence: {remaining_life.confidence:.2%}")
    print(f"Repair recommended: {remaining_life.repair_recommended}")
    print(f"Repair urgency: {remaining_life.repair_urgency}")
    print(f"Next inspection: {remaining_life.next_inspection.date()}")
    
    # Test 5: Repair Plan Generation
    print("\n5️⃣ TEST: Repair Plan Generation")
    print("-"*30)
    
    repair_plan = analyzer.generate_repair_plan(
        asset_id="vessel_005",
        component=component,
        flaw=flaw2,
        crack=crack
    )
    
    print(f"Repair method: {repair_plan['repair_method']}")
    print(f"Estimated cost: ${repair_plan['estimated_cost_usd']:,}")
    print(f"Estimated duration: {repair_plan['estimated_duration_days']} days")
    print(f"Prerequisites: {', '.join(repair_plan['prerequisites'][:2])}")
    
    # Test 6: System Statistics
    print("\n6️⃣ TEST: System Statistics")
    print("-"*30)
    
    stats = analyzer.get_ffs_statistics()
    print(f"Total assessments: {stats['total_assessments']}")
    print(f"Assets assessed: {stats['assets_assessed']}")
    print(f"Unacceptable assessments: {stats['unacceptable_assessments']}")
    print(f"Critical cracks: {stats['critical_cracks']}")
    
    if stats['common_flaw_types']:
        print("Common flaw types:")
        for flaw_type, count in stats['common_flaw_types'].items():
            print(f"  {flaw_type}: {count}")
    
    # Test 7: Report Generation
    print("\n7️⃣ TEST: Report Generation")
    print("-"*30)
    
    report = analyzer.export_ffs_report("vessel_001")
    if report:
        print(f"Report for {report['asset_id']}")
        summary = report['assessment_summary']
        print(f"Status: {summary['status']}")
        print(f"Criticality: {summary['criticality']}")
        print(f"Recommendations: {len(report['recommendations'])}")
    else:
        print("No report available")
    
    print(f"\n✅ FFS/FIT Analysis tests complete!")