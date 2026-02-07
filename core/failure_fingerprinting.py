"""
🔍 FAILURE FINGERPRINTING MODULE
Identifies and classifies failure patterns from sensor data
Uses pattern matching and machine learning for failure detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

@dataclass
class FailureSignature:
    """Represents a failure pattern signature"""
    name: str
    description: str
    sensor_pattern: Dict[str, float]  # Expected sensor values for this failure
    weights: Dict[str, float]  # Importance of each sensor for this failure
    thresholds: Dict[str, Tuple[float, float]]  # Min/max values for each sensor
    severity: str  # low, medium, high, critical
    typical_assets: List[str]  # Asset types where this failure is common
    learned_from: List[str] = field(default_factory=list)  # Assets that contributed to learning
    
    def to_vector(self, sensor_order: List[str]) -> np.ndarray:
        """Convert signature to vector for comparison"""
        vector = []
        for sensor in sensor_order:
            vector.append(self.sensor_pattern.get(sensor, 0.0))
        return np.array(vector)

@dataclass
class FingerprintMatch:
    """Result of fingerprint matching"""
    failure_type: str
    confidence: float  # 0-1
    similarity_scores: Dict[str, float]  # Per-sensor similarity
    severity: str
    suggested_actions: List[str]
    time_to_failure: Optional[float] = None  # Estimated days to failure
    risk_score: float = 0.0  # 0-100 risk score

class FailureFingerprinter:
    """
    Main fingerprinting engine that:
    1. Maintains library of known failure patterns
    2. Matches current sensor data against patterns
    3. Learns new patterns from detected failures
    4. Provides confidence scores and recommendations
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.signatures: Dict[str, FailureSignature] = {}
        self.learned_patterns: List[FailureSignature] = []
        self.sensor_order = ["vibration", "temperature", "acoustic", "pressure", "strain", "corrosion"]
        
        # Machine learning components
        self.scaler = StandardScaler()
        self.cluster_model = DBSCAN(eps=0.5, min_samples=3)
        
        # Statistics
        self.detection_history = []
        self.confusion_matrix = {}  # For accuracy tracking
        
        # Initialize with known failure patterns
        self._initialize_known_failures()
        
        # Load previously learned patterns
        self._load_learned_patterns()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _initialize_known_failures(self):
        """Initialize with industry-standard failure patterns"""
        
        # Bearing Failure Signature
        bearing_failure = FailureSignature(
            name="bearing_failure",
            description="Rolling element bearing wear or fatigue",
            sensor_pattern={
                "vibration": 3.8,  # High vibration (mm/s)
                "temperature": 92.0,  # Elevated temperature (°C)
                "acoustic": 88.0,  # High acoustic emission (dB)
                "pressure": 10.2,  # Normal pressure
                "strain": 180.0,  # Moderate strain
                "corrosion": 0.0  # Not applicable
            },
            weights={
                "vibration": 0.35,
                "temperature": 0.25,
                "acoustic": 0.25,
                "pressure": 0.05,
                "strain": 0.10,
                "corrosion": 0.0
            },
            thresholds={
                "vibration": (2.5, 20.0),
                "temperature": (85.0, 120.0),
                "acoustic": (75.0, 120.0)
            },
            severity="high",
            typical_assets=["centrifugal_pump", "motor", "compressor", "fan"]
        )
        
        # Shaft Misalignment Signature
        misalignment = FailureSignature(
            name="shaft_misalignment",
            description="Shaft or coupling misalignment",
            sensor_pattern={
                "vibration": 2.8,
                "temperature": 82.0,
                "acoustic": 72.0,
                "pressure": 10.0,
                "strain": 240.0,  # High strain due to bending
                "corrosion": 0.0
            },
            weights={
                "vibration": 0.30,
                "temperature": 0.15,
                "acoustic": 0.10,
                "pressure": 0.10,
                "strain": 0.35,
                "corrosion": 0.0
            },
            thresholds={
                "vibration": (2.0, 10.0),
                "strain": (200.0, 500.0)
            },
            severity="medium",
            typical_assets=["centrifugal_pump", "compressor", "turbine", "gear_box"]
        )
        
        # Cavitation Signature
        cavitation = FailureSignature(
            name="cavitation",
            description="Pump cavitation causing vibration and noise",
            sensor_pattern={
                "vibration": 2.5,
                "temperature": 78.0,
                "acoustic": 95.0,  # Very high acoustic
                "pressure": 6.5,  # Low pressure
                "strain": 130.0,
                "corrosion": 0.0
            },
            weights={
                "vibration": 0.20,
                "temperature": 0.10,
                "acoustic": 0.40,
                "pressure": 0.20,
                "strain": 0.10,
                "corrosion": 0.0
            },
            thresholds={
                "acoustic": (85.0, 120.0),
                "pressure": (4.0, 8.0)
            },
            severity="medium",
            typical_assets=["centrifugal_pump", "positive_displacement_pump"]
        )
        
        # Corrosion Signature
        corrosion = FailureSignature(
            name="corrosion",
            description="Material corrosion and thickness loss",
            sensor_pattern={
                "vibration": 1.5,
                "temperature": 85.0,
                "acoustic": 68.0,
                "pressure": 8.0,
                "strain": 200.0,  # Increased strain due to wall thinning
                "corrosion": 1.8  # Corrosion rate in mm/year
            },
            weights={
                "vibration": 0.10,
                "temperature": 0.20,
                "acoustic": 0.10,
                "pressure": 0.20,
                "strain": 0.20,
                "corrosion": 0.20
            },
            thresholds={
                "corrosion": (0.5, 5.0),
                "strain": (150.0, 400.0)
            },
            severity="high",
            typical_assets=["pressure_vessel", "storage_tank", "heat_exchanger", "piping"]
        )
        
        # Unbalance Signature
        unbalance = FailureSignature(
            name="rotor_unbalance",
            description="Rotating component unbalance",
            sensor_pattern={
                "vibration": 3.2,
                "temperature": 75.0,
                "acoustic": 70.0,
                "pressure": 10.0,
                "strain": 160.0,
                "corrosion": 0.0
            },
            weights={
                "vibration": 0.45,
                "temperature": 0.15,
                "acoustic": 0.20,
                "pressure": 0.10,
                "strain": 0.10,
                "corrosion": 0.0
            },
            thresholds={
                "vibration": (2.0, 15.0)
            },
            severity="low",
            typical_assets=["centrifugal_pump", "motor", "fan", "compressor"]
        )
        
        # Add all signatures
        self.signatures = {
            "bearing_failure": bearing_failure,
            "misalignment": misalignment,
            "cavitation": cavitation,
            "corrosion": corrosion,
            "rotor_unbalance": unbalance
        }
        
        print(f"📚 Initialized {len(self.signatures)} known failure signatures")
    
    def _load_learned_patterns(self):
        """Load previously learned patterns from file"""
        try:
            with open("data/models/learned_patterns.pkl", "rb") as f:
                self.learned_patterns = pickle.load(f)
            print(f"📖 Loaded {len(self.learned_patterns)} learned patterns")
        except FileNotFoundError:
            print("No learned patterns found. Starting fresh.")
    
    def _save_learned_patterns(self):
        """Save learned patterns to file"""
        with open("data/models/learned_patterns.pkl", "wb") as f:
            pickle.dump(self.learned_patterns, f)
    
    def normalize_sensor_data(self, sensor_data: Dict[str, float]) -> Dict[str, float]:
        """Normalize sensor data to 0-1 range"""
        normalized = {}
        
        # Normalization ranges (min, max) for each sensor
        ranges = {
            "vibration": (0, 10),     # 0-10 mm/s
            "temperature": (20, 120),  # 20-120°C
            "acoustic": (30, 100),     # 30-100 dB
            "pressure": (0, 20),       # 0-20 bar
            "strain": (0, 500),        # 0-500 με
            "corrosion": (0, 5)        # 0-5 mm/year
        }
        
        for sensor, value in sensor_data.items():
            if sensor in ranges:
                min_val, max_val = ranges[sensor]
                # Clamp value to range
                clamped = max(min_val, min(max_val, value))
                # Normalize to 0-1
                normalized[sensor] = (clamped - min_val) / (max_val - min_val)
            else:
                normalized[sensor] = 0.0
        
        return normalized
    
    def extract_features(self, sensor_data: Dict[str, float], window_size: int = 10) -> Dict[str, float]:
        """Extract statistical features from sensor data"""
        # For now, use simple features. In production, this would use historical data
        features = {}
        
        # Basic statistics
        for sensor, value in sensor_data.items():
            features[f"{sensor}_value"] = value
        
        # Rate of change features (if historical data available)
        # This is simplified - real implementation would use time-series data
        
        return features
    
    def calculate_similarity(self, current_data: Dict[str, float], 
                           signature: FailureSignature) -> Dict[str, float]:
        """Calculate similarity between current data and failure signature"""
        similarities = {}
        
        for sensor, weight in signature.weights.items():
            if sensor in current_data and weight > 0:
                current_val = current_data[sensor]
                pattern_val = signature.sensor_pattern.get(sensor, 0)
                
                # Calculate similarity (1 - normalized difference)
                max_val = max(current_val, pattern_val)
                if max_val > 0:
                    similarity = 1 - abs(current_val - pattern_val) / max_val
                else:
                    similarity = 1.0
                
                # Apply sensor-specific weighting
                weighted_similarity = similarity * weight
                similarities[sensor] = weighted_similarity
        
        return similarities
    
    def detect_failure(self, sensor_data: Dict[str, float], 
                      asset_type: str, 
                      asset_id: str = None) -> Optional[FingerprintMatch]:
        """
        Main method to detect failures from sensor data
        
        Args:
            sensor_data: Dictionary of sensor readings
            asset_type: Type of asset (e.g., "centrifugal_pump")
            asset_id: Optional asset ID for learning
            
        Returns:
            FingerprintMatch object if failure detected, None otherwise
        """
        
        # Normalize sensor data
        normalized_data = self.normalize_sensor_data(sensor_data)
        
        # Extract features
        features = self.extract_features(normalized_data)
        
        best_match = None
        best_confidence = 0.0
        
        # Check against known signatures
        for signature_name, signature in self.signatures.items():
            # Skip if this failure is not typical for this asset type
            if asset_type not in signature.typical_assets:
                continue
            
            # Calculate similarity
            similarities = self.calculate_similarity(normalized_data, signature)
            
            # Calculate overall confidence (weighted average)
            total_weight = sum(signature.weights.values())
            if total_weight > 0:
                total_similarity = sum(similarities.values())
                confidence = total_similarity / total_weight
            else:
                confidence = 0.0
            
            # Apply threshold checks
            confidence = self._apply_thresholds(confidence, normalized_data, signature)
            
            # Update best match
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = (signature_name, signature, similarities, confidence)
        
        # Check against learned patterns
        for learned_pattern in self.learned_patterns:
            if asset_type in learned_pattern.typical_assets:
                similarities = self.calculate_similarity(normalized_data, learned_pattern)
                
                total_weight = sum(learned_pattern.weights.values())
                if total_weight > 0:
                    confidence = sum(similarities.values()) / total_weight
                    
                    # Boost confidence for learned patterns from similar assets
                    if asset_id in learned_pattern.learned_from:
                        confidence *= 1.1  # 10% boost
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = (f"learned_{learned_pattern.name}", learned_pattern, 
                                    similarities, confidence)
        
        # Only return if confidence exceeds threshold
        confidence_threshold = self.config["analytics"]["failure_detection"]["confidence_threshold"]
        
        if best_match and best_confidence >= confidence_threshold:
            signature_name, signature, similarities, confidence = best_match
            
            # Calculate risk score (0-100)
            risk_score = self._calculate_risk_score(confidence, signature.severity)
            
            # Estimate time to failure
            time_to_failure = self._estimate_time_to_failure(normalized_data, signature)
            
            # Get suggested actions
            suggested_actions = self._get_suggested_actions(signature_name, confidence)
            
            # Create fingerprint match
            match = FingerprintMatch(
                failure_type=signature_name,
                confidence=confidence,
                similarity_scores=similarities,
                severity=signature.severity,
                suggested_actions=suggested_actions,
                time_to_failure=time_to_failure,
                risk_score=risk_score
            )
            
            # Record detection for learning
            self._record_detection(asset_id, signature_name, confidence, sensor_data)
            
            # Learn from this detection if confidence is high
            if confidence > 0.8 and asset_id:
                self._learn_from_detection(asset_id, asset_type, normalized_data, signature_name)
            
            return match
        
        return None
    
    def _apply_thresholds(self, confidence: float, 
                         sensor_data: Dict[str, float],
                         signature: FailureSignature) -> float:
        """Apply sensor-specific thresholds to confidence score"""
        
        adjusted_confidence = confidence
        
        for sensor, (min_val, max_val) in signature.thresholds.items():
            if sensor in sensor_data:
                value = sensor_data[sensor]
                
                # Check if value is within expected range for this failure
                if value < min_val or value > max_val:
                    # Reduce confidence if outside expected range
                    adjustment = 0.7  # Reduce by 30%
                    adjusted_confidence *= adjustment
        
        return adjusted_confidence
    
    def _calculate_risk_score(self, confidence: float, severity: str) -> float:
        """Calculate risk score (0-100)"""
        severity_weights = {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.8,
            "critical": 1.0
        }
        
        severity_weight = severity_weights.get(severity, 0.5)
        risk_score = confidence * severity_weight * 100
        
        return min(100, risk_score)
    
    def _estimate_time_to_failure(self, sensor_data: Dict[str, float], 
                                 signature: FailureSignature) -> Optional[float]:
        """Estimate time to failure in days (simplified)"""
        
        # This is a simplified estimation. Real implementation would use:
        # 1. Historical degradation rates
        # 2. Physics-based models
        # 3. Machine learning predictions
        
        if signature.name == "bearing_failure":
            # Simple model based on vibration and temperature
            vibration = sensor_data.get("vibration", 0)
            temperature = sensor_data.get("temperature", 0)
            
            if vibration > 0.5 and temperature > 0.6:  # Normalized values
                # Estimate based on how far from normal
                vibration_factor = min(1.0, vibration / 0.8)  # 0.8 is threshold
                temperature_factor = min(1.0, temperature / 0.7)
                
                # Combined factor (0-1, where 1 is imminent failure)
                combined_factor = (vibration_factor + temperature_factor) / 2
                
                # Convert to days (30 days at 0.1, 1 day at 0.9)
                days_to_failure = 30 * (1 - combined_factor)
                return max(1, days_to_failure)
        
        elif signature.name == "corrosion":
            corrosion_rate = sensor_data.get("corrosion", 0)
            if corrosion_rate > 0.2:  # Normalized
                # Assuming remaining thickness based on corrosion rate
                remaining_life_years = (1.0 - corrosion_rate) / corrosion_rate
                return remaining_life_years * 365  # Convert to days
        
        return None
    
    def _get_suggested_actions(self, failure_type: str, confidence: float) -> List[str]:
        """Get suggested maintenance actions for detected failure"""
        
        actions = {
            "bearing_failure": [
                "Increase vibration monitoring frequency",
                "Check lubrication levels and quality",
                "Schedule bearing inspection",
                "Prepare for bearing replacement"
            ],
            "misalignment": [
                "Perform laser alignment check",
                "Inspect coupling condition",
                "Check foundation bolts",
                "Schedule alignment correction"
            ],
            "cavitation": [
                "Check pump suction conditions",
                "Verify NPSH available vs required",
                "Inspect impeller for damage",
                "Consider pump speed reduction"
            ],
            "corrosion": [
                "Perform ultrasonic thickness testing",
                "Review corrosion protection system",
                "Schedule internal inspection",
                "Consider corrosion inhibitor addition"
            ],
            "rotor_unbalance": [
                "Perform dynamic balancing",
                "Clean rotor surfaces",
                "Check for material buildup",
                "Inspect for missing/blades"
            ]
        }
        
        base_actions = actions.get(failure_type, ["Increase monitoring frequency"])
        
        # Adjust actions based on confidence
        if confidence > 0.9:
            base_actions.append("Consider immediate shutdown")
        elif confidence > 0.7:
            base_actions.append("Schedule maintenance within 7 days")
        else:
            base_actions.append("Monitor closely for progression")
        
        return base_actions
    
    def _record_detection(self, asset_id: str, failure_type: str, 
                         confidence: float, sensor_data: Dict[str, float]):
        """Record detection for accuracy tracking and learning"""
        detection = {
            "timestamp": datetime.now().isoformat(),
            "asset_id": asset_id,
            "failure_type": failure_type,
            "confidence": confidence,
            "sensor_data": sensor_data
        }
        
        self.detection_history.append(detection)
        
        # Keep only last 1000 detections
        if len(self.detection_history) > 1000:
            self.detection_history = self.detection_history[-1000:]
    
    def _learn_from_detection(self, asset_id: str, asset_type: str,
                             sensor_data: Dict[str, float], failure_type: str):
        """Learn new patterns from detected failures"""
        
        # Check if we should create a new learned pattern
        # Look for similar detections
        similar_detections = []
        for detection in self.detection_history[-100:]:  # Last 100 detections
            if detection["failure_type"] == failure_type:
                # Calculate similarity
                similarity = self._calculate_data_similarity(sensor_data, detection["sensor_data"])
                if similarity > 0.8:
                    similar_detections.append(detection)
        
        # If we have at least 3 similar detections, create/update learned pattern
        if len(similar_detections) >= 3:
            # Calculate average sensor values
            avg_sensor_data = self._calculate_average_sensor_data(similar_detections)
            
            # Create or update learned pattern
            pattern_name = f"{failure_type}_pattern_{len(self.learned_patterns) + 1}"
            
            # Check if pattern already exists
            existing_pattern = None
            for pattern in self.learned_patterns:
                pattern_similarity = self._calculate_data_similarity(
                    avg_sensor_data, pattern.sensor_pattern
                )
                if pattern_similarity > 0.9:
                    existing_pattern = pattern
                    break
            
            if existing_pattern:
                # Update existing pattern
                existing_pattern.learned_from.append(asset_id)
                print(f"📚 Updated learned pattern from {asset_id}")
            else:
                # Create new pattern
                new_pattern = FailureSignature(
                    name=pattern_name,
                    description=f"Learned {failure_type} pattern",
                    sensor_pattern=avg_sensor_data,
                    weights=self.signatures.get(failure_type, self.signatures["bearing_failure"]).weights,
                    thresholds=self.signatures.get(failure_type, self.signatures["bearing_failure"]).thresholds,
                    severity=self.signatures.get(failure_type, self.signatures["bearing_failure"]).severity,
                    typical_assets=[asset_type],
                    learned_from=[asset_id]
                )
                
                self.learned_patterns.append(new_pattern)
                print(f"📚 Learned new {failure_type} pattern from {asset_id}")
                
                # Save learned patterns
                self._save_learned_patterns()
    
    def _calculate_data_similarity(self, data1: Dict[str, float], 
                                  data2: Dict[str, float]) -> float:
        """Calculate similarity between two sensor data dictionaries"""
        common_sensors = set(data1.keys()) & set(data2.keys())
        if not common_sensors:
            return 0.0
        
        similarities = []
        for sensor in common_sensors:
            val1 = data1[sensor]
            val2 = data2[sensor]
            max_val = max(abs(val1), abs(val2))
            if max_val > 0:
                similarity = 1 - abs(val1 - val2) / max_val
                similarities.append(max(0, similarity))
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_average_sensor_data(self, detections: List[Dict]) -> Dict[str, float]:
        """Calculate average sensor data from multiple detections"""
        sensor_sums = {}
        sensor_counts = {}
        
        for detection in detections:
            sensor_data = detection["sensor_data"]
            for sensor, value in sensor_data.items():
                sensor_sums[sensor] = sensor_sums.get(sensor, 0) + value
                sensor_counts[sensor] = sensor_counts.get(sensor, 0) + 1
        
        avg_data = {}
        for sensor, total in sensor_sums.items():
            avg_data[sensor] = total / sensor_counts[sensor]
        
        return avg_data
    
    def get_failure_statistics(self) -> Dict[str, Any]:
        """Get statistics about detected failures"""
        stats = {
            "total_detections": len(self.detection_history),
            "failure_counts": {},
            "average_confidence": 0.0,
            "recent_detections": []
        }
        
        # Count failures by type
        for detection in self.detection_history:
            failure_type = detection["failure_type"]
            stats["failure_counts"][failure_type] = stats["failure_counts"].get(failure_type, 0) + 1
        
        # Calculate average confidence
        if self.detection_history:
            total_confidence = sum(d["confidence"] for d in self.detection_history)
            stats["average_confidence"] = total_confidence / len(self.detection_history)
        
        # Get recent detections
        stats["recent_detections"] = self.detection_history[-10:] if self.detection_history else []
        
        return stats
    
    def export_signatures(self, filepath: str = "data/models/failure_signatures.json"):
        """Export failure signatures to JSON file"""
        export_data = {
            "known_signatures": {},
            "learned_patterns": []
        }
        
        # Export known signatures
        for name, signature in self.signatures.items():
            export_data["known_signatures"][name] = {
                "description": signature.description,
                "sensor_pattern": signature.sensor_pattern,
                "weights": signature.weights,
                "thresholds": signature.thresholds,
                "severity": signature.severity,
                "typical_assets": signature.typical_assets
            }
        
        # Export learned patterns
        for pattern in self.learned_patterns:
            export_data["learned_patterns"].append({
                "name": pattern.name,
                "description": pattern.description,
                "sensor_pattern": pattern.sensor_pattern,
                "weights": pattern.weights,
                "thresholds": pattern.thresholds,
                "severity": pattern.severity,
                "typical_assets": pattern.typical_assets,
                "learned_from": pattern.learned_from
            })
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"📤 Exported signatures to {filepath}")

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Failure Fingerprinting Module")
    print("="*50)
    
    # Initialize fingerprinter
    fingerprinter = FailureFingerprinter()
    
    # Test data simulating a bearing failure
    test_sensor_data = {
        "vibration": 4.2,  # High vibration
        "temperature": 95.0,  # High temperature
        "acoustic": 90.0,  # High acoustic
        "pressure": 10.5,  # Normal pressure
        "strain": 175.0,  # Moderate strain
        "corrosion": 0.0  # No corrosion
    }
    
    print("\n📊 Test Sensor Data:")
    for sensor, value in test_sensor_data.items():
        print(f"  {sensor}: {value}")
    
    # Detect failure
    match = fingerprinter.detect_failure(
        sensor_data=test_sensor_data,
        asset_type="centrifugal_pump",
        asset_id="pump_001"
    )
    
    if match:
        print(f"\n🔍 Failure Detected:")
        print(f"  Type: {match.failure_type}")
        print(f"  Confidence: {match.confidence:.2%}")
        print(f"  Severity: {match.severity}")
        print(f"  Risk Score: {match.risk_score:.1f}/100")
        
        if match.time_to_failure:
            print(f"  Estimated Time to Failure: {match.time_to_failure:.1f} days")
        
        print(f"\n📈 Sensor Similarities:")
        for sensor, similarity in match.similarity_scores.items():
            print(f"  {sensor}: {similarity:.2%}")
        
        print(f"\n🚨 Suggested Actions:")
        for i, action in enumerate(match.suggested_actions, 1):
            print(f"  {i}. {action}")
    else:
        print("\n✅ No failure detected")
    
    # Show statistics
    stats = fingerprinter.get_failure_statistics()
    print(f"\n📊 Statistics:")
    print(f"  Total Detections: {stats['total_detections']}")
    print(f"  Average Confidence: {stats['average_confidence']:.2%}")
    
    print(f"\n✅ Testing complete!")