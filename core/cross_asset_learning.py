"""
🤝 CROSS-ASSET LEARNING MODULE
Shares failure knowledge across similar assets
Prevents failures by applying lessons learned from other assets
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import pickle
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

@dataclass
class AssetProfile:
    """Profile of an asset with its characteristics and history"""
    asset_id: str
    asset_type: str
    manufacturer: str
    model: str
    installation_date: datetime
    operational_parameters: Dict[str, float]  # RPM, pressure, temperature ranges, etc.
    failure_history: List[Dict] = field(default_factory=list)
    maintenance_history: List[Dict] = field(default_factory=list)
    sensor_baselines: Dict[str, float] = field(default_factory=dict)
    similarity_scores: Dict[str, float] = field(default_factory=dict)  # Similarity to other assets
    
    def add_failure(self, failure_data: Dict):
        """Add a failure event to history"""
        self.failure_history.append({
            "timestamp": datetime.now().isoformat(),
            **failure_data
        })
    
    def get_recent_failures(self, days: int = 90) -> List[Dict]:
        """Get failures from the last N days"""
        cutoff = datetime.now() - timedelta(days=days)
        return [
            f for f in self.failure_history
            if datetime.fromisoformat(f["timestamp"]) > cutoff
        ]
    
    def calculate_asset_vector(self) -> np.ndarray:
        """Convert asset characteristics to numerical vector for similarity comparison"""
        vector = []
        
        # Encode asset type (one-hot like)
        asset_types = ["centrifugal_pump", "positive_displacement_pump", 
                      "compressor", "motor", "pressure_vessel", "heat_exchanger"]
        type_vector = [1 if self.asset_type == t else 0 for t in asset_types]
        vector.extend(type_vector)
        
        # Operational parameters (normalized)
        if "rpm" in self.operational_parameters:
            vector.append(min(self.operational_parameters["rpm"] / 10000, 1.0))
        else:
            vector.append(0.0)
        
        if "pressure" in self.operational_parameters:
            vector.append(min(self.operational_parameters["pressure"] / 100, 1.0))
        else:
            vector.append(0.0)
        
        if "temperature" in self.operational_parameters:
            vector.append(min(self.operational_parameters["temperature"] / 200, 1.0))
        else:
            vector.append(0.0)
        
        # Failure rate (failures per year)
        if self.installation_date:
            years_operating = (datetime.now() - self.installation_date).days / 365.25
            if years_operating > 0:
                failure_rate = len(self.failure_history) / years_operating
                vector.append(min(failure_rate / 10, 1.0))  # Normalize
            else:
                vector.append(0.0)
        else:
            vector.append(0.0)
        
        return np.array(vector)

@dataclass
class CrossAssetAlert:
    """Alert generated from cross-asset learning"""
    source_asset: str
    target_asset: str
    failure_type: str
    similarity_score: float
    predicted_risk: float  # 0-100
    evidence: List[str]  # Reasons for the alert
    recommended_actions: List[str]
    confidence: float  # 0-1
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "source_asset": self.source_asset,
            "target_asset": self.target_asset,
            "failure_type": self.failure_type,
            "similarity_score": self.similarity_score,
            "predicted_risk": self.predicted_risk,
            "evidence": self.evidence,
            "recommended_actions": self.recommended_actions,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class LearnedPattern:
    """Pattern learned from multiple assets"""
    pattern_id: str
    failure_type: str
    sensor_pattern: Dict[str, float]
    asset_types: List[str]
    learned_from: List[str]  # Asset IDs that contributed
    occurrence_count: int
    first_seen: datetime
    last_seen: datetime
    effectiveness: float  # How often this pattern correctly predicted failures
    
    def update(self, new_data: Dict[str, float], asset_id: str):
        """Update pattern with new data"""
        # Moving average update
        alpha = 0.1  # Learning rate
        for sensor, value in new_data.items():
            if sensor in self.sensor_pattern:
                current = self.sensor_pattern[sensor]
                self.sensor_pattern[sensor] = current * (1 - alpha) + value * alpha
            else:
                self.sensor_pattern[sensor] = value
        
        self.learned_from.append(asset_id)
        self.occurrence_count += 1
        self.last_seen = datetime.now()

class CrossAssetLearner:
    """
    Main cross-asset learning engine that:
    1. Builds profiles of all assets
    2. Calculates similarities between assets
    3. Transfers failure knowledge from similar assets
    4. Generates preventive alerts
    5. Learns patterns that apply across multiple assets
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.asset_profiles: Dict[str, AssetProfile] = {}
        self.learned_patterns: Dict[str, LearnedPattern] = {}
        self.asset_similarity_matrix: pd.DataFrame = None
        self.alerts_history: List[CrossAssetAlert] = []
        
        # Machine learning components
        self.scaler = StandardScaler()
        self.pattern_clusters = {}
        
        # Statistics
        self.knowledge_transfers = 0
        self.prevented_failures = 0
        self.false_positives = 0
        
        # Load existing data
        self._load_existing_data()
        
        print(f"🤝 Cross-Asset Learning initialized with {len(self.asset_profiles)} assets")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_existing_data(self):
        """Load previously saved data"""
        try:
            # Load asset profiles
            with open("data/models/asset_profiles.pkl", "rb") as f:
                self.asset_profiles = pickle.load(f)
            print(f"📖 Loaded {len(self.asset_profiles)} asset profiles")
        except FileNotFoundError:
            print("No existing asset profiles found")
        
        try:
            # Load learned patterns
            with open("data/models/cross_asset_patterns.pkl", "rb") as f:
                self.learned_patterns = pickle.load(f)
            print(f"📚 Loaded {len(self.learned_patterns)} cross-asset patterns")
        except FileNotFoundError:
            print("No existing cross-asset patterns found")
    
    def _save_data(self):
        """Save all data to files"""
        # Save asset profiles
        with open("data/models/asset_profiles.pkl", "wb") as f:
            pickle.dump(self.asset_profiles, f)
        
        # Save learned patterns
        with open("data/models/cross_asset_patterns.pkl", "wb") as f:
            pickle.dump(self.learned_patterns, f)
        
        # Save alerts history
        alerts_data = [alert.to_dict() for alert in self.alerts_history[-1000:]]
        with open("data/models/cross_asset_alerts.json", "w") as f:
            json.dump(alerts_data, f, indent=2, default=str)
    
    def register_asset(self, asset_id: str, asset_type: str, 
                      manufacturer: str, model: str,
                      installation_date: datetime,
                      operational_params: Dict[str, float]):
        """Register a new asset in the system"""
        if asset_id in self.asset_profiles:
            print(f"Asset {asset_id} already registered")
            return
        
        profile = AssetProfile(
            asset_id=asset_id,
            asset_type=asset_type,
            manufacturer=manufacturer,
            model=model,
            installation_date=installation_date,
            operational_parameters=operational_params
        )
        
        self.asset_profiles[asset_id] = profile
        print(f"📝 Registered new asset: {asset_id} ({asset_type})")
        
        # Update similarity matrix
        self._update_similarity_matrix()
        
        # Check for preventive alerts from similar assets
        self._check_preventive_alerts(asset_id)
    
    def record_failure(self, asset_id: str, failure_data: Dict):
        """
        Record a failure on an asset and share knowledge with similar assets
        
        Args:
            asset_id: Asset where failure occurred
            failure_data: Dictionary with failure details including:
                - failure_type: Type of failure
                - sensor_data: Sensor readings at failure
                - severity: Severity level
                - root_cause: Identified root cause (if known)
        """
        if asset_id not in self.asset_profiles:
            print(f"Asset {asset_id} not registered")
            return
        
        # Add to asset's failure history
        self.asset_profiles[asset_id].add_failure(failure_data)
        
        # Extract failure pattern
        failure_pattern = self._extract_failure_pattern(failure_data)
        
        # Share knowledge with similar assets
        self._share_knowledge(asset_id, failure_data["failure_type"], failure_pattern)
        
        # Learn cross-asset pattern
        self._learn_cross_asset_pattern(asset_id, failure_data["failure_type"], failure_pattern)
        
        # Update statistics
        print(f"📝 Recorded {failure_data['failure_type']} on {asset_id}")
        
        # Save data
        self._save_data()
    
    def record_maintenance(self, asset_id: str, maintenance_data: Dict):
        """Record maintenance performed on an asset"""
        if asset_id in self.asset_profiles:
            self.asset_profiles[asset_id].maintenance_history.append({
                "timestamp": datetime.now().isoformat(),
                **maintenance_data
            })
            print(f"🔧 Recorded maintenance on {asset_id}")
    
    def update_sensor_baselines(self, asset_id: str, sensor_data: Dict[str, float]):
        """Update normal operating baselines for an asset"""
        if asset_id in self.asset_profiles:
            profile = self.asset_profiles[asset_id]
            
            for sensor, value in sensor_data.items():
                if sensor not in profile.sensor_baselines:
                    profile.sensor_baselines[sensor] = value
                else:
                    # Moving average update
                    profile.sensor_baselines[sensor] = (
                        profile.sensor_baselines[sensor] * 0.9 + value * 0.1
                    )
    
    def _extract_failure_pattern(self, failure_data: Dict) -> Dict[str, float]:
        """Extract pattern from failure data"""
        pattern = {}
        
        # Include sensor data if available
        if "sensor_data" in failure_data:
            for sensor, value in failure_data["sensor_data"].items():
                pattern[f"sensor_{sensor}"] = value
        
        # Include metadata
        if "severity" in failure_data:
            severity_map = {"low": 0.3, "medium": 0.6, "high": 0.8, "critical": 1.0}
            pattern["severity"] = severity_map.get(failure_data["severity"], 0.5)
        
        if "root_cause" in failure_data:
            # Encode root cause as categorical
            causes = ["bearing", "alignment", "lubrication", "corrosion", "fatigue", "other"]
            pattern["root_cause"] = causes.index(failure_data["root_cause"]) / len(causes)
        
        return pattern
    
    def _update_similarity_matrix(self):
        """Update similarity matrix between all assets"""
        if len(self.asset_profiles) < 2:
            return
        
        asset_ids = list(self.asset_profiles.keys())
        vectors = []
        
        # Calculate vectors for all assets
        for asset_id in asset_ids:
            vector = self.asset_profiles[asset_id].calculate_asset_vector()
            vectors.append(vector)
        
        # Create similarity matrix
        vectors_array = np.array(vectors)
        
        # Handle different vector lengths
        max_len = max(v.shape[0] for v in vectors)
        padded_vectors = []
        for v in vectors_array:
            if len(v) < max_len:
                padded = np.pad(v, (0, max_len - len(v)), mode='constant')
                padded_vectors.append(padded)
            else:
                padded_vectors.append(v[:max_len])
        
        padded_vectors = np.array(padded_vectors)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(padded_vectors)
        
        # Create DataFrame
        self.asset_similarity_matrix = pd.DataFrame(
            similarity,
            index=asset_ids,
            columns=asset_ids
        )
        
        # Update similarity scores in profiles
        for i, asset_id in enumerate(asset_ids):
            profile = self.asset_profiles[asset_id]
            profile.similarity_scores = dict(zip(asset_ids, similarity[i]))
    
    def get_similar_assets(self, asset_id: str, 
                          min_similarity: float = 0.7,
                          limit: int = 5) -> List[Tuple[str, float]]:
        """Get assets similar to the given asset"""
        if asset_id not in self.asset_profiles or self.asset_similarity_matrix is None:
            return []
        
        similarities = self.asset_similarity_matrix[asset_id]
        similar = []
        
        for other_id, score in similarities.items():
            if other_id != asset_id and score >= min_similarity:
                similar.append((other_id, score))
        
        # Sort by similarity (descending)
        similar.sort(key=lambda x: x[1], reverse=True)
        
        return similar[:limit]
    
    def _share_knowledge(self, source_asset: str, 
                        failure_type: str, 
                        failure_pattern: Dict[str, float]):
        """Share failure knowledge with similar assets"""
        similar_assets = self.get_similar_assets(source_asset, min_similarity=0.6)
        
        for target_asset, similarity in similar_assets:
            # Don't share if target asset recently had same failure
            recent_failures = self.asset_profiles[target_asset].get_recent_failures(30)
            if any(f.get("failure_type") == failure_type for f in recent_failures):
                continue
            
            # Calculate risk based on similarity and failure severity
            risk_score = self._calculate_risk_score(similarity, failure_pattern)
            
            if risk_score > 50:  # Only alert for significant risk
                alert = self._create_preventive_alert(
                    source_asset, target_asset, failure_type, 
                    similarity, risk_score, failure_pattern
                )
                
                self.alerts_history.append(alert)
                self.knowledge_transfers += 1
                
                print(f"🔔 Shared knowledge: {source_asset} → {target_asset}")
                print(f"   Failure: {failure_type}, Similarity: {similarity:.2%}")
                print(f"   Risk Score: {risk_score:.1f}/100")
    
    def _calculate_risk_score(self, similarity: float, 
                            failure_pattern: Dict[str, float]) -> float:
        """Calculate risk score for preventive alert"""
        base_risk = similarity * 100
        
        # Adjust based on failure severity
        severity = failure_pattern.get("severity", 0.5)
        adjusted_risk = base_risk * (0.5 + severity * 0.5)
        
        # Adjust based on pattern confidence
        if "confidence" in failure_pattern:
            adjusted_risk *= failure_pattern["confidence"]
        
        return min(100, adjusted_risk)
    
    def _create_preventive_alert(self, source_asset: str, 
                                target_asset: str,
                                failure_type: str,
                                similarity: float,
                                risk_score: float,
                                pattern: Dict[str, float]) -> CrossAssetAlert:
        """Create a preventive alert for similar asset"""
        
        evidence = [
            f"Similar asset ({source_asset}) experienced {failure_type}",
            f"Asset similarity: {similarity:.2%}",
            f"Pattern confidence: {pattern.get('confidence', 0.7):.2%}"
        ]
        
        # Get recommended actions based on failure type
        actions = self._get_preventive_actions(failure_type, risk_score)
        
        # Calculate alert confidence
        confidence = min(1.0, similarity * 0.8 + 0.2)  # Base confidence on similarity
        
        alert = CrossAssetAlert(
            source_asset=source_asset,
            target_asset=target_asset,
            failure_type=failure_type,
            similarity_score=similarity,
            predicted_risk=risk_score,
            evidence=evidence,
            recommended_actions=actions,
            confidence=confidence
        )
        
        return alert
    
    def _get_preventive_actions(self, failure_type: str, risk_score: float) -> List[str]:
        """Get preventive actions based on failure type and risk"""
        
        base_actions = {
            "bearing_failure": [
                "Perform vibration analysis",
                "Check lubrication system",
                "Inspect bearing temperatures",
                "Review bearing maintenance history"
            ],
            "misalignment": [
                "Perform laser alignment check",
                "Inspect coupling condition",
                "Check foundation and baseplate",
                "Monitor vibration trends"
            ],
            "cavitation": [
                "Check pump suction conditions",
                "Verify NPSH margins",
                "Inspect impeller and casing",
                "Review operating parameters"
            ],
            "corrosion": [
                "Perform ultrasonic thickness testing",
                "Review corrosion protection system",
                "Check process chemistry",
                "Inspect for coating damage"
            ]
        }
        
        actions = base_actions.get(failure_type, [
            "Increase monitoring frequency",
            "Review maintenance history",
            "Check operational parameters"
        ])
        
        # Add urgency based on risk
        if risk_score > 80:
            actions.insert(0, "Schedule immediate inspection")
        elif risk_score > 60:
            actions.insert(0, "Schedule inspection within 7 days")
        elif risk_score > 40:
            actions.insert(0, "Schedule inspection within 30 days")
        
        return actions
    
    def _learn_cross_asset_pattern(self, asset_id: str, 
                                  failure_type: str, 
                                  pattern: Dict[str, float]):
        """Learn patterns that apply across multiple assets"""
        
        # Check if similar pattern already exists
        existing_pattern_id = None
        max_similarity = 0
        
        for pattern_id, learned_pattern in self.learned_patterns.items():
            if learned_pattern.failure_type == failure_type:
                # Calculate pattern similarity
                similarity = self._calculate_pattern_similarity(pattern, learned_pattern.sensor_pattern)
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    existing_pattern_id = pattern_id
        
        if existing_pattern_id and max_similarity > 0.8:
            # Update existing pattern
            self.learned_patterns[existing_pattern_id].update(pattern, asset_id)
            print(f"📚 Updated cross-asset pattern {existing_pattern_id}")
        else:
            # Create new pattern
            pattern_id = f"{failure_type}_{len(self.learned_patterns) + 1}"
            
            new_pattern = LearnedPattern(
                pattern_id=pattern_id,
                failure_type=failure_type,
                sensor_pattern=pattern,
                asset_types=[self.asset_profiles[asset_id].asset_type],
                learned_from=[asset_id],
                occurrence_count=1,
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                effectiveness=0.7  # Initial effectiveness
            )
            
            self.learned_patterns[pattern_id] = new_pattern
            print(f"📚 Learned new cross-asset pattern: {pattern_id}")
    
    def _calculate_pattern_similarity(self, pattern1: Dict[str, float], 
                                    pattern2: Dict[str, float]) -> float:
        """Calculate similarity between two patterns"""
        common_keys = set(pattern1.keys()) & set(pattern2.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            val1 = pattern1[key]
            val2 = pattern2[key]
            max_val = max(abs(val1), abs(val2))
            if max_val > 0:
                similarity = 1 - abs(val1 - val2) / max_val
                similarities.append(max(0, similarity))
        
        return np.mean(similarities) if similarities else 0.0
    
    def _check_preventive_alerts(self, new_asset_id: str):
        """Check if new asset should receive preventive alerts"""
        for pattern_id, pattern in self.learned_patterns.items():
            # Check if pattern applies to this asset type
            asset_type = self.asset_profiles[new_asset_id].asset_type
            if asset_type not in pattern.asset_types:
                continue
            
            # Check if similar assets had this failure
            similar_assets = self.get_similar_assets(new_asset_id, min_similarity=0.6)
            
            for similar_id, similarity in similar_assets:
                # Check if similar asset has experienced this failure
                profile = self.asset_profiles.get(similar_id)
                if profile:
                    recent_failures = profile.get_recent_failures(90)
                    matching_failures = [
                        f for f in recent_failures 
                        if f.get("failure_type") == pattern.failure_type
                    ]
                    
                    if matching_failures:
                        # Create preventive alert
                        risk_score = similarity * 100 * pattern.effectiveness
                        
                        if risk_score > 40:
                            alert = self._create_preventive_alert(
                                similar_id, new_asset_id, pattern.failure_type,
                                similarity, risk_score, pattern.sensor_pattern
                            )
                            
                            self.alerts_history.append(alert)
                            print(f"🔔 Preventive alert for {new_asset_id} from {similar_id}")
    
    def monitor_asset(self, asset_id: str, sensor_data: Dict[str, float]) -> List[CrossAssetAlert]:
        """
        Monitor asset against learned patterns and generate alerts
        
        Args:
            asset_id: Asset to monitor
            sensor_data: Current sensor readings
            
        Returns:
            List of alerts if patterns match
        """
        alerts = []
        
        if asset_id not in self.asset_profiles:
            return alerts
        
        profile = self.asset_profiles[asset_id]
        
        # Check against learned patterns
        for pattern_id, pattern in self.learned_patterns.items():
            if profile.asset_type not in pattern.asset_types:
                continue
            
            # Calculate similarity to pattern
            pattern_similarity = self._calculate_pattern_similarity(
                {"sensor_" + k: v for k, v in sensor_data.items()},
                pattern.sensor_pattern
            )
            
            if pattern_similarity > 0.7:
                # Check if similar assets had this failure
                similar_assets = self.get_similar_assets(asset_id, min_similarity=0.6)
                
                for similar_id, similarity in similar_assets:
                    similar_profile = self.asset_profiles.get(similar_id)
                    if similar_profile:
                        recent_failures = similar_profile.get_recent_failures(90)
                        matching_failures = [
                            f for f in recent_failures
                            if f.get("failure_type") == pattern.failure_type
                        ]
                        
                        if matching_failures:
                            # Create alert
                            risk_score = pattern_similarity * similarity * 100 * pattern.effectiveness
                            
                            alert = CrossAssetAlert(
                                source_asset=similar_id,
                                target_asset=asset_id,
                                failure_type=pattern.failure_type,
                                similarity_score=similarity,
                                predicted_risk=risk_score,
                                evidence=[
                                    f"Pattern match: {pattern_similarity:.2%}",
                                    f"Similar asset ({similar_id}) had same failure",
                                    f"Pattern effectiveness: {pattern.effectiveness:.2%}"
                                ],
                                recommended_actions=self._get_preventive_actions(
                                    pattern.failure_type, risk_score
                                ),
                                confidence=pattern_similarity * pattern.effectiveness
                            )
                            
                            alerts.append(alert)
        
        return alerts
    
    def get_asset_insights(self, asset_id: str) -> Dict[str, Any]:
        """Get insights for a specific asset"""
        if asset_id not in self.asset_profiles:
            return {}
        
        profile = self.asset_profiles[asset_id]
        
        # Get similar assets
        similar_assets = self.get_similar_assets(asset_id, min_similarity=0.5, limit=10)
        
        # Get preventive alerts
        preventive_alerts = [
            alert for alert in self.alerts_history[-100:]
            if alert.target_asset == asset_id
        ]
        
        # Calculate risk scores from patterns
        pattern_risks = []
        for pattern_id, pattern in self.learned_patterns.items():
            if profile.asset_type in pattern.asset_types:
                # Check if similar assets had this failure
                for similar_id, similarity in similar_assets:
                    similar_profile = self.asset_profiles.get(similar_id)
                    if similar_profile:
                        recent_failures = similar_profile.get_recent_failures(90)
                        if any(f.get("failure_type") == pattern.failure_type for f in recent_failures):
                            risk_score = similarity * 100 * pattern.effectiveness
                            pattern_risks.append({
                                "pattern": pattern_id,
                                "failure_type": pattern.failure_type,
                                "risk_score": risk_score,
                                "similar_asset": similar_id
                            })
        
        insights = {
            "asset_id": asset_id,
            "asset_type": profile.asset_type,
            "failure_history": {
                "total_failures": len(profile.failure_history),
                "recent_failures": len(profile.get_recent_failures(90)),
                "failure_types": list(set(
                    f.get("failure_type", "unknown") 
                    for f in profile.failure_history
                ))
            },
            "similar_assets": [
                {
                    "asset_id": asset_id,
                    "similarity": similarity,
                    "asset_type": self.asset_profiles[asset_id].asset_type,
                    "recent_failures": len(self.asset_profiles[asset_id].get_recent_failures(90))
                }
                for asset_id, similarity in similar_assets
            ],
            "preventive_alerts": [alert.to_dict() for alert in preventive_alerts[-5:]],
            "pattern_risks": pattern_risks,
            "knowledge_benefit": self._calculate_knowledge_benefit(asset_id)
        }
        
        return insights
    
    def _calculate_knowledge_benefit(self, asset_id: str) -> Dict[str, float]:
        """Calculate the benefit of cross-asset knowledge for this asset"""
        if asset_id not in self.asset_profiles:
            return {"total_benefit": 0, "prevented_failures": 0}
        
        profile = self.asset_profiles[asset_id]
        
        # Count preventive alerts
        preventive_alerts = [
            alert for alert in self.alerts_history
            if alert.target_asset == asset_id
        ]
        
        # Estimate prevented failures (simplified)
        # Assume each high-confidence alert prevented 0.5 failures on average
        prevented_failures = sum(
            0.5 for alert in preventive_alerts 
            if alert.confidence > 0.7
        )
        
        # Calculate cost savings
        avg_failure_cost = 50000  # Average cost per failure
        cost_savings = prevented_failures * avg_failure_cost
        
        return {
            "total_benefit": cost_savings,
            "prevented_failures": prevented_failures,
            "preventive_alerts": len(preventive_alerts),
            "knowledge_transfers": self.knowledge_transfers
        }
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get overall system statistics"""
        stats = {
            "total_assets": len(self.asset_profiles),
            "total_failures": sum(
                len(p.failure_history) for p in self.asset_profiles.values()
            ),
            "learned_patterns": len(self.learned_patterns),
            "knowledge_transfers": self.knowledge_transfers,
            "preventive_alerts": len(self.alerts_history),
            "asset_types": list(set(
                p.asset_type for p in self.asset_profiles.values()
            )),
            "recent_activity": {
                "last_24h_failures": sum(
                    1 for p in self.asset_profiles.values()
                    for f in p.failure_history
                    if datetime.now() - datetime.fromisoformat(f["timestamp"]) < timedelta(hours=24)
                ),
                "last_24h_alerts": sum(
                    1 for alert in self.alerts_history
                    if datetime.now() - alert.timestamp < timedelta(hours=24)
                )
            }
        }
        
        # Calculate effectiveness metrics
        if self.alerts_history:
            # Simplified effectiveness calculation
            high_confidence_alerts = [
                alert for alert in self.alerts_history
                if alert.confidence > 0.7
            ]
            
            if high_confidence_alerts:
                stats["alert_effectiveness"] = len(high_confidence_alerts) / len(self.alerts_history)
            else:
                stats["alert_effectiveness"] = 0
        
        return stats
    
    def export_knowledge_base(self, filepath: str = "data/models/cross_asset_knowledge.json"):
        """Export the complete knowledge base to JSON"""
        export_data = {
            "asset_profiles": {},
            "learned_patterns": {},
            "statistics": self.get_system_statistics(),
            "export_timestamp": datetime.now().isoformat()
        }
        
        # Export asset profiles
        for asset_id, profile in self.asset_profiles.items():
            export_data["asset_profiles"][asset_id] = {
                "asset_type": profile.asset_type,
                "manufacturer": profile.manufacturer,
                "model": profile.model,
                "installation_date": profile.installation_date.isoformat(),
                "operational_parameters": profile.operational_parameters,
                "failure_count": len(profile.failure_history),
                "maintenance_count": len(profile.maintenance_history),
                "similar_assets": profile.similarity_scores
            }
        
        # Export learned patterns
        for pattern_id, pattern in self.learned_patterns.items():
            export_data["learned_patterns"][pattern_id] = {
                "failure_type": pattern.failure_type,
                "sensor_pattern": pattern.sensor_pattern,
                "asset_types": pattern.asset_types,
                "learned_from": pattern.learned_from,
                "occurrence_count": pattern.occurrence_count,
                "first_seen": pattern.first_seen.isoformat(),
                "last_seen": pattern.last_seen.isoformat(),
                "effectiveness": pattern.effectiveness
            }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"📤 Exported knowledge base to {filepath}")

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Cross-Asset Learning Module")
    print("="*50)
    
    # Initialize learner
    learner = CrossAssetLearner()
    
    # Register sample assets
    print("\n📝 Registering assets...")
    
    # Asset 1: Pump in Plant A
    learner.register_asset(
        asset_id="pump_plant_a_001",
        asset_type="centrifugal_pump",
        manufacturer="Grundfos",
        model="CR 32-5",
        installation_date=datetime(2020, 1, 15),
        operational_params={"rpm": 1800, "pressure": 10.5, "temperature": 75.0}
    )
    
    # Asset 2: Similar pump in Plant B
    learner.register_asset(
        asset_id="pump_plant_b_001",
        asset_type="centrifugal_pump",
        manufacturer="Grundfos",
        model="CR 32-5",
        installation_date=datetime(2020, 3, 20),
        operational_params={"rpm": 1750, "pressure": 11.0, "temperature": 78.0}
    )
    
    # Asset 3: Different type of pump
    learner.register_asset(
        asset_id="pump_plant_c_001",
        asset_type="positive_displacement_pump",
        manufacturer="Seepex",
        model="MD 100-16",
        installation_date=datetime(2019, 6, 10),
        operational_params={"rpm": 300, "pressure": 15.0, "temperature": 85.0}
    )
    
    # Record a failure on Asset 1
    print("\n📝 Recording failure on pump_plant_a_001...")
    learner.record_failure(
        asset_id="pump_plant_a_001",
        failure_data={
            "failure_type": "bearing_failure",
            "sensor_data": {
                "vibration": 4.5,
                "temperature": 95.0,
                "acoustic": 90.0
            },
            "severity": "high",
            "root_cause": "lubrication",
            "confidence": 0.85
        }
    )
    
    # Monitor Asset 2 (should receive preventive alert)
    print("\n🔍 Monitoring pump_plant_b_001...")
    alerts = learner.monitor_asset(
        asset_id="pump_plant_b_001",
        sensor_data={
            "vibration": 2.0,
            "temperature": 80.0,
            "acoustic": 70.0
        }
    )
    
    if alerts:
        print(f"🚨 Generated {len(alerts)} preventive alert(s):")
        for alert in alerts:
            print(f"  • {alert.failure_type} (Risk: {alert.predicted_risk:.1f})")
            print(f"    Source: {alert.source_asset}, Confidence: {alert.confidence:.2%}")
    else:
        print("✅ No alerts generated")
    
    # Get insights for Asset 2
    print("\n📊 Getting insights for pump_plant_b_001...")
    insights = learner.get_asset_insights("pump_plant_b_001")
    
    if insights:
        print(f"Asset: {insights['asset_id']} ({insights['asset_type']})")
        print(f"Similar assets: {len(insights['similar_assets'])}")
        print(f"Preventive alerts: {len(insights['preventive_alerts'])}")
        
        if insights['pattern_risks']:
            print("\n📈 Pattern Risks:")
            for risk in insights['pattern_risks'][:3]:
                print(f"  • {risk['failure_type']}: {risk['risk_score']:.1f} "
                      f"(from {risk['similar_asset']})")
    
    # Get system statistics
    print("\n📈 System Statistics:")
    stats = learner.get_system_statistics()
    print(f"  Total assets: {stats['total_assets']}")
    print(f"  Learned patterns: {stats['learned_patterns']}")
    print(f"  Knowledge transfers: {stats['knowledge_transfers']}")
    print(f"  Preventive alerts: {stats['preventive_alerts']}")
    
    # Export knowledge base
    learner.export_knowledge_base()
    
    print(f"\n✅ Cross-asset learning test complete!")