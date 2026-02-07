"""
📡 Data Collection System
Collects data from multiple sources and normalizes it
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import threading
import time
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import yaml

@dataclass
class SensorReading:
    """Standardized sensor reading format"""
    timestamp: datetime
    asset_id: str
    sensor_type: str  # vibration, temperature, pressure, etc.
    value: float
    unit: str
    location: Optional[str] = None
    quality: float = 1.0  # 0-1 data quality score
    
@dataclass
class AssetMetadata:
    """Asset information database"""
    asset_id: str
    asset_type: str  # pump, compressor, vessel, etc.
    installation_date: datetime
    manufacturer: str
    model: str
    specifications: Dict
    location: Dict  # plant, unit, coordinates
    inspection_history: List[Dict]
    maintenance_history: List[Dict]

class DataCollector:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize data collection system"""
        self.config = self.load_config(config_path)
        self.asset_registry = {}  # asset_id -> AssetMetadata
        self.sensor_buffer = {}   # asset_id -> List[SensorReading]
        self.is_running = False
        self.data_lock = threading.Lock()
        
        # Initialize data sources
        self.sources = {
            "simulated": self.simulate_sensor_data,
            "csv": self.read_csv_data,
            "api": self.fetch_api_data,
            "database": self.query_database
        }
        
        self.load_asset_registry()
        
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_asset_registry(self):
        """Load asset database"""
        try:
            with open("data/assets/registry.json", 'r') as f:
                assets_data = json.load(f)
                
            for asset_id, asset_data in assets_data.items():
                self.asset_registry[asset_id] = AssetMetadata(
                    asset_id=asset_id,
                    asset_type=asset_data["type"],
                    installation_date=datetime.fromisoformat(asset_data["installation_date"]),
                    manufacturer=asset_data["manufacturer"],
                    model=asset_data["model"],
                    specifications=asset_data["specifications"],
                    location=asset_data["location"],
                    inspection_history=asset_data.get("inspection_history", []),
                    maintenance_history=asset_data.get("maintenance_history", [])
                )
            print(f"Loaded {len(self.asset_registry)} assets")
        except FileNotFoundError:
            print("No existing asset registry found. Creating new one.")
            self.create_sample_assets()
    
    def create_sample_assets(self):
        """Create sample assets for demonstration"""
        sample_assets = {
            "pump_001": {
                "type": "centrifugal_pump",
                "installation_date": "2020-01-15",
                "manufacturer": "Grundfos",
                "model": "CR 32-5",
                "specifications": {
                    "max_pressure": 25.0,
                    "max_temperature": 120.0,
                    "material": "stainless_steel_316",
                    "design_life": 20  # years
                },
                "location": {
                    "plant": "Plant A",
                    "unit": "PU-101",
                    "coordinates": {"x": 45.2, "y": 32.1, "z": 0}
                },
                "sensors": [
                    {"type": "vibration", "location": "bearing_housing", "unit": "mm/s"},
                    {"type": "temperature", "location": "bearing", "unit": "°C"},
                    {"type": "pressure", "location": "discharge", "unit": "bar"},
                    {"type": "acoustic", "location": "pump_housing", "unit": "dB"},
                    {"type": "strain", "location": "base", "unit": "με"}
                ]
            },
            "vessel_001": {
                "type": "pressure_vessel",
                "installation_date": "2018-03-10",
                "manufacturer": "ASME",
                "model": "PV-1500",
                "specifications": {
                    "design_pressure": 15.0,
                    "design_temperature": 150.0,
                    "material": "carbon_steel",
                    "corrosion_allowance": 3.0,  # mm
                    "thickness": 25.0  # mm
                },
                "location": {
                    "plant": "Plant B",
                    "unit": "V-205",
                    "coordinates": {"x": 52.7, "y": 41.3, "z": 0}
                },
                "sensors": [
                    {"type": "corrosion", "location": "bottom", "unit": "mm"},
                    {"type": "temperature", "location": "shell", "unit": "°C"},
                    {"type": "pressure", "location": "internal", "unit": "bar"},
                    {"type": "strain", "location": "support", "unit": "με"}
                ]
            },
            "compressor_001": {
                "type": "screw_compressor",
                "installation_date": "2019-06-20",
                "manufacturer": "Atlas Copco",
                "model": "GA 37",
                "specifications": {
                    "max_pressure": 10.0,
                    "max_temperature": 100.0,
                    "material": "cast_iron",
                    "design_life": 15
                },
                "location": {
                    "plant": "Plant C",
                    "unit": "C-301",
                    "coordinates": {"x": 38.9, "y": 29.4, "z": 0}
                },
                "sensors": [
                    {"type": "vibration", "location": "motor", "unit": "mm/s"},
                    {"type": "temperature", "location": "oil", "unit": "°C"},
                    {"type": "pressure", "location": "discharge", "unit": "bar"},
                    {"type": "acoustic", "location": "compressor", "unit": "dB"}
                ]
            }
        }
        
        # Save to file
        with open("data/assets/registry.json", 'w') as f:
            json.dump(sample_assets, f, indent=2, default=str)
        
        # Load into memory
        self.load_asset_registry()
    
    def simulate_sensor_data(self, asset_id: str, duration_seconds: int = 1) -> List[SensorReading]:
        """Generate realistic simulated sensor data"""
        asset = self.asset_registry.get(asset_id)
        if not asset:
            return []
        
        readings = []
        current_time = datetime.now()
        
        # Get sensor configurations for this asset
        sensor_configs = asset.specifications.get("sensors", [])
        
        for sensor in sensor_configs:
            sensor_type = sensor["type"]
            location = sensor["location"]
            unit = sensor["unit"]
            
            # Base values based on asset type and sensor type
            base_value, variation = self.get_sensor_parameters(asset.asset_type, sensor_type)
            
            # Add degradation over time
            age_days = (current_time - asset.installation_date).days
            degradation_factor = 1 + (age_days / 365.0 * 0.01)  # 1% degradation per year
            
            # Add random variation
            value = base_value * degradation_factor + np.random.normal(0, variation)
            
            # Ensure value is within reasonable bounds
            value = self.clamp_sensor_value(sensor_type, value)
            
            # Calculate data quality (simulate occasional bad readings)
            quality = 0.95 + np.random.normal(0, 0.05)
            quality = max(0.5, min(1.0, quality))
            
            reading = SensorReading(
                timestamp=current_time,
                asset_id=asset_id,
                sensor_type=sensor_type,
                value=float(value),
                unit=unit,
                location=location,
                quality=float(quality)
            )
            readings.append(reading)
        
        return readings
    
    def get_sensor_parameters(self, asset_type: str, sensor_type: str) -> tuple:
        """Get base values and variation for sensor types"""
        # Base values for different asset-sensor combinations
        parameters = {
            ("centrifugal_pump", "vibration"): (1.2, 0.3),
            ("centrifugal_pump", "temperature"): (75.0, 2.0),
            ("centrifugal_pump", "pressure"): (10.5, 0.5),
            ("centrifugal_pump", "acoustic"): (65.0, 5.0),
            ("centrifugal_pump", "strain"): (120.0, 15.0),
            
            ("pressure_vessel", "corrosion"): (0.5, 0.1),  # mm loss
            ("pressure_vessel", "temperature"): (85.0, 3.0),
            ("pressure_vessel", "pressure"): (8.0, 0.3),
            ("pressure_vessel", "strain"): (80.0, 10.0),
            
            ("screw_compressor", "vibration"): (1.8, 0.4),
            ("screw_compressor", "temperature"): (85.0, 3.0),
            ("screw_compressor", "pressure"): (7.5, 0.4),
            ("screw_compressor", "acoustic"): (72.0, 6.0),
        }
        
        return parameters.get((asset_type, sensor_type), (0.0, 0.1))
    
    def clamp_sensor_value(self, sensor_type: str, value: float) -> float:
        """Ensure sensor values are within reasonable physical limits"""
        limits = {
            "vibration": (0, 20),      # mm/s
            "temperature": (-20, 200),  # °C
            "pressure": (0, 50),        # bar
            "acoustic": (30, 120),      # dB
            "strain": (0, 1000),        # με
            "corrosion": (0, 10),       # mm loss
        }
        
        min_val, max_val = limits.get(sensor_type, (0, 100))
        return max(min_val, min(max_val, value))
    
    def start_collection(self, collection_interval: int = 1):
        """Start continuous data collection"""
        if self.is_running:
            print("Data collection already running")
            return
        
        self.is_running = True
        print(f"Starting data collection (interval: {collection_interval}s)")
        
        def collection_loop():
            while self.is_running:
                with self.data_lock:
                    # Collect data from all assets
                    for asset_id in self.asset_registry:
                        readings = self.simulate_sensor_data(asset_id)
                        
                        # Store in buffer
                        if asset_id not in self.sensor_buffer:
                            self.sensor_buffer[asset_id] = []
                        
                        self.sensor_buffer[asset_id].extend(readings)
                        
                        # Keep only last 1000 readings per sensor
                        if len(self.sensor_buffer[asset_id]) > 1000:
                            self.sensor_buffer[asset_id] = self.sensor_buffer[asset_id][-1000:]
                    
                    # Save to database/file
                    self.save_buffer_to_database()
                
                time.sleep(collection_interval)
        
        # Start collection thread
        self.collection_thread = threading.Thread(target=collection_loop, daemon=True)
        self.collection_thread.start()
    
    def save_buffer_to_database(self):
        """Save collected data to persistent storage"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/input/simulated_sensors/sensor_data_{timestamp}.json"
            
            # Convert buffer to serializable format
            serializable_buffer = {}
            for asset_id, readings in self.sensor_buffer.items():
                serializable_buffer[asset_id] = [
                    {
                        "timestamp": r.timestamp.isoformat(),
                        "sensor_type": r.sensor_type,
                        "value": r.value,
                        "unit": r.unit,
                        "location": r.location,
                        "quality": r.quality
                    }
                    for r in readings[-100:]  # Save last 100 readings
                ]
            
            with open(filename, 'w') as f:
                json.dump(serializable_buffer, f, indent=2)
            
            # Also update CSV for easy analysis
            self.update_csv_database(serializable_buffer)
            
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def update_csv_database(self, buffer_data: Dict):
        """Update CSV database with latest readings"""
        csv_rows = []
        for asset_id, readings in buffer_data.items():
            for reading in readings:
                csv_rows.append({
                    "timestamp": reading["timestamp"],
                    "asset_id": asset_id,
                    "sensor_type": reading["sensor_type"],
                    "value": reading["value"],
                    "unit": reading["unit"],
                    "location": reading["location"],
                    "quality": reading["quality"]
                })
        
        if csv_rows:
            df = pd.DataFrame(csv_rows)
            csv_file = "data/input/simulated_sensors/sensor_data_latest.csv"
            
            # Append to existing file or create new
            try:
                existing_df = pd.read_csv(csv_file)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                combined_df.to_csv(csv_file, index=False)
            except FileNotFoundError:
                df.to_csv(csv_file, index=False)
    
    def stop_collection(self):
        """Stop data collection"""
        self.is_running = False
        if hasattr(self, 'collection_thread'):
            self.collection_thread.join(timeout=5)
        print("Data collection stopped")
    
    def get_latest_readings(self, asset_id: str = None, 
                           sensor_type: str = None, 
                           limit: int = 100) -> List[SensorReading]:
        """Get latest sensor readings from buffer"""
        with self.data_lock:
            if asset_id:
                if asset_id in self.sensor_buffer:
                    readings = self.sensor_buffer[asset_id]
                    if sensor_type:
                        readings = [r for r in readings if r.sensor_type == sensor_type]
                    return readings[-limit:]
                return []
            else:
                # Return all readings
                all_readings = []
                for asset_readings in self.sensor_buffer.values():
                    if sensor_type:
                        asset_readings = [r for r in asset_readings if r.sensor_type == sensor_type]
                    all_readings.extend(asset_readings[-limit//len(self.sensor_buffer):])
                
                # Sort by timestamp
                all_readings.sort(key=lambda x: x.timestamp, reverse=True)
                return all_readings[:limit]
    
    def add_inspection_data(self, asset_id: str, inspection_data: Dict):
        """Add manual inspection data"""
        if asset_id in self.asset_registry:
            inspection_data["timestamp"] = datetime.now().isoformat()
            self.asset_registry[asset_id].inspection_history.append(inspection_data)
            self.save_asset_registry()
            print(f"Inspection data added for {asset_id}")
    
    def add_maintenance_record(self, asset_id: str, maintenance_data: Dict):
        """Add maintenance record"""
        if asset_id in self.asset_registry:
            maintenance_data["timestamp"] = datetime.now().isoformat()
            self.asset_registry[asset_id].maintenance_history.append(maintenance_data)
            self.save_asset_registry()
            print(f"Maintenance record added for {asset_id}")
    
    def save_asset_registry(self):
        """Save asset registry to file"""
        serializable_registry = {}
        for asset_id, asset in self.asset_registry.items():
            serializable_registry[asset_id] = {
                "type": asset.asset_type,
                "installation_date": asset.installation_date.isoformat(),
                "manufacturer": asset.manufacturer,
                "model": asset.model,
                "specifications": asset.specifications,
                "location": asset.location,
                "inspection_history": asset.inspection_history,
                "maintenance_history": asset.maintenance_history
            }
        
        with open("data/assets/registry.json", 'w') as f:
            json.dump(serializable_registry, f, indent=2)

# Example usage
if __name__ == "__main__":
    collector = DataCollector()
    collector.start_collection(collection_interval=2)
    
    try:
        # Run for 10 seconds
        time.sleep(10)
        
        # Get latest readings
        readings = collector.get_latest_readings("pump_001", "vibration", 5)
        for r in readings:
            print(f"{r.timestamp}: {r.sensor_type} = {r.value} {r.unit}")
        
    finally:
        collector.stop_collection()