"""
📦 Setup script for Step 1: Data Collection System
"""

import os
import json
import shutil
from datetime import datetime

def setup_directory_structure():
    """Create the complete directory structure"""
    directories = [
        "data/input/simulated_sensors",
        "data/input/csv_imports",
        "data/input/api_endpoints",
        "data/assets",
        "data/models/trained_models",
        "data/models/degradation_curves",
        "core",
        "analytics",
        "output/dashboard",
        "output/alerts",
        "output/api",
        "utils",
        "tests"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created: {directory}")
    
    # Create empty __init__.py files
    for package in ["core", "analytics", "utils"]:
        init_file = os.path.join(package, "__init__.py")
        with open(init_file, 'w') as f:
            f.write("# Package initialization\n")
    
    print("✅ Directory structure created")

def create_config_file():
    """Create configuration file"""
    config = {
        "system": {
            "name": "VIBE-GUARD Advanced",
            "version": "2.0",
            "mode": "development"
        },
        "data_collection": {
            "interval_seconds": 1,
            "buffer_size": 1000,
            "save_interval": 10,
            "sources": ["simulated", "csv", "api"]
        },
        "assets": {
            "registry_file": "data/assets/registry.json",
            "backup_interval_hours": 24
        },
        "analytics": {
            "health_score_update_interval": 60,
            "failure_prediction_horizon_days": 30,
            "confidence_threshold": 0.7
        },
        "alerts": {
            "email_enabled": False,
            "sms_enabled": False,
            "critical_threshold": 0.8,
            "warning_threshold": 0.6
        }
    }
    
    with open("config.yaml", 'w') as f:
        import yaml
        yaml.dump(config, f, default_flow_style=False)
    
    print("✅ Configuration file created")

def create_requirements_file():
    """Create requirements.txt"""
    requirements = [
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "plotly>=5.10.0",
        "streamlit>=1.12.0",
        "pyyaml>=6.0",
        "scipy>=1.7.0",
        "joblib>=1.2.0",
        "python-dateutil>=2.8.2"
    ]
    
    with open("requirements.txt", 'w') as f:
        f.write("\n".join(requirements))
    
    print("✅ Requirements file created")

def create_main_file():
    """Create main.py"""
    main_code = '''#!/usr/bin/env python3
"""
🎯 VIBE-GUARD Advanced - Industrial Asset Health Management
Main Entry Point
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.data_collector import DataCollector
import time

def main():
    """Main function"""
    print("="*60)
    print("🚀 VIBE-GUARD Advanced - Asset Health Management")
    print("="*60)
    
    print("\\n1. Starting Data Collection System...")
    collector = DataCollector()
    collector.start_collection(interval_seconds=2)
    
    try:
        print("2. Data collection running. Press Ctrl+C to stop.")
        print("\\nSample data will be saved to:")
        print("   • data/input/simulated_sensors/")
        print("   • data/assets/registry.json")
        print("\\nCollecting data for 30 seconds...")
        
        # Collect data for 30 seconds
        for i in range(30):
            time.sleep(1)
            if i % 10 == 0:
                print(f"   Collected {i+1} seconds of data...")
        
        print("\\n3. Data collection complete.")
        print("\\n4. Showing sample data:")
        
        # Display sample data
        readings = collector.get_latest_readings("pump_001", limit=3)
        for reading in readings:
            print(f"   • {reading.sensor_type}: {reading.value} {reading.unit}")
        
    except KeyboardInterrupt:
        print("\\nStopping data collection...")
    finally:
        collector.stop_collection()
        print("\\n✅ System stopped successfully.")
        print("="*60)

if __name__ == "__main__":
    main()
'''
    
    with open("main.py", 'w') as f:
        f.write(main_code)
    
    print("✅ Main file created")

def create_readme():
    """Create README.md"""
    readme = '''# VIBE-GUARD Advanced

## Industrial Asset Health Management System

A complete software-only solution for predictive maintenance, failure fingerprinting, cross-asset learning, corrosion detection, FFS/FIT analysis, and life extension.

## Features

### ✅ Data Collection
- Multi-source data collection (simulated, CSV, API)
- Real-time sensor data simulation
- Asset registry management
- Historical data storage

### 🔍 Analytics
- Failure fingerprinting
- Cross-asset learning
- Corrosion detection
- Fitness-for-Service (FFS) analysis
- Crack growth prediction
- Life extension calculations

### 📊 Output
- Real-time dashboard
- Alert system
- API endpoints
- Detailed reports

## Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
'''