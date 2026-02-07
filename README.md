Vibe Guard Advanced - maintenance optimizer and diagnostics toolkit.

See the `examples/` and `dashboard/` folders for usage and the `run_optimizer.py` runner for CLI execution.

Quick commands:
- `python -u run_optimizer.py --input examples/minimal_input.json`
- `python -u run_optimizer.py --interactive`
- `streamlit run dashboard/app.py`
# Vibe Guard Advanced

Vibe Guard Advanced is a lightweight maintenance optimization and diagnostics toolkit for industrial assets.

Key features
- Risk scoring and RBI-style analysis
- Maintenance scheduling optimizer (preventive / corrective)
- Failure fingerprinting and cross-asset learning (pluggable)
- Corrosion and FFS analysis helpers
- Streamlit dashboard for quick interactive runs

Quickstart

1. Create and activate a Python environment (Windows PowerShell):

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt

2. Run the non-interactive optimizer with the example input:

    python -u run_optimizer.py --input examples/minimal_input.json

3. Run the interactive CLI runner:

    python -u run_optimizer.py --interactive

4. Start the Streamlit dashboard (open browser to the printed URL):

    streamlit run dashboard/app.py

Tests

    python -u tests/test_maintenance_optimizer.py
    # or, with pytest:
    python -m pytest -q

Adding sensor data

- The codebase includes hooks for sensor-driven analysis (see core/data_collector.py, core/failure_fingerprinting.py, core/cross_asset_learning.py).
- I can add a --sensor-file flag or CSV ingest to the dashboard if you want to load real sensor files.

Repository

- Remote: https://github.com/Rushi1696/SHM_project.git

Next steps

- Add `.gitignore` (recommended) and CI workflow to run tests on push.
- Optionally wire a persistent datastore for inspection/maintenance history.

If you want, I can add `--sensor-file` ingestion to the runner and dashboard, or add a Save Report button to the dashboard.
# Vibe Guard Advanced

Vibe Guard Advanced is a lightweight maintenance optimization and diagnostics toolkit for industrial assets.

Key features
- Risk scoring and RBI-style analysis
- Maintenance scheduling optimizer (preventive / corrective)
- Failure fingerprinting and cross-asset learning (pluggable)
- Corrosion and FFS analysis helpers
- Streamlit dashboard for quick interactive runs

Quickstart
1. Create and activate a Python environment (Windows PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the non-interactive optimizer with the example input:
```powershell
python -u run_optimizer.py --input examples/minimal_input.json
```

3. Run the interactive CLI runner:
```powershell
python -u run_optimizer.py --interactive
```

4. Start the Streamlit dashboard (open browser to the printed URL):
```powershell
streamlit run dashboard/app.py
```

Tests
```powershell
python -u tests/test_maintenance_optimizer.py
# or, with pytest:
python -m pytest -q
```

Adding sensor data
- The codebase includes hooks for sensor-driven analysis (see `core/data_collector.py`, `core/failure_fingerprinting.py`, `core/cross_asset_learning.py`).
- I can add a `--sensor-file` flag or CSV ingest to the dashboard if you want to load real sensor files.

Repository
- Remote: https://github.com/Rushi1696/SHM_project.git

Next steps
- Add `.gitignore` (recommended) and CI workflow to run tests on push.
- Optionally wire a persistent datastore for inspection/maintenance history.

If you want, I can add `--sensor-file` ingestion to the runner and dashboard, or add a Save Report button to the dashboard.
# Vibe Guard Advanced

Vibe Guard Advanced is a lightweight maintenance optimization and diagnostics toolkit for industrial assets.

Key features
- Risk scoring and RBI-style analysis
- Maintenance scheduling optimizer (preventive / corrective)
- Failure fingerprinting and cross-asset learning (pluggable)
- Corrosion and FFS analysis helpers
- Streamlit dashboard for quick interactive runs

Quickstart
1. Create and activate a Python environment (Windows PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the non-interactive optimizer with the example input:
```powershell
python -u run_optimizer.py --input examples/minimal_input.json
```

3. Run the interactive CLI runner:
```powershell
python -u run_optimizer.py --interactive
```

4. Start the Streamlit dashboard (open browser to the printed URL):
```powershell
streamlit run dashboard/app.py
```

Tests
```powershell
python -u tests/test_maintenance_optimizer.py
# or, with pytest:
python -m pytest -q
```

Adding sensor data
- The codebase includes hooks for sensor-driven analysis (see `core/data_collector.py`, `core/failure_fingerprinting.py`, `core/cross_asset_learning.py`).
- I can add a `--sensor-file` flag or CSV ingest to the dashboard if you want to load real sensor files.

Repository
- Remote: https://github.com/Rushi1696/SHM_project.git

Next steps
- Add `.gitignore` (recommended) and CI workflow to run tests on push.
- Optionally wire a persistent datastore for inspection/maintenance history.

If you want, I can add `--sensor-file` ingestion to the runner and dashboard, or add a Save Report button to the dashboard.
