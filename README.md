
# AFMO

<<<<<<< HEAD
Paketorientiertes Toolkit für Zeitreihen (ARIMA/SARIMA/ETS) mit optionaler Streamlit-GUI.
=======
## Installation
In Powershell ausführen:

- py -m venv venv
- .\run_win.ps1

ODER:
- **Windows:** Rechtsklick → *Mit PowerShell ausführen* auf `run_win.ps1` (legt venv an, installiert *locked* Wheels, startet App).
- **macOS/Linux:** `./run_unix.sh` (legt venv an, installiert *locked* Wheels, startet App).
>>>>>>> main

## Start
```bash
streamlit run app/AFMo.py
```

## CLI
```bash
afmo forecast --input examples/sample.csv --column y --h 12 --p 1 --d 1 --q 1 --out forecast.csv
```

## One-command setup (zero friction)

- **Windows:** Rechtsklick → *Mit PowerShell ausführen* auf `run_win.ps1` (legt venv an, installiert *locked* Wheels, startet App).
- **macOS/Linux:** `./run_unix.sh` (legt venv an, installiert *locked* Wheels, startet App).
- **Conda:** `mamba env create -f environment.yml && mamba activate afmo && streamlit run app/AFMo.py`
- **Docker:** `docker compose up --build` und im Browser `http://localhost:8501` öffnen.

Die *locked* Dateien pinnen kompatible Binärpakete (NumPy 1.26.x + pmdarima 2.0.4), damit es ohne Compiler & ohne ABI-Fehler läuft.

## Extensibility (features & metrics)

AFMO is now extensible by *design*. You can add new time‑series features
or forecast accuracy metrics:

```python
# features.py
from afmo.core.registry import register_feature

@register_feature
def my_feature(y: pd.Series) -> float:
    ...
```

```python
# metrics.py
from afmo.core.registry import register_metric

@register_metric
def my_metric(y_true, y_pred) -> float:
    ...
```

Registered functions automatically appear in the public registries
`afmo.FEATURES` / `afmo.METRICS` and can be computed programmatically via
`afmo.compute_features(...)` and `afmo.compute_metrics(...)`.