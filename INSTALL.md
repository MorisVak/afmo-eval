# Installation notes

This project now declares Seaborn/Matplotlib for the association heatmaps.

## Pip (recommended)
```bash
python -m venv .venv
# Windows:
#   .venv\Scripts\activate
# macOS/Linux:
#   source .venv/bin/activate
pip install -r requirements.txt
```

## Conda (optional)
```bash
conda env create -f environment.yml  # if available
conda activate afmo
```

If Seaborn is missing at runtime, the app falls back to Matplotlib or CSS-styled heatmaps automatically.
