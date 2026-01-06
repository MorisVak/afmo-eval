$ErrorActionPreference = "Stop"
# Auto-setup venv and run app with locked Windows deps (no manual steps)
$HERE = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $HERE

if (!(Test-Path ".venv")) {
  py -3.11 -m venv .venv
}
. .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .[ui]

streamlit run app/AFMo.py
