# Run this from PowerShell with: .\start.ps1
$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

if (-not (Test-Path "models\best_model.pkl")) {
    & .\.venv\Scripts\python.exe train_model.py
}

& .\.venv\Scripts\python.exe app.py
