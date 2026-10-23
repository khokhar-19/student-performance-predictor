# EduPredict — Student Performance Predictor

A responsive Flask dashboard for predicting a student's final mathematics grade. It compares Linear Regression, Decision Tree, and Random Forest models, then serves the best model in a simple web interface.

## Run on Windows

1. Install Python 3.11 or newer from [python.org](https://www.python.org/downloads/). During setup, select **Add Python to PATH**.
2. Open PowerShell in this directory and run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\start.ps1
```

The first run creates a virtual environment, installs packages, downloads the UCI Student Performance dataset, trains the models, and starts the server. Visit `http://127.0.0.1:5000`.

To stop the server, press `Ctrl+C` in PowerShell.

## Project layout

```text
app.py              Flask dashboard and prediction routes
train_model.py      Dataset download, model training, and evaluation
templates/          Responsive dashboard and prediction pages
static/style.css    Responsive visual design
models/             Generated model and evaluation metrics
```
