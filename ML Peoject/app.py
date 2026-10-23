from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.pkl"
METRICS_PATH = ROOT / "models" / "metrics.json"

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "local-development-key")


def metrics_data() -> dict | None:
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@app.route("/")
def index():
    return render_template("index.html", report=metrics_data())


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    values = request.form.to_dict()
    if request.method == "POST":
        if not MODEL_PATH.exists():
            flash("Model not trained yet. Run python train_model.py in the project folder, then try again.", "error")
            return redirect(url_for("predict"))
        try:
            row = pd.DataFrame([{
                "school": values["school"], "sex": values["sex"],
                "age": float(values["age"]), "studytime": float(values["studytime"]),
                "failures": float(values["failures"]), "absences": float(values["absences"]),
                "G1": float(values["G1"]), "G2": float(values["G2"]),
            }])
            score = float(joblib.load(MODEL_PATH).predict(row)[0])
            result = max(0, min(20, score))
        except (KeyError, TypeError, ValueError):
            flash("Please provide valid values for every field.", "error")
    return render_template("predict.html", result=result, values=values, report=metrics_data())


if __name__ == "__main__":
    app.run(debug=True)
