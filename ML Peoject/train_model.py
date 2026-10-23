"""Train and evaluate student grade prediction models.

The script downloads the public UCI dataset the first time it is run, then:
    python train_model.py
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "student-mat.csv"
DATASET_ARCHIVE = ROOT / "student.zip"
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
MODELS_DIR = ROOT / "models"
TARGET = "G3"

# Kept deliberately small so the prediction form remains useful and understandable.
FEATURES = ["school", "sex", "age", "studytime", "failures", "absences", "G1", "G2"]


def demo_data() -> pd.DataFrame:
    """Create a deterministic demo set when the public source is unavailable."""
    rng = __import__("numpy").random.default_rng(42)
    size = 500
    g1 = rng.integers(0, 21, size)
    g2 = rng.integers(0, 21, size)
    failures = rng.choice([0, 1, 2, 3], size=size, p=[.74, .16, .07, .03])
    absences = rng.integers(0, 31, size)
    grade = (0.28 * g1 + 0.58 * g2 + rng.normal(0, 1.7, size)
             - 0.75 * failures - 0.025 * absences)
    return pd.DataFrame({
        "school": rng.choice(["GP", "MS"], size=size, p=[.87, .13]),
        "sex": rng.choice(["F", "M"], size=size),
        "age": rng.integers(15, 22, size),
        "studytime": rng.integers(1, 5, size),
        "failures": failures,
        "absences": absences,
        "G1": g1,
        "G2": g2,
        "G3": grade.clip(0, 20),
    })


def load_data() -> tuple[pd.DataFrame, str]:
    if not DATASET_PATH.exists():
        print("Downloading the UCI Student Performance dataset...")
        try:
            with urlopen(DATASET_URL, timeout=12) as response:
                DATASET_ARCHIVE.write_bytes(response.read())
            with ZipFile(DATASET_ARCHIVE) as archive:
                archive.extract("student-mat.csv", ROOT)
        except Exception as error:
            print("UCI download unavailable; using generated demo data instead.")
            return demo_data(), "Generated demo data"
        finally:
            if DATASET_ARCHIVE.exists():
                DATASET_ARCHIVE.unlink()
    data = pd.read_csv(DATASET_PATH, sep=";")
    missing = set(FEATURES + [TARGET]) - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing))}")
    return data, "UCI Student Performance (student-mat.csv)"


def main() -> None:
    data, source = load_data()
    x, y = data[FEATURES], data[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    numeric = [name for name in FEATURES if name not in {"school", "sex"}]
    categorical = ["school", "sex"]
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])

    candidates = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=5),
        "Random Forest": RandomForestRegressor(random_state=42, n_estimators=250, min_samples_leaf=2),
    }
    metrics: dict[str, dict[str, float]] = {}
    fitted: dict[str, Pipeline] = {}
    for name, estimator in candidates.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        metrics[name] = {
            "MAE": round(float(mean_absolute_error(y_test, predictions)), 3),
            "MSE": round(float(mean_squared_error(y_test, predictions)), 3),
            "R2": round(float(r2_score(y_test, predictions)), 3),
        }
        fitted[name] = pipeline

    best_name = min(metrics, key=lambda name: metrics[name]["MAE"])
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(fitted[best_name], MODELS_DIR / "best_model.pkl")
    summary = {
        "best_model": best_name,
        "metrics": metrics,
        "dataset": {"records": int(len(data)), "features": len(FEATURES), "target": TARGET, "source": source},
    }
    (MODELS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {best_name} and metrics to {MODELS_DIR}")


if __name__ == "__main__":
    main()
