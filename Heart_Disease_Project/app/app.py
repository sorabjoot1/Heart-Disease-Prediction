# Simple Flask backend: loads the saved pickle model and predicts from the form input.
import json
import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request

BASE = Path(__file__).resolve().parent
with open(BASE / "model" / "best_model.pkl", "rb") as f:
    model = pickle.load(f)                      # full pipeline: preprocessing + best model
with open(BASE / "model" / "model_info.json") as f:
    info = json.load(f)
FEATURES = info["feature_order"]                # same column order used in training

# Form fields: name -> (label, type, options or (min, max, step))
FIELDS = {
    "age": ("Age (years)", "number", (1, 120, 1)),
    "sex": ("Sex", "select", {0: "Female", 1: "Male"}),
    "cp": ("Chest pain type", "select", {0: "Typical angina", 1: "Atypical angina", 2: "Non-anginal pain", 3: "Asymptomatic"}),
    "trestbps": ("Resting blood pressure (mm Hg)", "number", (50, 250, 1)),
    "chol": ("Serum cholesterol (mg/dl)", "number", (50, 700, 1)),
    "fbs": ("Fasting blood sugar > 120 mg/dl", "select", {0: "No", 1: "Yes"}),
    "restecg": ("Resting ECG result", "select", {0: "Normal", 1: "ST-T wave abnormality", 2: "Left ventricular hypertrophy"}),
    "thalach": ("Maximum heart rate achieved", "number", (50, 250, 1)),
    "exang": ("Exercise-induced angina", "select", {0: "No", 1: "Yes"}),
    "oldpeak": ("ST depression (oldpeak)", "number", (0, 10, 0.1)),
    "slope": ("Slope of peak exercise ST segment", "select", {0: "Upsloping", 1: "Flat", 2: "Downsloping"}),
    "ca": ("Major vessels colored by fluoroscopy", "select", {0: "0", 1: "1", 2: "2", 3: "3", 4: "4"}),
    "thal": ("Thalassemia (dataset code)", "select", {0: "0 - Null", 1: "1 - Fixed defect", 2: "2 - Normal", 3: "3 - Reversible defect"}),
}

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    result, values = None, {}
    if request.method == "POST":
        values = request.form.to_dict()
        # Build one-row DataFrame with the exact training column order
        row = pd.DataFrame([[float(values[c]) for c in FEATURES]], columns=FEATURES)
        pred = int(model.predict(row)[0])
        prob = float(model.predict_proba(row)[0][1])
        result = {"pred": pred, "prob": round(prob * 100, 1)}
    return render_template("index.html", fields=FIELDS, values=values, result=result,
                           model_name=info["model_name"], metrics=info["test_metrics"])


if __name__ == "__main__":
    app.run(debug=False, port=5000)
