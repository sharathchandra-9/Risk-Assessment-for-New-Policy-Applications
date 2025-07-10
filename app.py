from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS  # simple CORS – handy for local dev

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "risk_assessment_model.pkl"

app = Flask(__name__, template_folder="templates")
CORS(app)  # remove or tighten for production

# -------- Load trained model --------
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found ({MODEL_PATH}). Run train_model.py first."
    )
model = joblib.load(MODEL_PATH)


# -------- Routes --------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Receive JSON from frontend, massage it into the
    same feature schema used during training, and return a risk label.
    """
    try:
        data = request.get_json(force=True)
        # ---------- Feature engineering ----------
        dob = datetime.strptime(data["dob"], "%Y-%m-%d")
        today = datetime.today()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

        # Map/clean raw inputs ------------
        def safe_int(x, default=0):
            try:
                return int(x)
            except Exception:
                return default

        medical_conditions = data.get("medicalConditions")
        if isinstance(medical_conditions, list):
            medical_conditions = (
                "none" if "none" in medical_conditions else medical_conditions[0]
            )
        medical_conditions = medical_conditions or "none"

        hobbies = (data.get("hobbies") or "").strip().lower()
        high_risk_hobbies = (
            hobbies if hobbies in {"skydiving", "scuba diving"} else "none"
        )

        # Build single-row DataFrame -------------
        row = {
            "age": age,
            "ssn_last4": safe_int(data.get("ssn")),
            "policy_type": data.get("policyType"),
            "coverage_amount": safe_int(data.get("coverageAmount")),
            "term_length": safe_int(data.get("termLength")),
            "payment_frequency": data.get("paymentFrequency"),
            "height": safe_int(data.get("height") or 60),
            "weight": safe_int(data.get("weight") or 150),
            "tobacco_use": data.get("tobaccoUse"),
            "medical_conditions": medical_conditions,
            "annual_income": safe_int(data.get("annualIncome")),
            "credit_score": safe_int(data.get("creditScore")),
            "bankruptcies": safe_int(data.get("bankruptcies")),
            "high_risk_hobbies": high_risk_hobbies,
        }
        input_df = pd.DataFrame([row])

        # -------- Predict ----------
        class_idx = int(model.predict(input_df)[0])
        risk_map = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}

        return jsonify({"risk": risk_map.get(class_idx, "Unknown")})

    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    # Gunicorn/production servers should set app.run(..., debug=False)
    app.run(debug=True, host="0.0.0.0", port=5000)
