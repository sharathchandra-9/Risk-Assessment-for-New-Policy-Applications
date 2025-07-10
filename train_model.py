import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


def generate_synthetic_data(n: int = 10_000) -> pd.DataFrame:
    """
    Create a synthetic underwriting dataset and label each row
    with a discrete risk class (0 = low, 1 = medium, 2 = high).
    """
    rng = np.random.default_rng(seed=42)
    data = pd.DataFrame({
        "age": rng.integers(18, 85, n),
        "ssn_last4": rng.integers(0, 9999, n),
        "policy_type": rng.choice(
            ["term_life", "whole_life", "health", "auto", "homeowners"], n
        ),
        "coverage_amount": rng.integers(1_000, 1_000_000, n),
        "term_length": rng.integers(1, 31, n),
        "payment_frequency": rng.choice(
            ["monthly", "quarterly", "semiannually", "annually"], n
        ),
        "height": rng.integers(48, 96, n),      # inches
        "weight": rng.integers(50, 500, n),      # lbs
        "tobacco_use": rng.choice(["yes", "no"], n, p=[0.2, 0.8]),
        "medical_conditions": rng.choice(
            ["none", "diabetes", "hypertension", "heart_disease", "cancer", "respiratory"],
            n,
        ),
        "annual_income": rng.integers(10_000, 500_000, n),
        "credit_score": rng.integers(300, 850, n),
        "bankruptcies": rng.integers(0, 4, n),
        "high_risk_hobbies": rng.choice(
            ["none", "skydiving", "scuba diving"], n, p=[0.8, 0.1, 0.1]
        ),
    })

    def calc_risk(row) -> int:
        score = 0
        if row["coverage_amount"] > 250_000:
            score += 1
        if row["tobacco_use"] == "yes":
            score += 2
        if row["medical_conditions"] != "none":
            score += 2
        if row["credit_score"] < 600:
            score += 1
        if row["bankruptcies"] > 0:
            score += 1
        if row["high_risk_hobbies"] != "none":
            score += 1
        return 0 if score <= 1 else (1 if score <= 3 else 2)

    data["risk_class"] = data.apply(calc_risk, axis=1)
    return data


if __name__ == "__main__":
    df = generate_synthetic_data()
    X, y = df.drop("risk_class", axis=1), df["risk_class"]

    cat_cols = [
        "policy_type",
        "payment_frequency",
        "tobacco_use",
        "medical_conditions",
        "high_risk_hobbies",
    ]
    num_cols = [c for c in X.columns if c not in cat_cols + ["ssn_last4"]]

    preprocessor = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="passthrough",
    )

    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
            )),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    model.fit(X_train, y_train)

    print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))
    print(classification_report(y_test, model.predict(X_test),
                                target_names=["Low", "Medium", "High"]))

    joblib.dump(model, "risk_assessment_model.pkl")
    print("Model saved to risk_assessment_model.pkl")
