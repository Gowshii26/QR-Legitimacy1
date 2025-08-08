# test_model.py
import os
import json
import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score, recall_score

# === Paths ===
DATA_PATH = os.path.join("data", "test_split.csv")
MODEL_PATH = os.path.join("model", "qr_model.pkl")
SCALER_PATH = os.path.join("model", "scaler.pkl")
RESULTS_PATH = os.path.join("all_results.json")

# === Must match train_model.py ===
FEATURE_COLUMNS = [
    "length","vpa_local_entropy","vpa_handle_entropy","vpa_length",
    "digits_ratio","amount_value","vpa_age_proxy",
    "num_parameters","has_suspicious_keywords","is_valid_vpa",
    "bank_handle","risky_handle","amount_is_integer","amount_suspicious",
    "location_risk_proxy",
]
SCALED_COLS = ["length","vpa_local_entropy","vpa_handle_entropy","vpa_length","digits_ratio","amount_value","vpa_age_proxy"]

def main():
    print("Starting test_model.py...")

    # Load test split
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Test split file not found → {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if not set(FEATURE_COLUMNS).issubset(df.columns):
        raise ValueError(f"Missing required feature columns in test CSV. Found: {df.columns.tolist()}")

    X_test = df[FEATURE_COLUMNS].copy()
    y_test = df["label"].astype(int).values
    qr_content = df["qr_content"].tolist()

    # Load scaler and model
    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)

    # Scale
    X_test[SCALED_COLS] = scaler.transform(X_test[SCALED_COLS])

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    rec_fraud = recall_score(y_test, y_pred, pos_label=1)
    rec_legit = recall_score(y_test, y_pred, pos_label=0)

    print("\n=== Test Metrics ===")
    print(f"Accuracy       : {acc:.3f}")
    print(f"Recall Fraud   : {rec_fraud:.3f}")
    print(f"Recall Legit   : {rec_legit:.3f}\n")
    print(classification_report(y_test, y_pred, target_names=["Legitimate","Fraudulent"]))

    # Save results to JSON
    results = []
    for i, content in enumerate(qr_content):
        results.append({
            "qr_content": content,
            "true_label": int(y_test[i]),
            "predicted_label": int(y_pred[i]),
            "status": "Fraudulent" if y_pred[i] == 1 else "Legitimate"
        })

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Saved predictions → {RESULTS_PATH}")

if __name__ == "__main__":
    main()
