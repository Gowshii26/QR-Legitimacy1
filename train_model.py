# train_model.py (only the splitting part changed)
import os, json, pandas as pd, joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score, recall_score
from sklearn.ensemble import HistGradientBoostingClassifier

DATA_PATH = os.path.join("data", "qr_data_upi.csv")
TEST_SPLIT_PATH = os.path.join("data", "test_split.csv")
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
MODEL_PATH  = os.path.join(MODEL_DIR, "qr_model.pkl")

FEATURE_COLUMNS = [
    "length","vpa_local_entropy","vpa_handle_entropy","vpa_length",
    "digits_ratio","amount_value","vpa_age_proxy",
    "num_parameters","has_suspicious_keywords","is_valid_vpa",
    "bank_handle","risky_handle","amount_is_integer","amount_suspicious",
    "location_risk_proxy",
]
SCALED_COLS = ["length","vpa_local_entropy","vpa_handle_entropy","vpa_length","digits_ratio","amount_value","vpa_age_proxy"]

def main():
    print("Starting train_model.py...")
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLUMNS].copy()
    y = df["label"].astype(int).values
    groups = df["group_vpa"].astype(str).values

    # Grouped split (80/20)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups=groups))
    X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
    y_train, y_test = y[tr_idx], y[te_idx]
    meta_test = df.iloc[te_idx][["qr_content","label"]]

    scaler = StandardScaler()
    X_tr_s = X_train.copy(); X_te_s = X_test.copy()
    X_tr_s[SCALED_COLS] = scaler.fit_transform(X_tr_s[SCALED_COLS])
    X_te_s[SCALED_COLS]  = scaler.transform(X_te_s[SCALED_COLS])

    smote = SMOTE(random_state=42, k_neighbors=5)
    X_bal, y_bal = smote.fit_resample(X_tr_s, y_train)

    clf = HistGradientBoostingClassifier(
        max_depth=8, learning_rate=0.05, max_iter=400,
        early_stopping=True, random_state=42
    )
    clf.fit(X_bal, y_bal)

    y_pred = clf.predict(X_te_s)
    acc = accuracy_score(y_test, y_pred)
    rec_fraud = recall_score(y_test, y_pred, pos_label=1)
    rec_legit = recall_score(y_test, y_pred, pos_label=0)

    print("\n=== Test Metrics (Grouped Holdout) ===")
    print(f"Accuracy       : {acc:.3f}")
    print(f"Recall Fra u d : {rec_fraud:.3f}")
    print(f"Recall Legit   : {rec_legit:.3f}\n")
    print(classification_report(y_test, y_pred, target_names=["Legitimate","Fraudulent"]))

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(clf, MODEL_PATH)
    print(f"✅ Saved scaler → {SCALER_PATH}")
    print(f"✅ Saved model  → {MODEL_PATH}")

    out = meta_test.copy()
    out = pd.concat([out.reset_index(drop=True), X_test.reset_index(drop=True)], axis=1)
    out.to_csv(TEST_SPLIT_PATH, index=False)
    print(f"✅ Saved holdout test split → {TEST_SPLIT_PATH}")

    with open("training_metrics.json","w",encoding="utf-8") as f:
        json.dump({"accuracy": acc, "recall_fraud": rec_fraud, "recall_legit": rec_legit}, f, indent=2)

if __name__ == "__main__":
    main()
