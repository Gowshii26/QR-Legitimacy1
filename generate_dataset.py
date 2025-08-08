# generate_dataset.py
import os
import random
import numpy as np
import pandas as pd
from urllib.parse import quote

from features_upi import (
    BANK_HANDLES, RISKY_HANDLES,
    extract_upi_features, FEATURE_COLUMNS
)

# -----------------------
# Controls: tune these if you want the task easier/harder
# -----------------------
SEED = 42
N_LEGIT = 2000
N_FRAUD = 4000

# Overlap probabilities (make classes look more alike)
# Legit sometimes looks a bit risky; Fraud sometimes looks legit.
LEGIT_RISKY_HANDLE_P = 0.15     # legit using a risky-looking handle
LEGIT_SUS_NAME_P      = 0.10     # legit with suspicious-ish payee name
LEGIT_ROUND_AMT_P     = 0.20     # legit using round amounts occasionally

FRAUD_BANK_HANDLE_P   = 0.40     # fraud camouflaging with bank handle
FRAUD_NORMAL_NAME_P   = 0.40     # fraud sometimes uses normal human names
FRAUD_NONROUND_AMT_P  = 0.30     # fraud sometimes uses non-round amounts

# Label noise (simulates annotation errors / ambiguous cases)
LABEL_FLIP_P = 0.04

# Small numeric jitter (keeps features realistic but non-deterministic)
JITTER_LENGTH_RANGE   = (-3, 3)
JITTER_AMOUNT_SCALE   = (0.97, 1.03)
JITTER_AGE_SHIFT      = (-4, 4)

random.seed(SEED)
np.random.seed(SEED)

OUT_PATH = os.path.join("data", "qr_data_upi.csv")
os.makedirs("data", exist_ok=True)

FIRST_NAMES = ["Rahul", "Sneha", "Arjun", "Priya", "Vikram", "Anita", "Kiran", "Neha", "Aman", "Pooja"]
LAST_NAMES  = ["Sharma", "Patel", "Reddy", "Iyer", "Khan", "Gupta", "Nair", "Das", "Singh", "Chawla"]

SUS_NAMES = ["Verify Account", "Secure Login", "Payment Support", "Update KYC", "Help Desk", "Service Team"]

def random_human_name():
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"

def make_local_from_name(name: str) -> str:
    base = (name.split()[0] + random.choice(["", ".", "_"]) + name.split()[-1]).replace(" ", "").lower()
    if random.random() < 0.5:
        base += str(random.randint(1, 999))
    return base

def legit_upi_uri():
    """
    Legitimate but with some noise/overlap:
    - 85% bank handle, 15% risky-looking handle
    - 10% suspicious-ish name
    - 20% round amounts (to overlap with fraud)
    """
    handle_pool = BANK_HANDLES if random.random() >= LEGIT_RISKY_HANDLE_P else RISKY_HANDLES
    handle = random.choice(handle_pool)

    name = random_human_name()
    if random.random() < LEGIT_SUS_NAME_P:
        name = random.choice(SUS_NAMES)

    local = make_local_from_name(name)
    pa = f"{local}{handle}"

    if random.random() < LEGIT_ROUND_AMT_P:
        amount = float(random.choice([100, 200, 300, 500, 1000]))
    else:
        amount = round(random.uniform(10, 500), 2)
        if amount.is_integer():
            amount += 0.25  # push away from perfect integer

    pn = quote(name)
    return f"upi://pay?pa={pa}&pn={pn}&am={amount:.2f}", pa

def fraud_upi_uri():
    """
    Fraudulent with camouflage overlap:
    - 60% risky handle, 40% bank handle
    - 60% suspicious name, 40% normal name
    - 70% round/large amounts; 30% non-round amounts
    """
    handle_pool = RISKY_HANDLES if random.random() >= FRAUD_BANK_HANDLE_P else BANK_HANDLES
    handle = random.choice(handle_pool)

    if random.random() < FRAUD_NORMAL_NAME_P:
        name = random_human_name()
    else:
        name = random.choice(SUS_NAMES)

    # machine-ish local
    base = name.split()[0].lower()
    local = base + str(random.randint(100, 9999))
    if random.random() < 0.3:
        local += random.choice(["help", "secure", "pay", "care"])
    pa = f"{local}{handle}"

    if random.random() >= FRAUD_NONROUND_AMT_P:
        amount = float(random.choice([100, 200, 300, 500, 900, 1000, 1500, 5000]))
    else:
        amount = round(random.uniform(10, 500), 2)
        if amount.is_integer():
            amount += 0.20

    pn = quote(name)
    return f"upi://pay?pa={pa}&pn={pn}&am={amount:.2f}", pa

def build_rows(n_legit=N_LEGIT, n_fraud=N_FRAUD):
    rows = []

    # Legit rows
    for _ in range(n_legit):
        qr, pa = legit_upi_uri()
        feats = extract_upi_features(qr)

        # Soften vpa_age_proxy to overlapping ranges
        if feats["is_valid_vpa"] == 1 and feats["bank_handle"] == 1:
            feats["vpa_age_proxy"] = random.randint(55, 85)
        else:
            feats["vpa_age_proxy"] = random.randint(10, 55)  # overlapping with fraud

        feats["label"] = 0
        feats["qr_content"] = qr
        feats["group_vpa"] = pa
        rows.append(feats)

    # Fraud rows
    for _ in range(n_fraud):
        qr, pa = fraud_upi_uri()
        feats = extract_upi_features(qr)

        if feats["is_valid_vpa"] == 1 and feats["bank_handle"] == 1:
            feats["vpa_age_proxy"] = random.randint(35, 75)
        else:
            feats["vpa_age_proxy"] = random.randint(5, 60)

        feats["label"] = 1
        feats["qr_content"] = qr
        feats["group_vpa"] = pa
        rows.append(feats)

    random.shuffle(rows)
    df = pd.DataFrame(rows)

    # -----------------------
    # Controlled label noise
    # -----------------------
    flip_mask = np.random.rand(len(df)) < LABEL_FLIP_P
    df.loc[flip_mask, "label"] = 1 - df.loc[flip_mask, "label"]

    # -----------------------
    # Numeric jitter (keep realistic)
    # -----------------------
    # length jitter
    df["length"] = df["length"] + np.random.randint(JITTER_LENGTH_RANGE[0], JITTER_LENGTH_RANGE[1] + 1, size=len(df))
    df["length"] = df["length"].clip(lower=1)

    # amount jitter → then recompute amount flags for internal consistency
    scale = np.random.uniform(JITTER_AMOUNT_SCALE[0], JITTER_AMOUNT_SCALE[1], size=len(df))
    df["amount_value"] = (df["amount_value"] * scale).round(2)
    # recompute amount flags
    df["amount_is_integer"] = (np.isclose(df["amount_value"] % 1, 0)).astype(int)
    df["amount_suspicious"] = ((df["amount_value"] >= 100) & (df["amount_is_integer"] == 1)).astype(int)

    # vpa_age_proxy small shift
    df["vpa_age_proxy"] = (df["vpa_age_proxy"] + np.random.randint(JITTER_AGE_SHIFT[0], JITTER_AGE_SHIFT[1] + 1, size=len(df))).clip(lower=0, upper=100)

    # Keep column order stable
    ordered = ["qr_content", "label", "group_vpa"] + FEATURE_COLUMNS
    df = df[ordered]
    return df

def main():
    df = build_rows()
    df.to_csv(OUT_PATH, index=False)
    print(f"✅ Wrote {len(df):,} rows → {OUT_PATH}")

if __name__ == "__main__":
    main()
