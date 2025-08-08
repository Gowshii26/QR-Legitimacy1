# features_upi.py
import math
import random
import re
from urllib.parse import urlparse, parse_qs, unquote

# Known bank handles & risky handles (expand if you like)
BANK_HANDLES = [
    "@upi", "@ybl", "@oksbi", "@okaxis", "@okhdfcbank",
    "@paytm", "@icici", "@sbi", "@hdfc", "@axis", "@kotak", "@barodampay"
]
RISKY_HANDLES = [
    "@unknown", "@online", "@secure", "@verify", "@service",
    "@support", "@help", "@bonus", "@gift", "@pay", "@xpay"
]
SUS_NAME_KWS = {"verify", "account", "secure", "update", "signin", "login", "kYC", "kyc"}

VPA_REGEX = re.compile(r"^[a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+$")

def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freqs = {}
    for ch in s:
        freqs[ch] = freqs.get(ch, 0) + 1
    n = len(s)
    return -sum((c/n) * math.log2(c/n) for c in freqs.values())

def parse_upi_uri(upi_uri: str):
    """
    Returns dict with pa, pn, am (strings). Missing keys become ''.
    """
    if not upi_uri.startswith("upi://"):
        return {"pa": "", "pn": "", "am": ""}
    parsed = urlparse(upi_uri)
    q = parse_qs(parsed.query)
    pa = q.get("pa", [""])[0]
    pn = unquote(q.get("pn", [""])[0])
    am = q.get("am", [""])[0]
    return {"pa": pa, "pn": pn, "am": am}

def is_valid_vpa(vpa: str) -> bool:
    return bool(VPA_REGEX.match(vpa))

def extract_upi_features(upi_uri: str) -> dict:
    """
    Returns a pure-numeric feature dict for ML.
    """
    parsed = urlparse(upi_uri)
    q = parse_qs(parsed.query)
    pa = q.get("pa", [""])[0].lower()
    pn = unquote(q.get("pn", [""])[0]).lower()
    am_str = q.get("am", [""])[0]

    # core base features
    length = len(upi_uri)
    num_parameters = len(q)
    has_suspicious_keywords = int(any(k in pn for k in (kw.lower() for kw in SUS_NAME_KWS)))

    # VPA validity & handle
    valid_vpa = int(is_valid_vpa(pa))
    handle = pa.split("@", 1)[1] if "@" in pa else ""
    bank_handle = int(any(pa.endswith(h) for h in BANK_HANDLES))
    risky_handle = int(any(pa.endswith(h) for h in RISKY_HANDLES))

    # entropy and ratios
    local = pa.split("@", 1)[0] if "@" in pa else pa
    vpa_local_entropy = shannon_entropy(local)
    vpa_handle_entropy = shannon_entropy(handle)
    vpa_length = len(pa)
    digits_ratio = sum(ch.isdigit() for ch in pa) / max(1, len(pa))

    # amount flags
    try:
        amount = float(am_str)
    except Exception:
        amount = 0.0

    # round & large amounts are suspicious
    amount_is_integer = int(amount == int(amount))
    amount_suspicious = int(amount >= 100 and amount_is_integer)

    # proxies
    vpa_age_proxy = 75 if (valid_vpa and bank_handle) else 5
    location_risk_proxy = 0 if (valid_vpa and bank_handle) else 1

    features = {
        "is_upi": 1,
        "length": length,
        "num_parameters": num_parameters,
        "has_suspicious_keywords": has_suspicious_keywords,
        "is_valid_vpa": valid_vpa,
        "bank_handle": bank_handle,
        "risky_handle": risky_handle,
        "vpa_local_entropy": vpa_local_entropy,
        "vpa_handle_entropy": vpa_handle_entropy,
        "vpa_length": vpa_length,
        "digits_ratio": digits_ratio,
        "amount_value": amount,
        "amount_is_integer": amount_is_integer,
        "amount_suspicious": amount_suspicious,
        "vpa_age_proxy": vpa_age_proxy,
        "location_risk_proxy": location_risk_proxy,
    }
    return features

# fields used by the model (order matters for scaler)
FEATURE_COLUMNS = [
    # scaled
    "length",
    "vpa_local_entropy",
    "vpa_handle_entropy",
    "vpa_length",
    "digits_ratio",
    "amount_value",
    "vpa_age_proxy",
    # unscaled binary flags
    "num_parameters",
    "has_suspicious_keywords",
    "is_valid_vpa",
    "bank_handle",
    "risky_handle",
    "amount_is_integer",
    "amount_suspicious",
    "location_risk_proxy",
]
