"""Pathway B – localised short-term pest-risk forecaster (separate from the leaf-image model).

Inputs : crop + 7 days of past weather + 3 days of forecast (Open-Meteo, or manual / cached for offline).
Model  : multinomial logistic regression on agro-meteorological features
         -> Low / Moderate / High probabilities + exact per-feature contributions (fully explainable).
Data   : if data/pest_observations.csv exists (real scouting data) it is used for training,
         otherwise a SYNTHETIC agronomy-rule dataset is generated (clearly labelled in the UI).

CLI:  python pest_forecast.py train      # (re)train and print metrics vs. heat/humidity baseline
      python pest_forecast.py demo       # offline demo prediction
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "pest_model.joblib"
CACHE_DIR = ROOT / "cache"
REAL_CSV = ROOT / "data" / "pest_observations.csv"

# Optimum temperature (C) and tolerance width for pest/disease build-up, with typical pests (informational).
CROPS = {
    "Cotton":    {"topt": 29, "w": 6, "pests": "Whitefly, jassids, aphids, pink bollworm"},
    "Groundnut": {"topt": 27, "w": 6, "pests": "Leaf miner, tikka leaf spot, aphids"},
    "Wheat":     {"topt": 20, "w": 6, "pests": "Aphids, rust"},
    "Tomato":    {"topt": 26, "w": 6, "pests": "Fruit borer, whitefly, blight"},
    "Potato":    {"topt": 18, "w": 5, "pests": "Late blight, aphids"},
    "Cumin":     {"topt": 22, "w": 5, "pests": "Aphids, blight, powdery mildew"},
}

DISTRICTS = {  # approximate district HQ coordinates (lat, lon)
    "Ahmedabad": (23.03, 72.58), "Gandhinagar": (23.22, 72.65), "Surat": (21.17, 72.83),
    "Vadodara": (22.31, 73.18), "Rajkot": (22.30, 70.80), "Bhavnagar": (21.76, 72.15),
    "Junagadh": (21.52, 70.46), "Jamnagar": (22.47, 70.07), "Mehsana": (23.59, 72.37),
    "Anand": (22.56, 72.95), "Bharuch": (21.71, 72.99), "Kutch (Bhuj)": (23.25, 69.67),
    "Banaskantha (Palanpur)": (24.17, 72.43), "Amreli": (21.60, 71.22),
    "Surendranagar": (22.73, 71.64), "Patan": (23.85, 72.13),
    "Sabarkantha (Himmatnagar)": (23.60, 72.97), "Navsari": (20.95, 72.93), "Valsad": (20.60, 72.93),
}

FEATURES = ["temp_suitability", "mean_rh", "rain_total", "humid_warm_days",
            "hot_dry_days", "fc_mean_rh", "fc_rain_total"]
FEATURE_TEXT = {
    "temp_suitability": "Temperature suitability for pests of this crop (7-day)",
    "mean_rh": "Average humidity, last 7 days (%)",
    "rain_total": "Rainfall, last 7 days (mm)",
    "humid_warm_days": "Warm + humid days, last 7 days",
    "hot_dry_days": "Hot + dry days, last 7 days",
    "fc_mean_rh": "Forecast humidity, next 3 days (%)",
    "fc_rain_total": "Forecast rainfall, next 3 days (mm)",
}
LEVEL_ORDER = ["Low", "Moderate", "High"]


# ----------------------------------------------------------------------------- features
def build_features(crop: str, past: pd.DataFrame, fc: pd.DataFrame) -> dict:
    """past/fc: DataFrames with columns tmean (C), rh (%), rain (mm/day)."""
    c = CROPS[crop]
    t, rh, rain = past["tmean"].to_numpy(float), past["rh"].to_numpy(float), past["rain"].to_numpy(float)
    suit = np.exp(-(((t - c["topt"]) / c["w"]) ** 2))
    f_rh = fc["rh"].to_numpy(float) if len(fc) else rh[-3:]
    f_rain = fc["rain"].to_numpy(float) if len(fc) else rain[-3:]
    return {
        "temp_suitability": float(suit.mean()),
        "mean_rh": float(rh.mean()),
        "rain_total": float(rain.sum()),
        "humid_warm_days": float(((rh >= 75) & (t >= 20) & (t <= 32)).sum()),
        "hot_dry_days": float(((t >= 35) & (rh < 40)).sum()),
        "fc_mean_rh": float(f_rh.mean()),
        "fc_rain_total": float(f_rain.sum()),
    }


# ----------------------------------------------------------------------------- training data
def synthetic_dataset(n=8000, seed=7) -> pd.DataFrame:
    """Agronomy-rule proxy data (NOT field observations). Replace with data/pest_observations.csv when available."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        crop = rng.choice(list(CROPS))
        base_t, base_rh = rng.uniform(10, 40), rng.uniform(20, 95)
        def days(k):
            tt = base_t + rng.normal(0, 2.5, k)
            rr = np.clip(base_rh + rng.normal(0, 8, k), 10, 100)
            wet = rng.random(k) < np.clip((base_rh - 45) / 80, 0.02, 0.6)
            rain = np.where(wet, rng.gamma(2.0, 4.0, k), 0.0)
            return pd.DataFrame({"tmean": tt, "rh": rr, "rain": rain})
        past, fc = days(7), days(3)
        f = build_features(crop, past, fc)
        score = (1.8 * f["temp_suitability"] + 0.035 * (f["mean_rh"] - 55) + 0.9 * f["humid_warm_days"] / 7
                 + 0.012 * min(f["rain_total"], 80) + 0.3 * f["hot_dry_days"] / 7
                 + 0.01 * (f["fc_mean_rh"] - 55) + 0.006 * min(f["fc_rain_total"], 40)
                 + rng.normal(0, 0.25))
        rows.append({**f, "crop": crop, "score": score})
    df = pd.DataFrame(rows)
    q1, q2 = df["score"].quantile([0.45, 0.80])
    df["risk"] = np.where(df["score"] < q1, "Low", np.where(df["score"] < q2, "Moderate", "High"))
    return df.drop(columns="score")


def baseline_rule(df: pd.DataFrame) -> np.ndarray:
    """Simple heat/humidity baseline to compare against (humidity-only thinking)."""
    return np.where(df["mean_rh"] >= 80, "High", np.where(df["mean_rh"] >= 60, "Moderate", "Low"))


def train(verbose=True) -> dict:
    if REAL_CSV.exists():
        df, source = pd.read_csv(REAL_CSV), "real observations (data/pest_observations.csv)"
    else:
        df, source = synthetic_dataset(), "SYNTHETIC agronomy-rule proxy data (replace with real scouting data)"
    X, y = df[FEATURES], df["risk"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    acc = accuracy_score(yte, pred)
    base_acc = accuracy_score(yte, baseline_rule(Xte))
    if verbose:
        print(f"Data source      : {source}")
        print(f"Model accuracy   : {acc:.3f}")
        print(f"Humidity baseline: {base_acc:.3f}")
        print(classification_report(yte, pred, digits=3))
    MODEL_PATH.parent.mkdir(exist_ok=True)
    bundle = {"pipe": pipe, "source": source, "accuracy": float(acc), "baseline_accuracy": float(base_acc),
              "trained_at": datetime.now().isoformat(timespec="seconds")}
    joblib.dump(bundle, MODEL_PATH)
    return bundle


def load_model() -> dict:
    if not MODEL_PATH.exists():
        return train(verbose=False)
    try:
        return joblib.load(MODEL_PATH)
    except Exception:  # e.g. different scikit-learn version -> retrain transparently
        return train(verbose=False)


# ----------------------------------------------------------------------------- weather
def _split_daily(hourly: dict) -> pd.DataFrame:
    h = pd.DataFrame(hourly)
    h["time"] = pd.to_datetime(h["time"])
    h["date"] = h["time"].dt.date
    d = h.groupby("date").agg(tmean=("temperature_2m", "mean"), rh=("relative_humidity_2m", "mean"),
                              rain=("precipitation", "sum")).reset_index()
    return d.dropna().reset_index(drop=True)


def fetch_weather(lat: float, lon: float, timeout=8):
    """Returns (daily_df, meta). Falls back to the last cached response if offline. Raises RuntimeError if neither."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / f"wx_{lat:.2f}_{lon:.2f}.json"
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=timeout, params={
            "latitude": lat, "longitude": lon, "past_days": 7, "forecast_days": 3, "timezone": "auto",
            "hourly": "temperature_2m,relative_humidity_2m,precipitation"})
        r.raise_for_status()
        js = r.json()
        js["_fetched_at"] = datetime.now().isoformat(timespec="seconds")
        cache.write_text(json.dumps(js))
        source = "live (Open-Meteo)"
    except Exception:
        if not cache.exists():
            raise RuntimeError("No internet and no cached weather for this location.")
        js = json.loads(cache.read_text())
        source = "cached copy (offline)"
    daily = _split_daily(js["hourly"])
    meta = {"source": source, "fetched_at": js.get("_fetched_at", "unknown"), "lat": lat, "lon": lon,
            "first_day": str(daily["date"].iloc[0]), "last_day": str(daily["date"].iloc[-1])}
    return daily, meta


def manual_daily(t, rh, rain7, fc_rh, fc_rain3) -> pd.DataFrame:
    """Offline fallback: build a 10-day frame from a few values entered by the farmer / extension worker."""
    dates = pd.date_range(end=pd.Timestamp.today().normalize() + pd.Timedelta(days=2), periods=10).date
    rhs = [rh] * 7 + [fc_rh] * 3
    rains = [rain7 / 7] * 7 + [fc_rain3 / 3] * 3
    return pd.DataFrame({"date": dates, "tmean": [t] * 10, "rh": rhs, "rain": rains})


# ----------------------------------------------------------------------------- prediction
def predict_risk(crop: str, daily: pd.DataFrame, meta: dict | None = None) -> dict:
    bundle = load_model()
    pipe = bundle["pipe"]
    daily = daily.reset_index(drop=True)
    if len(daily) < 4:
        raise ValueError("Need at least 4 days of weather data.")
    past, fc = daily.iloc[:-3].tail(7), daily.iloc[-3:]
    feats = build_features(crop, past, fc)
    x = pd.DataFrame([feats])[FEATURES]
    proba = pipe.predict_proba(x)[0]
    classes = list(pipe.classes_)
    probs = {c: float(p) for c, p in zip(classes, proba)}
    level = max(probs, key=probs.get)
    confidence = probs[level]

    # conservative rule: low confidence or short data window -> lean toward caution
    conservative = False
    if confidence < 0.55 or len(past) < 5:
        bumped = LEVEL_ORDER[min(LEVEL_ORDER.index(level) + 1, 2)]
        conservative = bumped != level
        level = bumped

    # exact linear contributions to the "High" logit (standardised features)
    scaler, lr = pipe.steps[0][1], pipe.steps[1][1]
    z = scaler.transform(x)[0]
    coef = lr.coef_[classes.index("High")]
    contrib = coef * z
    order = np.argsort(-np.abs(contrib))[:4]
    factors = [{"feature": FEATURE_TEXT[FEATURES[i]], "value": round(feats[FEATURES[i]], 2),
                "effect": "raises risk" if contrib[i] > 0 else "lowers risk",
                "strength": round(float(abs(contrib[i])), 2)} for i in order]

    return {"level": level, "probs": probs, "confidence": confidence, "conservative": conservative,
            "factors": factors, "features": feats, "crop": crop, "typical_pests": CROPS[crop]["pests"],
            "data_source": bundle["source"], "model_accuracy": bundle["accuracy"],
            "baseline_accuracy": bundle["baseline_accuracy"], "meta": meta or {},
            "window": f"{past['date'].iloc[0]} to {fc['date'].iloc[-1]}" if "date" in daily else "n/a"}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd == "train":
        train()
    else:
        out = predict_risk("Cotton", manual_daily(30, 82, 45, 85, 20), {"source": "manual demo"})
        print(json.dumps({k: out[k] for k in ("level", "probs", "confidence", "conservative", "factors")}, indent=2))
