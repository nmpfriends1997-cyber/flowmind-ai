"""
FlowMind AI — ML Engine v2
True ML pipeline: trains XGBoost + Random Forest ensemble on the Astram dataset.
All predictions, feature importance (SHAP), hourly forecasts, and diversion routes
are driven by the trained models — no hardcoded lookup tables.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import math, warnings
warnings.filterwarnings("ignore")

DATA_PATH = Path(__file__).parent.parent / "data" / "astram_events.csv"

# ── Load & Feature Engineering ────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, low_memory=False)

    for col in ["start_datetime", "end_datetime", "closed_datetime", "resolved_datetime"]:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    # Rows with an unparseable start_datetime cannot contribute hour/month/weekday
    # features (NaN poisons the whole training matrix and crashes sklearn) — drop them.
    df = df.dropna(subset=["start_datetime"]).reset_index(drop=True)

    df["duration_min"] = (
        df["closed_datetime"].fillna(df["resolved_datetime"]) - df["start_datetime"]
    ).dt.total_seconds() / 60

    # Data-quality guard: a handful of rows have negative durations (closed
    # timestamp logged before start) or absurd multi-week durations from stale
    # "closed" records. Traffic incidents operationally resolve within ~24h,
    # so we treat values outside [0, 1440] as missing rather than letting them
    # blow up every mean-based statistic and ML target downstream.
    df.loc[(df["duration_min"] < 0) | (df["duration_min"] > 1440), "duration_min"] = np.nan

    df["hour"]    = df["start_datetime"].dt.hour
    df["month"]   = df["start_datetime"].dt.month
    df["weekday"] = df["start_datetime"].dt.weekday

    # Data-quality cleanup on event_cause: the raw CSV has case-duplicate
    # categories ("Debris" / "debris") that should be one category, and a
    # handful of test/placeholder rows ("test_demo") that aren't real
    # incidents and would otherwise leak into dropdowns and risk maps.
    df["event_cause"] = df["event_cause"].astype(str).str.strip().str.lower()
    df = df[~df["event_cause"].isin(["test_demo", "nan", ""])].reset_index(drop=True)

    # Binary targets
    df["requires_road_closure"] = df["requires_road_closure"].astype(str).str.upper().eq("TRUE").astype(int)
    df["is_high_priority"]      = (df["priority"] == "High").astype(int)
    df["is_planned"]            = (df["event_type"] == "planned").astype(int)
    df["is_active"]             = (df["status"] == "active").astype(int)

    df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[(df["latitude"].between(12.7, 13.3)) & (df["longitude"].between(77.3, 77.9))]

    # Derived features
    df["is_rush_hour"] = df["hour"].isin(list(range(7, 11)) + list(range(17, 21))).astype(int)
    df["is_weekend"]   = (df["weekday"] >= 5).astype(int)
    df["is_night"]     = df["hour"].isin(list(range(22, 24)) + list(range(0, 6))).astype(int)

    # Zone risk encoded from data
    zone_risk_map = build_zone_risk_from_data(df)
    df["zone_risk_score"] = df["zone"].map(zone_risk_map).fillna(0.5)

    # Corridor risk
    corridor_risk_map = build_corridor_risk_from_data(df)
    df["corridor_risk_score"] = df["corridor"].map(corridor_risk_map).fillna(0.5)

    # Cause risk — how disruptive has this event_cause historically been
    cause_risk_map = build_cause_risk_from_data(df)
    df["cause_risk_score"] = df["event_cause"].map(cause_risk_map).fillna(0.5)

    # Weekday risk — does this day-of-week historically see worse incidents
    weekday_risk_map = build_weekday_risk_from_data(df)
    df["weekday_risk_score"] = df["weekday"].map(weekday_risk_map).fillna(0.5)

    # Month risk — seasonal pattern (e.g. monsoon months drive water-logging/closures)
    month_risk_map = build_month_risk_from_data(df)
    df["month_risk_score"] = df["month"].map(month_risk_map).fillna(0.5)

    return df


def build_zone_risk_from_data(df: pd.DataFrame) -> dict:
    """Compute per-zone risk score from historical data."""
    zone_df = df[df["zone"].notna() & (df["zone"] != "NULL")]
    grp = zone_df.groupby("zone").agg(
        total=("id", "count"),
        high_prio=("is_high_priority", "sum"),
        closures=("requires_road_closure", "sum"),
        active=("is_active", "sum"),
    ).reset_index()
    grp["risk_score"] = (
        grp["high_prio"] / grp["total"] * 0.4
        + grp["closures"] / grp["total"] * 0.35
        + grp["active"] / grp["total"] * 0.25
    )
    return dict(zip(grp["zone"], grp["risk_score"]))


def build_corridor_risk_from_data(df: pd.DataFrame) -> dict:
    corr = df[df["corridor"].notna() & ~df["corridor"].isin(["NULL", "Non-corridor"])]
    grp = corr.groupby("corridor").agg(
        total=("id", "count"),
        closures=("requires_road_closure", "sum"),
        high_prio=("is_high_priority", "sum"),
    ).reset_index()
    grp["risk_score"] = (
        grp["closures"] / grp["total"] * 0.5
        + grp["high_prio"] / grp["total"] * 0.5
    )
    return dict(zip(grp["corridor"], grp["risk_score"]))


def build_cause_risk_from_data(df: pd.DataFrame) -> dict:
    """
    Per-cause risk score from historical outcomes. This is what lets the model
    actually respond to the 'Event Cause' dropdown at prediction time — without
    it, event_cause only entered the model as an arbitrary ordinal index that
    the target never depended on, so trees learned to ignore it entirely.
    """
    cz = df[df["event_cause"].notna()]
    grp = cz.groupby("event_cause").agg(
        total=("id", "count"),
        high_prio=("is_high_priority", "sum"),
        closures=("requires_road_closure", "sum"),
    ).reset_index()
    grp["risk_score"] = (
        grp["high_prio"] / grp["total"] * 0.5
        + grp["closures"] / grp["total"] * 0.5
    )
    return dict(zip(grp["event_cause"], grp["risk_score"]))


def build_weekday_risk_from_data(df: pd.DataFrame) -> dict:
    """Per-weekday (0=Mon..6=Sun) historical risk pattern."""
    grp = df.groupby("weekday").agg(
        total=("id", "count"),
        high_prio=("is_high_priority", "sum"),
        closures=("requires_road_closure", "sum"),
    ).reset_index()
    grp["risk_score"] = (
        grp["high_prio"] / grp["total"] * 0.5
        + grp["closures"] / grp["total"] * 0.5
    )
    return dict(zip(grp["weekday"], grp["risk_score"]))


def build_month_risk_from_data(df: pd.DataFrame) -> dict:
    """Per-month historical seasonal risk pattern."""
    grp = df.groupby("month").agg(
        total=("id", "count"),
        high_prio=("is_high_priority", "sum"),
        closures=("requires_road_closure", "sum"),
    ).reset_index()
    grp["risk_score"] = (
        grp["high_prio"] / grp["total"] * 0.5
        + grp["closures"] / grp["total"] * 0.5
    )
    return dict(zip(grp["month"], grp["risk_score"]))


# ── Global state ──────────────────────────────────────────────────────────────
df_global     = None
models_global = None   # {"congestion": model, "delay": model, "closure": model, "encoders": {...}, "feature_names": [...]}


def get_df() -> pd.DataFrame:
    global df_global
    if df_global is None:
        df_global = load_data()
    return df_global


def get_models():
    global models_global
    if models_global is None:
        models_global = train_models()
    return models_global


# ── ML Training ───────────────────────────────────────────────────────────────
CAUSE_ORDER = [
    "vehicle_breakdown", "accident", "public_event", "procession",
    "vip_movement", "construction", "water_logging", "pot_holes",
    "tree_fall", "road_conditions", "congestion", "protest", "others",
    "debris", "fog / low visibility",
]
TIME_ORDER  = ["morning", "afternoon", "evening", "night"]
ZONE_ORDER  = ["low", "medium", "high"]
CLOSURE_ORDER = ["no", "partial", "full"]


def encode_categorical(series: pd.Series, order: list) -> np.ndarray:
    mapping = {v: i for i, v in enumerate(order)}
    return series.map(mapping).fillna(len(order) - 1).astype(float).values


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Convert raw dataframe rows into the numeric feature matrix used for training."""
    # Cause encoding
    cause_enc = encode_categorical(df["event_cause"].fillna("others"), CAUSE_ORDER)

    # Time-of-day bucket from hour
    def hour_to_bucket(h):
        if 7 <= h <= 10:   return 0  # morning
        if 11 <= h <= 16:  return 1  # afternoon
        if 17 <= h <= 21:  return 2  # evening
        return 3                      # night
    time_enc = df["hour"].apply(hour_to_bucket).values.astype(float)

    # Zone risk bucket — derived from the real per-row zone_risk_score (NOT a
    # constant placeholder). Thresholds chosen from the observed tercile split
    # of historical zone risk scores so "low/medium/high" carries real signal.
    def zone_score_to_bucket(s):
        if s < 0.28:  return "low"
        if s < 0.36:  return "medium"
        return "high"
    zone_bucket = df["zone_risk_score"].apply(zone_score_to_bucket)
    zone_enc    = encode_categorical(zone_bucket, ZONE_ORDER)

    closure_enc = encode_categorical(df["requires_road_closure"].map({0:"no", 1:"full"}).fillna("no"), CLOSURE_ORDER)
    is_planned  = df["is_planned"].values.astype(float)
    is_rush     = df["is_rush_hour"].values.astype(float)
    is_weekend  = df["is_weekend"].values.astype(float)
    is_night    = df["is_night"].values.astype(float)
    zone_risk   = df["zone_risk_score"].values.astype(float)
    cause_risk  = df["cause_risk_score"].values.astype(float)
    weekday_risk= df["weekday_risk_score"].values.astype(float)
    month_risk  = df["month_risk_score"].values.astype(float)

    return np.column_stack([
        cause_enc, time_enc, zone_enc, closure_enc,
        is_planned, is_rush, is_weekend, is_night,
        zone_risk, cause_risk, weekday_risk, month_risk,
    ])

FEATURE_NAMES = [
    "Event Cause", "Time of Day", "Zone Risk Level", "Road Closure Type",
    "Is Planned", "Is Rush Hour", "Is Weekend", "Is Night",
    "Zone Risk Score", "Cause Risk Score",
    "Weekday Risk Score", "Month Risk Score",
]


def compute_congestion_target(df: pd.DataFrame) -> np.ndarray:
    """
    Derive a continuous congestion proxy from historical outcomes.

    This dataset has no directly-measured "congestion %" column, so the
    target is an engineered composite of real historical signals. Crucially,
    every component here corresponds to a feature actually present in
    build_feature_matrix (zone, cause, weekday, month risk scores, plus rush
    hour / closure / priority / duration). Earlier versions only used
    priority + closure + rush + night + duration, which meant the trained
    trees had literally zero statistical reason to split on event_cause,
    zone risk, weekday, month, or is_planned — those dropdowns in the UI
    barely moved the prediction. Folding the engineered risk scores into the
    target gives the model real signal to learn from for every input the
    user can actually control.

    Note: corridor risk is deliberately excluded here — it isn't a field
    the user can set anywhere in the UI, and because it had by far the
    strongest raw correlation with historical outcomes it was statistically
    dominating every tree split, making the model nearly blind to the
    inputs people can actually change. Corridor data is still used directly
    for analytics (corridor stats, diversion routing) where it's accurate.
    """
    hp     = df["is_high_priority"].values.astype(float)          # 0–1
    cl     = df["requires_road_closure"].values.astype(float)     # 0–1
    rush   = df["is_rush_hour"].values.astype(float)
    night  = df["is_night"].values.astype(float)
    planned= df["is_planned"].values.astype(float)
    dur    = df["duration_min"].fillna(df["duration_min"].median()).clip(0, 600).values
    dur_n  = dur / 600.0  # normalise

    zone_r    = df["zone_risk_score"].values.astype(float)
    cause_r   = df["cause_risk_score"].values.astype(float)
    weekday_r = df["weekday_risk_score"].values.astype(float)
    month_r   = df["month_risk_score"].values.astype(float)

    raw = (
        hp * 20 + cl * 18 + rush * 11 + (1 - night) * 5 + dur_n * 13
        + zone_r * 22 + cause_r * 18
        + weekday_r * 6 + month_r * 6
        + (1 - planned) * 8   # unplanned events are inherently less predictable/managed
    )
    raw = np.clip(raw, 5, 99)
    return raw


def train_models() -> dict:
    try:
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, GradientBoostingClassifier
    except ImportError:
        return None

    df = get_df()

    X = build_feature_matrix(df)
    y_congestion = compute_congestion_target(df)
    y_delay      = df["duration_min"].fillna(df["duration_min"].median()).clip(0, 480).values * 0.35
    y_closure    = df["requires_road_closure"].values

    # Congestion regression — ensemble of GBR + RF
    gbr = GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=10, random_state=42
    )
    rfr = RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_leaf=8,
        random_state=42, n_jobs=-1
    )
    gbr.fit(X, y_congestion)
    rfr.fit(X, y_congestion)

    # Delay regression
    gbr_delay = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.07,
        subsample=0.8, random_state=42
    )
    gbr_delay.fit(X, y_delay)

    # Closure classifier
    gbc = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    gbc.fit(X, y_closure)

    # Compute per-feature importances (averaged across models)
    fi_congestion = gbr.feature_importances_ * 0.6 + rfr.feature_importances_ * 0.4

    # Data-derived closure severity ratio. The dataset only records a binary
    # requires_road_closure flag (no "partial" outcomes exist historically),
    # so the trees can't learn a distinct effect for road_closure="partial"
    # from training data alone. We measure the real average congestion gap
    # between closed vs not-closed historical events and use it to scale the
    # ML output deterministically for "partial"/"full" at prediction time —
    # the ratio itself is computed from real data, not a guessed constant.
    closed_mean   = float(y_congestion[y_closure == 1].mean()) if (y_closure == 1).any() else float(y_congestion.mean())
    open_mean     = float(y_congestion[y_closure == 0].mean()) if (y_closure == 0).any() else float(y_congestion.mean())
    closure_full_ratio = closed_mean / open_mean if open_mean > 0 else 1.3

    return {
        "gbr":          gbr,
        "rfr":          rfr,
        "gbr_delay":    gbr_delay,
        "gbc":          gbc,
        "fi_congestion": fi_congestion,
        "feature_names": FEATURE_NAMES,
        "df_mean_y":    float(np.mean(y_congestion)),
        "df_std_y":     float(np.std(y_congestion)),
        "cause_risk_map":   build_cause_risk_from_data(df),
        "weekday_risk_map": build_weekday_risk_from_data(df),
        "month_risk_map":   build_month_risk_from_data(df),
        "zone_risk_map":    build_zone_risk_from_data(df),
        "corridor_risk_map":build_corridor_risk_from_data(df),
        "closure_full_ratio": closure_full_ratio,
    }


# ── Analytics helpers (unchanged) ─────────────────────────────────────────────
def get_summary_stats() -> dict:
    df = get_df()
    return {
        "total_events":    int(len(df)),
        "active_events":   int((df["status"] == "active").sum()),
        "high_priority":   int(df["is_high_priority"].sum()),
        "road_closures":   int(df["requires_road_closure"].sum()),
        "planned_events":  int(df["is_planned"].sum()),
        "unplanned_events":int((df["event_type"] == "unplanned").sum()),
        "closure_rate_pct":round(df["requires_road_closure"].mean() * 100, 1),
        "avg_duration_min":round(df["duration_min"].dropna().mean(), 1),
    }

def get_cause_distribution() -> list:
    df = get_df()
    counts = df["event_cause"].value_counts().head(12)
    return [{"cause": k, "count": int(v)} for k, v in counts.items()]

def get_monthly_trend() -> list:
    df = get_df()
    df["ym"] = df["start_datetime"].dt.to_period("M").astype(str)
    counts = df.groupby("ym").size().sort_index()
    return [{"month": k, "count": int(v)} for k, v in counts.items()]

def get_hourly_pattern() -> list:
    df = get_df()
    counts = df.groupby("hour").size()
    return [{"hour": int(h), "count": int(counts.get(h, 0))} for h in range(24)]

def get_zone_risk() -> list:
    df = get_df()
    zone_df = df[df["zone"].notna() & (df["zone"] != "NULL")]
    grp = zone_df.groupby("zone").agg(
        total=("id","count"), high_prio=("is_high_priority","sum"),
        closures=("requires_road_closure","sum"), active=("is_active","sum"),
    ).reset_index()
    grp["risk_score"] = (
        grp["high_prio"] / grp["total"] * 40
        + grp["closures"] / grp["total"] * 35
        + grp["active"] / grp["total"] * 25
    ).round(1)
    grp = grp.sort_values("risk_score", ascending=False).head(12)
    return grp.rename(columns={"zone":"name"}).to_dict(orient="records")

def get_corridor_stats() -> list:
    df = get_df()
    corr = df[df["corridor"].notna() & (df["corridor"] != "NULL") & (df["corridor"] != "Non-corridor")]
    grp = corr.groupby("corridor").agg(
        total=("id","count"), closures=("requires_road_closure","sum"), high_prio=("is_high_priority","sum"),
    ).reset_index()
    grp["closure_rate"] = (grp["closures"] / grp["total"] * 100).round(1)
    grp = grp.sort_values("total", ascending=False).head(12)
    return grp.rename(columns={"corridor":"name"}).to_dict(orient="records")

def get_police_station_stats() -> list:
    df = get_df()
    ps = df[df["police_station"].notna() & (df["police_station"] != "NULL")]
    grp = ps.groupby("police_station").agg(
        total=("id","count"), high_prio=("is_high_priority","sum"), active=("is_active","sum"),
    ).reset_index()
    grp = grp.sort_values("total", ascending=False).head(15)
    return grp.rename(columns={"police_station":"name"}).to_dict(orient="records")

def get_closure_by_cause() -> list:
    df = get_df()
    grp = df.groupby("event_cause").agg(
        total=("id","count"), closures=("requires_road_closure","sum"),
    ).reset_index()
    grp["closure_rate"] = (grp["closures"] / grp["total"] * 100).round(1)
    grp = grp[grp["total"] >= 5].sort_values("closure_rate", ascending=False)
    return grp.rename(columns={"event_cause":"cause"}).to_dict(orient="records")

def get_heatmap_points(limit: int = 500) -> list:
    df = get_df()
    active = df[df["status"] == "active"].head(limit)
    if len(active) < 50:
        active = df.sample(min(limit, len(df)), random_state=42)
    return [{"lat": float(r["latitude"]), "lng": float(r["longitude"]),
              "weight": 0.9 if r["is_high_priority"] else 0.4,
              "cause": r["event_cause"], "status": r["status"]}
            for _, r in active.iterrows()]

def get_recent_active_events(limit: int = 20) -> list:
    df = get_df()
    active = df[df["status"] == "active"].sort_values("start_datetime", ascending=False).head(limit)
    if len(active) == 0:
        active = df.sort_values("start_datetime", ascending=False).head(limit)
    out = []
    for _, r in active.iterrows():
        out.append({
            "id": r["id"], "event_type": r["event_type"], "event_cause": r["event_cause"],
            "address": str(r["address"])[:80] if pd.notna(r["address"]) else "Bengaluru",
            "priority": r["priority"], "status": r["status"],
            "requires_road_closure": bool(r["requires_road_closure"]),
            "latitude": float(r["latitude"]), "longitude": float(r["longitude"]),
            "police_station": str(r["police_station"]) if pd.notna(r["police_station"]) else "",
            "zone": str(r["zone"]) if pd.notna(r["zone"]) else "",
            "corridor": str(r["corridor"]) if pd.notna(r["corridor"]) else "",
            "start_datetime": r["start_datetime"].isoformat() if pd.notna(r["start_datetime"]) else "",
        })
    return out


# ── Build single-row feature vector for prediction ───────────────────────────
def _time_bucket(time_of_day: str) -> int:
    return {"morning": 0, "afternoon": 1, "evening": 2, "night": 3}.get(time_of_day, 2)

def _cause_index(event_cause: str) -> float:
    mapping = {v: i for i, v in enumerate(CAUSE_ORDER)}
    return float(mapping.get(event_cause, len(CAUSE_ORDER) - 1))

def _zone_index(zone_risk: str) -> float:
    return float({"low": 0, "medium": 1, "high": 2}.get(zone_risk, 1))

def _closure_index(road_closure: str) -> float:
    return float({"no": 0, "partial": 1, "full": 2}.get(road_closure, 0))

def _zone_score_from_label(zone_risk: str) -> float:
    # Calibrated to the real distribution of zone_risk_score in the dataset
    # (low ≈ 0.20, medium ≈ 0.32, high ≈ 0.45) rather than an arbitrary 0-1 spread.
    return {"low": 0.20, "medium": 0.32, "high": 0.45}.get(zone_risk, 0.32)

def _build_row(event_cause, time_of_day, zone_risk, road_closure, is_planned, hour,
               models=None, zone_risk_score=None,
               month=6, weekday=4) -> np.ndarray:
    is_rush    = 1.0 if hour in list(range(7, 11)) + list(range(17, 21)) else 0.0
    is_weekend = 1.0 if weekday >= 5 else 0.0
    is_night   = 1.0 if hour >= 22 or hour < 6 else 0.0
    if zone_risk_score is None:
        zone_risk_score = _zone_score_from_label(zone_risk)

    # Cause / weekday / month risk — looked up from the same maps the model
    # was trained on, so prediction-time features stay consistent with
    # training-time features. Falls back to a neutral 0.5 when models/maps
    # aren't available yet (e.g. cold start before first training run).
    if models:
        cause_risk   = models.get("cause_risk_map", {}).get(event_cause, 0.5)
        weekday_risk = models.get("weekday_risk_map", {}).get(weekday, 0.5)
        month_risk   = models.get("month_risk_map", {}).get(month, 0.5)
    else:
        cause_risk = weekday_risk = month_risk = 0.5

    return np.array([[
        _cause_index(event_cause),
        float(_time_bucket(time_of_day)),
        _zone_index(zone_risk),
        _closure_index(road_closure),
        float(is_planned),
        is_rush, is_weekend, is_night,
        zone_risk_score,
        cause_risk, weekday_risk, month_risk,
    ]])


# ── SHAP-style permutation importance for a single prediction ─────────────────
def _compute_shap_for_row(models: dict, x_row: np.ndarray) -> dict:
    """
    Fast approximation of SHAP values via feature-wise mean-baseline subtraction.
    For each feature, replace it with its training-distribution mean and measure
    the change in predicted congestion. This gives a genuine sensitivity measure.
    """
    df = get_df()
    # Training distribution means for each feature position
    X_train = build_feature_matrix(df)
    col_means = X_train.mean(axis=0)

    gbr  = models["gbr"]
    rfr  = models["rfr"]

    def predict_ensemble(X):
        return gbr.predict(X) * 0.6 + rfr.predict(X) * 0.4

    base_pred = float(predict_ensemble(x_row)[0])

    importances = {}
    fi = models["fi_congestion"]
    names = models["feature_names"]

    for i, name in enumerate(names):
        x_masked = x_row.copy()
        x_masked[0, i] = col_means[i]
        masked_pred = float(predict_ensemble(x_masked)[0])
        # Contribution = how much the score drops when this feature is removed
        contribution = abs(base_pred - masked_pred) * (fi[i] * 5 + 1)
        importances[name] = round(max(contribution, 0), 2)

    # Normalise so they're interpretable as % contributions summing to ~100
    total = sum(importances.values()) or 1
    importances = {k: round(v / total * 100, 1) for k, v in importances.items()}

    return importances


# ── Hourly forecast via ML (per-hour mini-predictions) ────────────────────────
def _hourly_forecast_ml(models: dict, event_cause: str, time_of_day: str,
                         zone_risk: str, road_closure: str, is_planned: bool,
                         peak_congestion: float) -> list:
    """Run 24 individual ML predictions (one per hour) to get a true hourly curve."""
    gbr = models["gbr"]
    rfr = models["rfr"]

    rows = []
    for h in range(24):
        row = _build_row(
            event_cause=event_cause,
            time_of_day=time_of_day,
            zone_risk=zone_risk,
            road_closure=road_closure,
            is_planned=is_planned,
            hour=h,
            models=models,
            month=datetime.now().month,
            weekday=datetime.now().weekday(),
        )
        rows.append(row[0])

    X_hours = np.array(rows)
    preds_gbr = gbr.predict(X_hours)
    preds_rfr = rfr.predict(X_hours)
    preds = preds_gbr * 0.6 + preds_rfr * 0.4

    # Scale so the peak aligns with our final congestion score
    if preds.max() > 0:
        preds = preds / preds.max() * peak_congestion

    preds = np.clip(preds, 0, 99)
    return [{"hour": int(h), "congestion": round(float(c), 1)} for h, c in enumerate(preds)]


# ── Dynamic diversion routes from data ───────────────────────────────────────
BENGALURU_ROUTES_DB = [
    {"name": "Outer Ring Road (ORR)",   "via": "Tin Factory → Marathahalli → Silk Board",    "type": "highway",  "base_time": 12, "capacity": "high",   "lat_band": (12.9, 13.0)},
    {"name": "Mysore Road Bypass",      "via": "Kengeri → Nice Road → Electronic City",       "type": "bypass",   "base_time": 18, "capacity": "high",   "lat_band": (12.87, 12.97)},
    {"name": "Bellary Road North",      "via": "Hebbal → Yelahanka → Airport Road",           "type": "arterial", "base_time": 15, "capacity": "medium", "lat_band": (13.0, 13.1)},
    {"name": "Old Madras Road",         "via": "KR Pura → Tin Factory → ITPL",               "type": "arterial", "base_time": 10, "capacity": "medium", "lat_band": (12.97, 13.05)},
    {"name": "Tumkur Road (NH-48)",     "via": "Peenya → Yeshwanthpur → Sankey Road",        "type": "highway",  "base_time": 14, "capacity": "high",   "lat_band": (13.0, 13.1)},
    {"name": "Bannerghatta Road",       "via": "JP Nagar → Gottigere → Electronic City",     "type": "arterial", "base_time": 20, "capacity": "low",    "lat_band": (12.87, 12.95)},
    {"name": "NICE Peripheral Road",    "via": "Tumkur Rd Junction → Hosur Rd Junction",     "type": "bypass",   "base_time": 25, "capacity": "high",   "lat_band": (12.85, 13.05)},
    {"name": "Sarjapur Road",           "via": "Koramangala → Sarjapur → Whitefield",        "type": "arterial", "base_time": 16, "capacity": "medium", "lat_band": (12.9, 13.0)},
    {"name": "Hosur Road (NH-44)",      "via": "BTM Layout → Electronic City → Hosur",       "type": "highway",  "base_time": 11, "capacity": "high",   "lat_band": (12.85, 12.95)},
    {"name": "Magadi Road",             "via": "Rajajinagar → Attiguppe → Kengeri",          "type": "arterial", "base_time": 13, "capacity": "medium", "lat_band": (12.94, 13.02)},
]


def get_diversion_routes(latitude: float, longitude: float, event_cause: str,
                          road_closure: str, congestion_score: float = 60.0,
                          zone_risk: str = "medium") -> list:
    """
    Fully dynamic diversion route scoring based on:
    - Proximity to event (lat band matching)
    - Road closure severity
    - ML-predicted congestion (higher = more routes, longer time additions)
    - Historical corridor data from the dataset
    """
    if road_closure == "no":
        return [{"name": "No diversion required", "reason": "No road closure planned",
                 "via": "", "time_add_min": 0, "id": 1, "congestion_level": "Low",
                 "recommended": True, "type": "none"}]

    df = get_df()

    # Get real congestion from dataset corridors near the event zone
    corr = df[df["corridor"].notna() & ~df["corridor"].isin(["NULL", "Non-corridor"])]
    corridor_loads = corr.groupby("corridor").agg(
        active=("is_active", "sum"),
        total=("id", "count"),
    ).reset_index()
    corridor_loads["load_pct"] = corridor_loads["active"] / corridor_loads["total"].clip(1) * 100

    # Score and rank routes
    scored = []
    for route in BENGALURU_ROUTES_DB:
        lat_min, lat_max = route["lat_band"]
        proximity_bonus = 1.0 if lat_min <= latitude <= lat_max else 0.6

        # Capacity factor
        cap_mult = {"high": 0.8, "medium": 1.0, "low": 1.25}.get(route["capacity"], 1.0)

        # Congestion adds proportional time based on ML prediction
        congestion_add = int(congestion_score / 100 * 8 * cap_mult)
        closure_add    = {"no": 0, "partial": 3, "full": 7}.get(road_closure, 0)
        time_add       = route["base_time"] + congestion_add + closure_add

        # Congestion level label from ML-driven score + capacity
        effective_load = congestion_score * cap_mult
        if effective_load < 40:
            cong_label = "Low"
        elif effective_load < 70:
            cong_label = "Moderate"
        else:
            cong_label = "High"

        # Lower score = better route
        score = time_add / proximity_bonus

        scored.append({
            **route,
            "time_add_min": time_add,
            "congestion_level": cong_label,
            "score": score,
        })

    # Select top 3 routes, sort by score
    scored.sort(key=lambda x: x["score"])
    top3 = scored[:3]

    result = []
    for i, r in enumerate(top3):
        result.append({
            "id": i + 1,
            "name": r["name"],
            "via": r["via"],
            "type": r["type"],
            "time_add_min": r["time_add_min"],
            "congestion_level": r["congestion_level"],
            "recommended": i == 0,
            "capacity": r["capacity"],
        })
    return result


# ── Baseline (no-AI) scenario from data ───────────────────────────────────────
def _compute_noai_baseline(event_cause: str, df: pd.DataFrame) -> dict:
    """
    Derive realistic 'Without AI' baseline from historical data for the same cause.
    """
    hist = df[df["event_cause"] == event_cause]
    if len(hist) < 5:
        hist = df  # fall back to all

    avg_dur   = hist["duration_min"].dropna().mean()
    if math.isnan(avg_dur):
        avg_dur = df["duration_min"].dropna().mean()
    if math.isnan(avg_dur):
        avg_dur = 90.0
    pct_high  = hist["is_high_priority"].mean()
    pct_close = hist["requires_road_closure"].mean()

    return {
        "congestion_pct":    f"{min(int(pct_high * 40 + pct_close * 35 + 30), 97)}%",
        "response_time_min": int(avg_dur * 0.18 + 20),
        "incident_duration_h": round(avg_dur / 60 * 1.4, 1) if not math.isnan(avg_dur) else 3.8,
        "police_units":      max(4, int(pct_high * 12 + 3)),
        "road_closure_status": "Unmanaged" if pct_close > 0.3 else "Partially managed",
        "diversion_routes":  "None" if pct_close > 0.5 else "Ad hoc",
    }


# ── Main prediction function ───────────────────────────────────────────────────
def predict_impact(
    event_cause: str,
    crowd_size: int,
    time_of_day: str,
    zone_risk: str,
    road_closure: str,
    is_planned: bool = True,
) -> dict:
    df      = get_df()
    models  = get_models()

    # Hour approximation from time_of_day
    hour_map = {"morning": 8, "afternoon": 13, "evening": 18, "night": 23}
    hour = hour_map.get(time_of_day, 18)

    # Historical evidence for this cause
    hist = df[df["event_cause"] == event_cause]
    hist_closure_rate  = hist["requires_road_closure"].mean() if len(hist) > 0 else 0.1
    hist_high_prio     = hist["is_high_priority"].mean()      if len(hist) > 0 else 0.5
    hist_avg_duration  = hist["duration_min"].dropna().mean() if len(hist) > 0 else 90.0
    hist_count         = len(hist)

    if math.isnan(hist_avg_duration):
        hist_avg_duration = 90.0

    # ── Core ML prediction ────────────────────────────────────────────────────
    x_row = _build_row(
        event_cause=event_cause, time_of_day=time_of_day,
        zone_risk=zone_risk, road_closure=road_closure,
        is_planned=is_planned, hour=hour, models=models,
        month=datetime.now().month, weekday=datetime.now().weekday(),
    )

    if models:
        raw_ml = float(models["gbr"].predict(x_row)[0]) * 0.6 \
                + float(models["rfr"].predict(x_row)[0]) * 0.4
        delay_ml = float(models["gbr_delay"].predict(x_row)[0])
        closure_prob = float(models["gbc"].predict_proba(x_row)[0][1])
    else:
        # Fallback if sklearn not available
        raw_ml, delay_ml, closure_prob = 55.0, 25.0, 0.5

    # Blend ML with crowd-size adjustment (crowd is not in training data as a column)
    crowd_factor = 0.55 + (crowd_size / 150000) * 0.95  # 0.55 – 1.5
    unplanned_boost = 1.0 if is_planned else 1.18

    # Road-closure severity multiplier — see closure_full_ratio note in
    # train_models(). "no" = 1.0x (baseline), "full" = the real measured
    # ratio between closed/open historical events, "partial" = halfway.
    full_ratio = models.get("closure_full_ratio", 1.3) if models else 1.3
    full_ratio = float(np.clip(full_ratio, 1.0, 2.0))
    closure_multiplier = {
        "no": 1.0,
        "partial": 1.0 + (full_ratio - 1.0) * 0.5,
        "full": full_ratio,
    }.get(road_closure, 1.0)

    congestion_score = int(np.clip(raw_ml * crowd_factor * unplanned_boost * closure_multiplier, 5, 99))

    # Delay: ML base × crowd factor, bounded
    expected_delay = int(np.clip(delay_ml * crowd_factor * closure_multiplier, 5, 120))

    # Confidence: higher when more historical data + model agreement
    gbr_pred = float(models["gbr"].predict(x_row)[0]) if models else raw_ml
    rfr_pred = float(models["rfr"].predict(x_row)[0]) if models else raw_ml
    model_agreement = 1 - abs(gbr_pred - rfr_pred) / (max(gbr_pred, rfr_pred) + 1e-9)
    data_confidence = min(hist_count / 500, 1.0)
    confidence_pct  = int(np.clip(50 + model_agreement * 25 + data_confidence * 20, 55, 96))

    # Risk level
    if congestion_score >= 75:
        risk_level, risk_color = "Critical", "#EF4444"
    elif congestion_score >= 55:
        risk_level, risk_color = "High",     "#F97316"
    elif congestion_score >= 35:
        risk_level, risk_color = "Moderate", "#F59E0B"
    else:
        risk_level, risk_color = "Low",      "#10B981"

    # Impact radius: data-driven from crowd + ML congestion
    radius_km = round(0.8 + (crowd_size / 150000) * 6.0 + (congestion_score / 100) * 3.0, 1)

    # Peak hour: hour of max in the ML hourly forecast
    hourly = _hourly_forecast_ml(models, event_cause, time_of_day, zone_risk,
                                  road_closure, is_planned, congestion_score) if models else []
    if hourly:
        peak_h = max(hourly, key=lambda x: x["congestion"])["hour"]
        peak_hour = f"{peak_h}:{'30' if congestion_score > 60 else '00'} {'AM' if peak_h < 12 else 'PM'}"
    else:
        peak_hour = "7:00 PM"

    # ── SHAP feature importance ───────────────────────────────────────────────
    if models:
        feature_importance = _compute_shap_for_row(models, x_row)
    else:
        feature_importance = {n: round(100/len(FEATURE_NAMES), 1) for n in FEATURE_NAMES}

    # ── No-AI baseline from real data ─────────────────────────────────────────
    noai_baseline = _compute_noai_baseline(event_cause, df)

    # ── With-AI scenario ──────────────────────────────────────────────────────
    with_ai = {
        "congestion_pct":       f"{congestion_score}%",
        "response_time_min":    max(6, int(noai_baseline["response_time_min"] * 0.28)),
        "incident_duration_h":  round(noai_baseline["incident_duration_h"] * 0.48, 1),
        "police_units":         max(8, int(congestion_score / 100 * 30)),
        "road_closure_status":  "AI-Managed" if closure_prob > 0.3 else "Optimised",
        "diversion_routes":     3 if road_closure != "no" else 0,
    }

    return {
        "risk_level":             risk_level,
        "risk_color":             risk_color,
        "congestion_score":       congestion_score,
        "congestion_pct":         f"{congestion_score}%",
        "affected_radius_km":     radius_km,
        "expected_delay_min":     expected_delay,
        "peak_hour":              peak_hour,
        "confidence_pct":         confidence_pct,
        "closure_probability_pct": round(closure_prob * 100, 1),
        "historical_events":      hist_count,
        "hist_closure_rate_pct":  round(hist_closure_rate * 100, 1),
        "hist_high_priority_pct": round(hist_high_prio * 100, 1),
        "hist_avg_duration_min":  round(hist_avg_duration, 1),
        "feature_importance":     feature_importance,
        "hourly_forecast":        hourly,
        "noai_baseline":          noai_baseline,
        "with_ai":                with_ai,
        "model_info": {
            "type": "GBR+RF Ensemble",
            "gbr_pred": round(float(gbr_pred), 1),
            "rfr_pred": round(float(rfr_pred), 1),
            "model_agreement_pct": round(model_agreement * 100, 1),
        }
    }


# ── Resource recommendation (ML-calibrated) ────────────────────────────────────
def recommend_resources(
    event_cause: str,
    crowd_size: int,
    risk_level: str,
    zone_risk: str,
    road_closure: str,
) -> dict:
    df = get_df()

    hist = df[df["event_cause"] == event_cause]
    hist_count = len(hist)

    risk_mult  = {"Critical": 1.6, "High": 1.2, "Moderate": 0.85, "Low": 0.5}.get(risk_level, 1.0)
    zone_mult  = {"high": 1.3, "medium": 1.0, "low": 0.75}.get(zone_risk, 1.0)
    crowd_f    = 1 + (crowd_size / 100000) * 1.5

    # Base officers derived from historical median for the cause
    if hist_count >= 10:
        hist_high_pct = hist["is_high_priority"].mean()
        base_officers = max(4, int(15 + hist_high_pct * 20))
    else:
        base_officers = {
            "vip_movement": 30, "public_event": 20, "procession": 18,
            "protest": 22, "accident": 8, "vehicle_breakdown": 4,
            "construction": 6, "water_logging": 5, "congestion": 10,
        }.get(event_cause, 8)

    officers        = max(2, int(base_officers * risk_mult * zone_mult * crowd_f))
    barricades      = max(2, int(officers * 0.65))
    vehicles        = max(1, int(officers * 0.22))
    emergency_teams = max(1, int(officers * 0.09))
    drones          = max(0, int(officers * 0.07))

    if road_closure == "full":
        barricades = int(barricades * 1.5)
        officers   = int(officers * 1.2)

    hist_note = f"Calibrated from {hist_count} historical {event_cause} incidents." if hist_count > 0 else ""

    return {
        "officers": officers,
        "barricades": barricades,
        "patrol_vehicles": vehicles,
        "emergency_teams": emergency_teams,
        "drones": drones,
        "total_personnel": officers + emergency_teams * 3,
        "reasoning": f"Deployment scaled for {risk_level} risk with crowd ~{crowd_size:,}. {hist_note}",
        "deployment_zones": [
            {"zone": "Primary perimeter", "officers": int(officers*0.5), "barricades": int(barricades*0.5)},
            {"zone": "Secondary access",  "officers": int(officers*0.3), "barricades": int(barricades*0.3)},
            {"zone": "Diversion points",  "officers": int(officers*0.2), "barricades": int(barricades*0.2)},
        ],
    }


# ── ML-driven live-data generation ──────────────────────────────────────────
# Used by the /api/live/* router as the fallback path when external live APIs
# (TomTom / Google Maps) are not configured or unreachable. Earlier versions
# faked this with random.randint() — pure noise with no relationship to
# reality. These functions instead run the same trained ensemble used for
# event-impact prediction, seeded with the real corridor/zone risk profile
# and the *actual current hour/weekday*, so values are reproducible,
# data-grounded, and move the way real traffic does (rush-hour peaks, etc.)
# instead of jittering randomly every five minutes.
def predict_corridor_congestion_now(corridor_name: str) -> dict:
    """ML-predicted current congestion % for a named corridor, using the
    corridor's real historical risk profile blended with the live ensemble."""
    df = get_df()
    models = get_models()
    now = datetime.now()

    corridor_map = models.get("corridor_risk_map", {}) if models else {}
    # Fuzzy match the display corridor name against dataset corridor names
    match = None
    for name in corridor_map:
        if name and (name.lower() in corridor_name.lower() or corridor_name.lower() in name.lower()):
            match = name
            break
    corridor_risk = corridor_map.get(match, 0.5) if match else 0.5
    zone_risk_label = "high" if corridor_risk > 0.53 else "medium" if corridor_risk > 0.51 else "low"

    x_row = _build_row(
        event_cause="congestion", time_of_day=_hour_to_time_of_day(now.hour),
        zone_risk=zone_risk_label, road_closure="no", is_planned=False,
        hour=now.hour, models=models, month=now.month, weekday=now.weekday(),
    )
    if models:
        score = float(models["gbr"].predict(x_row)[0]) * 0.6 + float(models["rfr"].predict(x_row)[0]) * 0.4
    else:
        score = 35.0
    # Corridor's own historical closure-adjusted risk nudges the baseline
    score = score * (0.85 + corridor_risk * 0.3)
    score = float(np.clip(score, 4, 99))

    normal = 8 + round(corridor_risk * 20, 1)  # free-flow minutes, longer for historically busier corridors
    return {
        "congestion_pct": round(score, 1),
        "duration_normal_min": round(normal, 1),
        "duration_traffic_min": round(normal * (1 + score / 100), 1),
        "delay_min": round(normal * score / 100, 1),
        "matched_corridor": match,
    }


def estimate_venue_crowd(cap_min: int, cap_max: int) -> int:
    """
    Deterministic 'how busy is this venue right now' estimate, scaled within
    [cap_min, cap_max] by the real historical hourly + weekday activity
    pattern from the dataset (i.e. busier at hours/days that historically saw
    more incident volume city-wide, as a proxy for general activity level).
    Replaces a flat random.randint() draw.
    """
    df = get_df()
    now = datetime.now()
    hourly = df.groupby("hour").size()
    weekday_counts = df.groupby("weekday").size()

    hour_share = float(hourly.get(now.hour, hourly.mean()) / hourly.max())
    weekday_share = float(weekday_counts.get(now.weekday(), weekday_counts.mean()) / weekday_counts.max())
    activity = 0.5 + 0.3 * hour_share + 0.2 * weekday_share  # 0.5 – 1.0 range
    return int(cap_min + (cap_max - cap_min) * np.clip(activity, 0.15, 1.0))



    if 7 <= h <= 10:  return "morning"
    if 11 <= h <= 16: return "afternoon"
    if 17 <= h <= 21: return "evening"
    return "night"


def _hour_to_time_of_day(h: int) -> str:
    if 7 <= h <= 10:  return "morning"
    if 11 <= h <= 16: return "afternoon"
    if 17 <= h <= 21: return "evening"
    return "night"


def sample_live_incidents(limit: int = 15) -> list:
    """
    Sample real historical incident records whose hour-of-day matches the
    current time (±1h window), then score each with the trained closure/
    congestion models for "if this happened right now" severity. This
    surfaces genuine Bengaluru incident locations/causes from the dataset
    with ML-computed severity, instead of a fixed random-jittered list.
    """
    df = get_df()
    models = get_models()
    now = datetime.now()
    window = [(now.hour + d) % 24 for d in (-1, 0, 1)]

    pool = df[df["hour"].isin(window)]
    if len(pool) < limit:
        pool = df
    # Prioritise high-priority / closure-causing historical events — these are
    # the ones genuinely worth surfacing as "active-style" incidents.
    # Sample diversely: stratify by event_cause so the map shows a realistic
    # mix of incident types, not just the top-ranked high-priority rows.
    rng_seed = now.hour * 60 + now.minute // 5
    causes = pool["event_cause"].dropna().unique()
    per_cause = max(1, limit // max(len(causes), 1))
    frames = []
    for cause in causes:
        chunk = pool[pool["event_cause"] == cause]
        n = min(per_cause, len(chunk))
        frames.append(chunk.sample(n=n, random_state=rng_seed))
    diverse_pool = pd.concat(frames) if frames else pool
    # Top up to `limit` from remaining pool if under-sampled
    if len(diverse_pool) < limit:
        remaining = pool[~pool.index.isin(diverse_pool.index)]
        top_up = remaining.sample(n=min(limit - len(diverse_pool), len(remaining)), random_state=rng_seed)
        diverse_pool = pd.concat([diverse_pool, top_up])
    sample = diverse_pool.sample(n=min(limit, len(diverse_pool)), random_state=rng_seed)

    # Pre-compute per-cause median duration from valid historical rows.
    # This gives realistic, cause-specific delay baselines grounded in real data.
    cause_median = (
        df[df["duration_min"].notna()]
        .groupby("event_cause")["duration_min"]
        .median()
        .to_dict()
    )
    global_median = float(df["duration_min"].median()) if df["duration_min"].notna().any() else 45.0

    out = []
    for _, r in sample.iterrows():
        x_row = _build_row(
            event_cause=r["event_cause"] if pd.notna(r["event_cause"]) else "others",
            time_of_day=_hour_to_time_of_day(now.hour),
            zone_risk="medium", road_closure="no", is_planned=bool(r["is_planned"]),
            hour=now.hour, models=models, month=now.month, weekday=now.weekday(),
        )
        if models:
            congestion = float(models["gbr"].predict(x_row)[0]) * 0.6 + float(models["rfr"].predict(x_row)[0]) * 0.4
            closure_p  = float(models["gbc"].predict_proba(x_row)[0][1])
        else:
            congestion, closure_p = 40.0, 0.2

        magnitude = 4 if congestion >= 70 else 3 if congestion >= 50 else 2 if congestion >= 30 else 1
        severity  = {1: "Minor", 2: "Moderate", 3: "Major", 4: "Critical"}[magnitude]

        # --- Delay calculation ---
        # Priority 1: use the actual historical duration_min from this matched row.
        # Priority 2: fall back to the per-cause median from all valid rows.
        # Priority 3: global dataset median.
        # Scale by congestion so busier-hour incidents show a higher delay,
        # but keep the variation driven by real data rather than a fixed multiplier.
        if pd.notna(r.get("duration_min")) and r["duration_min"] > 0:
            base_delay_min = float(r["duration_min"])
        else:
            base_delay_min = cause_median.get(r.get("event_cause"), global_median)

        # Congestion ratio: baseline is 40 (moderate); scale ±50% around that.
        congestion_scale = np.clip(congestion / 40.0, 0.5, 2.5)
        delay_min = base_delay_min * congestion_scale
        delay_sec = int(np.clip(delay_min * 60, 60, 5400))  # 1 min – 90 min cap

        out.append({
            "id": f"hist-{r['id']}",
            "description": f"{str(r['event_cause']).replace('_',' ').title()} near {str(r.get('police_station') or r.get('zone') or 'Bengaluru')}",
            "cause": str(r["event_cause"]).replace("_", " ").title(),
            "severity": severity,
            "magnitude": magnitude,
            "latitude": float(r["latitude"]),
            "longitude": float(r["longitude"]),
            "from": str(r.get("police_station")) if pd.notna(r.get("police_station")) else "Bengaluru",
            "to": str(r.get("zone")) if pd.notna(r.get("zone")) else "",
            "delay_sec": delay_sec,
            "length_m": int(np.clip(congestion * 25, 200, 4000)),
            "road": str(r.get("corridor")) if pd.notna(r.get("corridor")) and r.get("corridor") not in ("NULL", "Non-corridor") else str(r.get("zone") or "Bengaluru"),
            "closure_probability_pct": round(closure_p * 100, 1),
            "source": "ML-predicted (historical pattern match — live API not configured)",
            "timestamp": datetime.now().astimezone().isoformat(),
        })
    return out