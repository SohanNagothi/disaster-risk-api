
import pickle
import numpy as np
import pandas as pd
import json
import os

# ---- LOAD ALL MODELS ----
BASE = os.path.dirname(os.path.abspath(__file__))

def load(filename):
    return pickle.load(open(os.path.join(BASE, filename), "rb"))

rf_model    = load("rf_model.pkl")
gb_model    = load("gb_model.pkl")
lr_model    = load("lr_model.pkl")
scaler      = load("scaler.pkl")
le_district = load("le_district.pkl")
le_state    = load("le_state.pkl")
le_disaster = load("le_disaster.pkl")
FEATURES    = load("feature_list.pkl")

with open(os.path.join(BASE, "model_config.json"), "r") as f:
    model_config = json.load(f)

best_threshold = model_config["threshold"]
w_rf = model_config["w_rf"]
w_gb = model_config["w_gb"]
w_lr = model_config["w_lr"]

lkp_s_freq    = pd.read_csv(os.path.join(BASE, "lookup_state_freq.csv"))
lkp_s_mon     = pd.read_csv(os.path.join(BASE, "lookup_state_monthly.csv"))
lkp_s_death   = pd.read_csv(os.path.join(BASE, "lookup_state_deaths.csv"))
lkp_s_mon_any = pd.read_csv(os.path.join(BASE, "lookup_state_month_any.csv"))
lkp_d_freq    = pd.read_csv(os.path.join(BASE, "lookup_district_freq.csv"))
lkp_d_mon     = pd.read_csv(os.path.join(BASE, "lookup_district_monthly.csv"))
lkp_d_death   = pd.read_csv(os.path.join(BASE, "lookup_district_deaths.csv"))
lkp_g_freq    = pd.read_csv(os.path.join(BASE, "lookup_grid_freq.csv"))
lkp_g_mon     = pd.read_csv(os.path.join(BASE, "lookup_grid_monthly.csv"))
lkp_g_death   = pd.read_csv(os.path.join(BASE, "lookup_grid_deaths.csv"))
lkp_g_dis     = pd.read_csv(os.path.join(BASE, "lookup_global_disaster_prior.csv"))
lkp_g_mon_p   = pd.read_csv(os.path.join(BASE, "lookup_global_month_prior.csv"))
lkp_g_dis_mon = pd.read_csv(os.path.join(BASE, "lookup_global_dis_month_prior.csv"))

INDIA_STATES = {
    "Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chhattisgarh",
    "Goa","Gujarat","Haryana","Himachal Pradesh","Jharkhand","Karnataka",
    "Kerala","Madhya Pradesh","Maharashtra","Manipur","Meghalaya","Mizoram",
    "Nagaland","Odisha","Punjab","Rajasthan","Sikkim","Tamil Nadu","Telangana",
    "Tripura","Uttar Pradesh","Uttarakhand","West Bengal",
    "Jammu And Kashmir","Delhi","Ladakh","Puducherry",
    "Andaman And Nicobar Islands","Lakshadweep","Chandigarh",
    "Dadra And Nagar Haveli","Daman And Diu"
}

MONTH_NAMES = {
    "january":1,"february":2,"march":3,"april":4,
    "may":5,"june":6,"july":7,"august":8,
    "september":9,"october":10,"november":11,"december":12
}

def safe_enc(encoder, val):
    if not val or pd.isna(val): val = "Unknown"
    val = str(val).strip().title()
    if val in encoder.classes_:
        return int(encoder.transform([val])[0])
    return int(encoder.transform(["Unknown"])[0]) if "Unknown" in encoder.classes_ else 0

def lookup_val(table, filters, target_col):
    mask = pd.Series([True] * len(table))
    for col, val in filters.items():
        if col in table.columns and val is not None:
            mask = mask & (table[col] == val)
        else:
            return 0.0
    result = table[mask]
    return float(result[target_col].iloc[0]) if len(result) > 0 and target_col in result.columns else 0.0

def parse_month(val):
    try:
        m = int(val)
        return m if 1 <= m <= 12 else None
    except:
        return MONTH_NAMES.get(str(val).strip().lower()) if val else None

def predict_risk(disaster_type, month, district=None, state=None, latitude=None, longitude=None):

    valid = ["Flood","Earthquake","Landslide","Cyclone"]
    disaster_type = str(disaster_type).strip().title()
    if disaster_type not in valid:
        return {"error": f"disaster_type must be one of {valid}"}
    month = parse_month(month)
    if not month:
        return {"error": "month must be 1-12 or a month name"}
    if district: district = str(district).strip().title()
    if state:    state    = str(state).strip().title()

    contributing = []

    d_ev  = lookup_val(lkp_d_freq,  {"district":district,"disaster_type":disaster_type}, "district_event_count")   if district else 0
    d_mo  = lookup_val(lkp_d_mon,   {"district":district,"disaster_type":disaster_type,"month":month}, "district_monthly_count") if district else 0
    d_de  = lookup_val(lkp_d_death, {"district":district,"disaster_type":disaster_type}, "district_avg_deaths")    if district else 0
    if d_ev > 0: contributing.append(f"district ({district})")

    s_ev  = lookup_val(lkp_s_freq,    {"state":state,"disaster_type":disaster_type}, "state_event_count")          if state else 0
    s_mo  = lookup_val(lkp_s_mon,     {"state":state,"disaster_type":disaster_type,"month":month}, "state_monthly_count") if state else 0
    s_de  = lookup_val(lkp_s_death,   {"state":state,"disaster_type":disaster_type}, "state_avg_deaths")           if state else 0
    s_any = lookup_val(lkp_s_mon_any, {"state":state,"month":month}, "state_any_monthly_count")                    if state else 0
    if s_ev > 0: contributing.append(f"state ({state})")

    try:    grid_cell = f"{round(float(latitude),1)}_{round(float(longitude),1)}" if latitude and longitude else None
    except: grid_cell = None
    g_ev  = lookup_val(lkp_g_freq,  {"grid_cell":grid_cell,"disaster_type":disaster_type}, "grid_event_count")   if grid_cell else 0
    g_mo  = lookup_val(lkp_g_mon,   {"grid_cell":grid_cell,"disaster_type":disaster_type,"month":month}, "grid_monthly_count") if grid_cell else 0
    g_de  = lookup_val(lkp_g_death, {"grid_cell":grid_cell,"disaster_type":disaster_type}, "grid_avg_deaths")    if grid_cell else 0
    if g_ev > 0: contributing.append(f"lat/lon ({grid_cell})")

    has_district_data = int(d_ev > 0)
    has_state_data    = int(s_ev > 0)
    has_grid_data     = int(g_ev > 0)
    has_any_data      = int((d_ev + s_ev + g_ev) > 0)
    if not contributing: contributing.append("global priors only")

    total_ev  = d_ev + s_ev + g_ev
    total_mo  = d_mo + s_mo + g_mo
    best_ev   = d_ev if d_ev > 0 else (s_ev if s_ev > 0 else g_ev)
    best_mo   = d_mo if d_mo > 0 else (s_mo if s_mo > 0 else g_mo)
    best_de   = d_de if d_de > 0 else (s_de if s_de > 0 else g_de)
    mo_ratio  = total_mo / (total_ev + 1e-6) if total_ev > 0 else 0

    g_dis_pct = lookup_val(lkp_g_dis,     {"disaster_type":disaster_type},                "global_disaster_pct")
    g_mon_pct = lookup_val(lkp_g_mon_p,   {"month":month},                                "global_month_pct")
    g_dm_pct  = lookup_val(lkp_g_dis_mon, {"disaster_type":disaster_type,"month":month},  "global_dis_month_pct")
    dm_rate   = lookup_val(lkp_g_dis_mon, {"disaster_type":disaster_type,"month":month},  "disaster_month_rate")
    prior_score = g_dis_pct * dm_rate * 10

    disaster_enc = safe_enc(le_disaster, disaster_type)
    district_enc = safe_enc(le_district, district)
    state_enc    = safe_enc(le_state,    state)
    month_sin    = np.sin(2 * np.pi * month / 12)
    month_cos    = np.cos(2 * np.pi * month / 12)
    dis_month    = disaster_enc * 12 + month
    is_india     = 1 if (state and str(state).strip().title() in INDIA_STATES) else 0

    feature_values = {
        "disaster_enc"              : disaster_enc,
        "district_enc"              : district_enc,
        "state_enc"                 : state_enc,
        "month_sin"                 : month_sin,
        "month_cos"                 : month_cos,
        "disaster_month_interaction": dis_month,
        "is_india"                  : is_india,
        "state_event_count"         : s_ev,
        "state_monthly_count"       : s_mo,
        "state_avg_deaths"          : s_de,
        "state_any_monthly_count"   : s_any,
        "district_event_count"      : d_ev,
        "district_monthly_count"    : d_mo,
        "district_avg_deaths"       : d_de,
        "grid_event_count"          : g_ev,
        "grid_monthly_count"        : g_mo,
        "grid_avg_deaths"           : g_de,
        "has_district_data"         : has_district_data,
        "has_state_data"            : has_state_data,
        "has_grid_data"             : has_grid_data,
        "has_any_data"              : has_any_data,
        "best_event_count"          : best_ev,
        "best_monthly_count"        : best_mo,
        "best_avg_deaths"           : best_de,
        "total_event_count"         : total_ev,
        "total_monthly_count"       : total_mo,
        "monthly_ratio"             : mo_ratio,
        "log_best_event_count"      : np.log1p(best_ev),
        "log_total_event_count"     : np.log1p(total_ev),
        "log_best_monthly_count"    : np.log1p(best_mo),
        "log_best_avg_deaths"       : np.log1p(best_de),
        "global_disaster_pct"       : g_dis_pct,
        "global_month_pct"          : g_mon_pct,
        "global_dis_month_pct"      : g_dm_pct,
        "disaster_month_rate"       : dm_rate,
        "prior_score"               : prior_score,
        "state_x_prior"             : s_ev * g_dis_pct,
        "state_x_month_rate"        : s_ev * dm_rate,
    }

    fv = np.array([[feature_values.get(f, 0) for f in FEATURES]])

    prob_rf    = float(rf_model.predict_proba(fv)[0][1])
    prob_gb    = float(gb_model.predict_proba(fv)[0][1])
    prob_lr    = float(lr_model.predict_proba(scaler.transform(fv))[0][1])
    prob_final = w_rf * prob_rf + w_gb * prob_gb + w_lr * prob_lr
    is_high_risk = prob_final >= best_threshold

    if prob_final >= 0.91:   risk_level, emoji = "HIGH",     "🔴"
    elif prob_final >= 0.50: risk_level, emoji = "MODERATE", "🟡"
    else:                    risk_level, emoji = "LOW",       "🟢"

    advice = {
        "HIGH"    : f"Strong historical record of {disaster_type} here. Take precautions seriously.",
        "MODERATE": f"Moderate {disaster_type} risk. Stay informed and have an emergency plan.",
        "LOW"     : f"Low historical {disaster_type} risk for this location.",
    }[risk_level]

    data_confidence = "HIGH" if d_ev > 0 else ("MEDIUM" if s_ev > 0 or g_ev > 0 else "LOW")

    return {
        "inputs": {
            "disaster_type": disaster_type, "month": month,
            "district": district, "state": state,
            "latitude": latitude, "longitude": longitude,
        },
        "prediction": {
            "risk_level"         : risk_level,
            "emoji"              : emoji,
            "probability"        : round(prob_final, 4),
            "threshold_used"     : best_threshold,
            "is_high_risk"       : bool(is_high_risk),
            "advice"             : advice,
            "data_confidence"    : data_confidence,
            "contributing_levels": contributing,
        },
        "breakdown": {
            "district_events"    : int(d_ev),
            "district_this_month": int(d_mo),
            "state_events"       : int(s_ev),
            "state_this_month"   : int(s_mo),
            "grid_events"        : int(g_ev),
            "grid_this_month"    : int(g_mo),
            "disaster_month_rate": round(dm_rate, 4),
            "prior_score"        : round(prior_score, 4),
        },
        "model_scores": {
            "random_forest"      : round(prob_rf,    4),
            "gradient_boosting"  : round(prob_gb,    4),
            "logistic_regression": round(prob_lr,    4),
            "ensemble"           : round(prob_final, 4),
        }
    }
