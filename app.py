
from flask import Flask, request, jsonify
from predict import predict_risk, lkp_s_freq, lkp_d_freq
import os

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message"  : "Disaster Risk Assessment API",
        "status"   : "running",
        "endpoints": [
            "POST /predict",
            "GET  /predict",
            "GET  /states",
            "GET  /districts",
            "GET  /disaster_types",
            "GET  /location_profile",
            "GET  /health"
        ]
    })

@app.route("/predict", methods=["POST"])
def predict_post():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "Send JSON body"}), 400
    result = predict_risk(
        disaster_type = data.get("disaster_type"),
        month         = data.get("month"),
        district      = data.get("district"),
        state         = data.get("state"),
        latitude      = data.get("latitude"),
        longitude     = data.get("longitude"),
    )
    return jsonify(result), 400 if "error" in result else 200

@app.route("/predict", methods=["GET"])
def predict_get():
    result = predict_risk(
        disaster_type = request.args.get("disaster_type"),
        month         = request.args.get("month"),
        district      = request.args.get("district"),
        state         = request.args.get("state"),
        latitude      = request.args.get("latitude"),
        longitude     = request.args.get("longitude"),
    )
    return jsonify(result), 400 if "error" in result else 200

@app.route("/states", methods=["GET"])
def get_states():
    states = sorted(lkp_s_freq["state"].dropna().unique().tolist())
    return jsonify({"states": states, "count": len(states)})

@app.route("/districts", methods=["GET"])
def get_districts():
    q         = request.args.get("q", "").strip().title()
    districts = sorted(lkp_d_freq["district"].dropna().unique().tolist())
    if q:
        districts = [d for d in districts if q.lower() in d.lower()]
    return jsonify({"districts": districts, "count": len(districts)})

@app.route("/disaster_types", methods=["GET"])
def get_disaster_types():
    return jsonify({
        "disaster_types": ["Flood", "Earthquake", "Landslide", "Cyclone"],
        "note"          : "Storms and heavy rain counted under Flood"
    })

@app.route("/location_profile", methods=["GET"])
def location_profile():
    state    = request.args.get("state",    "").strip().title() or None
    district = request.args.get("district", "").strip().title() or None
    month    = request.args.get("month", 6)
    if not state and not district:
        return jsonify({"error": "Provide at least state or district"}), 400
    profile = {}
    for disaster in ["Flood", "Earthquake", "Landslide", "Cyclone"]:
        result = predict_risk(disaster, month, district, state)
        if "error" not in result:
            profile[disaster] = {
                "risk_level" : result["prediction"]["risk_level"],
                "emoji"      : result["prediction"]["emoji"],
                "probability": result["prediction"]["probability"],
            }
    return jsonify({"state": state, "district": district,
                    "month": month, "profile": profile})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "version": "1.0"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
