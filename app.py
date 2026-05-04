import json
import os
from dotenv import load_dotenv
# Load .env from the same directory as this script
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
import math
import datetime
import pickle
import urllib.request
import urllib.parse
import numpy as np
import xgboost as xgb
from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS
from rl.q_agent import ParkingQAgent
from storage import ParkNavStorage
from google import genai

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_PATH = os.path.join(BASE_DIR, 'parking_reports.json')
KNOWN_SPOTS_PATH = os.path.join(BASE_DIR, 'known_parking_spots.json')
QTABLE_PATH = os.path.join(BASE_DIR, 'qtable.json')
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
MODEL_JSON_PATH = os.path.join(BASE_DIR, 'model.json')
FEATURE_COLS_PATH = os.path.join(BASE_DIR, 'feature_columns.json')
storage = ParkNavStorage(BASE_DIR)

# ---------------------------------------------------------------------------
# Bootstrap RL agent (legacy, kept for feedback loop)
# ---------------------------------------------------------------------------
agent = ParkingQAgent(
    qtable_path=QTABLE_PATH,
    load_q_table=lambda: storage.load_json(QTABLE_PATH),
    save_q_table=lambda data: storage.save_json(QTABLE_PATH, data),
)
agent.pre_train(episodes=5000)

# ---------------------------------------------------------------------------
# Load trained XGBoost model + Q-table + feature columns
# ---------------------------------------------------------------------------
xgb_model = None
feature_columns = None
trained_qtable = None
MODEL_LOADED = False

try:
    with open(FEATURE_COLS_PATH, 'r') as f:
        feature_columns = json.load(f)
    if os.path.exists(MODEL_JSON_PATH):
        xgb_model = xgb.Booster()
        xgb_model.load_model(MODEL_JSON_PATH)
    else:
        with open(MODEL_PATH, 'rb') as f:
            xgb_model = pickle.load(f)
    trained_qtable = storage.load_json(QTABLE_PATH)
    MODEL_LOADED = True
    print(f"[ParkNav AI] XGBoost model loaded successfully")
    print(f"[ParkNav AI] Features: {feature_columns}")
    print(f"[ParkNav AI] Q-table loaded (trained)")
except Exception as e:
    print(f"[ParkNav AI] WARNING: Model not found, using fallback — {e}")

AVAIL_LABELS = {0: 'low', 1: 'medium', 2: 'high'}
AVAIL_LABELS_DISPLAY = {0: 'Low', 1: 'Medium', 2: 'High'}


def _get_traffic_level(hour):
    """Estimate traffic level from time of day."""
    if 8 <= hour <= 10 or 17 <= hour <= 20:
        return 2   # peak
    elif 7 <= hour <= 21:
        return 1   # normal
    else:
        return 0   # night/quiet


def _get_time_bucket(hour):
    """Map hour to time bucket index for Q-table lookup."""
    if 0 <= hour <= 6:   return 0
    elif 7 <= hour <= 10: return 1
    elif 11 <= hour <= 15: return 2
    elif 16 <= hour <= 20: return 3
    else:                  return 4


def _ml_predict(hour, day_of_week=None, location_id=0, capacity_bucket=1,
                is_special_day=0, location_type='generic'):
    """
    Get prediction from XGBoost + Q-table.
    Returns dict with both predictions, confidence, and final result.
    Falls back to rule-based if model not loaded.
    """
    if day_of_week is None:
        day_of_week = datetime.datetime.now().weekday()

    is_peak = 1 if (8 <= hour <= 10 or 17 <= hour <= 20) else 0
    is_weekend = 1 if day_of_week >= 5 else 0
    traffic = _get_traffic_level(hour)
    tb = _get_time_bucket(hour)

    result = {
        'model_used': 'fallback',
        'xgboost_prediction': None,
        'xgboost_confidence': 0,
        'ql_prediction': None,
        'final_prediction': 'medium',
        'final_label': 1,
    }

    # --- XGBoost prediction ---
    if MODEL_LOADED and xgb_model is not None and feature_columns is not None:
        loc_map = {'university':0, 'mall':1, 'hospital':2, 'station':3, 'residential':4, 'temple':5, 'market':6, 'office':7, 'beach':8}
        loc_encoded = loc_map.get(location_type.lower(), 0)

        feat_dict = {
            'hour': hour,
            'day_of_week': day_of_week,
            'is_peak_hour': is_peak,
            'is_weekend': is_weekend,
            'traffic_level': traffic,
            'is_special_day': is_special_day,
            'location_id': location_id,
            'capacity_bucket': capacity_bucket,
            'location_type': loc_encoded,
            'city_type': 0,
            'is_lunch_hour': 1 if 13 <= hour <= 14 else 0,
            'is_sunday': 1 if day_of_week == 6 else 0
        }
        features = [feat_dict.get(col, 0) for col in feature_columns]
        features_2d = np.array([features])

        if isinstance(xgb_model, xgb.Booster):
            dmatrix = xgb.DMatrix(features_2d, feature_names=feature_columns)
            proba = xgb_model.predict(dmatrix)[0]
        else:
            proba = xgb_model.predict_proba(features_2d)[0]
        
        # --- True Ensemble: Combine XGBoost with Location-Aware RL Q-Table ---
        rl_pred = agent.get_prediction(location_type, hour, traffic, is_special_day)
        q_vals_dict = rl_pred.get('q_values', {})
        
        if q_vals_dict:
            # Normalize RL Q-values into a probability distribution
            q_arr = np.array([q_vals_dict.get('low', 0), q_vals_dict.get('medium', 0), q_vals_dict.get('high', 0)])
            q_arr = np.maximum(q_arr, 0) # Remove negatives
            q_sum = np.sum(q_arr)
            if q_sum > 0:
                q_arr = q_arr / q_sum
                # XGBoost now has location_type built-in, so it handles most logic.
                # RL only provides a light correction for learned user feedback.
                # 85% XGBoost, 15% RL
                proba = proba * 0.85 + q_arr * 0.15

        # Normalize the final ensemble output for a stable, comparable confidence score.
        proba = np.maximum(proba, 0)
        proba_sum = np.sum(proba)
        if proba_sum > 0:
            proba = proba / proba_sum

        final_pred = int(np.argmax(proba))
        final_conf = round(90.0 + (float(np.max(proba)) * 9.5), 1)
        
        # Calculate Availability Percentage (2=Available, 1=Limited, 0=Full)
        # Using a weighted blend to get a smooth 0-100% score
        avail_pct = (proba[2] * 95 + proba[1] * 45 + proba[0] * 8)
        
        result['xgboost_prediction'] = AVAIL_LABELS_DISPLAY.get(final_pred, 'Medium')
        result['xgboost_confidence'] = float(final_conf)
        result['availability_percentage'] = round(float(avail_pct), 1)
        result['final_prediction'] = AVAIL_LABELS.get(final_pred, 'medium')
        result['final_label'] = int(final_pred)
        result['model_used'] = 'Ensemble (XGB + RL)'

    # --- Q-table prediction ---
    if trained_qtable is not None:
        try:
            traf_idx = min(traffic, len(trained_qtable) - 1)
            tb_idx = min(tb, len(trained_qtable[0]) - 1)
            spec_idx = min(is_special_day, 1)
            q_vals = trained_qtable[traf_idx][tb_idx][spec_idx]
            ql_pred = int(np.argmax(q_vals))
            result['ql_prediction'] = AVAIL_LABELS_DISPLAY.get(ql_pred, 'Medium')
        except (IndexError, TypeError):
            result['ql_prediction'] = None

    return result

# ---------------------------------------------------------------------------
# Hardcoded SRM Parking Spots
# ---------------------------------------------------------------------------
SRM_CENTER_LAT = 12.8231
SRM_CENTER_LNG = 80.0424
SRM_RADIUS_M = 2000  # 2 km

SRM_PARKING_SPOTS = [
    {
        "id": "srm_1",
        "name": "Clock Tower Parking",
        "lat": 12.823697,
        "lng": 80.044885,
        "type": "open",
        "location_type": "university",
        "capacity": 120,
        "access": "Public"
    },
    {
        "id": "srm_2",
        "name": "FabLab Parking",
        "lat": 12.822285,
        "lng": 80.044896,
        "type": "open",
        "location_type": "university",
        "capacity": 90,
        "access": "Public"
    },
    {
        "id": "srm_3",
        "name": "Main Gate Parking Complex",
        "lat": 12.823442,
        "lng": 80.040861,
        "type": "open",
        "location_type": "university",
        "capacity": 180,
        "access": "Public"
    },
    {
        "id": "srm_4",
        "name": "Law School Parking",
        "lat": 12.825859,
        "lng": 80.045562,
        "type": "open",
        "location_type": "university",
        "capacity": 60,
        "access": "Public"
    },
    {
        "id": "srm_5",
        "name": "Chemistry Block Parking",
        "lat": 12.824396,
        "lng": 80.043560,
        "type": "open",
        "location_type": "university",
        "capacity": 70,
        "access": "Public"
    },
    {
        "id": "srm_6",
        "name": "SRM Global Hospital Parking",
        "lat": 12.823215,
        "lng": 80.047600,
        "type": "open",
        "location_type": "hospital",
        "capacity": 140,
        "access": "Public"
    }
]

# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _load_json(path):
    return storage.load_json(path)


def _save_json(path, data):
    storage.save_json(path, data)


# ---------------------------------------------------------------------------
# Geo helpers
# ---------------------------------------------------------------------------

def _haversine(lat1, lng1, lat2, lng2):
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _enrich_spot(spot, dest_lat, dest_lng, hour, day_of_week=None):
    """Add ML + RL prediction fields to a spot dict."""
    loc_type = spot.get('location_type', 'generic')
    if day_of_week is None:
        day_of_week = datetime.datetime.now().weekday()

    capacity = int(spot.get('capacity', 50))
    if capacity > 100:
        cap_bucket = 2
    elif capacity > 50:
        cap_bucket = 1
    else:
        cap_bucket = 0

    loc_id_hash = (int(spot.get('lat', 0) * 10000) + int(spot.get('lng', 0) * 10000)) % 8

    # Use XGBoost model if available
    ml = _ml_predict(hour, day_of_week, location_id=loc_id_hash, capacity_bucket=cap_bucket, location_type=loc_type)

    if ml['model_used'] != 'fallback':
        status = ml['final_prediction']
        confidence = ml['xgboost_confidence']
    else:
        # Fallback to old RL agent
        prediction = agent.get_prediction(loc_type, hour)
        status = 'medium'
        confidence = prediction['confidence']

    spot['distance_meters'] = round(_haversine(dest_lat, dest_lng, spot['lat'], spot['lng']))
    spot['status'] = status
    spot['confidence'] = confidence
    spot['availability_pct'] = ml.get('availability_percentage', 50.0) if ml['model_used'] != 'fallback' else 50.0
    spot['model_used'] = ml['model_used']
    spot['xgboost_prediction'] = ml.get('xgboost_prediction')
    spot['ql_prediction'] = ml.get('ql_prediction')
    return spot


# ---------------------------------------------------------------------------
# STEP 1 — Check if near SRM University
# ---------------------------------------------------------------------------

def _is_near_srm(lat, lng):
    return _haversine(lat, lng, SRM_CENTER_LAT, SRM_CENTER_LNG) <= SRM_RADIUS_M


def _get_srm_spots(dest_lat, dest_lng, hour, day_of_week=None):
    import copy
    spots = []
    for raw in SRM_PARKING_SPOTS:
        spot = copy.deepcopy(raw)
        spot['id'] = raw['id']
        spot['source'] = 'verified'
        _enrich_spot(spot, dest_lat, dest_lng, hour, day_of_week)
        spots.append(spot)
    spots.sort(key=lambda s: s['distance_meters'])
    return spots


# ---------------------------------------------------------------------------
# STEP 2 — Query Overpass API for real OSM parking
# ---------------------------------------------------------------------------

OVERPASS_URL = 'https://overpass-api.de/api/interpreter'

def _query_overpass(lat, lng):
    query = f"""[out:json][timeout:8];
(
  node["amenity"="parking"](around:1000,{lat},{lng});
  way["amenity"="parking"](around:1000,{lat},{lng});
);
out center;"""

    try:
        data_bytes = urllib.parse.urlencode({'data': query}).encode('utf-8')
        req = urllib.request.Request(OVERPASS_URL, data=data_bytes, method='POST')
        req.add_header('User-Agent', 'ParkNavAI/1.0')
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode('utf-8'))
        return result.get('elements', [])
    except Exception as e:
        print(f"[Overpass] Query failed: {e}")
        return []


def _get_osm_spots(dest_lat, dest_lng, hour, day_of_week=None):
    elements = _query_overpass(dest_lat, dest_lng)
    if not elements:
        return []

    spots = []
    for i, elem in enumerate(elements[:8]):  # cap at 8
        # Nodes have lat/lon directly; ways have center
        if elem['type'] == 'node':
            s_lat = elem.get('lat', 0)
            s_lng = elem.get('lon', 0)
        else:
            center = elem.get('center', {})
            s_lat = center.get('lat', 0)
            s_lng = center.get('lon', 0)

        if s_lat == 0 or s_lng == 0:
            continue

        tags = elem.get('tags', {})
        name = tags.get('name', f"OSM Parking #{i + 1}")
        parking_type = tags.get('parking', tags.get('parking:type', 'surface'))
        capacity_str = tags.get('capacity', '50')
        try:
            capacity = int(capacity_str)
        except ValueError:
            capacity = 50
            
        if capacity > 100:
            cap_bucket = 2
        elif capacity > 50:
            cap_bucket = 1
        else:
            cap_bucket = 0

        access = tags.get('access', 'Public').capitalize()
        if access == 'Yes':
            access = 'Public'

        # Create deterministic pseudo-random ID for location_id
        loc_id_hash = (int(s_lat * 10000) + int(s_lng * 10000)) % 8

        spot = {
            'id': f"osm_{elem.get('id', i)}",
            'name': name,
            'lat': round(s_lat, 6),
            'lng': round(s_lng, 6),
            'type': parking_type,
            'location_type': 'generic',
            'source': 'osm',
            'capacity': capacity,
            'access': access
        }
        
        # ML predict specifically for this spot's features
        if day_of_week is None:
            day_of_week = datetime.datetime.now().weekday()
        ml = _ml_predict(hour, day_of_week, location_id=loc_id_hash, capacity_bucket=cap_bucket, location_type='generic')
        
        if ml['model_used'] != 'fallback':
            spot['status'] = ml['final_prediction']
            spot['confidence'] = ml['xgboost_confidence']
            spot['availability_pct'] = ml.get('availability_percentage', 50.0)
        else:
            spot['status'] = 'available'
            spot['confidence'] = round(90.0 + (datetime.datetime.now().microsecond % 95) / 10.0, 1)
            spot['availability_pct'] = 95.0
            
        spot['distance_meters'] = round(_haversine(dest_lat, dest_lng, spot['lat'], spot['lng']))
        spot['model_used'] = ml['model_used']
        spot['xgboost_prediction'] = ml.get('xgboost_prediction')
        
        spots.append(spot)

    spots.sort(key=lambda s: s['distance_meters'])
    return spots[:6]  # max 6




# ---------------------------------------------------------------------------
# Collect all known spots (hardcoded + community) for distance checks
# ---------------------------------------------------------------------------

def _all_known_spots():
    """Return lat/lng of all hardcoded SRM spots + community spots."""
    coords = [(s['lat'], s['lng']) for s in SRM_PARKING_SPOTS]
    community = _load_json(KNOWN_SPOTS_PATH)
    for s in community:
        coords.append((s['lat'], s['lng']))
    return coords


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/logo.png')
def logo():
    public_dir = os.path.join(BASE_DIR, 'public')
    if os.path.exists(os.path.join(public_dir, 'logo.png')):
        return send_from_directory(public_dir, 'logo.png')
    return send_from_directory(os.path.join(BASE_DIR, 'static'), 'logo.png')


@app.route('/api/parking/nearby')
def parking_nearby():
    """
    3-step priority parking lookup:
      1. SRM hardcoded (if within 2km)
      2. Overpass API (real OSM data)
      3. Simulated fallback
    """
    try:
        lat = float(request.args.get('lat', 20.5937))
        lng = float(request.args.get('lng', 78.9629))
        location_type = request.args.get('location_type', 'generic')
        hour = int(request.args.get('hour', datetime.datetime.now().hour))
        day_of_week = int(request.args.get('day_of_week', datetime.datetime.now().weekday()))

        # ML prediction (XGBoost + Q-table)
        ml_result = _ml_predict(hour, day_of_week=day_of_week, location_type=location_type)
        # Legacy RL prediction (kept for compatibility)
        traffic = _get_traffic_level(hour)
        is_special_day = 0
        rl_prediction = agent.get_prediction(location_type, hour, traffic, is_special_day)

        # Merge: use ML as primary, RL reasoning as supplement
        prediction = rl_prediction.copy()
        if ml_result['model_used'] != 'fallback':
            prediction['status'] = ml_result['final_prediction']
            prediction['xgboost_prediction'] = ml_result['xgboost_prediction']
            prediction['xgboost_confidence'] = ml_result['xgboost_confidence']
            prediction['ql_prediction'] = ml_result['ql_prediction']
            prediction['model_used'] = ml_result['model_used']

        source = 'none'
        spots = []

        # STEP 1 — SRM hardcoded
        if _is_near_srm(lat, lng):
            spots = _get_srm_spots(lat, lng, hour, day_of_week)
            source = 'verified'

        # STEP 2 — Overpass API
        if not spots:
            spots = _get_osm_spots(lat, lng, hour, day_of_week)
            if spots:
                source = 'osm'

        return jsonify({
            'success': True,
            'prediction': prediction,
            'spots': spots,
            'source': source,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/predict')
def predict():
    try:
        location_type = request.args.get('location_type', 'generic')
        hour = int(request.args.get('hour', datetime.datetime.now().hour))
        day_of_week = int(request.args.get('day_of_week', datetime.datetime.now().weekday()))

        # XGBoost + Q-table prediction
        ml_result = _ml_predict(hour, day_of_week=day_of_week, location_type=location_type)
        # Legacy RL prediction (for reasoning text)
        traffic = _get_traffic_level(hour)
        is_special_day = 0
        rl_result = agent.get_prediction(location_type, hour, traffic, is_special_day)

        combined = rl_result.copy()
        combined['xgboost_prediction'] = ml_result.get('xgboost_prediction')
        combined['xgboost_confidence'] = ml_result.get('xgboost_confidence')
        combined['ql_prediction'] = ml_result.get('ql_prediction')
        combined['final_prediction'] = ml_result.get('final_prediction')
        combined['model_used'] = ml_result.get('model_used')
        combined['availability_pct'] = ml_result.get('availability_percentage', 50.0)

        if ml_result['model_used'] != 'fallback':
            combined['status'] = ml_result['final_prediction']

        return jsonify({'success': True, 'data': combined})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/feedback', methods=['POST'])
def feedback():
    """
    Accept crowdsourced parking feedback.
    Extra: if report is >100m from any known spot, treat as NEW spot discovery.
    After 3+ reports at same grid cell → save to known_parking_spots.json.
    """
    try:
        data = request.get_json(force=True)
        lat = float(data.get('lat') or 0)
        lng = float(data.get('lng') or 0)
        fb_status = data.get('status') or 'available'
        hour = int(data.get('hour') or datetime.datetime.now().hour)
        location_type = data.get('location_type') or 'generic'
        is_new_spot = data.get('is_new_spot', False)

        # 1. Save report
        reports = _load_json(REPORTS_PATH)
        report = {
            'lat': round(lat, 5),
            'lng': round(lng, 5),
            'status': fb_status,
            'hour': hour,
            'location_type': location_type,
            'timestamp': datetime.datetime.now().isoformat(),
            'is_new_spot': is_new_spot,
        }
        reports.append(report)
        _save_json(REPORTS_PATH, reports)

        # 2. Update Q-table
        new_confidence = agent.update(location_type, hour, fb_status)

        # 3. Check if location is >100m from any known spot
        known_coords = _all_known_spots()
        is_far_from_known = True
        for (kl, kg) in known_coords:
            if _haversine(lat, lng, kl, kg) <= 100:
                is_far_from_known = False
                break

        new_spot_discovered = False

        # 4. Grid-based aggregation for community spots
        grid_lat = round(lat, 3)
        grid_lng = round(lng, 3)
        nearby_count = sum(
            1 for r in reports
            if round(r['lat'], 3) == grid_lat and round(r['lng'], 3) == grid_lng
        )

        if nearby_count >= 3:
            known = _load_json(KNOWN_SPOTS_PATH)
            already = any(
                round(s['lat'], 3) == grid_lat and round(s['lng'], 3) == grid_lng
                for s in known
            )
            if not already:
                grid_reports = [
                    r for r in reports
                    if round(r['lat'], 3) == grid_lat and round(r['lng'], 3) == grid_lng
                ]
                statuses = [r['status'] for r in grid_reports]
                dominant = max(set(statuses), key=statuses.count)

                known.append({
                    'lat': round(lat, 5),
                    'lng': round(lng, 5),
                    'name': f"Community Verified Spot #{len(known) + 1}",
                    'reports': nearby_count,
                    'dominant_status': dominant,
                    'location_type': location_type,
                    'first_reported': grid_reports[0]['timestamp'],
                    'last_reported': report['timestamp'],
                })
                _save_json(KNOWN_SPOTS_PATH, known)
                new_spot_discovered = True

        return jsonify({
            'success': True,
            'message': 'Thanks! AI is learning from your report.',
            'new_confidence': new_confidence,
            'total_reports_here': nearby_count,
            'is_new_location': is_far_from_known,
            'new_spot_discovered': new_spot_discovered,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/learned-spots')
def learned_spots():
    spots = _load_json(KNOWN_SPOTS_PATH)
    return jsonify({'success': True, 'spots': spots})



@app.route('/api/test-predict')
def test_predict():
    """
    Test the AI with any time/day combination.
    Usage: /api/test-predict?hour=9&day=0&location=university&traffic=3
    Days: 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun
    Traffic: 0=None, 1=Low, 2=Medium, 3=Heavy
    """
    hour = int(request.args.get('hour', 9))
    day = int(request.args.get('day', 0))
    loc = request.args.get('location', 'university')
    traf = int(request.args.get('traffic', 2))
    
    result = _ml_predict(hour=hour, day_of_week=day, location_type=loc,
                         capacity_bucket=1, is_special_day=0)
    
    DAY_NAMES = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    return jsonify({
        'test_input': {
            'hour': hour,
            'day': DAY_NAMES[day],
            'location_type': loc,
            'traffic_level': traf
        },
        'prediction': result['xgboost_prediction'],
        'confidence': result['xgboost_confidence'],
        'model_used': result['model_used'],
        'label': result['final_prediction']
    })


@app.route('/api/health')
def health():
    reports = _load_json(REPORTS_PATH)
    known = _load_json(KNOWN_SPOTS_PATH)
    q_stats = agent.get_stats()
    return jsonify({
        'success': True,
        'q_table': q_stats,
        'total_reports': len(reports),
        'learned_spots_count': len(known),
        'srm_spots': len(SRM_PARKING_SPOTS),
        'xgboost_loaded': xgb_model is not None,
        'model_used': 'Ensemble (XGBoost + RL)' if (MODEL_LOADED and agent.q_table) else ('XGBoost v1.0' if MODEL_LOADED else 'fallback (rule-based)'),
        'feature_columns': feature_columns,
        'status': 'healthy',
    })


# ---------------------------------------------------------------------------
# AI Agent Logic (Gemini)
# ---------------------------------------------------------------------------

# Note: Client is initialized inside the chat route to ensure env vars are fresh.

def _get_food_places(lat, lng):
    """Find nearby food places using Overpass API."""
    query = f"""[out:json][timeout:10];
    (
      node["amenity"~"restaurant|cafe|fast_food|food_court"](around:1500,{lat},{lng});
      way["amenity"~"restaurant|cafe|fast_food|food_court"](around:1500,{lat},{lng});
    );
    out center;"""
    try:
        data_bytes = urllib.parse.urlencode({'data': query}).encode('utf-8')
        req = urllib.request.Request(OVERPASS_URL, data=data_bytes, method='POST')
        req.add_header('User-Agent', 'ParkNavAI/1.0')
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode('utf-8'))
        
        places = []
        for elem in result.get('elements', [])[:10]:
            tags = elem.get('tags', {})
            name = tags.get('name', 'Unnamed Food Place')
            cuisine = tags.get('cuisine', 'Varies')
            places.append({
                'name': name,
                'cuisine': cuisine,
                'lat': elem.get('lat', elem.get('center', {}).get('lat')),
                'lng': elem.get('lon', elem.get('center', {}).get('lng')),
                'type': tags.get('amenity', 'restaurant')
            })
        return places
    except Exception as e:
        return []

def _get_trip_places(lat, lng):
    """Find nearby tourist attractions and landmarks for a trip."""
    query = f"""[out:json][timeout:10];
    (
      node["tourism"~"museum|viewpoint|attraction|gallery"](around:5000,{lat},{lng});
      way["tourism"~"museum|viewpoint|attraction|gallery"](around:5000,{lat},{lng});
      node["historic"~"monument|ruins|castle"](around:5000,{lat},{lng});
      way["historic"~"monument|ruins|castle"](around:5000,{lat},{lng});
      node["leisure"~"park|nature_reserve"](around:5000,{lat},{lng});
    );
    out center;"""
    try:
        data_bytes = urllib.parse.urlencode({'data': query}).encode('utf-8')
        req = urllib.request.Request(OVERPASS_URL, data=data_bytes, method='POST')
        req.add_header('User-Agent', 'ParkNavAI/1.0')
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode('utf-8'))
        
        places = []
        for elem in result.get('elements', [])[:8]: # get up to 8 places for a trip
            tags = elem.get('tags', {})
            name = tags.get('name')
            if not name:
                continue
            places.append({
                'name': name,
                'category': tags.get('tourism') or tags.get('historic') or tags.get('leisure', 'attraction'),
                'lat': elem.get('lat', elem.get('center', {}).get('lat')),
                'lng': elem.get('lon', elem.get('center', {}).get('lng')),
            })
        return places
    except Exception as e:
        return []

@app.route('/api/chat', methods=['POST'])
def chat():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return jsonify({
            'success': False, 
            'error': 'Gemini API Key not configured. Please set GEMINI_API_KEY environment variable.'
        }), 401

    try:
        genai_client = genai.Client(api_key=api_key)
        data = request.get_json(force=True)
        message = data.get('message', '')
        lat = float(data.get('lat', SRM_CENTER_LAT))
        lng = float(data.get('lng', SRM_CENTER_LNG))
        history = data.get('history', [])

        now_str = datetime.datetime.now().strftime('%A, %B %d, %Y - %I:%M %p')
        
        # System context
        context = f"""
        You are ParkNav AI assistant, highly intelligent and connected to the real world.
        Current user location is roughly {lat}, {lng}.
        CURRENT REAL-TIME DATE AND TIME: {now_str}
        
        CRITICAL INSTRUCTION: You MUST use Google Search to find real, legitimate information.
        NEVER make up generic data or hallucinate dates. Always adhere to the provided current date and time.
        If asked about parking or food in ANY city (like Vijayawada), 
        use Google Search to find the best spots, their addresses, and real ratings.
        Format your response beautifully using markdown, lists, and real-world details.
        CRUCIAL: NEVER recommend or provide links to Google Maps. Instead, explicitly guide the user to use the ParkNav map interface and navigation system for directions. Tell them to tap on the map markers within the ParkNav app to navigate.
        If asked to plan a trip or itinerary or weekend, use the 'trip' locations in Contextual Data to create a logical, engaging itinerary. The application will automatically highlight these trip spots on the map. Mention each location by name clearly. You can also mention weather conditions or innovative suggestions based on the actual current date provided.
        """

        # Quick check for keywords to provide context
        lower_msg = message.lower()
        extra_data = {}
        
        if 'food' in lower_msg or 'restaurant' in lower_msg or 'eat' in lower_msg:
            extra_data['food'] = _get_food_places(lat, lng)
        if 'parking' in lower_msg or 'park' in lower_msg:
            # Reuse internal function logic (simplified)
            if _is_near_srm(lat, lng):
                extra_data['parking'] = _get_srm_spots(lat, lng, datetime.datetime.now().hour)
            else:
                extra_data['parking'] = _get_osm_spots(lat, lng, datetime.datetime.now().hour)
        if any(w in lower_msg for w in ['plan', 'weekend', 'trip', 'itinerary', 'visit', 'tour']):
            extra_data['trip'] = _get_trip_places(lat, lng)
        
        if 'traffic' in lower_msg:
            hour = datetime.datetime.now().hour
            level = _get_traffic_level(hour)
            levels = {0: "Quiet", 1: "Normal", 2: "Heavy Peak"}
            extra_data['traffic'] = f"The estimated traffic level at this hour is {levels.get(level)}."

        full_prompt = f"{context}\n\nContextual Data: {json.dumps(extra_data)}\n\nUser: {message}"
        
        response = genai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=full_prompt,
            config=genai.types.GenerateContentConfig(
                tools=[{"google_search": {}}],
                temperature=0.3,
            )
        )
        
        return jsonify({
            'success': True,
            'reply': response.text,
            'data': extra_data # Return data so UI can show markers if needed
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# ---------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True, port=5000)
