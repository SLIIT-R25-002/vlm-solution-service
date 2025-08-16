from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os
import base64
import re
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# >>> In production, restrict origins:
CORS(app, resources={r"/*": {"origins": os.getenv("CORS_ORIGINS", "*")}})

# ---- Model & Scaler ----
MODEL_PATH = "heat_island_model.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Model or scaler file not found!")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("[INFO] Model and Scaler loaded successfully.")

# ---- Gemini ----
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY is not set in .env")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ---- Materials ----
material_mapping = {
    "asphalt": 0, "concrete": 1, "grass": 2, "metal": 3, "plastic": 4,
    "rubber": 5, "sand": 6, "soil": 7, "solar panel": 8, "steel": 9,
    "water": 10, "artificial turf": 11, "glass": 12
}
heat_materials = {"asphalt", "concrete", "metal", "steel", "solar panel", "rubber", "plastic", "glass"}
veg_materials = {"grass", "soil", "artificial turf"}

# ---- Helpers ----
def clean_markdown(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"[-•–]+", "", text)
    text = re.sub(r"^#+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^---+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def to_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def parse_segments(segments):
    """Return validated rows: [label, material, temp, humidity, area]"""
    rows = []
    errors = []
    for idx, s in enumerate(segments):
        if not isinstance(s, dict):
            errors.append(f"Segment {idx} not an object")
            continue
        label = s.get("label")
        material = (s.get("material") or "").strip().lower()
        temp = to_float(s.get("temp"))
        humidity = to_float(s.get("humidity"))
        area = to_float(s.get("area"))
        if material not in material_mapping:
            errors.append(f"Segment {idx} invalid material: {material}")
            continue
        if None in (label, temp, humidity, area):
            errors.append(f"Segment {idx} missing/invalid numeric fields")
            continue
        if area < 0 or temp < -80 or temp > 80 or humidity < 0 or humidity > 100:
            errors.append(f"Segment {idx} out-of-range values")
            continue
        rows.append([label, material, temp, humidity, area])
    return rows, errors

def compute_metrics(rows):
    total_area = sum(r[4] for r in rows) or 1.0
    heat_area = sum(r[4] for r in rows if r[1] in heat_materials)
    veg_area = sum(r[4] for r in rows if r[1] in veg_materials)
    avg_temp = float(np.mean([r[2] for r in rows])) if rows else 0.0
    avg_humidity = float(np.mean([r[3] for r in rows])) if rows else 0.0
    return {
        "total_area": float(total_area),
        "heat_pct": float((heat_area / total_area) * 100.0),
        "veg_pct": float((veg_area / total_area) * 100.0),
        "avg_temp": float(avg_temp),
        "avg_humidity": float(avg_humidity),
    }

def rule_based_uhi(m):
    return (m["avg_temp"] > 33 and m["heat_pct"] > 60 and m["veg_pct"] < 20 and m["avg_humidity"] < 45)

def model_vote(rows):
    """Majority vote of per-segment predictions (1=UHI)."""
    if not rows:
        return None
    X = np.array([[material_mapping[r[1]], r[2], r[3], r[4]] for r in rows], dtype=float)
    Xs = scaler.transform(X)
    preds = model.predict(Xs)
    vals, counts = np.unique(preds, return_counts=True)
    return int(vals[np.argmax(counts)]), preds.tolist()

def jsonify_float(x):
    # Ensure Python float (not numpy types)
    return float(x)

# ---------------------------- /predict ----------------------------
@app.route('/predict', methods=['POST'])
def predict_heat_island():
    try:
        payload = request.get_json(force=True, silent=False)
        segments = payload.get("segments", [])
        rows, errors = parse_segments(segments)
        if not rows:
            return jsonify({"error": "No valid segments found", "details": errors}), 400

        metrics = compute_metrics(rows)
        vote, preds = model_vote(rows)  # preds is list aligned to rows
        vote_text = {0: "No Heat Island", 1: "Heat Island"}.get(vote, "N/A")
        final_is_heat_island = (vote == 1) or rule_based_uhi(metrics)

        response = {
            "is_heat_island": bool(final_is_heat_island),
            "avg_temperature": round(jsonify_float(metrics["avg_temp"]), 1),
            "avg_humidity": round(jsonify_float(metrics["avg_humidity"]), 1),
            "heat_retaining_percent": round(jsonify_float(metrics["heat_pct"]), 2),
            "vegetation_percent": round(jsonify_float(metrics["veg_pct"]), 2),
            "model_vote": vote_text,
            "detailed_predictions": [
                {
                    "location": r[0],
                    "material": r[1],
                    "temperature": round(jsonify_float(r[2]), 1),
                    "humidity": round(jsonify_float(r[3]), 1),
                    "area": round(jsonify_float(r[4]), 2),
                    "heat_island": "Yes" if int(preds[i]) == 1 else "No"
                } for i, r in enumerate(rows)
            ],
            "validation_warnings": errors
        }
        return jsonify(response)
    except Exception as e:
        print("[ERROR] /predict exception:", str(e))
        return jsonify({"error": str(e)}), 500

# ---------------------------- /recommend ----------------------------
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        json_data = request.get_json(force=True, silent=False)
        segments = json_data.get("segments", [])
        image_b64 = (json_data.get("image_base64") or "").strip()

        if not segments or not image_b64:
            return jsonify({"error": "Missing image or segments"}), 400

        # Optional: strip data URL prefix if present
        if image_b64.startswith("data:"):
            try:
                image_b64 = image_b64.split(",", 1)[1]
            except Exception:
                return jsonify({"error": "Invalid data URL format"}), 400

        # Basic size guard (e.g., 10 MB)
        if len(image_b64) > 10_000_000 * 1.37:  # base64 ~37% overhead
            return jsonify({"error": "Image too large"}), 413

        rows, errors = parse_segments(segments)
        if not rows:
            return jsonify({"error": "No valid segments", "details": errors}), 400

        m = compute_metrics(rows)
        is_heat_island = rule_based_uhi(m) or (model_vote(rows)[0] == 1)

        if not is_heat_island:
            return jsonify({"message": "No heat island detected. Recommendation skipped."}), 200

        prompt = (
            "You are an expert AI assistant specializing in sustainable urban planning.\n\n"
            "Analyze the following metadata and city image to recommend strategies to reduce Urban Heat Island (UHI) effects:\n"
            f"- Average Temperature: {m['avg_temp']:.1f}°C\n"
            f"- Average Humidity: {m['avg_humidity']:.1f}%\n"
            f"- Heat-retaining Surface Coverage: {m['heat_pct']:.2f}%\n"
            f"- Vegetation Coverage: {m['veg_pct']:.2f}%\n\n"
            "Based on this data and the visible environment, provide exactly 3 actionable and affordable strategies to mitigate UHI.\n\n"
            "Guidelines:\n"
            "- Each recommendation must be a full sentence.\n"
            "- Focus on nature-based, low-cost, or passive cooling solutions.\n"
            "- Justify each suggestion with a brief explanation.\n"
            "- Avoid transportation-related suggestions.\n\n"
            "Format:\n"
            "1. [Action and explanation]\n"
            "2. [Action and explanation]\n"
            "3. [Action and explanation]"
        )

        try:
            image_bytes = base64.b64decode(image_b64, validate=True)
        except Exception:
            return jsonify({"error": "Invalid base64 image"}), 400

        # Accept common mime types (frontend can always send image/jpeg)
        mime_type = json_data.get("image_mime") or "image/jpeg"
        if mime_type not in ("image/jpeg", "image/png", "image/webp"):
            mime_type = "image/jpeg"

        image_part = {"mime_type": mime_type, "data": image_bytes}

        response = gemini_model.generate_content([prompt, image_part])
        cleaned = clean_markdown((response.text or "").strip())

        return jsonify({
            "is_heat_island": True,
            "metrics": {
                "avg_temperature": round(jsonify_float(m["avg_temp"]), 1),
                "avg_humidity": round(jsonify_float(m["avg_humidity"]), 1),
                "heat_retaining_percent": round(jsonify_float(m["heat_pct"]), 2),
                "vegetation_percent": round(jsonify_float(m["veg_pct"]), 2),
            },
            "gemini_recommendation": cleaned or "(no text returned)"
        })

    except Exception as e:
        print("[ERROR] /recommend exception:", str(e))
        return jsonify({"error": str(e)}), 500



# ---------------------------- Start Server ----------------------------
if __name__ == '__main__':
    print("[INFO] Starting Flask server at http://localhost:5000 ...")
    app.run(host="0.0.0.0", debug=True, port=5000)  # >>> debug=False in prod
