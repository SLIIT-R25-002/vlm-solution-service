# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os
import base64
import re
import requests
import google.generativeai as genai
from dotenv import load_dotenv

# ---------------------- Setup & Config ----------------------
load_dotenv()

app = Flask(__name__)
# In production, restrict origins via CORS_ORIGINS="https://your-frontend"
CORS(app, resources={r"/*": {
    "origins": os.getenv("CORS_ORIGINS", "*"),
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"],
}})

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
# Treat artificial turf as heat-retaining (synthetic) rather than vegetation
heat_materials = {"asphalt", "concrete", "metal", "steel", "solar panel", "rubber", "plastic", "glass", "artificial turf"}
veg_materials = {"grass", "soil"}

# ---------------------- Helpers ----------------------
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
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except Exception:
        return default

def parse_segments(segments):
    """
    Return (rows, errors) where rows are:
      [label, material(str), temp(float), humidity(float), area(float)]
    """
    rows = []
    errors = []
    for idx, s in enumerate(segments):
        if not isinstance(s, dict):
            errors.append(f"Segment {idx} not an object")
            continue
        label = s.get("label", "")
        material = (s.get("material") or "").strip().lower()
        temp = to_float(s.get("temp"))
        humidity = to_float(s.get("humidity"))
        area = to_float(s.get("area"))

        if material not in material_mapping:
            errors.append(f"Segment {idx} invalid material: {material or '(empty)'}")
            continue
        if None in (temp, humidity, area):
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

# Looser humidity gate for humid climates; rely primarily on vote + heat/veg/temperature
def rule_based_uhi(m):
    return (m["avg_temp"] > 34 and m["heat_pct"] > 55 and m["veg_pct"] < 25)

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
    return float(x)

def format_segments_for_prompt(segments):
    lines = []
    for s in segments:
        lines.append(
            f"- label:{s.get('label','')} | material:{s.get('material','')} | "
            f"temp:{s.get('temp', s.get('temperature','?'))}°C | "
            f"humidity:{s.get('humidity','?')}% | area:{s.get('area','?')} m²"
        )
    return "\n".join(lines)

# ---------------------- Routes ----------------------
@app.route('/healthz', methods=['GET'])
def healthz():
    return jsonify(status="ok"), 200

@app.route('/predict', methods=['POST'])
def predict_heat_island():
    try:
        payload = request.get_json(force=True, silent=False)
        segments = payload.get("segments", [])
        rows, errors = parse_segments(segments)

        if not rows:
            return jsonify({"error": "No valid segments found", "details": errors}), 400

        metrics = compute_metrics(rows)
        vote_tuple = model_vote(rows)
        vote = vote_tuple[0] if isinstance(vote_tuple, tuple) else vote_tuple
        preds = vote_tuple[1] if isinstance(vote_tuple, tuple) else []
        vote_text = {0: "No Heat Island", 1: "Heat Island"}.get(vote, "N/A")
        final_is_heat_island = (vote == 1) or rule_based_uhi(metrics)

        # vote details for debugging / transparency
        uniq, cnts = (np.unique(np.array(preds), return_counts=True) if len(preds) else (np.array([]), np.array([])))
        vote_details = {
            "0": int(cnts[uniq.tolist().index(0)]) if 0 in uniq.tolist() else 0,
            "1": int(cnts[uniq.tolist().index(1)]) if 1 in uniq.tolist() else 0
        }

        response = {
            "is_heat_island": bool(final_is_heat_island),
            "avg_temperature": round(jsonify_float(metrics["avg_temp"]), 1),
            "avg_humidity": round(jsonify_float(metrics["avg_humidity"]), 1),
            "heat_retaining_percent": round(jsonify_float(metrics["heat_pct"]), 2),
            "vegetation_percent": round(jsonify_float(metrics["veg_pct"]), 2),
            "model_vote": vote_text,
            "model_vote_details": vote_details,
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

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        json_data = request.get_json(force=True, silent=False)
        segments = json_data.get("segments", [])
        image_b64 = (json_data.get("image_base64") or "").strip()
        image_url = (json_data.get("image_url") or "").strip()
        mime_type = (json_data.get("image_mime") or "image/jpeg").lower()

        print("[DEBUG] /recommend keys:", list(json_data.keys()))
        print("[DEBUG] segments:", len(segments), "image_b64?", bool(image_b64), "image_url?", bool(image_url))

        rows, errors = parse_segments(segments)
        if not rows:
            return jsonify({
                "error": "No valid segments found",
                "details": errors or ["All segments were rejected (check material, temp, humidity, area)."]
            }), 400

        if not image_b64 and not image_url:
            return jsonify({"error": "Provide either image_base64 or image_url"}), 400

        image_bytes = None
        if image_b64:
            # Optional: strip data URL prefix if present
            if image_b64.startswith("data:"):
                try:
                    image_b64 = image_b64.split(",", 1)[1]
                except Exception:
                    return jsonify({"error": "Invalid data URL format"}), 400
            # Base64 guard (~13MB decoded -> ~18MB base64)
            if len(image_b64) > 18_000_000:
                return jsonify({"error": "Image too large"}), 413
            try:
                image_bytes = base64.b64decode(image_b64, validate=True)
            except Exception:
                return jsonify({"error": "Invalid base64 image"}), 400
        else:
            try:
                r = requests.get(image_url, timeout=15)
                r.raise_for_status()
                image_bytes = r.content
            except Exception as e:
                return jsonify({"error": f"Failed to download image_url: {e}"}), 400

        if mime_type not in ("image/jpeg", "image/png", "image/webp"):
            mime_type = "image/jpeg"

        m = compute_metrics(rows)
        vote_tuple = model_vote(rows)
        vote_val = vote_tuple[0] if isinstance(vote_tuple, tuple) else vote_tuple
        is_heat_island = rule_based_uhi(m) or (vote_val == 1)

        if not is_heat_island:
            return jsonify({
                "message": "No heat island detected. Recommendation skipped.",
                "metrics": {
                    "avg_temperature": round(float(m["avg_temp"]), 1),
                    "avg_humidity": round(float(m["avg_humidity"]), 1),
                    "heat_retaining_percent": round(float(m["heat_pct"]), 2),
                    "vegetation_percent": round(float(m["veg_pct"]), 2),
                },
                "validation_warnings": errors
            }), 200

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

        image_part = {"mime_type": mime_type, "data": image_bytes}
        try:
            response = gemini_model.generate_content([prompt, image_part])
            cleaned = clean_markdown((getattr(response, "text", "") or "").strip())
        except Exception as ge:
            cleaned = f"(generation failed: {ge})"

        return jsonify({
            "is_heat_island": True,
            "metrics": {
                "avg_temperature": round(float(m["avg_temp"]), 1),
                "avg_humidity": round(float(m["avg_humidity"]), 1),
                "heat_retaining_percent": round(float(m["heat_pct"]), 2),
                "vegetation_percent": round(float(m["veg_pct"]), 2),
            },
            "gemini_recommendation": cleaned or "(no text returned)",
            "validation_warnings": errors
        })

    except Exception as e:
        print("[ERROR] /recommend exception:", str(e))
        return jsonify({"error": str(e)}), 500

# ---------------------- ANSWER-ONLY follow-up chat ----------------------
@app.route('/recommend/chat', methods=['POST', 'OPTIONS'])
def recommend_chat():
    # CORS preflight
    if request.method == "OPTIONS":
        return ("", 204)

    try:
        data = request.get_json(force=True, silent=False) or {}

        message = (data.get("message") or "").strip()
        prior_recommendation = data.get("prior_recommendation") or ""
        history = data.get("history") or []  # [{role:'user'|'assistant', text:'...'}]

        segments = data.get("segments") or []
        image_url = (data.get("image_url") or "").strip()
        image_b64 = (data.get("image_base64") or "").strip()
        image_mime = (data.get("image_mime") or "image/jpeg").lower()

        if not message:
            return jsonify({"error": "message is required"}), 400

        rows, seg_errors = parse_segments(segments)
        if not rows:
            return jsonify({
                "error": "No valid segments found",
                "details": seg_errors or ["All segments were rejected (check material, temp, humidity, area)."]
            }), 400

        # Optional image retrieval (chat can proceed without image)
        image_bytes = None
        if image_b64:
            if image_b64.startswith("data:"):
                try:
                    image_b64 = image_b64.split(",", 1)[1]
                except Exception:
                    return jsonify({"error": "Invalid data URL format"}), 400
            if len(image_b64) > 18_000_000:  # ~13 MB decoded cap
                return jsonify({"error": "Image too large"}), 413
            try:
                image_bytes = base64.b64decode(image_b64, validate=True)
            except Exception:
                return jsonify({"error": "Invalid base64 image"}), 400
        elif image_url:
            try:
                r = requests.get(image_url, timeout=15)
                r.raise_for_status()
                image_bytes = r.content
            except Exception as e:
                print(f"[WARN] /recommend/chat image fetch failed: {e}")
                image_bytes = None

        if image_mime not in ("image/jpeg", "image/png", "image/webp"):
            image_mime = "image/jpeg"

        # -------- answer-only prompting --------
        history_lines = []
        for turn in history:
            role = str(turn.get("role", "")).lower()
            text = str(turn.get("text", "")).strip()
            if not text:
                continue
            who = "User" if role == "user" else "Assistant"
            history_lines.append(f"{who}: {text}")
        history_txt = "\n".join(history_lines)

        segments_txt = format_segments_for_prompt(segments)

        system_context = (
            "You are a UHI mitigation expert. "
            "CRITICAL: Reply with the direct answer to the user's question ONLY. "
            "Do NOT include preambles, summaries, disclaimers, tips, checklists, or next steps "
            "unless explicitly requested. Do NOT repeat the question."
        )

        user_context = (
            f"Context (do not restate): prior recommendation:\n{prior_recommendation}\n\n"
            f"Segments:\n{segments_txt}\n\n"
            + (f"Conversation so far:\n{history_txt}\n\n" if history_txt else "")
            + f"Question to answer directly:\n{message}\n"
        )

        parts = [system_context + "\n\n" + user_context]
        if image_bytes:
            parts.append({"mime_type": image_mime, "data": image_bytes})

        try:
            resp = gemini_model.generate_content(parts)
            reply = clean_markdown((getattr(resp, "text", "") or "").strip())
        except Exception as ge:
            reply = f"(model error: {ge})"

        return jsonify({"reply": reply}), 200

    except Exception as e:
        print("[ERROR] /recommend/chat exception:", str(e))
        return jsonify({"error": str(e)}), 500

# ---------------------- Start Server ----------------------
if __name__ == '__main__':
    print("[INFO] Starting Flask server at http://localhost:5000 ...")
    app.run(host="0.0.0.0", debug=True, port=5000)  # set debug=False in prod
