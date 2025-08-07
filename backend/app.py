from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os
import base64
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Load model and scaler
MODEL_PATH = "heat_island_model.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Model or scaler file not found!")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("[INFO] Model and Scaler loaded successfully.")

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY is not set in .env")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Material mapping
MATERIAL_MAPPING = {
    "asphalt": 0, "concrete": 1, "grass": 2, "metal": 3, "plastic": 4,
    "rubber": 5, "sand": 6, "soil": 7, "solar panel": 8, "steel": 9,
    "water": 10, "artificial turf": 11, "glass": 12
}

# ---------------------------- /predict ----------------------------
@app.route('/predict', methods=['POST'])
def predict_heat_island():
    try:
        data = request.json.get('data', [])
        if not data:
            return jsonify({"error": "No data provided"}), 400

        processed = []
        for item in data:
            material = item[1]
            if material in MATERIAL_MAPPING:
                processed.append([
                    MATERIAL_MAPPING[material], item[2], item[3], item[4]
                ])
        if not processed:
            return jsonify({"error": "No valid materials found"}), 400

        # Predict
        scaled = scaler.transform(processed)
        preds = model.predict(scaled)

        # Stats
        total_area = sum(x[4] for x in data)
        heat_area = sum(x[4] for x in data if x[1] in ["asphalt", "concrete", "metal", "steel", "solar panel", "rubber", "plastic", "glass"])
        veg_area = sum(x[4] for x in data if x[1] in ["grass", "soil", "artificial turf"])
        avg_temp = np.mean([x[2] for x in data])
        avg_humidity = np.mean([x[3] for x in data])
        heat_retaining_percent = (heat_area / total_area) * 100
        vegetation_percent = (veg_area / total_area) * 100

        is_heat_island = int(sum(preds) > len(preds) / 2)

        response_data = {
            "is_heat_island": bool(is_heat_island),
            "avg_temperature": round(avg_temp, 1),
            "avg_humidity": round(avg_humidity, 1),
            "heat_retaining_percent": round(heat_retaining_percent, 2),
            "vegetation_percent": round(vegetation_percent, 2),
            "detailed_predictions": [
                {
                    "location": x[0],
                    "material": x[1],
                    "temperature": x[2],
                    "humidity": x[3],
                    "area": x[4],
                    "heat_island": "Yes" if preds[i] == 1 else "No"
                } for i, x in enumerate(data)
            ]
        }

        print("[LOG] Prediction Result:", response_data)  # ✅ Console log
        return jsonify(response_data)

    except Exception as e:
        print("[ERROR] /predict exception:", str(e))  # ❌ Error log
        return jsonify({"error": str(e)}), 500

# ---------------------------- /recommend ----------------------------
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        json_data = request.get_json()
        data = json_data.get("data", [])
        image_b64 = json_data.get("image_base64", "")

        if not data or not image_b64:
            return jsonify({"error": "Missing data or image"}), 400

        total_area = sum(x[4] for x in data)
        heat_area = sum(x[4] for x in data if x[1] in ["asphalt", "concrete", "metal", "steel", "solar panel", "rubber", "plastic", "glass"])
        veg_area = sum(x[4] for x in data if x[1] in ["grass", "soil", "artificial turf"])
        avg_temp = np.mean([x[2] for x in data])
        avg_humidity = np.mean([x[3] for x in data])
        heat_retaining_percent = (heat_area / total_area) * 100
        vegetation_percent = (veg_area / total_area) * 100

        # Build prompt
        prompt = (
            "You are an expert AI assistant specializing in sustainable urban planning.\n\n"
            "Analyze the following metadata and city image to recommend strategies to reduce Urban Heat Island (UHI) effects:\n"
            f"- Average Temperature: {avg_temp:.1f}°C\n"
            f"- Average Humidity: {avg_humidity:.1f}%\n"
            f"- Heat-retaining Surface Coverage: {heat_retaining_percent:.2f}%\n"
            f"- Vegetation Coverage: {vegetation_percent:.2f}%\n\n"
            "Based on this data and the visible environment, provide **exactly 3 actionable and affordable strategies** that could mitigate UHI in this area.\n\n"
            " Guidelines:\n"
            "- Each recommendation must be a **full sentence**.\n"
            "- Focus on **nature-based**, **low-cost**, or **passive cooling** solutions (e.g., tree planting, reflective surfaces, green infrastructure).\n"
            "- Justify each suggestion with a brief explanation of how it helps reduce UHI.\n"
            "- Avoid transportation-related suggestions.\n\n"
            "Format:\n"
            "1. [Action and explanation]\n"
            "2. [Action and explanation]\n"
            "3. [Action and explanation]"
        )

        image_bytes = base64.b64decode(image_b64)
        image_part = {"mime_type": "image/jpeg", "data": image_bytes}

        response = gemini_model.generate_content([prompt, image_part])
        print("[LOG] Gemini Recommendation:\n", response.text)  # ✅ Console log
        return jsonify({"gemini_recommendation": response.text})

    except Exception as e:
        print("[ERROR] /recommend exception:", str(e))  # ❌ Error log
        return jsonify({"error": str(e)}), 500

# ---------------------------- Health Check ----------------------------
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for container orchestration"""
    try:
        # Basic health checks
        health_status = {
            "status": "healthy",
            "service": "vlm-solution-service",
            "model_loaded": model is not None,
            "scaler_loaded": scaler is not None,
            "gemini_configured": GEMINI_API_KEY is not None
        }
        return jsonify(health_status), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "service": "vlm-solution-service",
            "error": str(e)
        }), 500

# ---------------------------- Run App ----------------------------
if __name__ == '__main__':
    print("[INFO] Starting Flask server at http://localhost:5000 ...")
    app.run(debug=True, port=5000)
