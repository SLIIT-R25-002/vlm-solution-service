from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os
import time

app = Flask(__name__)
CORS(app)

MODEL_PATH = "heat_island_model.pkl"
SCALER_PATH = "scaler.pkl"

if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
    raise FileNotFoundError("Model or scaler files not found!")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print("[INFO] Model and Scaler loaded successfully.")

MATERIAL_MAPPING = {
    "asphalt": 0, "concrete": 1, "grass": 2, "metal": 3, "plastic": 4,
    "rubber": 5, "sand": 6, "soil": 7, "solar panel": 8, "steel": 9,
    "water": 10, "artificial turf": 11, "glass": 12
}

@app.route('/predict', methods=['POST'])
def predict_heat_island():
    try:
        print("[INFO] /predict endpoint hit")

        data = request.json.get('data', [])
        if not data:
            return jsonify({"error": "No data provided"}), 400

        processed_data = []
        for item in data:
            material = item[1]
            if material in MATERIAL_MAPPING:
                processed_data.append([
                    MATERIAL_MAPPING[material],
                    item[2], item[3], item[4]
                ])

        if not processed_data:
            return jsonify({"error": "No valid data to process"}), 400

        processed_data_scaled = scaler.transform(processed_data)
        predictions = model.predict(processed_data_scaled)

        total_area = sum(item[4] for item in data)
        heat_retaining_area = sum(
            item[4] for item in data if item[1] in ["asphalt", "concrete", "metal", "steel", "solar panel", "rubber", "plastic", "glass"]
        )
        vegetation_area = sum(
            item[4] for item in data if item[1] in ["grass", "soil", "artificial turf"]
        )

        avg_temp = np.mean([item[2] for item in data])
        avg_humidity = np.mean([item[3] for item in data])

        heat_retaining_percent = (heat_retaining_area / total_area) * 100
        vegetation_percent = (vegetation_area / total_area) * 100

        is_heat_island = (
            avg_temp > 33 and
            heat_retaining_percent > 60 and
            vegetation_percent < 20 and
            avg_humidity < 45
        )

        return jsonify({
            "summary": {
                "heat_retaining_percent": round(heat_retaining_percent, 2),
                "vegetation_percent": round(vegetation_percent, 2),
                "avg_temperature": round(avg_temp, 1),
                "avg_humidity": round(avg_humidity, 1),
                "final_decision": "Heat Island Detected" if is_heat_island else "No Heat Island Detected",
                "is_heat_island": "true" if is_heat_island else "false"
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


import base64
import google.generativeai as genai  # ensure you have gemini installed
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY not set in .env")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-pro-vision")

@app.route('/recommend', methods=['POST'])
def recommend_strategies():
    try:
        print("[INFO] /recommend endpoint hit")

        json_data = request.get_json()
        data = json_data.get('data', [])
        image_base64 = json_data.get('image_base64', '')

        if not data or not image_base64:
            return jsonify({"error": "Missing data or image_base64"}), 400

        # Extract metrics for prompt
        avg_temp = np.mean([item[2] for item in data])
        avg_humidity = np.mean([item[3] for item in data])
        total_area = sum(item[4] for item in data)

        heat_retaining_area = sum(
            item[4] for item in data if item[1] in ["asphalt", "concrete", "metal", "steel", "solar panel", "rubber", "plastic", "glass"]
        )
        vegetation_area = sum(
            item[4] for item in data if item[1] in ["grass", "soil", "artificial turf"]
        )

        heat_retaining_percent = (heat_retaining_area / total_area) * 100
        vegetation_percent = (vegetation_area / total_area) * 100

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

        image_bytes = base64.b64decode(image_base64)
        image_part = {
            "mime_type": "image/jpeg",
            "data": image_bytes
        }

        response = model.generate_content([prompt, image_part])
        reply = response.text

        return jsonify({"gemini_recommendation": reply})

    except Exception as e:
        print(f"[ERROR] /recommend failed: {e}")
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    print("[INFO] Starting Flask server on port 5000...")
    app.run(debug=True, port=5000)
