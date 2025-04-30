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

        time.sleep(5)

        data = request.json.get('data', [])
        if not data:
            print("[ERROR] No data provided in request.")
            return jsonify({"error": "No data provided"}), 400

        print(f"[INFO] Received {len(data)} data entries")

        processed_data = []
        unrecognized_materials = []

        for item in data:
            material = item[1]
            if material in MATERIAL_MAPPING:
                processed_data.append([
                    MATERIAL_MAPPING[material],
                    item[2], item[3], item[4]
                ])
            else:
                unrecognized_materials.append(material)

        if unrecognized_materials:
            print(f"[WARNING] Unrecognized materials: {unrecognized_materials}")

        if not processed_data:
            print("[ERROR] No valid data to process after filtering.")
            return jsonify({"error": "No valid data to process"}), 400

        print(f"[INFO] Processed data count: {len(processed_data)}")

        processed_data_scaled = scaler.transform(processed_data)
        predictions = model.predict(processed_data_scaled)
        print(f"[INFO] Predictions completed: {predictions}")

        total_area = sum(item[4] for item in data)
        heat_retaining_area = sum(
            item[4] for item in data
            if item[1] in ["asphalt", "concrete", "metal", "steel", "solar panel", "rubber", "plastic", "glass"]
        )
        vegetation_area = sum(
            item[4] for item in data
            if item[1] in ["grass", "soil", "artificial turf"]
        )

        avg_temp = np.mean([item[2] for item in data])
        avg_humidity = np.mean([item[3] for item in data])

        heat_retaining_percent = (heat_retaining_area / total_area) * 100
        vegetation_percent = (vegetation_area / total_area) * 100

        print(f"[INFO] Area stats: heat={heat_retaining_percent:.2f}%, vegetation={vegetation_percent:.2f}%")
        print(f"[INFO] Temp={avg_temp:.2f}°C, Humidity={avg_humidity:.2f}%")

        is_heat_island = (
            avg_temp > 33 and
            heat_retaining_percent > 60 and
            vegetation_percent < 20 and
            avg_humidity < 45
        )

        print(f"[INFO] Final Decision: {'Heat Island Detected' if is_heat_island else 'No Heat Island Detected'}")

        detailed_results = []
        for i, item in enumerate(data):
            detailed_results.append({
                "location": item[0],
                "material": item[1],
                "temperature": item[2],
                "humidity": item[3],
                "area": item[4],
                "heat_island": "Yes" if predictions[i] == 1 else "No"
            })

        return jsonify({
            "detailed_results": detailed_results,
            "summary": {
                "heat_retaining_percent": round(heat_retaining_percent, 2),
                "vegetation_percent": round(vegetation_percent, 2),
                "avg_temperature": round(avg_temp, 1),
                "avg_humidity": round(avg_humidity, 1),
                "final_decision": "Heat Island Detected" if is_heat_island else "No Heat Island Detected"
            }
        })

    except Exception as e:
        print(f"[ERROR] Exception occurred: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("[INFO] Starting Flask server on port 5000...")
    app.run(debug=True, port=5000)
