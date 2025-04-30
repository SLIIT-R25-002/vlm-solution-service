# 🌇 AI-Powered Urban Heat Island Detection and Mitigation System

### A Vision-Language and ML-Driven System to Detect and Mitigate Urban Heat Island (UHI) Effects Using Environmental Metadata

This project presents an AI-driven decision support system that combines segmented image metadata, environmental sensor data, and machine learning to detect and mitigate Urban Heat Islands (UHIs). The system identifies urban structures (buildings, roads, parks, ponds), analyzes material properties (e.g., concrete, glass, grass), evaluates real-time temperature and humidity, and predicts the presence of heat islands. It further generates mitigation recommendations using prompt engineering through a Vision-Language Model (VLM), aiding smart city planning.

---

## 🔍 Research Focus

### 📌 Research Question

How can AI-driven Vision-Language Models (VLMs) combined with environmental data be used to detect and mitigate Urban Heat Islands (UHIs)?

### 🎯 Objectives

- Detect urban heat islands using a logistic regression model trained on structured metadata.
- Analyze impact of material type, surface area, temperature, and humidity on UHI formation.
- Evaluate area-based heat retention and vegetation coverage.
- Generate contextual mitigation strategies using prompt engineering through a Vision-Language Model.
- Provide interactive visual summaries through a user-friendly web interface.

---

## 🧠 Technology Stack

| Component                      | Tools/Frameworks                                                              |
|-------------------------------|-------------------------------------------------------------------------------|
| Model Training & Inference     | `scikit-learn`, `pandas`, `NumPy`, `StandardScaler`, `LogisticRegression`     |
| Backend API                    | `Flask`, `Flask-CORS`, `joblib`                                               |
| Frontend UI                    | `React`, `React-Bootstrap`, `Axios`                                           |
| Prompt-Based Recommendation    | [BLIP-2 / Vision-Language Models](https://huggingface.co/Salesforce/blip2)   |


---

## 🧪 System Components

### 📊 Heat Island Detection
- Logistic regression model trained on:
  - Material type (numerically encoded)
  - Temperature (°C)
  - Humidity (%)
  - Surface area (m²)
- Returns binary classification: Heat Island = Yes/No

### 🧠 Recommendation Engine
- Uses detected regions and metadata to prompt a VLM.
- Generates personalized, context-aware urban cooling suggestions.

### 🖥️ Web Interface (React)
- Add/Edit/Delete location entries
- Visualize per-location predictions and area-based summaries
- Displays:
  - Heat-retaining material %
  - Vegetation coverage
  - Avg temperature & humidity
  - Final UHI decision

---

## 🚀 How to Run

### 🔧 Backend (Flask)

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Make sure `heat_island_model.pkl` and `scaler.pkl` exist in the same directory.

---

### 🌐 Frontend (React)

```bash
cd heat-island-frontend
npm install
npm start
```

Frontend will run at `http://localhost:3000`, connecting to Flask on `http://localhost:5000`.

---

## 📦 Output Example

```json
{
  "summary": {
    "heat_retaining_percent": 85.0,
    "vegetation_percent": 10.0,
    "avg_temperature": 36.5,
    "avg_humidity": 32.4,
    "final_decision": "Heat Island Detected"
  },
  "detailed_results": [
    {
      "location": "building_1",
      "material": "concrete",
      "temperature": 37.0,
      "humidity": 30.0,
      "area": 200,
      "heat_island": "Yes"
    }
  ]
}
```

---

## 📌 Future Enhancements

- 🧠 Extend VLM recommendation with BLIP-2 fine-tuning

---

## 👨‍💻 Author

**Silva G.M.S.S (IT21802126)**  
Research Component: AI-Powered Urban Heat Island Detection and Mitigation System
