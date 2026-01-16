# Sangai Weather – FastAPI & AI Backend

This repository contains the **FastAPI backend** and **machine learning models** used by the Sangai Weather Kivy application.
It provides weather forecasting and multi-hazard risk prediction using real-time API data and trained ML models.

---

## Overview

The backend system:

- Receives location data from Kivy UI
- Fetches weather forecast from OpenWeather API
- Uses trained ML models to predict:

  - Rainfall
  - Flood risk
  - Landslide risk
  - Earthquake zone

- Generates rule-based warnings
- Sends structured JSON response to the UI

---

## Features

- REST API built with FastAPI
- Real-time weather data integration
- Machine learning predictions
- Rule-based hazard warnings
- IST timezone support
- Production-ready API design

---

## Tech Stack

- Python
- FastAPI
- Scikit-learn
- Pandas
- Joblib
- OpenWeather API
- Uvicorn

---

## Project Structure

```
backend/
│
├── main.py                  # FastAPI server
├── WEATHERFINAL.py          # Model training script
├── models/
│   ├── rainfall_model.pkl
│   ├── flood_model.pkl
│   ├── landslide_model.pkl
│   └── earthquake_cluster_model.pkl
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set environment variable

```bash
export OPENWEATHER_API_KEY="your_api_key"
```

### 3. Run server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## API Endpoint

### POST /predict

Request body:

```json
{
  "lat": 24.8039,
  "lon": 93.942,
  "district": "Imphal East",
  "rainfall_3d": null,
  "soil_moisture": 0.4,
  "slope": 10,
  "elevation": 500
}
```

Response:

- Weather forecast (3 days)
- Rainfall prediction
- Flood risk probability
- Landslide risk probability
- Earthquake zone
- Warning alerts
- Overall risk status

---

## Machine Learning Models

- Rainfall prediction model
- Flood classification model
- Landslide classification model
- Earthquake clustering model

All models are pre-trained and loaded at server startup.

---

## Important Notes

- Server must be reachable on local network
- Run with --host 0.0.0.0 for external access
- Firewall should allow port 8000
- OpenWeather API key is required

---

## Author

Adi Prakash
GitHub: [https://github.com/adiorinder](https://github.com/adiorinder)

---

## License

MIT License
