from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import requests
from datetime import datetime
import pytz
import os


# FASTAPI APP
app = FastAPI(
    title="SANGAI WEATHER",
    version="1.0.0",
    description="Forecast-driven disaster prediction using ML + rule engine"
)

IST = pytz.timezone("Asia/Kolkata")

# ============================================================
# CONFIG (ENV VAR FOR SECURITY)
# ============================================================
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
if not OPENWEATHER_API_KEY:
    raise RuntimeError("Missing OPENWEATHER_API_KEY environment variable")


# LOAD MODELS (ONCE AT STARTUP)
required_models = [
    "MODEL/saved_models/rainfall_model.joblib",
    "MODEL/saved_models/flood_model.joblib",
    "MODEL/saved_models/landslide_model.joblib",
    "MODEL/saved_models/earthquake_kmeans.joblib"
]


rainfall_model = joblib.load("MODEL/saved_models/rainfall_model.joblib")
flood_model = joblib.load("MODEL/saved_models/flood_model.joblib")
landslide_model = joblib.load("MODEL/saved_models/landslide_model.joblib")
earthquake_model = joblib.load("MODEL/saved_models/earthquake_kmeans.joblib")


# INPUT SCHEMA (FROM KIVY)
class HazardRequest(BaseModel):
    lat: float
    lon: float
    district: str | None = "UNKNOWN"
    rainfall_3d: float | None = None
    soil_moisture: float | None = 0.4
    slope: float | None = 10
    elevation: float | None = 500


# WEATHER FORECAST FETCHER (CURRENT + 3 DAYS, SAFE)

def fetch_forecast_weather(lat, lon):
    url = (
        "https://api.openweathermap.org/data/2.5/forecast"
        f"?lat={lat}&lon={lon}"
        "&units=metric"
        f"&appid={OPENWEATHER_API_KEY}"
    )

    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()

    # current weather (first entry)
    current = data["list"][0]

    forecast = {
        "today": {
            "temp_max": round(current["main"]["temp_max"], 1),
            "temp_min": round(current["main"]["temp_min"], 1),
            "humidity": current["main"]["humidity"],
            "wind_speed": current["wind"]["speed"]
        },
        "tomorrow": {
            "temp_max": round(data["list"][8]["main"]["temp_max"], 1),
            "temp_min": round(data["list"][8]["main"]["temp_min"], 1),
            "humidity": data["list"][8]["main"]["humidity"],
            "wind_speed": data["list"][8]["wind"]["speed"]
        },
        "day_after": {
            "temp_max": round(data["list"][16]["main"]["temp_max"], 1),
            "temp_min": round(data["list"][16]["main"]["temp_min"], 1),
            "humidity": data["list"][16]["main"]["humidity"],
            "wind_speed": data["list"][16]["wind"]["speed"]
        }
    }

    return {
        "current": {
            "temp": current["main"]["temp"],
            "pressure": current["main"]["pressure"],
            "humidity": current["main"]["humidity"],
            "wind_speed": current["wind"]["speed"],
            "cloud_cover": current["clouds"]["all"]
        },
        **forecast
    }


# RULE-BASED WEATHER WARNINGS
def generate_weather_warnings(w):
    warnings = {}

    heatwave = w["temp"] >= 40 and w["humidity"] <= 40 and w["wind_speed"] <= 3
    coldwave = w["temp"] <= 5 and w["wind_speed"] >= 3
    hailstorm = w["wind_speed"] >= 8 and w["cloud_cover"] >= 60

    warnings["heatwave"] = {
        "active": heatwave,
        "severity": "SEVERE" if w["temp"] >= 45 else "MODERATE" if heatwave else "NONE",
        "reason": "Extreme temperature with low humidity" if heatwave else "Conditions not met"
    }

    warnings["coldwave"] = {
        "active": coldwave,
        "severity": "SEVERE" if w["temp"] <= 0 else "MODERATE" if coldwave else "NONE",
        "reason": "Low temperature with strong wind chill" if coldwave else "Conditions not met"
    }

    warnings["hailstorm"] = {
        "active": hailstorm,
        "severity": "SEVERE" if w["wind_speed"] >= 12 else "MODERATE" if hailstorm else "NONE",
        "reason": "Strong updrafts within thunderstorm cell" if hailstorm else "Conditions not met"
    }

    return warnings


# MAIN API ENDPOINT

@app.post("/predict")
def predict_hazards(req: HazardRequest):
    try:
        #WEATHER
        weather_data = fetch_forecast_weather(req.lat, req.lon)
        weather = weather_data["current"]

        #RAINFALL
        rain_X = pd.DataFrame([[
            datetime.now().month,
            req.lat, req.lon,
            weather["pressure"],
            weather["temp"],
            weather["humidity"],
            weather["wind_speed"],
            weather["cloud_cover"]
        ]], columns=[
            "month", "lat", "lon", "pressure",
            "temp_avg", "humidity", "wind_speed", "cloud_cover"
        ])

        rainfall = float(rainfall_model.predict(rain_X)[0])

        #FLOOD (ML)
        flood_X = pd.DataFrame([[
            rainfall,
            req.rainfall_3d or rainfall,
            req.soil_moisture,
            req.slope,
            req.elevation,
            req.lat,
            req.lon
        ]], columns=[
            "rainfall", "rainfall_3d", "soil_moisture",
            "slope", "elevation", "lat", "lon"
        ])

        flood_prob = float(flood_model.predict_proba(flood_X)[0][1])

        #  LANDSLIDE
        land_X = pd.DataFrame([[
            req.rainfall_3d or rainfall,
            req.soil_moisture,
            req.slope,
            0.5,  # NDVI fallback
            weather["wind_speed"],
            req.elevation
        ]], columns=[
            "rainfall_3d", "soil_moisture",
            "slope", "ndvi", "wind_speed", "elevation"
        ])

        land_prob = float(landslide_model.predict_proba(land_X)[0][1])

        # EARTHQUAKE 
        eq_X = pd.DataFrame([[
    req.lat,
    req.lon,
    req.elevation,
    req.slope,
    1   # earthquake_prone default
]], columns=[
    "lat",
    "lon",
    "elevation",
    "slope",
    "earthquake_prone"
])
        eq_cluster = earthquake_model.predict(eq_X)[0]
        eq_zone = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}.get(eq_cluster, "MEDIUM")

        
        warnings = generate_weather_warnings(weather)

        
        def level(p):
            if p > 0.75: return "HIGH"
            if p > 0.45: return "MODERATE"
            return "LOW"

        flood_level = level(flood_prob)
        land_level = level(land_prob)

        overall_alert = (
            "SEVERE" if flood_level == "HIGH" or land_level == "HIGH"
            else "MODERATE" if any(w["active"] for w in warnings.values())
            else "NORMAL"
        )
        print({
            "location": {
                "lat": req.lat,
                "lon": req.lon,
                "district": req.district
            },
            "timestamp": datetime.now(IST).isoformat(),
            "observed_weather": {
                "temperature_c": weather["temp"],
                "pressure_hpa": weather["pressure"],
                "humidity_percent": weather["humidity"],
                "wind_speed_mps": weather["wind_speed"]
            },
            "today": weather_data.get("today", {}),
            "tomorrow": weather_data.get("tomorrow", {}),
            "day_after": weather_data.get("day_after", {}),
           "rainfall_mm": round(rainfall, 2),

            "warnings": warnings,
            "risk_assessment": {
                "flood": {
                    "probability": round(flood_prob, 2),
                    "risk_level": flood_level
                },
                "landslide": {
                    "probability": round(land_prob, 2),
                    "risk_level": land_level
                },
                "earthquake": {
                    "zone": eq_zone
                }
            },
            "overall_alert": overall_alert
        })
        # ---------------- RESPONSE ----------------
        return {
            "location": {
                "lat": req.lat,
                "lon": req.lon,
                "district": req.district
            },
            "timestamp": datetime.now(IST).isoformat(),
            "observed_weather": {
                "temperature_c": weather["temp"],
                "pressure_hpa": weather["pressure"],
                "humidity_percent": weather["humidity"],
                "wind_speed_mps": weather["wind_speed"]
            },
            "today": weather_data.get("today", {}),
            "tomorrow": weather_data.get("tomorrow", {}),
            "day_after": weather_data.get("day_after", {}),
           "rainfall_mm": round(rainfall, 2),

            "warnings": warnings,
            "risk_assessment": {
                "flood": {
                    "probability": round(flood_prob, 2),
                    "risk_level": flood_level
                },
                "landslide": {
                    "probability": round(land_prob, 5),
                    "risk_level": land_level
                },
                "earthquake": {
                    "zone": eq_zone
                }
            },
            "overall_alert": overall_alert
        }

    except Exception as e:
        print("SERVER ERROR ",e)
        raise HTTPException(status_code=500, detail=str(e))
