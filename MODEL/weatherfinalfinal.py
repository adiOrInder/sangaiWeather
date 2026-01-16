import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from xgboost import XGBClassifier, XGBRegressor


#Create model directory
os.makedirs("saved_models", exist_ok=True)

#Load data
df = pd.read_csv("manipur_hazards_final_FIXED.csv")

# Dummy labels (only for ML-based hazards)
df["flood_label"] = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])
df["landslide_label"] = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])


#Rainfall Prediction
y = df["rainfall"]
X = df[
    ["month", "lat", "lon", "pressure",
     "temp_avg", "humidity", "wind_speed", "cloud_cover"]
]

rainfall_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=42
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rainfall_model.fit(X_train, y_train)
preds = rainfall_model.predict(X_test)

print("Rainfall MAE:", mean_absolute_error(y_test, preds))
print("Rainfall R2:", r2_score(y_test, preds))

# RULE-BASED MODELS
class HeatwaveRuleModel:
    def predict(self, X):
        results = []
        for temp_max, month, humidity, pressure, wind_speed in X:
            if month not in [3, 4, 5, 6]:
                results.append(0)
            elif temp_max >= 45:
                results.append(1)
            elif temp_max < 40:
                results.append(0)
            else:
                score = 0
                if 25 <= humidity <= 60:
                    score += 1
                if pressure >= 1004:
                    score += 1
                if wind_speed <= 3:
                    score += 1
                results.append(1 if score >= 2 else 0)
        return np.array(results)


class ColdwaveRuleModel:
    def predict(self, X):
        results = []
        for temp_min, month, humidity, pressure, wind_speed in X:
            if month not in [12, 1, 2]:
                results.append(0)
            elif temp_min > 5:
                results.append(0)
            else:
                score = 0
                if humidity <= 50:
                    score += 1
                if pressure >= 1010:
                    score += 1
                if wind_speed >= 3:
                    score += 1
                results.append(1 if score >= 2 else 0)
        return np.array(results)


class HailstormRuleModel:
    def predict(self, X):
        results = []
        for temp, humidity, pressure, wind_speed, cloud_cover in X:
            score = 0

            if temp >= 25:
                score += 1
            if humidity >= 70:
                score += 1
            if pressure <= 1000:
                score += 1
            if wind_speed >= 6:
                score += 1
            if cloud_cover >= 70:
                score += 1

            results.append(1 if score >= 3 else 0)
        return np.array(results)


#Flood Prediction
y = df["flood_label"]
X = df[
    ["rainfall", "rainfall_3d", "soil_moisture",
     "slope", "elevation", "lat", "lon"]
]

flood_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=5,
        eval_metric="logloss",
        random_state=42
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

flood_model.fit(X_train, y_train)
preds = flood_model.predict(X_test)

print("\nFlood Model")
print(classification_report(y_test, preds))


#Landslide Prediction
y = df["landslide_label"]
X = df[
    ["rainfall_3d", "soil_moisture",
     "slope", "ndvi", "wind_speed", "elevation"]
]

landslide_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        class_weight="balanced",
        random_state=42
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

landslide_model.fit(X_train, y_train)
preds = landslide_model.predict(X_test)

print("\nLandslide Model")
print(classification_report(y_test, preds))

# Earthquake Risk Clustering
eq_features = df[[
    "lat", "lon",
    "elevation",
    "slope",
    "earthquake_prone"
]]

earthquake_cluster_model = Pipeline([
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(n_clusters=3, random_state=42, n_init=20))
])

df["earthquake_cluster"] = earthquake_cluster_model.fit_predict(eq_features)

joblib.dump(rainfall_model, "saved_models/rainfall_model.joblib")
joblib.dump(flood_model, "saved_models/flood_model.joblib")
joblib.dump(landslide_model, "saved_models/landslide_model.joblib")
joblib.dump(earthquake_cluster_model, "saved_models/earthquake_kmeans.joblib")
joblib.dump(HeatwaveRuleModel(), "saved_models/heatwave_rule.joblib")
joblib.dump(ColdwaveRuleModel(), "saved_models/coldwave_rule.joblib")
joblib.dump(HailstormRuleModel(), "saved_models/hailstorm_rule.joblib")
