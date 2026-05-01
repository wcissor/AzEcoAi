import  os
import csv


import pandas as pd
import joblib

from config import SERVER_PORT,ICE_RISK_THRESHOLDS
from datetime import datetime
from flask import Flask,request,jsonify,render_template

app = Flask(__name__)
LOG_FILE = "sensor_log.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE,"w",newline = "") as f:
        writer = csv.writer(f)
        writer.writerow(["temperature","humidity","timestamp","road_moisture"])
ice_model = joblib.load("models/ice_model.pkl")
FEATURES = joblib.load("models/features.pkl")
def create_ice_features(data):
    temperature = data["temperature"]
    humidity = data["humidity"]
    road_moisture = data["road_moisture"]
    freezing_index = max(0,0- temperature)* road_moisture
    dew_point = temperature - ((100 - humidity)/5)
    features_data = {"temperature":temperature,"humidity":humidity,"road_moisture":road_moisture,"freezing_index":freezing_index,"dew_point":dew_point}
    return pd.DataFrame([features_data])[FEATURES]

@app.route("/ice_predict",methods = ["POST"])
def predict_ice():
    data = request.json
    required_fields = ["temperature","humidity","road_moisture"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error":f"{field} mising"}),400
    try:
        temperature = float(data["temperature"])
        humidity = float(data["humidity"])
        road_moisture = float(data["road_moisture"])
    except:
        return jsonify({"error":f"NZD"})
    with open(LOG_FILE,"a",newline="")as f:
        writer = csv.writer(f)
        writer.writerow([temperature,humidity,datetime.now().strftime("%H:%M:%S"),road_moisture])
    df = create_ice_features({"temperature":temperature,
    "humidity":humidity,"road_moisture":road_moisture})
    probability = ice_model.predict_proba(df)[0][1]
    if probability > ICE_RISK_THRESHOLDS["HIGH"]:
        risk  = "HIGH"
    elif probability > ICE_RISK_THRESHOLDS["LOW"]:
        risk = "MEDIUM"
    else:
        risk = "LOW"
    return jsonify({"ice_probability":probability,"risk":risk})
@app.route("/sensor_log")
def sensor_log():
    if not os.path.exists(LOG_FILE):
        return jsonify([])
    df = pd.read_csv(LOG_FILE,names = ["temperature","humidity","timestamp","road_moisture"],header = 0)
    df = df.tail(30)
    return jsonify(df.to_dict(orient="records"))
@app.route("/")
def create():

    return render_template("dashboard.html")
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")
if __name__ == "__main__":
     app.run(   host = "127.0.0.1", debug = True, port = SERVER_PORT)
