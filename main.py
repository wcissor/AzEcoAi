import os
import joblib
import numpy as np
import pandas as pd
from pyexpat import features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from api_server import FEATURES
DATA_PATH = "data/ice_data.csv"
MODEL_PATH = "models/ice_model.pkl"
FEATURES_PATH = "models/features.pkl"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"{DATA_PATH} OOPS")
df = pd.read_csv(DATA_PATH)
print(df.shape)
df =df.dropna()
df = df[df["humidity"] <= 100]
df =df[df["road_moisture"] >= 0]
def add_features(data):
    dew_point = data["temperature"] - ((100 - data["humidity"])/5)
    freezing_index = np.maximum(0,0 - data["temperature"])* data["road_moisture"]
    return data
df = add_features(df)
FEATURES = ["temperature","humidity","dew_point","road_moisture","freezing_index"]
TARGET = "ice_label"
X = df[FEATURES]
y = df[TARGET]
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=42)
print(len(X_train))
print(len(X_test))
model = RandomForestClassifier(n_estimators=200,max_depth=10,min_samples_split=5,random_state=42)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_pred,y_test)
print(classification_report(y_pred,y_test))
for FEATURES,importance in zip(FEATURES,model.feature_importances_):
    print(f"{features}:{importance:.3f}")

os.makedirs("models",exist_ok=True)
joblib.dump("model",MODEL_PATH)
joblib.dump("features",FEATURES_PATH)
sample = {"temperature":-1.5,"humidity":85,"road_moisture":0.8}
sample["freezing_index"] = max(0,0 - sample["temperature"])*sample["road_moisture"]
sample["dew_point"] = sample["temperature"] - ((100 - sample["humidity"])/5)
sample_df = pd.DataFrame([sample])[FEATURES]
probability = model.predict_proba(sample_df)[0][1]
