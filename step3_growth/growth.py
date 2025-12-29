import joblib
import numpy as np

model = joblib.load("models/growth_model.pkl")

def predict_growth(biomass, temp, humidity, ph, salinity):
    X = np.array([[biomass, temp, humidity, ph, salinity]])
    growth = model.predict(X)[0]
    return max(growth, 0)  # no negative growth
