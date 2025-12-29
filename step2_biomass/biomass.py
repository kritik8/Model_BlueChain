import joblib
import numpy as np

model = joblib.load("models/biomass_model.pkl")

def predict_biomass(mean_ndvi, area_ha):
    density = model.predict(np.array([[mean_ndvi]]))[0]
    biomass = density * area_ha
    return biomass
