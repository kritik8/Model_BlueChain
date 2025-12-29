from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import os

from step1_area.area import UNet
from step1_area.preprocess import preprocess_image
from step2_biomass.biomass import predict_biomass
from step3_growth.growth import predict_growth
from utils.carbon import biomass_to_cc

app = FastAPI(title="BlueChain ML API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models ONCE
area_model = UNet()

# Check if model exists, if not use untrained model
model_path = "models/area_segmentation.pth"
if os.path.exists(model_path):
    try:
        area_model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print(f"Successfully loaded model from {model_path}")
    except RuntimeError as e:
        print(f"Warning: Could not load model weights - {e}")
        print("Using untrained model instead")
else:
    print(f"Warning: {model_path} not found. Using untrained model.")

area_model.eval()


def predict_area(model, image_tensor):
    """Predict vegetation area from image tensor"""
    with torch.no_grad():
        out = model(image_tensor)
        mask = (out > 0.5).float()

    pixel_area = 0.01  # m² per pixel
    veg_pixels = mask.sum().item()
    area_m2 = veg_pixels * pixel_area
    area_ha = area_m2 / 10000
    return area_m2, area_ha


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "BlueChain ML API is running"}


@app.post("/predict/full")
async def full_pipeline(
    image: UploadFile = File(...),
    mean_ndvi: float = 0.6,
    temp: float = 28,
    humidity: float = 70,
    ph: float = 7,
    salinity: float = 1.2
):
    """Full pipeline: area -> biomass -> growth -> carbon credits"""
    try:
        img_tensor = preprocess_image(image.file)
        area_m2, area_ha = predict_area(area_model, img_tensor)

        biomass = predict_biomass(mean_ndvi, area_ha)
        growth = predict_growth(biomass, temp, humidity, ph, salinity)
        credits = biomass_to_cc(growth)

        return {
            "status": "success",
            "area_hectare": round(area_ha, 4),
            "biomass_tons": round(biomass, 2),
            "growth_next_year": round(growth, 2),
            "carbon_credits": credits
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
