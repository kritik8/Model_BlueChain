<h1 align="center">BlueChain ML Engine</h1>

<p align="center">
  <b>Neural Carbon Restoration Engine for MRV Systems</b><br/>
  A modular, auditable machine learning pipeline for verified carbon credit estimation
</p>

<p align="center">
  <img src="https://img.shields.io/badge/ML-PyTorch-orange" />
  <img src="https://img.shields.io/badge/API-FastAPI-green" />
  <img src="https://img.shields.io/badge/Domain-Climate%20Tech-blue" />
</p>

---

## Overview

The **BlueChain ML Engine** is a **four-stage, modular machine learning pipeline** designed for **Measurement, Reporting & Verification (MRV)** of land restoration and carbon sequestration projects.

The system transforms **raw satellite imagery and on-ground environmental data** into **auditable carbon credits**, ensuring that every issued credit is backed by measurable, explainable, and verifiable data.

This repository focuses **only on the ML and scientific computation layer** and is designed to integrate seamlessly with:
- Web frontends
- Blockchain minting logic
- Government or ESG reporting systems

---

## ML Pipeline Architecture
- Satellite / Drone Images + Environmental Data
  ↓
- Stage 1: Area Estimation
  ↓
- Stage 2: Biomass Estimation
  ↓
- Stage 3: Growth Prediction
  ↓
- Stage 4: Carbon Credit Calculation


Each stage is **independent, explainable, and deployable as an API**.

---

## Stage 1 — Vegetation Area Estimation (Segmentation)
**Purpose**  
Accurately measure the restored vegetation area using aerial imagery.
**Inputs**
- RGB satellite or drone images
**Method**
- Semantic image segmentation (U-Net)
- Binary classification: vegetation vs non-vegetation
**Outputs**
- Vegetation mask
- Restored land area (m² / hectares)

**Architecture / Output Visualization**
> <img width="1124" height="578" alt="image" src="https://github.com/user-attachments/assets/184d4dc1-67ff-4185-a33b-66e674e4a7e0" />


---

## Stage 2 — Biomass Estimation (NDVI-Based)
**Purpose**  
Estimate above-ground biomass using vegetation health indices.
**Inputs**
- Vegetation mask (Stage 1)
- Red & NIR spectral bands
- Plant type (validated on-ground)
**Method**
- NDVI calculation  
- Scientific biomass conversion equations
**Outputs**
- Estimated biomass (tons)
**NDVI & Biomass Visualization**
> <img width="1105" height="764" alt="image" src="https://github.com/user-attachments/assets/4d4f6489-da18-48be-92b3-97d133d77eb5" />


---

## Stage 3 — One-Year Growth Prediction
**Purpose**  
Predict incremental biomass growth for the next year using climate data.
**Inputs**
- Current biomass
- Temperature
- Humidity
- Soil pH
- Soil salinity
**Method**
- Regression-based ML model (Random Forest / Gradient Boosting)
- Synthetic + rule-guided training for hackathon demo
**Outputs**
- Predicted biomass growth (tons/year)
**Growth Prediction Visualization**
> <img width="1120" height="610" alt="image" src="https://github.com/user-attachments/assets/dad8c2b2-ae4f-43b9-bdd2-2b8c82dccc08" />


---

## Stage 4 — Carbon Credit Calculation (Formula-Based)
**Purpose**  
Convert predicted biomass growth into **verified carbon credits**.
**Formula**
Carbon = Biomass × 0.47
CO₂e = Carbon × 3.67
**Outputs**
- Carbon Credits (tCO₂/year)
<img width="1103" height="862" alt="image" src="https://github.com/user-attachments/assets/456931c8-19b9-4216-848a-aeaaf3b48c48" />

📚 Documentation
Full Technical Documentation

