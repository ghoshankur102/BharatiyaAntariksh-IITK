# 🌆 Urban Cooling Priority Framework for Delhi
### *AI-driven Heat Mitigation with Social Equity Overlay*

![Python](https://img.shields.io/badge/python-3.10+-blue?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-3.0+-orange?style=for-the-badge&logo=xgboost)
![Geopandas](https://img.shields.io/badge/Geopandas-0.14+-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [The Problem & Our Innovation](#-the-problem--our-innovation)
- [Methodology](#-methodology)
- [Key Results](#-key-results)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Outputs](#-outputs)
- [Future Work: GSM-UTCI Integration](#-future-work-gsm-utci-integration)
- [Scalability & Transferability](#-scalability--transferability)
- [Team](#-team)
- [License](#-license)

---

## 📝 Overview

Urban heat islands (UHIs) disproportionately affect low-income and vulnerable populations who lack adaptive capacity (air conditioning, green spaces, etc.). This project delivers a **spatially explicit decision-support framework** for Delhi that:

1. **Predicts** Land Surface Temperature (LST) using an XGBoost model trained on physical drivers (NDVI, NDBI, Albedo, Population).
2. **Estimates UTCI** (Universal Thermal Climate Index) – a human‑perceived heat stress metric that accounts for humidity, wind, and radiation.
3. **Derives** cooling strategies (Cool Roofs, Blue-Green Infrastructure, Tree Corridors) from the predicted UTCI.
4. **Overlays** the Heat Vulnerability Index (HVI) to prioritize interventions based on **emergency** (high heat + high vulnerability) and **equity** (low heat but high poverty).
5. **Generates interactive and static maps** with real satellite imagery and administrative boundaries.

**The Key Innovation:** By separating physical heat prediction from social vulnerability, we ensure that limited municipal resources are directed first to the communities that need them most – aligning with climate justice principles.


---

## 🧠 The Problem & Our Innovation

### The Problem
Most urban heat models stop at predicting temperature. They produce a map of *"where it is hot"* – but that alone doesn't tell policymakers *where to act first* or *who suffers most*.

### Our Innovation
We go one step further. We keep **HVI (Heat Vulnerability Index) entirely separate** from the ML model – and apply it *post-hoc* as a risk multiplier.

| Approach | What It Does | Who Benefits |
| :--- | :--- | :--- |
| **LST Model (XGBoost)** | Predicts physical heat using satellite + OSM data. | Science-driven, universal. |
| **UTCI Estimation** | Converts LST to human‑perceived heat (humidity + wind correction). | More relevant to human health. |
| **HVI Overlay** | Adds Poverty (MPI) and Population Density to create a *Priority Score*. | **Vulnerable communities** (equity focus). |
| **Final Map** | 4 Priority Tiers: IMMEDIATE, HIGH PRIORITY, EQUITY FOCUS, STANDARD. | Policymakers – tells them exactly *where* to spend first. |

---

## 🗺️ Final Output (Preview)

The project produces a **clean, administrative-level zoning map** of Delhi with four priority tiers:

| Priority | Color | Description |
| :--- | :--- | :--- |
| 🚨 **IMMEDIATE** | Dark Red | High Heat + High HVI (Act within 1 month) |
| 🔴 **HIGH PRIORITY** | Red | Moderate Heat + High HVI |
| 🛡️ **EQUITY FOCUS** | Orange | Low Heat + High Poverty (Cooling Centres/Water Access) |
| 🟢 **STANDARD** | Green | Low Heat + Low HVI (Monitor & Maintain) |

> **Visual:** *The script generates `delhi_priority_zoning_map_recalc.png` – a clean, publication‑ready map with a legend on the right side and a fully visible title.*

---

## 🧪 Methodology

### 1. Data & Features
| Data Source | Variables Used |
| :--- | :--- |
| **Landsat 8** (GEE) | LST, NDVI, NDBI, NDWI, ALBEDO |
| **OpenStreetMap (OSM)** | Building Density |
| **WorldPop** | Population Density |
| **NFHS-5** | Multidimensional Poverty Index (MPI) – used for HVI |

### 2. Machine Learning Model
- **Algorithm:** XGBoost Regressor
- **Input:** 9 physical/social features (NDBI, POPULATION, NIGHTLIGHTS, COOLING_EFFECT, etc.)
- **Performance:** R² = **0.6397**, RMSE = **1.34°C**

### 3. UTCI Estimation (Human‑Perceived Heat)

We use an empirical approximation to convert LST to UTCI:
UTCI = LST + 0.15 * (RH - 50) - 0.3 * WindSpeed

Based purely on UTCI (not HVI):
- `UTCI >= 45°C` → **Cool Roofs**
- `UTCI >= 42°C` → **Blue-Green Infrastructure**
- `UTCI >= 38°C` → **Tree Corridors**
- `UTCI >= 35°C` → **Perimeter Shading**
- `UTCI < 35°C` → **Maintain Green**


### 4. HVI Overlay (Post-hoc)
The HVI incorporates *Poverty (MPI)* and *Population Density* to create **urgency modifiers**:
- **High HVI + High Heat** → 🚨 **IMMEDIATE**
- **High HVI + Low Heat** → 🛡️ **EQUITY FOCUS** (e.g., North Delhi)

---

## 📊 Key Results

### Feature Importance
The XGBoost model ranks the top drivers of heat in Delhi:

| Feature | Importance |
| :--- | :--- |
| **NDBI** (Built-up) | ~24% |
| **POPULATION** | ~15% |
| **VEG_QUALITY** | ~13% |
| **NIGHTLIGHTS** | ~11% |

### District-Level Recommendations
Based on the final priority map:
- **West & Central Delhi**: 🚨 **IMMEDIATE** – Deploy Cool Roofs.
- **South West Delhi**: 🔴 **HIGH PRIORITY** – Plant Green Corridors.
- **North Delhi**: 🛡️ **EQUITY FOCUS** – Establish Cooling Centres and improve Water Access.
- **South Delhi**: 🟢 **STANDARD** – Maintain existing Ridge forest cover.

---

## 📂 Repository Structure
urban-heat-mitigation-delhi/
│
├── README.md # You are here
├── LICENSE # MIT License
├── .gitignore # Excludes large files & cache
├── requirements.txt # Python dependencies
│
├── data/
│ ├── raw/ # Original GeoTIFFs & GeoJSON
│ └── processed/ # Cleaned CSVs (predictions, HVI)
│
├── src/ # Python source code
│ ├── hackathon_submission_fixed.py # Main end-to-end pipeline
│ └── delhi_priority_zoning_map_recalc.py # Zoning map generator
│
├── models/ # Trained model files
│ └── urban_heat_model_final.pkl
│
└── outputs/
└── figures/ # All generated visualizations
└── delhi_priority_zoning_map_recalc.png

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+
- Git (optional)


## 🚀 Usage
Option 1: Run the Full Pipeline (Predict + Map)
This script loads the trained XGBoost model, predicts LST for all pixels, assigns cooling strategies, overlays HVI, and generates the final priority map.

## 📊 Outputs
File	                                Description
lst_predictions_with_priority.csv	    Pixel-level predictions, strategies, and priority labels.
delhi_priority_zoning_map_recalc.png	Final clean zoning map with 4 priority tiers.


## 🔮 Future Work: GSM-UTCI Integration
### We have identified the state‑of‑the‑art GSM-UTCI framework (Yi et al., 2025, arXiv:2507.23000v1) as a next‑generation upgrade.

Feature	       Current Implementation	           GSM-UTCI Upgrade
Heat Metric	   LST (estimated UTCI empirically)	    Direct UTCI prediction (human‑perceived)
Resolution	   ~30m (Landsat)	                    1‑meter (hyperlocal)
Accuracy	   R² = 0.64 (LST)	                    R² = 0.915, MAE = 0.41°C
Scenario 
Simulation	   Rule‑based	                        Simulate greening scenarios 
Computational 
Speed	       Fast	                                <5 minutes for an entire city



## 🌍 Scalability & Transferability
### Applying the Framework to Any City

Our methodology is designed to be **city-agnostic**. While this project focuses on Delhi, the pipeline can be applied to any urban area worldwide with minimal adjustments:

| Component | Data Required | Global Availability |
| :--- | :--- | :--- |
| **LST Prediction** | Landsat 8/9 imagery (NDVI, NDBI, ALBEDO) | ✅ Free & global (USGS) |
| **UTCI Estimation** | LST + humidity + wind (empirical) or GSM-UTCI | ✅ ERA5 / NSRDB (global) |
| **HVI Proxy** | Population density + nighttime lights (if no poverty data) | ✅ WorldPop + NASA (global) |
| **Cooling Strategies** | Urban morphology + land cover | ✅ OSM (global) |

### Steps to Apply to a New City

1. **Collect satellite data** – Download Landsat 8/9 for the city.
2. **Extract features** – Compute NDVI, NDBI, ALBEDO, and population density.
3. **Predict LST** – Use our pre‑trained XGBoost model (transfer learning) or retrain it on local data.
4. **Estimate UTCI** – Use our empirical formula, or integrate GSM-UTCI (future work).
5. **Overlay local HVI** – Use census poverty data, or compute a proxy from nighttime lights + population.
6. **Generate maps** – Run the same scripts to produce interactive and static maps.

### Example: CNCR Cities

We have already prepared population projections for the **Central NCR (CNCR) cities** (Ghaziabad, Noida, Gurugram, Faridabad, Bahadurgarh, Sonipat). By applying this framework to these rapidly urbanising satellite cities, planners can proactively mitigate heat stress before these areas reach Delhi-level density.

### Global Potential

With the upcoming integration of **GSM-UTCI** (a multimodal deep learning model trained on SOLWEIG-derived UTCI maps), our framework will be able to:
- Predict **human‑perceived heat stress** at **1‑meter resolution** for any city.
- **Simulate greening scenarios** (e.g., "what if we convert 10% of impervious surfaces to trees?").
- Run city‑wide scenarios in **under 5 minutes** – enabling rapid, iterative planning.

**This is a blueprint for climate‑responsive urban planning, not just a Delhi case study.**

## 👥 Team
Team Name: Oasis

Members: Ankur Ghosh, Shanya Singh, Aishik Mukherjee, Bharat Gupta

Institute: IIT Kanpur

Contact: ghoshankur102@gmail.com

## 📜 License
This project is open-source and available under the MIT License.


## 🙏 Acknowledgements
Data Sources: USGS (Landsat), OSM, NFHS-5, WorldPop.

Inspiration: Climate justice frameworks and Delhi Master Plan 2041.





