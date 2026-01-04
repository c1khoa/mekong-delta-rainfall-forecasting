# Mekong Delta Rainfall Forecasting

## Project Overview
This project focuses on **spatio-temporal rainfall forecasting** in the Mekong Delta, Vietnam, using multi-source remote sensing and meteorological data. The Mekong Delta is a critical agricultural region that is highly vulnerable to climate change, droughts, and floods. Accurate rainfall prediction is therefore essential for water resource management, agriculture planning, and disaster early warning systems.

Instead of directly predicting rainfall from current observations, we propose a **two-stage deep learning framework** that reflects the natural physical process of rainfall formation.

---

## Data Sources
The dataset is collected for the period **01/01/2019 – 31/12/2024** using the **Google Earth Engine (GEE)** API, combining three main data sources:

### Remote Sensing
- **Sentinel-1 (SAR)**  
  - C-band radar, all-weather capability  
  - Resolution: 10 m  
  - Polarizations: VV, VH  
  - Useful for soil moisture, floods, and surface water detection

- **Sentinel-2 (Optical)**  
  - 13 multispectral bands (10–60 m resolution)  
  - Vegetation indices (NDVI, NDWI), surface reflectance, radiation features  

### Meteorological Reanalysis
- **ERA5**  
  - Temperature, dew point, wind, surface pressure, soil moisture  
- **GLDAS**  
  - Land surface variables: evaporation, energy balance, soil temperature  

### Target Variable
- **Rainfall** represented as a single-channel spatial grid `(1, H, W)`

---

## Exploratory Data Analysis (EDA)
- Seasonal rainfall patterns clearly show the **dry season (May–November)** and **rainy season**
- Aerosol Optical Thickness (AOT) peaks in April due to post-harvest straw burning
- Rainfall intensity increases significantly toward the late dry season

---

## Data Preprocessing
Key preprocessing steps include:

- **Handling missing values**
  - Removing features with excessive null values (e.g., Sentinel-2 features before 2022)
- **Dimensionality Reduction**
  - Principal Component Analysis (PCA) is applied to highly correlated features to reduce multicollinearity
- **Normalization**
  - Robust Scaling to mitigate extreme values  
  - Min-Max Scaling to map all features into the range `[0, 1]`
- **Data Splitting (Chronological)**
  - Train: 2019 – 2023  
  - Validation: 01/2024 – 06/2024  
  - Test: 07/2024 – 12/2024  

---

## Methodology: Two-Stage Framework

### Stage 1 – Future Feature Prediction
Predict future meteorological and surface features using **spatio-temporal deep learning models**:

- ConvLSTM
- ConvGRU
- TrajGRU

---

### Stage 2 – Rainfall Map Reconstruction
Map predicted future features to rainfall intensity using encoder–decoder architectures:

- U-Net
- Feature Pyramid Network (FPN)
- U-Net 3+

The output is a high-resolution 2D rainfall map representing spatial precipitation distribution.

---

## Experimental Results

### Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

### Performance Summary

| Stage 1 | Stage 2 | MAE | RMSE |
|------|------|------|------|
| ConvLSTM | U-Net | 2.84 | 3.47 |
| ConvLSTM | FPN | 2.52 | 3.64 |
| ConvLSTM | U-Net 3+ | 2.02 | 2.60 |
| ConvGRU | U-Net | 2.92 | 3.82 |
| ConvGRU | FPN | 2.88 | 4.51 |
| ConvGRU | U-Net 3+ | 2.33 | 2.92 |
| **TrajGRU** | **U-Net** | **1.69** | **2.05** |
| TrajGRU | FPN | 2.66 | 3.63 |
| **TrajGRU** | **U-Net 3+** | **1.59** | **2.06** |

---

## Key Findings
- **TrajGRU** consistently outperforms ConvLSTM and ConvGRU due to its learnable trajectory mechanism
- **U-Net 3+** achieves the best reconstruction accuracy by leveraging full-scale skip connections
- The combination **TrajGRU + U-Net 3+** yields the best overall rainfall prediction performance

---

## Installation and Usage

### Environment Setup

```bash
python -m venv venv
```

Activate the environment:

- Windows:
```bash
venv\Scripts\activate
```

- Linux / macOS:
```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Data Preprocessing

Run the preprocessing pipeline to generate data for prediction:

```bash
python src/preprocessing/main_script.py
```

### Run the Application

Launch the Streamlit application:

```bash
streamlit run app.py
```

The application will be available at:
http://localhost:8501
