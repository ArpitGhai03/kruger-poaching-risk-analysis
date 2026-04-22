# 🐘 Kruger National Park - Poaching Risk Assessment

**Advanced Geospatial Analysis & Machine Learning Dataset for Wildlife Conservation**

## 📌 Project Overview

This project develops a comprehensive **multimodal geospatial dataset** to predict and identify high-risk poaching zones in Kruger National Park. By combining satellite imagery from Google Earth Engine with environmental feature engineering, this work supports conservation efforts through data-driven risk assessment.

### Key Achievement:
Developed a **custom geospatial dataset from scratch** using satellite data (Google Earth Engine) due to lack of publicly available data tailored for poaching risk assessment in African protected areas.

---

## 🎯 Objectives

- ✅ Design and implement grid-based spatial segmentation of Kruger National Park
- ✅ Process and extract satellite imagery for each geographic region
- ✅ Engineer environmental features (vegetation density, water proximity, elevation)
- ✅ Create risk scoring models combining multiple data sources
- ✅ Visualize spatial patterns and risk distribution through dashboards
- ✅ Build foundation for machine learning-based poaching risk prediction
- ✅ Support wildlife conservation decision-making with actionable insights

---

## 🛰️ Data Sources

| Source | Data | Resolution |
|--------|------|-----------|
| **Google Earth Engine** | Sentinel-2 Satellite Imagery | 10m bands (RGB, NIR) |
| **WDPA Database** | Kruger Park Boundary Shapefile | Polygon geometry |
| **Derived** | NDVI, Water Classification, Elevation | 30m |

---

## 📊 Methodology

### 1. **Spatial Segmentation**
- Created regular 0.018° grid overlay clipped to Kruger boundary
- Generated Grid_IDs and computed lat/lon centroids
- Enabled systematic, region-based analysis

### 2. **Satellite Data Processing**
- Retrieved Sentinel-2 median composites (2024, <20% cloud cover)
- Extracted multi-band imagery for RGB and vegetation analysis
- Generated 128×128 pixel patches for each grid cell

### 3. **Feature Engineering**
Engineered key environmental indicators:

| Feature | Source | Interpretation |
|---------|--------|-----------------|
| **NDVI** | Sentinel-2 B8/B4 | Vegetation density (proxy for accessibility) |
| **Water Presence** | Water classification layer | Wildlife concentration zones |
| **Distance to Water** | Proximity analysis | Poacher accessibility patterns |
| **Elevation** | DEM data | Terrain difficulty & accessibility |
| **Burned Area** | Fire scar detection | Recent disturbance & access corridors |
| **Landcover Type** | ESA WorldCover classification | Habitat & accessibility characteristics |

### 4. **Risk Scoring & Classification**
- Combined environmental factors using weighted scoring model
- Generated risk scores normalized to [0, 1] range
- Classified regions into **Low, Medium, High** risk categories using quantile binning

### 5. **Feature Normalization**
- Applied StandardScaler to all numerical features
- Prepared dataset for machine learning models

---

## 🗺️ Key Features Engineered

### Vegetation & Accessibility
- **NDVI (Normalized Difference Vegetation Index)**: Lower vegetation → easier access → higher risk
- **Landcover Classification**: Open areas and grasslands are more accessible to poachers

### Water & Wildlife Patterns
- **Water Presence**: Concentrates wildlife populations, attracts poachers
- **Proximity to Water**: Closer distances indicate higher poaching likelihood

### Terrain & Natural Barriers
- **Elevation**: Lowlands are easier to traverse and access
- **Burned Areas**: Recent burns remove vegetation barriers and create access corridors

### Composite Risk Score
```
Risk Score = (1 - NDVI) + Water% + dist_to_water/5000 + burned_area + (500 - elevation)/500
```

---

## 📈 Dataset Composition

- **Total Grid Cells**: 1,000+ regions
- **Feature Dimensions**: 6 engineered features + satellite imagery
- **Image Patches**: 128×128 pixel Sentinel-2 composites (RGB)
- **Risk Labels**: Categorical (Low/Medium/High) + continuous scores
- **Multimodal**: Combines tabular geospatial features with satellite imagery

---

## 📊 Tableau Dashboard

Interactive dashboard visualizing:
🗺️ Spatial risk distribution across Kruger park
📊 Risk category proportions (Low/Medium/High)
📈 NDVI and elevation patterns by risk category
🎯 Key performance indicators and metrics
📍 Risk count and grid-level segmentation

**Location**: `dashboard/` folder (to be added)

---

## 🤖 Machine Learning

### Collaboration
This project includes **collaborative machine learning model development** for predicting poaching risk.

---

## 🔍 Key Insights

1. **Spatial Clustering**: Poaching risk concentrates near water bodies and lower-elevation zones
2. **Accessibility Factor**: Vegetation density is a strong inverse predictor of risk
3. **Seasonal Patterns**: Burned areas show elevated risk due to reduced vegetation barriers
4. **Multi-scale Risk**: Risk varies significantly at 2km × 2km resolution, enabling targeted conservation efforts

---

## 🛠️ Tools & Technologies

| Category | Tools |
|----------|-------|
| **Geospatial** | GeoPandas, Rasterio, Shapely, GDAL |
| **Satellite Data** | Google Earth Engine, Sentinel-2 |
| **Data Processing** | Pandas, NumPy, OpenCV |
| **Visualization** | Matplotlib, Tableau |
| **Programming** | Python 3.9+ |
| **Version Control** | Git, GitHub |

---

## 📂 Repository Structure

```
kruger-poaching-risk-analysis/
├── notebooks/
│   └── wildlife_analysis.ipynb          # Main analysis & feature engineering
├── data/
│   ├── raw/
│   │   └── WDPA_WDOECM_Dec2025_Public_873_shp-polygons.shp
│   └── processed/
│       └── improved_kruger_data.csv     # Final dataset (upload here)
├── images/
│   └── (satellite patches & visualizations)
├── dashboard/
│   └── (Tableau workbook to be added)
├── README.md                             # This file
├── requirements.txt                      # Python dependencies
└── .gitignore
```

---

## 🚀 Project Status

✅ **Completed**:
- Grid creation and spatial segmentation
- Satellite imagery extraction and processing
- Feature engineering (NDVI, water, elevation, etc.)
- Risk scoring model development
- Dataset preparation (tabular + imagery)
- Tableau dashboard design
- Collaborative ML model development
---

## 📋 Dataset Documentation

### Features
- `Grid_ID`: Unique grid cell identifier
- `Lat`, `Lon`: Centroid coordinates (WGS84)
- `NDVI`: Normalized Difference Vegetation Index (-1 to 1)
- `Water`: Water presence percentage (0-100)
- `dist_to_water`: Distance to nearest water body (meters)
- `elevation`: Mean elevation (meters)
- `burned_area`: Percentage of burned area (0-1)
- `landcover`: ESA World Cover classification
- `risk_score`: Composite risk score (0-1)
- `poaching_risk`: Risk category (Low/Medium/High)
- `image_path`: Path to 128×128 satellite patch

### Labels
- **Low Risk**: score < 33rd percentile
- **Medium Risk**: score 33rd-66th percentile
- **High Risk**: score > 66th percentile

---

## 📞 Contact & Collaboration

For questions, collaborations, or contributions to this wildlife conservation project, please reach out!

**GitHub**: [@ArpitGhai03](https://github.com/ArpitGhai03)

---

## 📚 References

- [Google Earth Engine](https://earthengine.google.com/)
- [Sentinel-2 Data](https://sentinel.esa.int/web/sentinel/missions/sentinel-2)
- [WDPA - World Database on Protected Areas](https://www.protectedplanet.net/)
- [GeoPandas Documentation](https://geopandas.org/)

---

## 📄 License

This project is open-source and available for educational and conservation purposes.

---

**Last Updated**: April 2026  
**Status**: Active Development
