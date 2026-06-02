# 🐘 Kruger National Park - Poaching Risk Prediction

**Advanced Geospatial ML Pipeline for Wildlife Conservation**

A comprehensive two-phase project combining geospatial data generation with deep learning models to predict and identify high-risk poaching zones in Kruger National Park.

---

## 📌 Project Overview

This unified repository integrates both **data pipeline** and **ML modeling**:

### Phase 1: Data Generation (Geospatial Processing)
- **Boundary Data**: WDPA (World Database Protected Areas) Shapefile for Kruger boundary
- **Grid Creation**: Regular 0.018° × 0.018° (≈2km × 2km) grid clipped to park boundary = ~5,095 cells
- **Feature Extraction via GEE**: 
  - Sentinel-2 2024 data (median composite, <20% cloud cover)
  - 6 environmental features: NDVI, Water %, dist_to_water, burned_area, elevation, landcover
  - Batch downloads + merged into unified dataset
- **Risk Scoring**: Composite algorithm combining 6 features with weighted rules (see Feature Engineering)
- **Image Processing**: Sentinel-2 GeoTIFFs clipped per grid → resized to 128×128 pixels → PNG format
- **Dataset**: 5,095 grid cells with normalized features, risk labels, and satellite image patches

### Phase 2: ML & Prediction (Local - This Repo)
- Load and explore geospatial dataset
- Train multiple deep learning models (CNN, ResNet, EfficientNet)
- Evaluate performance with metrics & confusion matrices
- Test predictions on satellite imagery
- Generate visualizations & dashboards

---

## 📂 Project Structure

```
kruger-poaching-risk-analysis/
├── notebooks/
│   ├── 01_data_generation.ipynb          # Phase 1: Data pipeline (reference)
│   └── 02_ml_prediction.ipynb            # Phase 2: ML models & training
├── src/
│   ├── test_models.py                    # Test trained models on random samples
│   └── models.py                         # Model architectures (future)
├── data/
│   ├── raw/                              # Raw data (if needed)
│   └── processed/
│       ├── data.csv                      # Final dataset (5,095 rows)
│       └── grid_patches_final_png/       # Satellite image patches (5,600+ images)
├── models/                               # Trained model checkpoints
├── dashboard/
│   └── wildlife_dashboard.twb            # Tableau visualization
├── README.md                             # This file
├── requirements.txt                      # Python dependencies
└── .gitignore                           # Exclude large/local files
```

---

## 🛰️ Dataset Details

### Data Generation Pipeline
1. **Boundary**: WDPA Shapefile → Kruger Park polygon (WGS84)
2. **Grid**: Regular grid overlay clipped to boundary → 5,095 cells
3. **GEE Processing**: Sentinel-2 2024 imagery → extract 6 features per grid → batch downloads
4. **Image Extraction**: GeoTIFFs masked per grid cell → clipped patches
5. **Standardization**: Resized to 128×128 pixels, converted TIF→PNG
6. **Feature Scaling**: StandardScaler normalization for ML models
7. **Labeling**: Composite risk scoring algorithm → 3 categories

### Files
- **data.csv**: Main dataset (5,095 rows × 15 columns)
  - Grid identifiers: id, Grid_ID, Lat, Lon, centroid_x, centroid_y
  - Geometry: WKT polygon for each grid cell
  - Environmental features (normalized): NDVI, Water, dist_to_water, burned_area, elevation, landcover
  - Target: poaching_risk (Low/Medium/High)
  - Image reference: image_path (relative path to PNG patch)

### Satellite Images
- **grid_patches_final_png/**: 128×128 pixel RGB patches from Sentinel-2 (bands 4,3,2)
- Derived from GeoTIFFs clipped per grid cell and standardized to fixed size
- Organized by grid ID (grid_0.png, grid_1.png, ... grid_5337.png)
- ~5,600 valid image patches corresponding to grid cells with complete data

### Class Distribution
- **Medium Risk**: ~45% (majority)
- **Low Risk**: ~27%
- **High Risk**: ~28%

---

## 🤖 Models Implemented

1. **BaselineCNN**: Custom 4-layer CNN with dropout
2. **ResNetClassifier**: ResNet18/34 with transfer learning
3. **EfficientNetClassifier**: EfficientNet-B0/B1 for optimized performance
4. **CustomCNN**: Enhanced CNN with batch normalization

All models trained for 3-class classification (Low/Medium/High risk).

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Explore Data
```bash
jupyter notebook notebooks/02_ml_prediction.ipynb
```

### 3. Train Models
Run the notebook cells sequentially to:
- Load and visualize dataset
- Balance classes with WeightedRandomSampler
- Train all models
- Evaluate performance

### 4. Test Models
```bash
python src/test_models.py
```
Tests all trained models on random samples from the dataset.

---

## 📊 Key Insights

1. **Spatial Clustering**: Risk concentrates near water bodies and lower-elevation zones
2. **Accessibility**: Vegetation density (NDVI) inversely correlates with poaching risk
3. **Seasonal Patterns**: Burned areas show elevated risk (reduced vegetation barriers)
4. **Multi-scale Resolution**: 2km × 2km grid enables targeted conservation efforts

---

## 🔍 Feature Engineering & Risk Scoring

### Environmental Features (normalized via StandardScaler)
| Feature | Source | Interpretation |
|---------|--------|------------------|
| **NDVI** | Sentinel-2 B8/B4 | Vegetation density (lower = easier access) |
| **Water %** | JRC Water dataset | Water body coverage (higher = wildlife concentration) |
| **Distance to Water** | GEE calculation | Proximity to water sources (closer = easier poaching) |
| **Elevation** | SRTM DEM | Terrain height (lower = easier access) |
| **Burned Area** | MODIS/Sentinel-2 | Recent fire damage % (higher = reduced barriers) |
| **Landcover** | GEE classification | Terrain type (grassland/trees/etc.) |

### Risk Scoring Methodology
Composite algorithm combining features with weighted rules:
```
Score = 0
if NDVI < 0.3: Score += 2 (very low vegetation)
if NDVI < 0.4: Score += 1 (low vegetation)
if Water > 50%: Score += 2 (high water presence)
if Water > 20%: Score += 1 (moderate water)
if dist_to_water < 500m: Score += 2 (very close to water)
if dist_to_water < 1500m: Score += 1 (close to water)
if burned_area > 50%: Score += 2 (heavily burned)
if burned_area > 20%: Score += 1 (partially burned)
if elevation < 500m: Score += 1 (lowland)

Risk Category:
if Score >= 6: "High" risk
if Score >= 3: "Medium" risk
else: "Low" risk
```

Resulting class distribution: Medium ~45%, Low ~27%, High ~28%

---

## 📈 Performance Metrics

Models are evaluated using:
- **Accuracy**: Overall classification rate
- **Precision & Recall**: Per-class performance
- **F1-Score**: Balanced metric for imbalanced data
- **Confusion Matrix**: Misclassification patterns
- **ROC-AUC**: Discrimination ability

---

## 🛠️ Tools & Technologies

| Category | Tools |
|----------|-------|
| **Data Processing** | Pandas, NumPy, OpenCV |
| **Deep Learning** | PyTorch, TorchVision |
| **ML/Evaluation** | Scikit-learn |
| **Visualization** | Matplotlib, Seaborn, Tableau |
| **Geospatial** | GeoPandas, Rasterio, Shapely |
| **Environment** | Python 3.9+, Jupyter, Git |

---

## 📞 Contact & Collaboration

**GitHub**: [@ArpitGhai03](https://github.com/ArpitGhai03)

For questions, collaborations, or contributions to this wildlife conservation project, please reach out!

---

## 📚 References

- [Google Earth Engine](https://earthengine.google.com/)
- [Sentinel-2 Data](https://sentinel.esa.int/web/sentinel/missions/sentinel-2)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [GeoPandas Documentation](https://geopandas.org/)

---

## 📄 License

This project is open-source and available for educational and conservation purposes.

---

**Last Updated**: June 2026  
**Status**: Active Development
