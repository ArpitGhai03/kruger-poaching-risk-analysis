# 🐘 Kruger National Park - Poaching Risk Prediction

**Advanced Geospatial ML Pipeline for Wildlife Conservation**

A comprehensive two-phase project combining geospatial data generation with deep learning models to predict and identify high-risk poaching zones in Kruger National Park.

---

## 📌 Project Overview

This unified repository integrates both **data pipeline** and **ML modeling**:

### Phase 1: Data Generation (Geospatial Processing)
- Grid-based spatial segmentation of Kruger Park (1000+ cells at 2km × 2km)
- Satellite imagery extraction from Google Earth Engine (Sentinel-2)
- Feature engineering: NDVI, water proximity, elevation, burned areas
- Risk scoring using environmental indicators
- Dataset: 5,095 grid cells with satellite patches & risk labels

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

## 🛰️ Dataset

### Files
- **data.csv**: Main dataset (5,095 grid cells)
  - Grid location (lat/lon)
  - Environmental features: NDVI, water %, elevation, burned area
  - Risk labels: categorical (Low/Medium/High)
  - Image paths pointing to satellite patches

### Satellite Images
- **grid_patches_final_png/**: 128×128 pixel patches from Sentinel-2
- Organized by grid ID (grid_0.png, grid_1.png, ... grid_5592.png)
- RGB composite from Sentinel-2 bands

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

## 🔍 Feature Engineering

| Feature | Interpretation |
|---------|-----------------|
| **NDVI** | Vegetation density (lower = easier access) |
| **Water %** | Wildlife concentration zones |
| **Distance to Water** | Poacher accessibility patterns |
| **Elevation** | Terrain difficulty & accessibility |
| **Burned Area** | Recent disturbance & access corridors |

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
