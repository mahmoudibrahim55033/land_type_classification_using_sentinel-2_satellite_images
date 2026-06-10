# 🛰️ Land Type Classification using Sentinel-2 Satellite Images

**A Deep Learning Approach for Multi-Spectral Land Cover Classification**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)

**Live Demo**: [🌍 land-type-classification.vercel.app](https://land-type-classification.vercel.app/)

---

## 📌 Project Overview

This project classifies **10 land cover types** using **Sentinel-2 satellite imagery** (all 13 spectral bands). It explores multiple CNN architectures (AlexNet, ResNet, EfficientNet, and GoogLeNet), performs hyperparameter tuning, applies PCA for dimensionality reduction, and includes rich spectral visualizations.

---

## 🗂️ Repository Structure

| Folder / File | Description |
|---|---|
| **`EuroSATallBands/`** | Raw dataset organized by 10 land cover classes (Sentinel-2 patches with 13 bands). |
| **`Model Selection/`** | Core training and experimentation code. |
| **`Interface/`** | Web-based inference interface. |
| **`models/`** | Trained model weights and artifacts. |
| **`results/`** | Training logs, accuracy plots, confusion matrices, and reports. |
| **`visualizations/`** | EDA, sample images, and spectral analysis. |
| **`WorkFlow.pdf`** | Complete project workflow diagram. |

### Detailed File Descriptions

- **`WorkFlow.pdf`** — High-level project pipeline and methodology overview.
- **`Model Selection/`** — Contains `dataset.py`, `models.py`, training scripts (`train_*.py`), `hyperparameter_tuning.py`, and PCA experiments.
- **`Interface/`** — Contains `Land Type Classification.html`, `inference_api.py`, and supporting files for real-time prediction.
- **`visualizations/`** — Jupyter notebook (`Source Code.ipynb`), plotting functions, RGB/false-color samples, NDVI/NDWI/NDBI maps, and spectral signatures.

---

## 🚀 Quick Start

### Requirements

```bash
pip install torch torchvision torchaudio scikit-learn matplotlib seaborn pandas numpy rasterio joblib fastapi uvicorn
```

### Usage

1. **Explore data** → `visualizations/Source Code.ipynb`
2. **Train models** → Run scripts in `Model Selection/`
3. **Run inference locally** → Use files in `Interface/`

---

## 📊 Results

| Model | Val Accuracy | Notes |
|---|---|---|
| **AlexNet** | **91.80%** ⭐ Best | LR 0.0001 · Batch 32 · 8 Epochs · RMSprop |
| GoogLeNet | — | Inception multi-scale modules |
| ResNet-50 | — | 50-layer residual network |
| EfficientNet | — | Compound scaling |

### AlexNet Training Log (Best Trial)

| Epoch | Val Accuracy |
|---|---|
| 1 / 8 | 83.85% |
| 2 / 8 | 89.78% |
| 3 / 8 | 89.90% |
| 4 / 8 | 85.33% |
| 5 / 8 | 89.97% |
| 6 / 8 | 91.35% |
| 7 / 8 | 91.60% |
| **8 / 8** | **91.80% ✅** |

Best configuration: `lr=0.0001`, `batch_size=32`, `epochs=8`, `optimizer=RMSprop`, `momentum=0.9`

Full training curves, confusion matrices, and classification reports are available in the `results/` folder.

---

## 🧪 Methodology

### 1 · Spectral Normalisation
All 13 bands are normalised per-channel using min-max scaling to `[0, 1]`. Statistics are computed on the training split only and applied to validation and test sets to prevent data leakage.

### 2 · Dimensionality Reduction (PCA)
Principal Component Analysis reduces the 13 correlated spectral bands to **8 principal components**, compressing the input while retaining dominant spectral variance and reducing overfitting risk.

### 3 · Model Selection
Four pre-trained CNNs were benchmarked under identical conditions (same optimiser, schedule, and augmentation policy), all fine-tuned from ImageNet weights with the final head replaced for 10 EuroSAT classes.

### 4 · Hyperparameter Tuning
Grid search over learning rate, batch size, optimizer, and momentum — logged across 20 trials to find the best configuration.

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| Deep Learning | PyTorch, torchvision |
| Data Processing | NumPy, scikit-learn (PCA), Rasterio |
| Visualization | Matplotlib, Seaborn |
| API Backend | FastAPI, Uvicorn |
| Frontend | HTML, CSS, JavaScript |
| Environment | CUDA GPU-accelerated training |

---

## 📚 Dataset

**EuroSAT (Sentinel-2)** — 20,000 labelled 64×64 GeoTIFF patches across 10 classes, perfectly balanced at 2,000 images per class. All 13 multispectral bands (443–2190 nm) are preserved in `.tif` format.

| Class | Label |
|---|---|
| 🌾 | AnnualCrop |
| 🌲 | Forest |
| 🌿 | HerbaceousVegetation |
| 🛣️ | Highway |
| 🏭 | Industrial |
| 🌊 | Pasture |
| 🏘️ | PermanentCrop |
| 🏙️ | Residential |
| 💧 | River |
| 🏞️ | SeaLake |

> Dataset split: **70% train · 15% validation · 15% test**

---

## 🌐 Live Inference API

The project ships with a FastAPI backend (`inference_api.py`) for real-time `.tif` classification.

```bash
# Start the API server
uvicorn inference_api:app --reload
```

Then open `Interface/Land Type Classification.html` in your browser, set the endpoint to `http://127.0.0.1:8000`, upload a `.tif` patch, and click **Classify Image**.

**Endpoint:** `POST /predict` — accepts a `.tif` file, returns predicted class, confidence score, and full probability distribution.

---

## 🎯 Future Improvements

- [ ] Vision Transformer / Swin Transformer models
- [ ] Attention maps (Grad-CAM visualizations)
- [ ] Cloud masking and preprocessing pipeline
- [ ] Full production deployment (FastAPI on Render + frontend on Vercel)

---

## 👩‍💻 Author

Made with ❤️ for Remote Sensing & Deep Learning

**Sohaila Mostafa** — [GitHub @SohailaMMostafa](https://github.com/SohailaMMostafa)

⭐ Star the repo if you found it useful!
