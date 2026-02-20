# P8-Image-Processing-for-Autonomous-Vehicle-Embedded-System

Development of embedded computer vision systems with the objective of gaining experience in image segmentation.

## 📋 Project Documentation

This project includes the development of an image segmentation system for autonomous vehicles with:
- Semantic segmentation model (8 Cityscapes categories)
- Prediction API (FastAPI)
- Web demonstration application
- Complete technical documentation

## 🚀 Setup

### Virtual Environment Setup

This project uses [UV](https://github.com/astral-sh/uv) for fast Python package management.

1. **Install UV (if not already installed):**
   ```bash
   pip install uv
   ```
   Or follow the [official installation guide](https://github.com/astral-sh/uv#installation).

2. **Create a virtual environment:**
   ```bash
   uv venv
   ```

3. **Activate the virtual environment:**
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```

4. **Install dependencies:**
   ```bash
   uv pip install -r requirements.txt
   ```

5. **Deactivate the virtual environment (when done):**
   ```bash
   deactivate
   ```

## 🌐 Running the API

The inference API loads the segmentation model from Azure at startup and serves `/health` and `/predict`.

**Run from the project root:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Environment variables** (copy `.env.example` to `.env` and set):

| Variable | Description | Default |
|----------|-------------|---------|
| `AZURE_MODEL_BLOB_NAME` | Blob path of the model in Azure | `model/best_model.keras` |
| `AZURE_CONTAINER_NAME` | Azure Blob Storage container name | `training-outputs` |
| `AZURE_STORAGE_CONNECTION_STRING` | Full connection string (recommended) | — |
| `AZURE_STORAGE_ACCOUNT_NAME` | Account name (if not using connection string) | — |
| `AZURE_STORAGE_ACCOUNT_KEY` | Account key (if not using connection string) | — |

- **GET /health** — Returns `{"status": "ok", "model_loaded": true/false}`.
- **POST /predict** — Upload a PNG or JPEG image (max 20 MB); returns mask and colored mask as base64, plus categories.
   
## 📊 Project Steps

The project follows a structured development approach:

1. **Data Exploration** - Analysis of the Cityscapes dataset structure, class distribution, and image characteristics
2. **Model Development** - Implementation of U-Net architecture for semantic segmentation with data augmentation
3. **Training Pipeline** - Model training with callbacks, metrics tracking, and visualization
4. **API Development** - FastAPI-based prediction service for real-time inference
5. **Web Application** - Streamlit-based demonstration interface for interactive predictions
6. **Documentation** - Technical documentation and presentation materials

## 🔍 Data Exploration

The data exploration phase (`notebooks/01-data_exploration.ipynb`) provides a comprehensive analysis of the Cityscapes dataset:

### Dataset Structure
- **Training set**: 2,964 images across 18 cities
- **Test set**: 1,525 images across 6 cities
- **Image dimensions**: Standardized 1024×2048 pixels (2:1 aspect ratio)

### Class Mapping
The exploration notebook implements a mapping from the original 34 Cityscapes classes to 8 main categories:
- **Void** (0): Unlabeled, ego vehicle, borders, and ignored regions
- **Flat** (1): Road, sidewalk, parking, rail track
- **Construction** (2): Building, wall, fence, guard rail, bridge, tunnel
- **Object** (3): Pole, traffic sign, traffic light
- **Nature** (4): Vegetation, terrain
- **Sky** (5): Sky regions
- **Human** (6): Person, rider
- **Vehicle** (7): Car, truck, bus, motorcycle, bicycle, and other vehicles

### Key Findings
- **Class imbalance**: Significant imbalance detected (52.5× ratio between most and least frequent classes)
  - Flat surfaces: ~39.3% of pixels
  - Construction: ~22.9% of pixels
  - Human class: ~0.75% of pixels (most underrepresented)
- **Recommendations**: Use class weights or focal loss during training to handle imbalance

### Utility Functions
The notebook creates reusable utility functions for:
- Loading Cityscapes images and masks
- Converting 34-class masks to 8-category masks
- Visualizing images with colored segmentation masks
- Path management for dataset files

These functions are exported to `src/utils.py` for use in the training pipeline.


### 📖 Resources

- Cityscapes Dataset: https://www.cityscapes-dataset.com/
- Keras Documentation: https://keras.io/
- FastAPI Documentation: https://fastapi.tiangolo.com/
- Streamlit Documentation: https://docs.streamlit.io/
