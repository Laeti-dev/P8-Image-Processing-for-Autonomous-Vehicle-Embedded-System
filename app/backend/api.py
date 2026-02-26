"""
FastAPI application for semantic segmentation inference.

Loads the model from Azure at startup and exposes /health and /predict endpoints.

Install the project first: uv pip install -e .  (or pip install -e .)
Then from project root: uvicorn app.main:app --host 0.0.0.0 --port 8000

Environment variables (see .env.example):
    AZURE_MODEL_BLOB_NAME    Blob path of the model (default: model/best_model.keras)
    AZURE_CONTAINER_NAME    Azure container name (default: training-outputs)
    AZURE_STORAGE_CONNECTION_STRING  Or use AZURE_STORAGE_ACCOUNT_NAME + AZURE_STORAGE_ACCOUNT_KEY
"""

import base64
import os
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from dotenv import load_dotenv

# Load .env from project root (parent of app/)
_project_root = Path(__file__).resolve().parent.parent
_env_path = _project_root / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

from fastapi import FastAPI, File, HTTPException, Request, UploadFile

from src.predictor import SegmentationPredictor
from src.utils import CATEGORY_NAMES


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load model from Azure. Shutdown: cleanup if needed."""
    azure_blob_name = os.environ.get("AZURE_MODEL_BLOB_NAME", "model/best_model.keras")
    azure_container_name = os.environ.get("AZURE_CONTAINER_NAME", "training-outputs")

    predictor = SegmentationPredictor(
        azure_blob_name=azure_blob_name,
        azure_container_name=azure_container_name,
    )
    app.state.predictor = predictor
    model_loaded = False
    try:
        predictor.load_model()
        model_loaded = True
    except Exception:
        # Fail gracefully: API can still run and return 503 on /predict
        model_loaded = False
    app.state.model_loaded = model_loaded
    yield
    # Shutdown cleanup (optional)


# Allowed image types for /predict (PNG, JPEG)
ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg"}
# Max upload size: 20 MB
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

app = FastAPI(
    title="Segmentation Inference API",
    description="Semantic segmentation mask prediction from Azure-loaded model",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request) -> dict:
    """Return API status and whether the model is loaded."""
    model_loaded = getattr(request.app.state, "model_loaded", False)
    return {"status": "ok", "model_loaded": model_loaded}


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)) -> dict:
    """
    Run segmentation on an uploaded image.
    Accepts PNG or JPEG. Returns mask and colored mask as base64 PNG, plus categories.
    """
    model_loaded = getattr(request.app.state, "model_loaded", False)
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check Azure configuration and /health.",
        )

    if not file.content_type or file.content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}",
        )

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {MAX_UPLOAD_BYTES // (1024*1024)} MB",
        )

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        predictor: SegmentationPredictor = request.app.state.predictor
        mask = predictor.predict_from_image(content)
        colored_mask = predictor.color_predicted_mask(mask=mask)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}",
        ) from e

    # Encode mask (H, W) and colored mask (H, W, 3 RGB) as PNG base64
    _, mask_buf = cv2.imencode(".png", mask)
    mask_base64 = base64.b64encode(mask_buf.tobytes()).decode("ascii")

    # OpenCV expects BGR for correct PNG colors when decoding; mask_to_colored returns RGB
    colored_bgr = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)
    _, colored_buf = cv2.imencode(".png", colored_bgr)
    colored_mask_base64 = base64.b64encode(colored_buf.tobytes()).decode("ascii")

    categories = [
        {"id": k, "name": v}
        for k, v in sorted(CATEGORY_NAMES.items())
    ]

    return {
        "mask_shape": list(mask.shape),
        "mask_base64": mask_base64,
        "colored_mask_base64": colored_mask_base64,
        "categories": categories,
    }
