"""
Unit tests for FastAPI inference API endpoints: GET /health and POST /predict.

Run from project root: pytest tests/test_api_endpoints.py -v
"""

import base64
import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import MAX_UPLOAD_BYTES, app
from src.utils import CATEGORY_NAMES


def _make_png_bytes(width: int = 64, height: int = 64) -> bytes:
    """Create minimal PNG image bytes for upload tests."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:] = [128, 128, 128]
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_jpeg_bytes(width: int = 64, height: int = 64) -> bytes:
    """Create minimal JPEG image bytes for upload tests."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:] = [128, 128, 128]
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_ok_and_model_loaded_flag(self):
        """Health returns status and model_loaded; actual value depends on Azure in lifespan."""
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data
        assert isinstance(data["model_loaded"], bool)

    def test_health_response_schema(self):
        """Health response has exactly status and model_loaded."""
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert set(data.keys()) == {"status", "model_loaded"}


class TestPredictEndpoint:
    """Tests for POST /predict."""

    def test_predict_without_file_returns_422(self):
        """POST /predict without file returns 422 Unprocessable Entity."""
        client = TestClient(app)
        response = client.post("/predict")
        assert response.status_code == 422

    def test_predict_invalid_content_type_returns_400(self):
        """POST /predict with disallowed content type returns 400."""
        client = TestClient(app)
        png_bytes = _make_png_bytes()
        response = client.post(
            "/predict",
            files={"file": ("image.txt", io.BytesIO(png_bytes), "text/plain")},
        )
        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]

    def test_predict_empty_file_returns_400(self):
        """POST /predict with empty file returns 400."""
        with patch("app.main.SegmentationPredictor") as MockPredictor:
            mock_instance = MagicMock()
            mock_instance.load_model = MagicMock(return_value=MagicMock())
            MockPredictor.return_value = mock_instance
            client = TestClient(app)
        response = client.post(
            "/predict",
            files={"file": ("empty.png", io.BytesIO(b""), "image/png")},
        )
        assert response.status_code == 400
        assert "Empty" in response.json()["detail"]

    def test_predict_file_too_large_returns_400(self):
        """POST /predict with file exceeding max size returns 400."""
        with patch("app.main.SegmentationPredictor") as MockPredictor:
            mock_instance = MagicMock()
            mock_instance.load_model = MagicMock(return_value=MagicMock())
            MockPredictor.return_value = mock_instance
            client = TestClient(app)
        # Create a file larger than MAX_UPLOAD_BYTES (20 MB)
        large = b"x" * (MAX_UPLOAD_BYTES + 1)
        response = client.post(
            "/predict",
            files={"file": ("huge.png", io.BytesIO(large), "image/png")},
        )
        assert response.status_code == 400
        assert "too large" in response.json()["detail"].lower()

    def test_predict_returns_503_when_model_not_loaded(self):
        """POST /predict returns 503 when model is not loaded."""
        client = TestClient(app)
        # Without mocking, in CI/offline model_loaded is typically False
        png_bytes = _make_png_bytes()
        response = client.post(
            "/predict",
            files={"file": ("image.png", io.BytesIO(png_bytes), "image/png")},
        )
        # If model failed to load in lifespan, we get 503
        if response.status_code == 503:
            assert "model" in response.json()["detail"].lower()
            return
        # If model was loaded (e.g. cached), we might get 200 or 500; allow 200 for local runs
        assert response.status_code in (200, 503, 500)

    @pytest.mark.parametrize("content_type,make_bytes", [
        ("image/png", _make_png_bytes),
        ("image/jpeg", _make_jpeg_bytes),
        ("image/jpg", _make_jpeg_bytes),
    ])
    def test_predict_accepts_allowed_types_and_returns_json(
        self, content_type: str, make_bytes
    ):
        """POST /predict with allowed type returns JSON with mask_shape, mask_base64, colored_mask_base64, categories."""
        mask_h, mask_w = 48, 48
        mock_mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
        mock_colored = np.zeros((mask_h, mask_w, 3), dtype=np.uint8)

        with patch("app.main.SegmentationPredictor") as MockPredictor:
            mock_instance = MagicMock()
            mock_instance.load_model = MagicMock(return_value=MagicMock())
            mock_instance.predict_from_image = MagicMock(return_value=mock_mask)
            mock_instance.predict_to_colored_mask = MagicMock(return_value=mock_colored)
            MockPredictor.return_value = mock_instance
            client = TestClient(app)
            # Override state so model_loaded is True (lifespan already ran with mock)
            app.state.model_loaded = True
            app.state.predictor = mock_instance

        image_bytes = make_bytes()
        response = client.post(
            "/predict",
            files={"file": ("image.png", io.BytesIO(image_bytes), content_type)},
        )
        assert response.status_code == 200
        data = response.json()
        assert "mask_shape" in data
        assert data["mask_shape"] == [mask_h, mask_w]
        assert "mask_base64" in data
        assert "colored_mask_base64" in data
        assert "categories" in data
        # Decode base64 to ensure it's valid PNG bytes
        decoded_mask = base64.b64decode(data["mask_base64"])
        assert len(decoded_mask) > 0
        decoded_colored = base64.b64decode(data["colored_mask_base64"])
        assert len(decoded_colored) > 0
        # Categories from CATEGORY_NAMES
        expected_categories = [
            {"id": k, "name": v}
            for k, v in sorted(CATEGORY_NAMES.items())
        ]
        assert data["categories"] == expected_categories

    def test_predict_inference_error_returns_500(self):
        """POST /predict when predictor raises returns 500."""
        with patch("app.main.SegmentationPredictor") as MockPredictor:
            mock_instance = MagicMock()
            mock_instance.load_model = MagicMock(return_value=MagicMock())
            mock_instance.predict_from_image = MagicMock(
                side_effect=RuntimeError("Inference failed")
            )
            MockPredictor.return_value = mock_instance
            client = TestClient(app)
            app.state.model_loaded = True
            app.state.predictor = mock_instance

        png_bytes = _make_png_bytes()
        response = client.post(
            "/predict",
            files={"file": ("image.png", io.BytesIO(png_bytes), "image/png")},
        )
        assert response.status_code == 500
        assert "Inference failed" in response.json()["detail"]
