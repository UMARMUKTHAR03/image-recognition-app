"""
Unit and integration tests for FastAPI backend and computer vision pipeline.
Tests health, model info, image validation, predictions, batching, and Grad-CAM.
"""

import io
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.core.model_service import get_model_service

client = TestClient(app)


def create_test_image(format="JPEG", size=(224, 224), color=(255, 0, 0)):
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=format)
    buf.seek(0)
    return buf.getvalue()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["model"] == "MobileNetV3-Small"
    assert data["classes_count"] == 1000
    assert "Image Recognition App" in data["app_title"]


def test_model_info_endpoint():
    response = client.get("/api/v1/model-info")
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["architecture"] == "MobileNetV3-Small"
    assert data["num_classes"] == 1000
    assert data["weights"] == "MobileNet_V3_Small_Weights.DEFAULT"
    assert "explainability_layer" in data


def test_classes_endpoint():
    response = client.get("/api/v1/classes")
    assert response.status_code == 200
    data = response.json()
    assert data["total_classes"] == 1000
    assert len(data["classes"]) == 1000
    assert data["classes"][0]["label"] == "tench"


def test_predict_single_valid_image():
    img_bytes = create_test_image(format="JPEG", size=(100, 100), color=(100, 150, 200))
    files = {"file": ("test_sample.jpg", img_bytes, "image/jpeg")}
    response = client.post("/api/v1/predict", files=files, data={"top_k": 5})

    assert response.status_code == 200
    data = response.json()["data"]
    assert "predicted_class" in data
    assert "confidence_percent" in data
    assert "top_predictions" in data
    assert len(data["top_predictions"]) == 5
    assert data["latency_ms"] >= 0
    assert data["image_dimensions"]["width"] == 100
    assert data["image_dimensions"]["height"] == 100

    # Top predictions ordering
    confs = [p["confidence_percent"] for p in data["top_predictions"]]
    assert confs == sorted(confs, reverse=True)


def test_predict_invalid_extension():
    fake_bytes = b"Hello world text file"
    files = {"file": ("malicious.txt", fake_bytes, "text/plain")}
    response = client.post("/api/v1/predict", files=files)
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_predict_corrupted_image():
    corrupt_bytes = b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00corrupteddatajunkjunk"
    files = {"file": ("corrupt.jpg", corrupt_bytes, "image/jpeg")}
    response = client.post("/api/v1/predict", files=files)
    assert response.status_code == 400
    assert "Invalid or corrupted image" in response.json()["detail"]


def test_predict_empty_file():
    files = {"file": ("empty.png", b"", "image/png")}
    response = client.post("/api/v1/predict", files=files)
    assert response.status_code == 400
    assert "is empty" in response.json()["detail"]


def test_predict_batch_images():
    img1 = create_test_image(format="JPEG", size=(50, 50), color=(255, 0, 0))
    img2 = create_test_image(format="PNG", size=(60, 60), color=(0, 255, 0))
    files = [
        ("files", ("img1.jpg", img1, "image/jpeg")),
        ("files", ("img2.png", img2, "image/png"))
    ]
    response = client.post("/api/v1/predict/batch", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["total_files"] == 2
    assert data["successful_predictions"] == 2
    assert len(data["results"]) == 2
    assert data["results"][0]["filename"] == "img1.jpg"
    assert data["results"][1]["filename"] == "img2.png"


def test_predict_batch_limit_exceeded():
    img = create_test_image()
    files = [("files", (f"img_{i}.jpg", img, "image/jpeg")) for i in range(9)]
    response = client.post("/api/v1/predict/batch", files=files)
    assert response.status_code == 400
    assert "Batch limit exceeded" in response.json()["detail"]


def test_explain_grad_cam():
    img_bytes = create_test_image(format="JPEG", size=(120, 120), color=(200, 100, 50))
    files = {"file": ("dog_test.jpg", img_bytes, "image/jpeg")}
    response = client.post("/api/v1/explain", files=files)
    assert response.status_code == 200
    data = response.json()["data"]
    assert "heatmap_base64" in data
    assert "overlay_base64" in data
    assert data["heatmap_base64"].startswith("data:image/jpeg;base64,")
    assert data["overlay_base64"].startswith("data:image/jpeg;base64,")
    assert "target_class_label" in data
    assert data["latency_ms"] >= 0
