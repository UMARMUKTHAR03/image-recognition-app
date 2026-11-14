"""
FastAPI Routes for Image Recognition App.
Provides single and batch prediction, Grad-CAM explainability, model metadata, and health status.
"""

import io
import json
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

from backend.core.model_service import get_model_service

router = APIRouter()

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB limit


def validate_image_upload(file_bytes: bytes, filename: str, content_type: Optional[str] = None) -> Image.Image:
    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File '{filename}' is empty."
        )

    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File '{filename}' exceeds maximum allowed size of 10 MB."
        )

    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        )

    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.verify()  # Verify image integrity
        # Re-open because verify() closes or damages the internal file pointer
        image = Image.open(io.BytesIO(file_bytes))
        image.load()
        return image
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid or corrupted image file '{filename}': {str(e)}"
        )


@router.get("/health")
def health_check():
    service = get_model_service()
    return {
        "status": "online",
        "app_title": "Image Recognition App – Deep Learning Computer Vision Platform",
        "tagline": "See. Recognize. Understand.",
        "version": "2.0.0",
        "model": service.model_name,
        "classes_count": service.num_classes
    }


@router.get("/api/v1/model-info")
def get_model_info():
    service = get_model_service()
    info = service.get_model_info()

    # Check for custom training artifacts
    artifacts_dir = Path("backend/artifacts")
    custom_metrics_path = artifacts_dir / "evaluation_metrics.json"
    custom_history_path = artifacts_dir / "training_history.json"

    custom_metrics = None
    custom_history = None

    if custom_metrics_path.exists():
        try:
            with open(custom_metrics_path, "r", encoding="utf-8") as f:
                custom_metrics = json.load(f)
        except Exception:
            pass

    if custom_history_path.exists():
        try:
            with open(custom_history_path, "r", encoding="utf-8") as f:
                custom_history = json.load(f)
        except Exception:
            pass

    return {
        "status": "success",
        "data": {
            **info,
            "has_custom_artifacts": custom_metrics is not None,
            "custom_evaluation": custom_metrics,
            "custom_training_history": custom_history
        }
    }


@router.get("/api/v1/classes")
def get_classes():
    service = get_model_service()
    return {
        "status": "success",
        "total_classes": service.num_classes,
        "classes": [
            {"index": i, "label": label}
            for i, label in enumerate(service.categories)
        ]
    }


@router.post("/api/v1/predict")
async def predict_single(
    file: UploadFile = File(...),
    top_k: int = Form(default=5)
):
    try:
        file_bytes = await file.read()
        image = validate_image_upload(file_bytes, file.filename or "image.jpg", file.content_type)
        service = get_model_service()
        result = service.predict(image, top_k=top_k)
        result["filename"] = file.filename or "image.jpg"
        return {
            "status": "success",
            "data": result
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference processing error: {str(e)}"
        )


@router.post("/api/v1/predict/batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    top_k: int = Form(default=5)
):
    if len(files) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided in batch request."
        )

    if len(files) > 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch limit exceeded. A maximum of 8 images can be submitted per request."
        )

    service = get_model_service()
    batch_results = []

    for file in files:
        fname = file.filename or "unknown.jpg"
        try:
            content = await file.read()
            image = validate_image_upload(content, fname, file.content_type)
            pred = service.predict(image, top_k=top_k)
            pred["filename"] = fname
            pred["status"] = "success"
            batch_results.append(pred)
        except HTTPException as he:
            batch_results.append({
                "filename": fname,
                "status": "error",
                "error": he.detail,
                "status_code": he.status_code
            })
        except Exception as e:
            batch_results.append({
                "filename": fname,
                "status": "error",
                "error": str(e),
                "status_code": 500
            })

    return {
        "status": "success",
        "total_files": len(files),
        "successful_predictions": sum(1 for r in batch_results if r["status"] == "success"),
        "results": batch_results
    }


@router.post("/api/v1/explain")
async def explain_prediction(
    file: UploadFile = File(...),
    target_class_index: Optional[int] = Form(default=None)
):
    try:
        file_bytes = await file.read()
        image = validate_image_upload(file_bytes, file.filename or "image.jpg", file.content_type)
        service = get_model_service()
        grad_cam_result = service.generate_grad_cam(image, target_class_idx=target_class_index)
        return {
            "status": "success",
            "data": {
                "filename": file.filename or "image.jpg",
                **grad_cam_result
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Grad-CAM explainability computation error: {str(e)}"
        )
