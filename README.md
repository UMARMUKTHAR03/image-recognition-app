# Image Recognition App – Deep Learning Computer Vision Platform

[![CI/CD Pipeline](https://github.com/UMARMUKTHAR03/image-recognition-app/actions/workflows/ci.yml/badge.svg)](https://github.com/UMARMUKTHAR03/image-recognition-app/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20CPU-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/React-19-61DAFB.svg?logo=react&logoColor=black)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178C6.svg?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Vite](https://img.shields.io/badge/Vite-6.x-646CFF.svg?logo=vite&logoColor=white)](https://vitejs.dev/)

> **Tagline:** See. Recognize. Understand.  
> **Author & Engineer:** Umar Mukthar  
> **Production Frontend:** [https://umarmukthar03.github.io/image-recognition-app/](https://umarmukthar03.github.io/image-recognition-app/)  
> **Production API & Swagger Docs:** [https://image-recognition-api-jq01.onrender.com/docs](https://image-recognition-api-jq01.onrender.com/docs)  
> **GitHub Repository:** [https://github.com/UMARMUKTHAR03/image-recognition-app](https://github.com/UMARMUKTHAR03/image-recognition-app)

---

## 1. Executive Summary & Problem Statement

Modern computer vision systems often suffer from two major production pitfalls: **high computational latency** on commodity hardware, and the **black-box problem**, where neural network predictions lack visual explanation.

**Image Recognition App** is an end-to-end deep learning computer vision platform engineered to deliver:
1. **Real-time, CPU-optimized image classification** across the full ImageNet-1K taxonomy (1,000 distinct object categories) with zero fabricated confidences or simulated latency.
2. **Transparent Explainable AI (XAI)** through genuine **Grad-CAM (Gradient-weighted Class Activation Mapping)** computed on the final convolutional feature extraction layer.
3. **High-throughput batch processing** supporting concurrent multi-image evaluation with CSV telemetry export.
4. **Reproducible transfer learning training pipeline** for fine-tuning custom classification heads with early stopping, confusion matrices, and validation reports.

---

## 2. Core Architectural Pillars

```
┌────────────────────────────────────────────────────────────────────────┐
│                        React 19 + Vite Frontend                        │
│   (Studio • Batch • Grad-CAM XAI • Dashboard • History • Taxonomy)     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP / Multipart Form-Data
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        FastAPI REST API Gateway                        │
│   (/health • /api/v1/predict • /api/v1/predict/batch • /api/v1/explain)│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│              PyTorch Deep Learning Engine (MobileNetV3-Small)          │
│   ├── Torchvision Pretrained Weights: MobileNet_V3_Small_Weights.DEFAULT│
│   ├── Input Transform: 256 Resize → 224 Center Crop → RGB Norm        │
│   ├── Softmax Top-K Probabilities Calculation                          │
│   └── Grad-CAM Hook: features.12 Conv2d Layer Backpropagation         │
└────────────────────────────────────────────────────────────────────────┘
```

### Why MobileNetV3-Small?
- **Efficiency:** Utilizes depthwise separable convolutions and Squeeze-and-Excitation (SE) blocks with Hardswish activations, reducing parameter count to ~2.5M.
- **Low Memory Footprint:** Executes in **<100 ms** on standard cloud CPU instances without GPU acceleration.
- **Resource Compatibility:** Perfectly optimized for Render Free-tier memory constraints while retaining high top-1 ImageNet accuracy.

---

## 3. Key Capabilities

### A. Real-Time Vision Studio
- Drag-and-drop file upload, file browsing, and live browser camera snapshot ingestion.
- Supports JPEG, PNG, and WEBP formats with a strict 10 MB file validation filter.
- Calculates genuine Softmax confidence distributions and top-k rankings.
- Automatically flags low-confidence predictions (<15%) when subjects fall outside ImageNet categories.

### B. Explainable AI (Grad-CAM)
- Extracts feature activations and backward gradients from `model.features[-1]`.
- Computes global average pooling weights to construct a 2D saliency heatmap.
- Generates alpha-blended Jet colormap overlays directly highlighting the visual regions that informed the classification decision.

### C. Concurrent Batch Recognition
- Accepts up to 8 images per request with per-file validation and error isolation.
- Displays structured batch summaries with individual confidence scores and latency metrics.
- One-click CSV export for downstream data science inspection.

### D. ImageNet-1K Taxonomy Explorer
- Real-time client search across all 1,000 ImageNet categories and class IDs.

### E. Reproducible Transfer Learning Pipeline
- Standard `ImageFolder` dataset loader (`dataset/train/`, `dataset/val/`).
- MobileNetV3 feature backbone freezing with custom classifier head replacement.
- AdamW optimizer with StepLR learning rate scheduling.
- Generation of training curves, confusion matrices, and classification metrics.

---

## 4. API Specification & Endpoints

| Method | Endpoint | Description |
| :---: | :--- | :--- |
| `GET` | `/health` | Server heartbeat, application status, active model, and class count. |
| `GET` | `/api/v1/model-info` | Architecture specifications, input dimensions, and custom training metrics. |
| `GET` | `/api/v1/classes` | Complete ImageNet-1K taxonomy mapping (class IDs and labels). |
| `POST` | `/api/v1/predict` | Multipart image upload returning top-5 softmax predictions and latency. |
| `POST` | `/api/v1/predict/batch` | Concurrently classifies up to 8 images with CSV-ready metadata. |
| `POST` | `/api/v1/explain` | Computes Grad-CAM heatmap and overlay base64 visualizations. |

Interactive API documentation is available via Swagger UI at `/docs`.

---

## 5. Local Installation & Development

### Prerequisites
- Python 3.11+
- Node.js 20+ & npm

### Backend Setup
```bash
# Clone the repository
git clone https://github.com/UMARMUKTHAR03/image-recognition-app.git
cd image-recognition-app

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install CPU-optimized PyTorch and dependencies
pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Run backend API server
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend Setup
```bash
cd frontend

# Install dependencies
npm ci

# Start Vite development server
npm run dev
```
Open [http://localhost:5173/image-recognition-app/](http://localhost:5173/image-recognition-app/) in your browser.

---

## 6. Testing & Quality Assurance

### Run Backend PyTest Suite
```bash
pytest backend/tests/test_api.py -v
```
Verified tests:
- `test_health_endpoint`
- `test_model_info_endpoint`
- `test_classes_endpoint`
- `test_predict_single_valid_image`
- `test_predict_invalid_extension`
- `test_predict_corrupted_image`
- `test_predict_empty_file`
- `test_predict_batch_images`
- `test_predict_batch_limit_exceeded`
- `test_explain_grad_cam`

### Run Frontend Vitest Suite & Build
```bash
cd frontend
npm test -- --run
npm run build
```

---

## 7. Production Deployment

### Frontend (GitHub Pages)
The GitHub Actions workflow at `.github/workflows/ci.yml` runs automatically on pushes to `main`:
1. Executes backend PyTest test suite.
2. Installs frontend dependencies and runs Vitest unit tests.
3. Validates TypeScript types (`tsc -b`).
4. Builds the production bundle with base path `/image-recognition-app/`.
5. Deploys artifacts directly to GitHub Pages.

### Backend (Render Blueprint)
The repository contains a production-ready `render.yaml` Blueprint:
1. Log in to [Render Dashboard](https://dashboard.render.com/).
2. Click **New +** → **Blueprint**.
3. Connect your repository: `https://github.com/UMARMUKTHAR03/image-recognition-app`.
4. Render detects `render.yaml` and deploys the `image-recognition-api` service automatically with CPU PyTorch on Python 3.11.

---

## 8. License

This project is licensed under the [MIT License](LICENSE).
