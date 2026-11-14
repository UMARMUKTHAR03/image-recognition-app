"""
Singleton Deep Learning Inference Service for Image Recognition App.
Uses torchvision MobileNetV3-Small with official MobileNet_V3_Small_Weights.DEFAULT.
Provides genuine top-5 softmax predictions and Grad-CAM visual explanations.
"""

import base64
import io
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class ModelService:
    _instance: Optional["ModelService"] = None

    def __new__(cls) -> "ModelService":
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        self.device = torch.device("cpu")
        self.model_name = "MobileNetV3-Small"
        self.weights_name = "MobileNet_V3_Small_Weights.DEFAULT"
        self.weights = MobileNet_V3_Small_Weights.DEFAULT
        self.model = mobilenet_v3_small(weights=self.weights)
        self.model.to(self.device)
        self.model.eval()

        # Official transforms and categories from pretrained weights
        self.transforms = self.weights.transforms()
        self.categories: List[str] = list(self.weights.meta["categories"])
        self.num_classes = len(self.categories)

        # Target feature extraction layer for Grad-CAM: last conv layer in features
        self.grad_cam_layer = self.model.features[-1]

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "architecture": "MobileNetV3-Small",
            "weights": self.weights_name,
            "num_classes": self.num_classes,
            "device": str(self.device),
            "input_resolution": "224x224 (preprocessed from 256 crop)",
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            },
            "explainability_layer": "model.features[-1] (Conv2dNormActivation 576ch)"
        }

    def preprocess_image(self, image: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int]]:
        orig_width, orig_height = image.size
        # Guarantee 3-channel RGB
        rgb_img = image.convert("RGB")
        tensor = self.transforms(rgb_img).unsqueeze(0).to(self.device)
        return tensor, (orig_width, orig_height)

    def predict(self, image: Image.Image, top_k: int = 5) -> Dict[str, Any]:
        start_time = time.perf_counter()
        tensor, (width, height) = self.preprocess_image(image)

        with torch.inference_mode():
            logits = self.model(tensor)
            probabilities = F.softmax(logits, dim=1)[0]
            top_probs, top_indices = torch.topk(probabilities, k=min(top_k, self.num_classes))

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        top_predictions: List[Dict[str, Any]] = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            confidence = round(prob * 100, 2)
            top_predictions.append({
                "class_index": idx,
                "label": self.categories[idx],
                "confidence_percent": confidence,
                "probability": round(prob, 5)
            })

        best = top_predictions[0]
        # Low confidence threshold: below 15% on 1000-class ImageNet
        is_low_confidence = best["confidence_percent"] < 15.0

        return {
            "predicted_class": best["label"],
            "class_index": best["class_index"],
            "confidence_percent": best["confidence_percent"],
            "top_predictions": top_predictions,
            "is_low_confidence": is_low_confidence,
            "warning": "Low classification confidence. Input may contain an out-of-domain subject or multiple objects." if is_low_confidence else None,
            "latency_ms": latency_ms,
            "image_dimensions": {"width": width, "height": height},
            "model": self.model_name
        }

    def generate_grad_cam(self, image: Image.Image, target_class_idx: Optional[int] = None) -> Dict[str, Any]:
        """
        Executes genuine Grad-CAM on the final convolutional features layer.
        Returns base64 heatmap, overlay, and target class diagnostics.
        """
        start_time = time.perf_counter()
        rgb_img = image.convert("RGB")
        width, height = rgb_img.size
        tensor, _ = self.preprocess_image(rgb_img)

        activations: List[torch.Tensor] = []
        gradients: List[torch.Tensor] = []

        def forward_hook(module, inp, out):
            activations.append(out)

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        handle_f = self.grad_cam_layer.register_forward_hook(forward_hook)
        handle_b = self.grad_cam_layer.register_full_backward_hook(backward_hook)

        try:
            self.model.zero_grad()
            output = self.model(tensor)
            if target_class_idx is None:
                target_class_idx = int(output.argmax(dim=1).item())

            target_class_idx = max(0, min(target_class_idx, self.num_classes - 1))
            score = output[0, target_class_idx]
            score.backward()

            act = activations[0].detach()  # [1, 576, 7, 7]
            grad = gradients[0].detach()   # [1, 576, 7, 7]

            weights = torch.mean(grad, dim=(2, 3), keepdim=True)  # Global average pooling
            cam = torch.sum(weights * act, dim=1, keepdim=True)
            cam = F.relu(cam)
            cam_np = cam.squeeze().cpu().numpy()

            # Normalize to [0, 1]
            c_min, c_max = cam_np.min(), cam_np.max()
            if c_max > c_min:
                cam_norm = (cam_np - c_min) / (c_max - c_min)
            else:
                cam_norm = np.zeros_like(cam_np)

            # Resize CAM to match original image dimensions
            cam_img = Image.fromarray((cam_norm * 255).astype(np.uint8)).resize(
                (width, height), Image.Resampling.BILINEAR
            )
            cam_resized = np.array(cam_img, dtype=np.float32) / 255.0

            # Jet colormap formula
            r = np.clip(1.5 - np.abs(4 * cam_resized - 3), 0, 1)
            g = np.clip(1.5 - np.abs(4 * cam_resized - 2), 0, 1)
            b = np.clip(1.5 - np.abs(4 * cam_resized - 1), 0, 1)
            heatmap = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

            # Create alpha overlay (55% original, 45% heatmap)
            orig_np = np.array(rgb_img, dtype=np.uint8)
            overlay = (0.55 * orig_np + 0.45 * heatmap).astype(np.uint8)

            # Encode images to base64
            buf_heat = io.BytesIO()
            Image.fromarray(heatmap).save(buf_heat, format="JPEG", quality=90)
            heatmap_b64 = base64.b64encode(buf_heat.getvalue()).decode("utf-8")

            buf_overlay = io.BytesIO()
            Image.fromarray(overlay).save(buf_overlay, format="JPEG", quality=90)
            overlay_b64 = base64.b64encode(buf_overlay.getvalue()).decode("utf-8")

            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

            return {
                "target_class_index": target_class_idx,
                "target_class_label": self.categories[target_class_idx],
                "heatmap_base64": f"data:image/jpeg;base64,{heatmap_b64}",
                "overlay_base64": f"data:image/jpeg;base64,{overlay_b64}",
                "latency_ms": latency_ms,
                "feature_layer": "features.12",
                "cam_resolution": f"{cam_np.shape[0]}x{cam_np.shape[1]}"
            }
        finally:
            handle_f.remove()
            handle_b.remove()
            self.model.zero_grad()


# Module-level getter
def get_model_service() -> ModelService:
    return ModelService()
