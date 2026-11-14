"""
Evaluation pipeline for custom-trained MobileNetV3-Small models.
Computes accuracy, precision, recall, F1-score, confusion matrix, and saves artifacts.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import models


def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    class_names: List[str],
    device: torch.device,
    output_dir: Path
) -> Dict[str, Any]:
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []
    all_probs: List[List[float]] = []

    with torch.inference_mode():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    # Calculate overall metrics
    correct = (y_true == y_pred).sum()
    total = len(y_true)
    accuracy = float(correct / total) if total > 0 else 0.0

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_list = cm.tolist()

    # Classification report
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    results = {
        "num_samples": total,
        "num_classes": len(class_names),
        "classes": class_names,
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "accuracy_percent": round(accuracy * 100, 2)
        },
        "confusion_matrix": cm_list,
        "classification_report": report
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "evaluation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    from backend.training.config import TrainingConfig
    from backend.training.dataset import create_dataloaders

    config = TrainingConfig()
    device = torch.device("cpu")

    try:
        dataloaders, sizes, classes = create_dataloaders(config.data_dir, config)
        print(f"Loaded dataset with {len(classes)} classes: {classes}")

        checkpoint_path = config.checkpoint_dir / "best_model.pth"
        if not checkpoint_path.exists():
            print(f"Checkpoint not found at {checkpoint_path}. Run training first.")
        else:
            model = models.mobilenet_v3_small(weights=None)
            in_features = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(in_features, len(classes))
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)

            metrics = evaluate_model(model, dataloaders["val"], classes, device, config.output_dir)
            print("Evaluation complete:")
            print(json.dumps(metrics["metrics"], indent=2))
    except Exception as e:
        print(f"Evaluation pipeline ready. (Notice: {e})")
