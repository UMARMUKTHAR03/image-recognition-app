"""
Reproducible transfer learning training pipeline using MobileNetV3-Small.
Trains a custom classification head on user-provided datasets, logs loss/accuracy curves,
and saves the best model checkpoint.
"""

import json
import random
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights

from backend.training.config import TrainingConfig
from backend.training.dataset import create_dataloaders
from backend.training.evaluate import evaluate_model


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(num_classes: int, config: TrainingConfig) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.DEFAULT if config.pretrained else None
    model = models.mobilenet_v3_small(weights=weights)

    if config.freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace classifier head: MobileNetV3-Small classifier has 4 layers: Linear(576, 1024), Hardswish, Dropout, Linear(1024, 1000)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def train_model(config: TrainingConfig) -> Dict[str, Any]:
    set_seed(config.seed)
    device = torch.device("cpu")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataloaders, dataset_sizes, class_names = create_dataloaders(config.data_dir, config)
    num_classes = len(class_names)

    model = build_model(num_classes, config)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)

    best_acc = 0.0
    best_epoch = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    start_time = time.time()

    for epoch in range(config.num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders["train"]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

        scheduler.step()
        epoch_train_loss = running_loss / dataset_sizes["train"]
        epoch_train_acc = running_corrects / dataset_sizes["train"]

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.inference_mode():
            for inputs, labels in dataloaders["val"]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data).item()

        epoch_val_loss = val_loss / dataset_sizes["val"]
        epoch_val_acc = val_corrects / dataset_sizes["val"]

        history["train_loss"].append(round(epoch_train_loss, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["train_acc"].append(round(epoch_train_acc, 4))
        history["val_acc"].append(round(epoch_val_acc, 4))

        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_epoch = epoch
            checkpoint_path = config.checkpoint_dir / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "accuracy": best_acc,
                "class_names": class_names,
                "num_classes": num_classes
            }, checkpoint_path)

    elapsed_time = round(time.time() - start_time, 2)

    # Save training history curves
    history_path = config.output_dir / "training_history.json"
    training_summary = {
        "best_epoch": best_epoch + 1,
        "best_val_accuracy": round(best_acc, 4),
        "total_epochs": config.num_epochs,
        "training_time_seconds": elapsed_time,
        "classes": class_names,
        "num_classes": num_classes,
        "history": history
    }
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)

    # Run final evaluation using the best model
    checkpoint = torch.load(config.checkpoint_dir / "best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    eval_results = evaluate_model(model, dataloaders["val"], class_names, device, config.output_dir)

    return {
        "training_summary": training_summary,
        "evaluation_results": eval_results
    }


if __name__ == "__main__":
    cfg = TrainingConfig()
    try:
        results = train_model(cfg)
        print("Training successfully finished!")
        print(f"Best Val Accuracy: {results['training_summary']['best_val_accuracy'] * 100:.2f}%")
    except Exception as e:
        print(f"Transfer learning engine ready. Notice: {e}")
