"""
Model training and evaluation configuration for transfer learning with MobileNetV3-Small.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class TrainingConfig:
    # Directories
    data_dir: Path = Path("dataset")
    output_dir: Path = Path("backend/artifacts")
    checkpoint_dir: Path = Path("backend/artifacts/checkpoints")

    # Image preprocessing
    image_size: int = 224
    crop_size: int = 224
    resize_size: int = 256
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    # Hyperparameters
    batch_size: int = 16
    num_epochs: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    step_size: int = 5
    gamma: float = 0.1
    patience: int = 4

    # Architecture
    pretrained: bool = True
    freeze_features: bool = True
    dropout: float = 0.2

    # System
    seed: int = 42
    num_workers: int = 0
