"""
Dataset loading and data augmentation pipelines for transfer learning.
"""

from pathlib import Path
from typing import Dict, Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from backend.training.config import TrainingConfig


def get_data_transforms(config: TrainingConfig) -> Dict[str, transforms.Compose]:
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config.image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.mean, std=config.std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(config.resize_size),
        transforms.CenterCrop(config.crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.mean, std=config.std)
    ])

    return {"train": train_transform, "val": val_transform}


def create_dataloaders(
    data_dir: Path,
    config: TrainingConfig
) -> Tuple[Dict[str, DataLoader], Dict[str, int], list]:
    data_transforms = get_data_transforms(config)

    train_path = data_dir / "train"
    val_path = data_dir / "val"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Dataset folders missing. Expected '{train_path}' and '{val_path}'. "
            "Please structure datasets as dataset/train/<class_name>/ and dataset/val/<class_name>/."
        )

    image_datasets = {
        "train": datasets.ImageFolder(str(train_path), data_transforms["train"]),
        "val": datasets.ImageFolder(str(val_path), data_transforms["val"])
    }

    class_names = image_datasets["train"].classes
    if len(class_names) < 2:
        raise ValueError(f"At least 2 classes are required for classification, found {len(class_names)}.")

    dataloaders = {
        "train": DataLoader(
            image_datasets["train"],
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers
        ),
        "val": DataLoader(
            image_datasets["val"],
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    return dataloaders, dataset_sizes, class_names
