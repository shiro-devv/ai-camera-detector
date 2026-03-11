"""
Dataset loader — reads captured images, applies augmentation,
and builds PyTorch DataLoaders for training and validation.

Also generates the YAML dataset spec required by Ultralytics/YOLOv8.
"""

from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
from config import CONFIG, DatasetConfig
from utils import resize_with_padding


# ── Augmentation pipeline ─────────────────────────────────────────────────────

def build_transforms(
    image_size: Tuple[int, int],
    augment: bool = True,
) -> transforms.Compose:
    """
    Build a torchvision transform pipeline.

    When augment=True applies random flips, colour jitter, rotation, and erasing
    to boost generalisation for small custom datasets.
    """
    steps = [transforms.ToPILImage()]
    if augment:
        steps += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        ]
    steps += [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if augment:
        steps.append(transforms.RandomErasing(p=0.1))
    return transforms.Compose(steps)


# ── Dataset class ─────────────────────────────────────────────────────────────

class WebcamDataset(Dataset):
    """
    Classification dataset built from flat per-class image folders.

    folder structure:
        dataset_dir/
            class_a/  *.jpg
            class_b/  *.jpg
            …
    """

    def __init__(
        self,
        image_paths: List[Path],
        labels: List[int],
        class_names: List[str],
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.class_names = class_names
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.image_paths[idx]
        image = cv2.imread(str(path))
        if image is None:
            logger.warning(f"Could not read {path}, using black frame.")
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        return image, self.labels[idx]


# ── Builder ───────────────────────────────────────────────────────────────────

class DatasetLoader:
    """Creates train/val DataLoaders from captured images."""

    def __init__(self, cfg: DatasetConfig = CONFIG.dataset) -> None:
        self._cfg = cfg

    # ── public API ────────────────────────────────────────────────────────────

    def scan(self) -> Dict[str, int]:
        """Return {class_name: image_count} for all classes."""
        counts: Dict[str, int] = {}
        for cls in self._cfg.classes:
            folder = self._cfg.dataset_dir / cls
            counts[cls] = len(list(folder.glob("*.jpg"))) if folder.exists() else 0
        return counts

    def validate(self) -> bool:
        """
        Check dataset health.  Returns True if every class meets the
        minimum image threshold.
        """
        counts = self.scan()
        ok = True
        for cls, n in counts.items():
            if n < self._cfg.min_images_per_class:
                logger.warning(f"Class '{cls}' has only {n} images "
                               f"(min={self._cfg.min_images_per_class}).")
                ok = False
            else:
                logger.info(f"  {cls:<12}: {n} images ✓")
        return ok

    def build_loaders(
        self,
        batch_size: int = CONFIG.training.batch_size,
        num_workers: int = CONFIG.training.num_workers,
        image_size: Tuple[int, int] = CONFIG.dataset.image_size,
        augment: bool = CONFIG.dataset.augmentation_enabled,
        seed: int = 42,
    ) -> Tuple[DataLoader, DataLoader, List[str]]:
        """
        Build and return (train_loader, val_loader, class_names).
        """
        random.seed(seed)
        all_paths: List[Path] = []
        all_labels: List[int] = []
        class_names = self._cfg.classes

        for label_idx, cls in enumerate(class_names):
            folder = self._cfg.dataset_dir / cls
            if not folder.exists():
                continue
            paths = sorted(folder.glob("*.jpg"))
            all_paths.extend(paths)
            all_labels.extend([label_idx] * len(paths))

        if not all_paths:
            raise RuntimeError("No images found. Run dataset_capture.py first.")

        # Shuffle together
        combined = list(zip(all_paths, all_labels))
        random.shuffle(combined)
        all_paths, all_labels = zip(*combined)
        all_paths, all_labels = list(all_paths), list(all_labels)

        # Split
        split = int(len(all_paths) * (1 - self._cfg.val_split))
        train_paths, val_paths = all_paths[:split], all_paths[split:]
        train_labels, val_labels = all_labels[:split], all_labels[split:]

        logger.info(f"Dataset split — train: {len(train_paths)}  val: {len(val_paths)}")

        train_tf = build_transforms(image_size, augment=augment)
        val_tf   = build_transforms(image_size, augment=False)

        train_ds = WebcamDataset(train_paths, train_labels, class_names, train_tf)
        val_ds   = WebcamDataset(val_paths,   val_labels,   class_names, val_tf)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        return train_loader, val_loader, class_names

    # ── YOLO YAML ─────────────────────────────────────────────────────────────

    def generate_yolo_yaml(self, output_path: Optional[Path] = None) -> Path:
        """
        Generate a dataset.yaml file for Ultralytics training.

        Copies images into train/val splits under a temporary YOLO-style
        directory tree and writes the YAML spec.
        """
        yolo_dir = self._cfg.dataset_dir.parent / "yolo_dataset"
        output_path = output_path or (self._cfg.dataset_dir.parent / "dataset.yaml")

        random.seed(42)
        class_names = self._cfg.classes

        for split in ("train", "val"):
            for cls in class_names:
                (yolo_dir / split / "images" / cls).mkdir(parents=True, exist_ok=True)

        counts: Dict[str, Tuple[int, int]] = {}
        for cls in class_names:
            folder = self._cfg.dataset_dir / cls
            images = sorted(folder.glob("*.jpg")) if folder.exists() else []
            random.shuffle(images)
            split_idx = int(len(images) * (1 - self._cfg.val_split))
            train_imgs, val_imgs = images[:split_idx], images[split_idx:]

            for img in train_imgs:
                dst = yolo_dir / "train" / "images" / cls / img.name
                if not dst.exists():
                    shutil.copy(img, dst)

            for img in val_imgs:
                dst = yolo_dir / "val" / "images" / cls / img.name
                if not dst.exists():
                    shutil.copy(img, dst)

            counts[cls] = (len(train_imgs), len(val_imgs))

        spec = {
            "path":  str(yolo_dir.resolve()),
            "train": "train/images",
            "val":   "val/images",
            "nc":    len(class_names),
            "names": class_names,
        }
        with open(output_path, "w") as f:
            yaml.dump(spec, f, default_flow_style=False)

        logger.success(f"YOLO dataset YAML written to {output_path}")
        for cls, (tr, va) in counts.items():
            logger.info(f"  {cls:<12}: train={tr}  val={va}")

        return output_path


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from utils import setup_logging
    setup_logging(CONFIG.training.logs_dir)

    loader = DatasetLoader()
    valid = loader.validate()
    if valid:
        yaml_path = loader.generate_yolo_yaml()
        logger.info(f"Ready for training. YAML: {yaml_path}")
    else:
        logger.error("Dataset not ready. Capture more images first.")
