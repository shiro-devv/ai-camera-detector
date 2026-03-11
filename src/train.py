"""
Training pipeline.

Supports two modes:
  1. ultralytics_yolo  — fine-tune YOLOv8 for object detection (recommended)
  2. pytorch_classifier — lightweight PyTorch classifier (fallback / demo)

The main entry-point is Trainer.run().
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import CONFIG, TrainingConfig
from dataset_loader import DatasetLoader
from utils import ensure_dirs, get_device, latest_checkpoint, setup_logging


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  PyTorch classifier (used when full YOLO detection isn't needed)
# ═══════════════════════════════════════════════════════════════════════════════

class TransferClassifier(nn.Module):
    """
    MobileNetV3-Small backbone + custom head for N classes.
    Fast, lightweight, suitable for real-time inference.
    """

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_features, num_classes)
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Trainer
# ═══════════════════════════════════════════════════════════════════════════════

class Trainer:
    """
    Orchestrates training, validation, checkpointing, and TensorBoard logging.

    Tries to use Ultralytics YOLOv8 fine-tuning when a valid YOLO dataset YAML
    is available; falls back to the PyTorch classifier otherwise.
    """

    def __init__(self, cfg: TrainingConfig = CONFIG.training) -> None:
        self._cfg = cfg
        self._device = get_device()
        ensure_dirs(cfg.checkpoint_dir, cfg.models_dir, cfg.logs_dir)
        self._writer = SummaryWriter(log_dir=str(cfg.logs_dir / "tensorboard"))

    # ── public API ────────────────────────────────────────────────────────────

    def run(self, use_yolo: bool = True) -> Path:
        """
        Train the model.

        Parameters
        ----------
        use_yolo:
            If True, attempt YOLOv8 fine-tuning via Ultralytics.
            Falls back to the PyTorch classifier on failure.

        Returns
        -------
        Path to the best model weights.
        """
        if use_yolo:
            try:
                return self._train_yolo()
            except Exception as exc:
                logger.warning(f"YOLOv8 training failed ({exc}). Falling back to classifier.")

        return self._train_classifier()

    # ── YOLOv8 fine-tuning ────────────────────────────────────────────────────

    def _train_yolo(self) -> Path:
        """Fine-tune YOLOv8 using the Ultralytics API."""
        from ultralytics import YOLO

        dataset_loader = DatasetLoader()
        yaml_path = dataset_loader.generate_yolo_yaml()

        logger.info(f"Starting YOLOv8 fine-tuning — model={self._cfg.model_name}")
        model = YOLO(self._cfg.model_name)

        results = model.train(
            data=str(yaml_path),
            epochs=self._cfg.epochs,
            imgsz=self._cfg.image_size,
            batch=self._cfg.batch_size,
            lr0=self._cfg.learning_rate,
            warmup_epochs=self._cfg.warmup_epochs,
            patience=self._cfg.patience,
            device=0 if self._device.type == "cuda" else "cpu",
            project=str(self._cfg.models_dir),
            name="yolov8_custom",
            exist_ok=True,
            amp=self._cfg.use_amp,
            workers=self._cfg.num_workers,
            save_period=self._cfg.save_every_n_epochs,
            verbose=True,
        )

        best = Path(results.save_dir) / "weights" / "best.pt"
        if not best.exists():
            # Fallback — grab latest checkpoint
            best = latest_checkpoint(self._cfg.models_dir / "yolov8_custom" / "weights") or best

        # Copy to canonical location
        final = self._cfg.models_dir / "best_yolo.pt"
        if best.exists():
            import shutil
            shutil.copy(best, final)
            logger.success(f"Best YOLO model → {final}")

        self._writer.close()
        return final

    # ── PyTorch classifier ────────────────────────────────────────────────────

    def _train_classifier(self) -> Path:
        """Train a MobileNetV3 classifier as a fallback."""
        loader = DatasetLoader()
        if not loader.validate():
            raise RuntimeError("Dataset validation failed. Capture more images.")

        train_dl, val_dl, class_names = loader.build_loaders(
            batch_size=self._cfg.batch_size,
            num_workers=self._cfg.num_workers,
        )

        logger.info(f"Training PyTorch classifier — classes: {class_names}")
        model = TransferClassifier(
            num_classes=len(class_names),
            pretrained=self._cfg.pretrained,
        ).to(self._device)

        # Freeze backbone for first N epochs
        self._set_backbone_grad(model, requires_grad=False)

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self._cfg.learning_rate,
            weight_decay=self._cfg.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self._cfg.epochs
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        scaler = GradScaler(enabled=self._cfg.use_amp and self._device.type == "cuda")

        best_val_acc = 0.0
        best_path = self._cfg.models_dir / "best_classifier.pt"
        patience_counter = 0

        for epoch in range(1, self._cfg.epochs + 1):
            # Unfreeze backbone after warmup
            if epoch == self._cfg.freeze_backbone_epochs + 1:
                self._set_backbone_grad(model, requires_grad=True)
                logger.info("Backbone unfrozen — fine-tuning entire network.")

            train_loss, train_acc = self._train_epoch(
                model, train_dl, optimizer, criterion, scaler, epoch
            )
            val_loss, val_acc = self._val_epoch(model, val_dl, criterion, epoch)

            scheduler.step()

            self._writer.add_scalars("Loss",    {"train": train_loss, "val": val_loss},    epoch)
            self._writer.add_scalars("Accuracy", {"train": train_acc,  "val": val_acc},    epoch)
            self._writer.add_scalar("LR",        optimizer.param_groups[0]["lr"],          epoch)

            logger.info(
                f"Epoch {epoch:3d}/{self._cfg.epochs}  "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.2%}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.2%}"
            )

            # Save best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "class_names": class_names,
                    "val_acc": val_acc,
                }, best_path)
                logger.success(f"  ✓ New best val_acc={val_acc:.2%} → {best_path.name}")
            else:
                patience_counter += 1
                if patience_counter >= self._cfg.patience:
                    logger.info(f"Early stopping at epoch {epoch}.")
                    break

            # Periodic checkpoint
            if epoch % self._cfg.save_every_n_epochs == 0:
                ckpt = self._cfg.checkpoint_dir / f"epoch_{epoch:04d}.pt"
                torch.save(model.state_dict(), ckpt)

        self._writer.close()
        logger.success(f"Training complete. Best val_acc={best_val_acc:.2%}")
        return best_path

    # ── epoch helpers ─────────────────────────────────────────────────────────

    def _train_epoch(
        self,
        model: nn.Module,
        loader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        scaler: GradScaler,
        epoch: int,
    ) -> Tuple[float, float]:
        model.train()
        total_loss = correct = total = 0
        pbar = tqdm(loader, desc=f"Train {epoch}", leave=False, unit="batch")

        for images, labels in pbar:
            images = images.to(self._device, non_blocking=True)
            labels = labels.to(self._device, non_blocking=True)

            optimizer.zero_grad()
            with autocast(enabled=self._cfg.use_amp and self._device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            bs = labels.size(0)
            total_loss += loss.item() * bs
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += bs
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.2%}")

        return total_loss / total, correct / total

    @torch.no_grad()
    def _val_epoch(
        self,
        model: nn.Module,
        loader,
        criterion: nn.Module,
        epoch: int,
    ) -> Tuple[float, float]:
        model.eval()
        total_loss = correct = total = 0

        for images, labels in tqdm(loader, desc=f"Val   {epoch}", leave=False, unit="batch"):
            images = images.to(self._device, non_blocking=True)
            labels = labels.to(self._device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            bs = labels.size(0)
            total_loss += loss.item() * bs
            correct += (logits.argmax(1) == labels).sum().item()
            total += bs

        return total_loss / total, correct / total

    # ── utils ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _set_backbone_grad(model: TransferClassifier, requires_grad: bool) -> None:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = requires_grad


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    setup_logging(CONFIG.training.logs_dir)
    import argparse

    ap = argparse.ArgumentParser(description="Train the AI Camera Detector model.")
    ap.add_argument("--epochs",    type=int,   default=CONFIG.training.epochs)
    ap.add_argument("--batch",     type=int,   default=CONFIG.training.batch_size)
    ap.add_argument("--lr",        type=float, default=CONFIG.training.learning_rate)
    ap.add_argument("--no-yolo",   action="store_true")
    args = ap.parse_args()

    CONFIG.training.epochs       = args.epochs
    CONFIG.training.batch_size   = args.batch
    CONFIG.training.learning_rate = args.lr

    trainer = Trainer()
    best = trainer.run(use_yolo=not args.no_yolo)
    logger.success(f"Best model: {best}")
