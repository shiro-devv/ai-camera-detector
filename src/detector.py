"""
Detector module — loads trained weights and runs single-frame inference.

Supports:
  • Ultralytics YOLOv8 .pt models  (primary)
  • PyTorch classifier .pt models   (fallback)
  • ONNX models                     (optional, fastest)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))
from config import CONFIG, InferenceConfig
from utils import get_device, latest_checkpoint, resize_with_padding


@dataclass
class Detection:
    """A single object detection result."""
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


class Detector:
    """
    Unified inference interface.

    Automatically picks the best available backend:
      1. ONNX  (if cfg.use_onnx and a .onnx file exists)
      2. YOLOv8 ultralytics  (if model_path ends with .pt and YOLO is available)
      3. PyTorch classifier  (checkpoint saved by train.py)
    """

    def __init__(self, cfg: InferenceConfig = CONFIG.inference) -> None:
        self._cfg = cfg
        self._device = get_device(cfg.device)
        self._model = None
        self._class_names: List[str] = CONFIG.dataset.classes
        self._backend: str = "none"

        self._load()

    # ── loading ───────────────────────────────────────────────────────────────

    def _resolve_model_path(self) -> Optional[Path]:
        if self._cfg.model_path and Path(self._cfg.model_path).exists():
            return Path(self._cfg.model_path)
        # Auto-detect latest
        for pattern, directory in [
            ("best_yolo.pt",       CONFIG.training.models_dir),
            ("best_classifier.pt", CONFIG.training.models_dir),
            ("*.pt",               CONFIG.training.checkpoint_dir),
        ]:
            p = CONFIG.training.models_dir / pattern
            if p.exists():
                return p
            found = latest_checkpoint(directory, pattern)
            if found:
                return found
        return None

    def _load(self) -> None:
        # ── ONNX ──────────────────────────────────────────────────────────────
        if self._cfg.use_onnx:
            onnx_path = self._cfg.onnx_path or (CONFIG.training.models_dir / "model.onnx")
            if onnx_path and Path(onnx_path).exists():
                self._load_onnx(Path(onnx_path))
                return
            logger.warning("ONNX requested but no .onnx file found. Trying .pt …")

        model_path = self._resolve_model_path()
        if model_path is None:
            logger.warning("No trained model found. Run training first.")
            return

        logger.info(f"Loading model from {model_path}")

        # ── YOLOv8 ────────────────────────────────────────────────────────────
        if "yolo" in model_path.name.lower():
            try:
                self._load_yolo(model_path)
                return
            except Exception as exc:
                logger.warning(f"YOLO load failed: {exc}. Trying classifier.")

        # ── PyTorch classifier ─────────────────────────────────────────────────
        self._load_classifier(model_path)

    def _load_yolo(self, path: Path) -> None:
        from ultralytics import YOLO
        self._model = YOLO(str(path))
        # Update class names from model meta if available
        if hasattr(self._model, "names"):
            self._class_names = list(self._model.names.values())
        self._backend = "yolo"
        logger.success(f"YOLOv8 detector ready — {len(self._class_names)} classes.")

    def _load_classifier(self, path: Path) -> None:
        from torchvision.models import mobilenet_v3_small
        ckpt = torch.load(path, map_location=self._device)

        if isinstance(ckpt, dict):
            class_names = ckpt.get("class_names", self._class_names)
            state = ckpt.get("model_state", ckpt)
        else:
            class_names = self._class_names
            state = ckpt

        self._class_names = class_names
        num_classes = len(class_names)

        from train import TransferClassifier
        model = TransferClassifier(num_classes=num_classes, pretrained=False)
        model.load_state_dict(state, strict=False)
        model.to(self._device).eval()

        self._model = model
        self._backend = "classifier"
        logger.success(f"Classifier detector ready — {num_classes} classes.")

    def _load_onnx(self, path: Path) -> None:
        import onnxruntime as ort
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if self._device.type == "cuda" else ["CPUExecutionProvider"])
        self._model = ort.InferenceSession(str(path), providers=providers)
        self._backend = "onnx"
        logger.success(f"ONNX runtime detector ready (providers={providers}).")

    # ── inference ─────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def predict(self, frame: np.ndarray) -> List[Detection]:
        """
        Run inference on a BGR frame.

        Returns a list of Detection objects sorted by confidence (descending).
        """
        if not self.is_ready:
            return []

        if self._backend == "yolo":
            return self._predict_yolo(frame)
        elif self._backend == "classifier":
            return self._predict_classifier(frame)
        elif self._backend == "onnx":
            return self._predict_onnx(frame)
        return []

    def _predict_yolo(self, frame: np.ndarray) -> List[Detection]:
        results = self._model(
            frame,
            conf=self._cfg.confidence_threshold,
            iou=self._cfg.iou_threshold,
            max_det=self._cfg.max_detections,
            verbose=False,
        )[0]

        detections: List[Detection] = []
        h, w = frame.shape[:2]
        for box in results.boxes:
            conf = float(box.conf[0])
            cls_idx = int(box.cls[0])
            name = results.names.get(cls_idx, str(cls_idx))
            # Filter to our target classes
            if name not in self._class_names and name not in CONFIG.dataset.classes:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            detections.append(Detection(name, conf, x1, y1, x2, y2))

        return sorted(detections, key=lambda d: d.confidence, reverse=True)

    @torch.no_grad()
    def _predict_classifier(self, frame: np.ndarray) -> List[Detection]:
        from torchvision import transforms
        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = tf(rgb).unsqueeze(0).to(self._device)
        logits = self._model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        conf, cls_idx = probs.max(0)
        conf = float(conf)
        if conf < self._cfg.confidence_threshold:
            return []
        h, w = frame.shape[:2]
        return [Detection(
            class_name=self._class_names[int(cls_idx)],
            confidence=conf,
            x1=0, y1=0, x2=w, y2=h,   # Full-frame box for classifier
        )]

    def _predict_onnx(self, frame: np.ndarray) -> List[Detection]:
        """Run ONNX inference (YOLOv8 ONNX output format)."""
        img, scale, (px, py) = resize_with_padding(frame, (640, 640))
        blob = img.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0
        input_name = self._model.get_inputs()[0].name
        outputs = self._model.run(None, {input_name: blob})
        # outputs[0] shape: (1, 84, 8400) for COCO YOLOv8
        preds = outputs[0][0].T  # (8400, 84)
        detections: List[Detection] = []
        h_orig, w_orig = frame.shape[:2]

        for row in preds:
            cls_scores = row[4:]
            cls_idx = int(cls_scores.argmax())
            conf = float(cls_scores[cls_idx])
            if conf < self._cfg.confidence_threshold:
                continue
            cx, cy, bw, bh = row[:4]
            x1 = int((cx - bw / 2 - px) / scale)
            y1 = int((cy - bh / 2 - py) / scale)
            x2 = int((cx + bw / 2 - px) / scale)
            y2 = int((cy + bh / 2 - py) / scale)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_orig, x2), min(h_orig, y2)
            name = (self._class_names[cls_idx]
                    if cls_idx < len(self._class_names) else str(cls_idx))
            detections.append(Detection(name, conf, x1, y1, x2, y2))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections[: self._cfg.max_detections]
