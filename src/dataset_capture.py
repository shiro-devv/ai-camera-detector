"""
Interactive dataset capture tool.

Controls
--------
S          – Capture image into current class folder
C          – Cycle to next class
A          – Toggle auto-labelling via a pretrained YOLOv8n (if ultralytics available)
Q / ESC    – Quit
"""

from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
import sys
import time

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from loguru import logger

# Make src/ importable when run directly
sys.path.insert(0, str(Path(__file__).parent))

from camera import Camera
from config import CONFIG, DatasetConfig
from utils import draw_overlay_text, ensure_dirs, setup_logging


class DatasetCapture:
    """Captures labelled images from the webcam into class-specific folders."""

    # Key bindings
    KEY_SAVE       = ord("s")
    KEY_NEXT_CLASS = ord("c")
    KEY_AUTO       = ord("a")
    KEY_QUIT_Q     = ord("q")
    KEY_QUIT_ESC   = 27

    def __init__(
        self,
        cfg: DatasetConfig = CONFIG.dataset,
        camera: Optional[Camera] = None,
    ) -> None:
        self._cfg = cfg
        self._camera = camera or Camera()
        self._classes: List[str] = list(cfg.classes)
        self._class_idx: int = 0
        self._auto_label: bool = False
        self._auto_model = None
        self._counters: dict[str, int] = {}

        ensure_dirs(*[cfg.dataset_dir / cls for cls in self._classes])
        self._init_counters()

    # ── init ──────────────────────────────────────────────────────────────────

    def _init_counters(self) -> None:
        """Count existing images to avoid overwriting."""
        for cls in self._classes:
            folder = self._cfg.dataset_dir / cls
            self._counters[cls] = len(list(folder.glob("*.jpg")))
        logger.info("Existing image counts: " +
                    ", ".join(f"{c}={n}" for c, n in self._counters.items()))

    def _load_auto_model(self) -> bool:
        """Attempt to load a pretrained YOLOv8n for auto-labelling."""
        try:
            from ultralytics import YOLO
            logger.info("Loading pretrained YOLOv8n for auto-labelling …")
            self._auto_model = YOLO("yolov8n.pt")
            logger.success("Auto-label model loaded.")
            return True
        except Exception as exc:
            logger.warning(f"Could not load auto-label model: {exc}")
            return False

    # ── capture ───────────────────────────────────────────────────────────────

    @property
    def current_class(self) -> str:
        return self._classes[self._class_idx]

    def _save_frame(self, frame: np.ndarray) -> None:
        cls = self.current_class
        folder = self._cfg.dataset_dir / cls
        self._counters[cls] += 1
        filename = folder / f"{cls}_{self._counters[cls]:05d}.jpg"
        cv2.imwrite(str(filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        logger.success(f"Saved [{cls}] → {filename.name}")

    def _auto_save(self, frame: np.ndarray) -> None:
        """Use the pretrained model to detect and save matching classes."""
        if self._auto_model is None:
            return
        try:
            results = self._auto_model(frame, verbose=False)[0]
            for box in results.boxes:
                label = results.names[int(box.cls[0])]
                if label in self._classes:
                    # Temporarily switch class index for saving
                    orig = self._class_idx
                    self._class_idx = self._classes.index(label)
                    self._save_frame(frame)
                    self._class_idx = orig
        except Exception as exc:
            logger.warning(f"Auto-label inference error: {exc}")

    # ── rendering ─────────────────────────────────────────────────────────────

    def _render_hud(self, frame: np.ndarray) -> np.ndarray:
        display = frame.copy()
        h, w = display.shape[:2]
        cls = self.current_class
        count = self._counters[cls]

        # Class indicator bar
        cv2.rectangle(display, (0, h - 45), (w, h), (0, 0, 0), -1)
        draw_overlay_text(display, f"Class: {cls.upper()}  ({count} images)",
                          (10, h - 15), font_scale=0.7, color=(0, 255, 100))

        # Controls hint
        hints = "S=Capture  C=NextClass  A=AutoLabel  Q=Quit"
        draw_overlay_text(display, hints, (10, h - 65),
                          font_scale=0.5, color=(200, 200, 200))

        # Auto-label indicator
        if self._auto_label:
            draw_overlay_text(display, "AUTO-LABEL ON", (w - 160, 30),
                              font_scale=0.6, color=(0, 100, 255))

        # Flash feedback border on save
        return display

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the interactive capture session."""
        own_camera = not self._camera.is_open
        if own_camera:
            self._camera.open()

        logger.info("Dataset capture started. Press S to capture, C to change class, Q to quit.")
        window_name = "Dataset Capture"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        flash_until: float = 0.0

        try:
            for frame in self._camera.stream():
                display = self._render_hud(frame)

                # Green flash feedback
                if time.time() < flash_until:
                    cv2.rectangle(display, (0, 0),
                                  (display.shape[1], display.shape[0]),
                                  (0, 255, 0), 6)

                cv2.imshow(window_name, display)
                key = cv2.waitKey(1) & 0xFF

                if key in (self.KEY_QUIT_Q, self.KEY_QUIT_ESC):
                    break

                elif key == self.KEY_SAVE:
                    self._save_frame(frame)
                    flash_until = time.time() + 0.15

                elif key == self.KEY_NEXT_CLASS:
                    self._class_idx = (self._class_idx + 1) % len(self._classes)
                    logger.info(f"Class → {self.current_class}")

                elif key == self.KEY_AUTO:
                    self._auto_label = not self._auto_label
                    if self._auto_label and self._auto_model is None:
                        if not self._load_auto_model():
                            self._auto_label = False

                if self._auto_label:
                    self._auto_save(frame)

        finally:
            cv2.destroyWindow(window_name)
            if own_camera:
                self._camera.release()

        self._print_summary()

    def _print_summary(self) -> None:
        total = sum(self._counters.values())
        logger.info(f"Capture session complete — {total} total images.")
        for cls in self._classes:
            logger.info(f"  {cls:<12}: {self._counters[cls]} images")


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    setup_logging(CONFIG.training.logs_dir)
    capture = DatasetCapture()
    capture.run()
