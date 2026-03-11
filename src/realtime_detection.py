"""
Real-time detection engine.

Continuously reads webcam frames, runs inference, and displays
annotated video with bounding boxes, labels, confidence scores,
and an FPS counter.

Keyboard controls
-----------------
Q / ESC → Quit
S       → Save current annotated frame to disk
R       → Reload model (if you retrained without restarting)
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

sys.path.insert(0, str(Path(__file__).parent))
from camera import Camera
from config import CONFIG, InferenceConfig
from detector import Detection, Detector
from utils import FPSCounter, draw_detection, draw_overlay_text, ensure_dirs, setup_logging


# ── Colour palette ────────────────────────────────────────────────────────────

_COLORS: dict[str, tuple[int, int, int]] = CONFIG.class_colors
_FALLBACK_COLOR = (0, 200, 255)


def _color(class_name: str) -> tuple[int, int, int]:
    return _COLORS.get(class_name, _FALLBACK_COLOR)


# ── Real-time runner ──────────────────────────────────────────────────────────

class RealtimeDetector:
    """
    Reads frames from the webcam, runs the detector, and displays results.
    """

    KEY_QUIT_Q  = ord("q")
    KEY_QUIT_ESC = 27
    KEY_SAVE    = ord("s")
    KEY_RELOAD  = ord("r")

    def __init__(
        self,
        cfg: InferenceConfig = CONFIG.inference,
        camera: Optional[Camera] = None,
    ) -> None:
        self._cfg = cfg
        self._camera = camera or Camera()
        self._detector = Detector(cfg)
        self._fps = FPSCounter(window=30)
        self._frame_count = 0
        self._last_detections: List[Detection] = []
        self._save_dir = Path("detections")
        ensure_dirs(self._save_dir)

    # ── rendering ─────────────────────────────────────────────────────────────

    def _annotate(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        fps: float,
    ) -> np.ndarray:
        display = frame.copy()

        for det in detections:
            draw_detection(
                display,
                det.x1, det.y1, det.x2, det.y2,
                det.class_name,
                det.confidence,
                _color(det.class_name),
            )

        # FPS counter
        draw_overlay_text(display, f"FPS: {fps:.1f}", (10, 30),
                          font_scale=0.7, color=(0, 255, 200))

        # Detection count
        draw_overlay_text(display, f"Objects: {len(detections)}",
                          (10, 60), font_scale=0.7, color=(200, 200, 0))

        # Backend / device hint
        backend_label = f"Backend: {self._detector._backend.upper()}  "  # noqa
        backend_label += f"Device: {self._detector._device.type.upper()}"
        h = display.shape[0]
        draw_overlay_text(display, backend_label, (10, h - 15),
                          font_scale=0.5, color=(180, 180, 180))

        # Controls hint
        draw_overlay_text(display, "S=Save  R=Reload  Q=Quit",
                          (10, h - 40), font_scale=0.5, color=(150, 150, 150))

        return display

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the real-time detection loop."""
        if not self._detector.is_ready:
            logger.error("No trained model found. Please run training first.")
            return

        own_camera = not self._camera.is_open
        if own_camera:
            self._camera.open()

        window = "AI Object Detector"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        logger.info("Real-time detection started. Press Q to quit.")

        skip = max(0, self._cfg.frame_skip)

        try:
            for frame in self._camera.stream():
                self._frame_count += 1
                self._fps.tick()

                # Frame skipping for higher display FPS
                if skip > 0 and (self._frame_count % (skip + 1) != 0):
                    detections = self._last_detections
                else:
                    detections = self._detector.predict(frame)
                    self._last_detections = detections

                fps = self._fps.tick()   # second tick just reads value
                display = self._annotate(frame, detections, fps)
                cv2.imshow(window, display)

                key = cv2.waitKey(1) & 0xFF
                if key in (self.KEY_QUIT_Q, self.KEY_QUIT_ESC):
                    break
                elif key == self.KEY_SAVE:
                    self._save_frame(display)
                elif key == self.KEY_RELOAD:
                    logger.info("Reloading model …")
                    self._detector = Detector(self._cfg)

        finally:
            cv2.destroyWindow(window)
            if own_camera:
                self._camera.release()

        logger.info(f"Detection session ended. {self._frame_count} frames processed.")

    def _save_frame(self, frame: np.ndarray) -> None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = self._save_dir / f"detection_{ts}.jpg"
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        logger.success(f"Frame saved → {path}")


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    setup_logging(CONFIG.training.logs_dir)

    ap = argparse.ArgumentParser(description="Run real-time object detection.")
    ap.add_argument("--model",      type=str,   default=None, help="Path to .pt or .onnx model.")
    ap.add_argument("--conf",       type=float, default=CONFIG.inference.confidence_threshold)
    ap.add_argument("--iou",        type=float, default=CONFIG.inference.iou_threshold)
    ap.add_argument("--skip",       type=int,   default=CONFIG.inference.frame_skip)
    ap.add_argument("--onnx",       action="store_true", help="Use ONNX runtime.")
    ap.add_argument("--camera",     type=int,   default=CONFIG.camera.device_index)
    args = ap.parse_args()

    CONFIG.inference.confidence_threshold = args.conf
    CONFIG.inference.iou_threshold        = args.iou
    CONFIG.inference.frame_skip           = args.skip
    CONFIG.inference.use_onnx             = args.onnx
    if args.model:
        CONFIG.inference.model_path       = Path(args.model)
    CONFIG.camera.device_index            = args.camera

    rd = RealtimeDetector()
    rd.run()
