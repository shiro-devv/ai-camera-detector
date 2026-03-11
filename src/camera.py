"""
Camera module — wraps OpenCV VideoCapture with graceful error handling,
configurable resolution/FPS, and a context-manager interface.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple, Generator

import cv2
import numpy as np
from loguru import logger

from config import CameraConfig, CONFIG


class CameraError(RuntimeError):
    """Raised when the webcam cannot be opened or read."""


class Camera:
    """
    OpenCV webcam wrapper.

    Usage
    -----
    >>> with Camera() as cam:
    ...     for frame in cam.stream():
    ...         process(frame)
    """

    def __init__(self, cfg: CameraConfig = CONFIG.camera) -> None:
        self._cfg = cfg
        self._cap: Optional[cv2.VideoCapture] = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def open(self) -> None:
        """Open the webcam; raises CameraError on failure."""
        logger.info(f"Opening camera index={self._cfg.device_index} "
                    f"@ {self._cfg.width}×{self._cfg.height} {self._cfg.fps}fps")
        cap = cv2.VideoCapture(self._cfg.device_index, cv2.CAP_ANY)
        if not cap.isOpened():
            raise CameraError(f"Cannot open camera {self._cfg.device_index}.")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.height)
        cap.set(cv2.CAP_PROP_FPS,          self._cfg.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   self._cfg.buffer_size)

        # Try MJPEG for higher throughput
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        # Verify actual resolution (driver may cap it)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Camera opened: {actual_w}×{actual_h} @ {actual_fps:.0f}fps")

        self._cap = cap

    def release(self) -> None:
        """Release the camera resource."""
        if self._cap and self._cap.isOpened():
            self._cap.release()
            logger.debug("Camera released.")
        self._cap = None

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.release()

    # ── frame access ──────────────────────────────────────────────────────────

    def read(self) -> np.ndarray:
        """
        Capture and return a single BGR frame.

        Raises
        ------
        CameraError
            If the camera is not open or the frame cannot be read.
        """
        if self._cap is None or not self._cap.isOpened():
            raise CameraError("Camera is not open. Call open() first.")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise CameraError("Failed to capture frame.")
        return frame

    def read_safe(self) -> Optional[np.ndarray]:
        """Return a frame, or None on error (no exception)."""
        try:
            return self.read()
        except CameraError as exc:
            logger.warning(f"Frame read error: {exc}")
            return None

    def stream(
        self,
        max_retries: int = 5,
        retry_delay: float = 0.5,
    ) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields frames continuously.

        Parameters
        ----------
        max_retries:
            Number of consecutive read failures before giving up.
        retry_delay:
            Seconds to wait between retries.
        """
        consecutive_failures = 0
        while True:
            frame = self.read_safe()
            if frame is None:
                consecutive_failures += 1
                logger.warning(f"Read failure {consecutive_failures}/{max_retries}")
                if consecutive_failures >= max_retries:
                    raise CameraError("Too many consecutive read failures.")
                time.sleep(retry_delay)
                continue
            consecutive_failures = 0
            yield frame

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def resolution(self) -> Tuple[int, int]:
        """Return (width, height) from the capture object."""
        if self._cap is None:
            return (self._cfg.width, self._cfg.height)
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    # ── static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def list_devices(max_index: int = 8) -> list[int]:
        """Probe device indices and return those that open successfully."""
        available = []
        for i in range(max_index):
            cap = cv2.VideoCapture(i, cv2.CAP_ANY)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available
