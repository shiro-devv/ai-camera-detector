"""
Utility helpers: logging, device selection, filesystem, timing, ONNX export.
"""

import time
import shutil
import hashlib
from pathlib import Path
from typing import Optional, Tuple, List
from contextlib import contextmanager

import cv2
import numpy as np
import torch
from loguru import logger


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(log_dir: Path, level: str = "INFO") -> None:
    """Configure loguru to write both to stderr and a rotating log file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=level,
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )
    logger.add(
        log_dir / "detector_{time:YYYY-MM-DD}.log",
        rotation="50 MB",
        retention="7 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} | {message}",
    )


# ── Device ────────────────────────────────────────────────────────────────────

def get_device(preference: str = "auto") -> torch.device:
    """Return the best available torch device."""
    if preference == "cuda" or (preference == "auto" and torch.cuda.is_available()):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            return device
        logger.warning("CUDA requested but not available — falling back to CPU.")
    if preference == "mps" or (preference == "auto" and torch.backends.mps.is_available()):
        logger.info("Using Apple MPS backend.")
        return torch.device("mps")
    logger.info("Using CPU.")
    return torch.device("cpu")


def gpu_info() -> str:
    """Return a formatted string with GPU info."""
    if not torch.cuda.is_available():
        return "No CUDA GPU detected."
    lines = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / 1024 ** 3
        lines.append(f"  [{i}] {props.name}  {mem_gb:.1f} GB  CC={props.major}.{props.minor}")
    return "\n".join(lines)


# ── Filesystem ────────────────────────────────────────────────────────────────

def ensure_dirs(*dirs: Path) -> None:
    """Create directories if they don't exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def latest_checkpoint(checkpoint_dir: Path, glob: str = "*.pt") -> Optional[Path]:
    """Return the most recently modified checkpoint, or None."""
    checkpoints = sorted(Path(checkpoint_dir).glob(glob), key=lambda p: p.stat().st_mtime)
    return checkpoints[-1] if checkpoints else None


def file_hash(path: Path, algo: str = "md5") -> str:
    """Compute file hash for integrity checks."""
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Timing ────────────────────────────────────────────────────────────────────

class FPSCounter:
    """Rolling-average FPS counter."""

    def __init__(self, window: int = 30) -> None:
        self._times: List[float] = []
        self._window = window

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])


@contextmanager
def timer(label: str = ""):
    """Simple context-manager stopwatch."""
    t0 = time.perf_counter()
    yield
    elapsed = (time.perf_counter() - t0) * 1000
    logger.debug(f"{label}: {elapsed:.1f} ms")


# ── Image helpers ─────────────────────────────────────────────────────────────

def resize_with_padding(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    pad_color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Letterbox resize keeping aspect ratio.

    Returns:
        resized image, scale factor, (pad_w, pad_h)
    """
    h, w = image.shape[:2]
    tw, th = target_size
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((th, tw, 3), pad_color, dtype=np.uint8)
    pad_x, pad_y = (tw - nw) // 2, (th - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, scale, (pad_x, pad_y)


def draw_overlay_text(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int] = (10, 30),
    font_scale: float = 0.8,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> None:
    """Draw text with a dark background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    cv2.rectangle(frame, (x - 2, y - th - 4), (x + tw + 2, y + baseline + 2), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_detection(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    label: str,
    confidence: float,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> None:
    """Draw a bounding box + label on the frame (in-place)."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    caption = f"{label} {confidence:.0%}"
    (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, caption, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)


# ── ONNX export ───────────────────────────────────────────────────────────────

def export_to_onnx(model_path: Path, output_path: Path, imgsz: int = 640) -> Path:
    """
    Export a trained Ultralytics model to ONNX format.

    Args:
        model_path: Path to the .pt weights file.
        output_path: Destination for the .onnx file.
        imgsz: Input image size.

    Returns:
        Path to the exported ONNX file.
    """
    try:
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        exported = model.export(format="onnx", imgsz=imgsz, simplify=True)
        src = Path(exported)
        shutil.copy(src, output_path)
        logger.success(f"ONNX model saved to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
        return output_path
    except Exception as exc:
        logger.error(f"ONNX export failed: {exc}")
        raise
