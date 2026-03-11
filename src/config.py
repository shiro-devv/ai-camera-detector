"""
Configuration module for the AI Camera Detector system.
Centralizes all hyperparameters and system settings.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import yaml


# ── Project root ──────────────────────────────────────────────────────────────
ROOT_DIR: Path = Path(__file__).resolve().parent.parent


@dataclass
class CameraConfig:
    """Webcam capture settings."""
    device_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    buffer_size: int = 1  # Minimize latency


@dataclass
class DatasetConfig:
    """Dataset collection and processing settings."""
    dataset_dir: Path = ROOT_DIR / "dataset"
    classes: List[str] = field(default_factory=lambda: [
        "human", "apple", "banana", "orange",
        "cat", "dog", "bottle", "phone", "keyboard", "laptop"
    ])
    image_size: Tuple[int, int] = (640, 640)
    val_split: float = 0.2
    min_images_per_class: int = 20
    augmentation_enabled: bool = True


@dataclass
class TrainingConfig:
    """Model training hyperparameters."""
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    image_size: int = 640
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    patience: int = 10                      # Early stopping patience
    warmup_epochs: int = 3
    model_name: str = "yolov8n.pt"          # nano = fastest; also s/m/l/x
    pretrained: bool = True
    freeze_backbone_epochs: int = 5         # Freeze backbone for first N epochs
    checkpoint_dir: Path = ROOT_DIR / "checkpoints"
    models_dir: Path = ROOT_DIR / "models"
    logs_dir: Path = ROOT_DIR / "logs"
    save_every_n_epochs: int = 5
    use_amp: bool = True                    # Automatic Mixed Precision
    num_workers: int = 4


@dataclass
class InferenceConfig:
    """Real-time inference settings."""
    model_path: Optional[Path] = None       # None → auto-detect latest
    confidence_threshold: float = 0.4
    iou_threshold: float = 0.45
    max_detections: int = 100
    frame_skip: int = 0                     # Skip N frames between inferences
    use_onnx: bool = False
    onnx_path: Optional[Path] = None
    device: str = "auto"                    # "auto" | "cuda" | "cpu"


@dataclass
class AppConfig:
    """Top-level application configuration."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # UI colours per class (BGR)
    class_colors: dict = field(default_factory=lambda: {
        "human":    (0,   255, 0),
        "apple":    (0,   0,   255),
        "banana":   (0,   255, 255),
        "orange":   (0,   128, 255),
        "cat":      (255, 0,   0),
        "dog":      (255, 0,   128),
        "bottle":   (128, 255, 0),
        "phone":    (255, 255, 0),
        "keyboard": (128, 0,   255),
        "laptop":   (0,   128, 128),
    })


# ── YAML persistence ──────────────────────────────────────────────────────────

def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """Load configuration from YAML file, falling back to defaults."""
    cfg = AppConfig()
    if config_path and config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        # Shallow merge for each section
        for section, section_cfg in [
            ("camera",   cfg.camera),
            ("dataset",  cfg.dataset),
            ("training", cfg.training),
            ("inference", cfg.inference),
        ]:
            if section in data:
                for k, v in data[section].items():
                    if hasattr(section_cfg, k):
                        setattr(section_cfg, k, v)
    return cfg


def save_config(cfg: AppConfig, config_path: Path) -> None:
    """Persist configuration to YAML."""
    import dataclasses
    def _serialize(obj):
        if dataclasses.is_dataclass(obj):
            return {k: _serialize(v) for k, v in dataclasses.asdict(obj).items()}
        if isinstance(obj, Path):
            return str(obj)
        return obj

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(_serialize(cfg), f, default_flow_style=False)


# ── Singleton ─────────────────────────────────────────────────────────────────
_CONFIG_PATH = ROOT_DIR / "config.yaml"
CONFIG: AppConfig = load_config(_CONFIG_PATH)
