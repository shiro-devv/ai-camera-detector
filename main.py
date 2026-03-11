"""
ai-camera-detector — main CLI entry point.

Usage
-----
  python main.py capture    # Collect webcam images
  python main.py train      # Train model
  python main.py detect     # Run real-time detection
  python main.py export     # Export model to ONNX
  python main.py info       # Show dataset / GPU info
  python main.py app        # Interactive multi-mode app (keyboard-driven)
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from loguru import logger
from config import CONFIG
from utils import setup_logging, gpu_info


def cmd_capture(args) -> None:
    from dataset_capture import DatasetCapture
    from camera import Camera
    cam = Camera()
    cam.open()
    try:
        DatasetCapture(camera=cam).run()
    finally:
        cam.release()


def cmd_train(args) -> None:
    CONFIG.training.epochs        = args.epochs
    CONFIG.training.batch_size    = args.batch
    CONFIG.training.learning_rate = args.lr
    from train import Trainer
    trainer = Trainer()
    best = trainer.run(use_yolo=not args.no_yolo)
    logger.success(f"Training complete → {best}")


def cmd_detect(args) -> None:
    if args.model:
        CONFIG.inference.model_path = Path(args.model)
    CONFIG.inference.confidence_threshold = args.conf
    CONFIG.inference.frame_skip           = args.skip
    CONFIG.inference.use_onnx             = args.onnx
    from realtime_detection import RealtimeDetector
    RealtimeDetector().run()


def cmd_export(args) -> None:
    from utils import export_to_onnx
    src = Path(args.model) if args.model else CONFIG.training.models_dir / "best_yolo.pt"
    dst = CONFIG.training.models_dir / "model.onnx"
    export_to_onnx(src, dst, imgsz=CONFIG.training.image_size)


def cmd_info(_args) -> None:
    from dataset_loader import DatasetLoader
    print("\n── GPU Info ──────────────────────────────────────")
    print(gpu_info())
    print("\n── Dataset ───────────────────────────────────────")
    counts = DatasetLoader().scan()
    for cls, n in counts.items():
        bar = "█" * min(n, 40) + " " * max(0, 40 - n)
        print(f"  {cls:<12} [{bar}] {n}")
    print()


def cmd_app(_args) -> None:
    """Interactive multi-mode app driven by keyboard."""
    import cv2
    import numpy as np
    from camera import Camera
    from dataset_capture import DatasetCapture
    from realtime_detection import RealtimeDetector
    from utils import draw_overlay_text, FPSCounter

    logger.info("Launching interactive app …")

    KEY_QUIT  = ord("q")
    KEY_ESC   = 27
    KEY_CAP   = ord("c")
    KEY_DET   = ord("d")
    KEY_TRAIN = ord("t")

    MODE_MENU = "menu"
    MODE_CAP  = "capture"
    MODE_DET  = "detect"

    cam = Camera()
    cam.open()
    mode = MODE_MENU
    fps = FPSCounter()

    window = "AI Camera Detector — Interactive"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = cam.read_safe()
            if frame is None:
                continue

            display = frame.copy()
            fps.tick()

            if mode == MODE_MENU:
                # Draw menu overlay
                h, w = display.shape[:2]
                overlay = display.copy()
                cv2.rectangle(overlay, (w//4, h//4), (3*w//4, 3*h//4), (20, 20, 20), -1)
                display = cv2.addWeighted(overlay, 0.7, display, 0.3, 0)

                lines = [
                    "AI CAMERA DETECTOR",
                    "",
                    "C  →  Capture Dataset",
                    "D  →  Real-Time Detection",
                    "T  →  Train Model",
                    "Q  →  Quit",
                ]
                for i, line in enumerate(lines):
                    draw_overlay_text(
                        display, line,
                        (w//4 + 20, h//4 + 40 + i * 36),
                        font_scale=0.9 if i == 0 else 0.7,
                        color=(0, 255, 180) if i == 0 else (220, 220, 220),
                    )

            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (KEY_QUIT, KEY_ESC):
                break
            elif key == KEY_CAP:
                cam.release()
                cam.open()
                DatasetCapture(camera=cam).run()
            elif key == KEY_DET:
                cam.release()
                cam.open()
                RealtimeDetector(camera=cam).run()
            elif key == KEY_TRAIN:
                logger.info("Starting training in background …")
                cv2.destroyWindow(window)
                from train import Trainer
                Trainer().run()
                cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                cam.open()

    finally:
        cam.release()
        cv2.destroyAllWindows()


# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="ai-detector",
        description="Real-time webcam object detector — collect, train, detect.",
    )
    root.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    sub = root.add_subparsers(dest="command")

    # capture
    sub.add_parser("capture", help="Collect labelled images from webcam.")

    # train
    tr = sub.add_parser("train", help="Train / fine-tune the detection model.")
    tr.add_argument("--epochs",   type=int,   default=CONFIG.training.epochs)
    tr.add_argument("--batch",    type=int,   default=CONFIG.training.batch_size)
    tr.add_argument("--lr",       type=float, default=CONFIG.training.learning_rate)
    tr.add_argument("--no-yolo",  action="store_true", help="Use PyTorch classifier instead of YOLOv8.")

    # detect
    det = sub.add_parser("detect", help="Run real-time object detection.")
    det.add_argument("--model",  type=str,   default=None)
    det.add_argument("--conf",   type=float, default=CONFIG.inference.confidence_threshold)
    det.add_argument("--skip",   type=int,   default=CONFIG.inference.frame_skip)
    det.add_argument("--onnx",   action="store_true")

    # export
    exp = sub.add_parser("export", help="Export model to ONNX.")
    exp.add_argument("--model", type=str, default=None)

    # info
    sub.add_parser("info", help="Show dataset stats and GPU info.")

    # app
    sub.add_parser("app", help="Launch interactive multi-mode app.")

    return root


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(CONFIG.training.logs_dir, level=args.log_level)

    dispatch = {
        "capture": cmd_capture,
        "train":   cmd_train,
        "detect":  cmd_detect,
        "export":  cmd_export,
        "info":    cmd_info,
        "app":     cmd_app,
        None:      lambda _: parser.print_help(),
    }
    dispatch.get(args.command, lambda _: parser.print_help())(args)


if __name__ == "__main__":
    main()
