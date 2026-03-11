# 🎯 AI Camera Detector

A **production-grade, real-time object detection system** that lets you:
1. **Collect** labelled images straight from your webcam  
2. **Train** a YOLOv8 model (or a lightweight PyTorch classifier) on your data  
3. **Detect** objects in real time with bounding boxes, confidence scores, and FPS  

Detectable classes: `human · apple · banana · orange · cat · dog · bottle · phone · keyboard · laptop`  
(Easily extensible — see [Adding New Classes](#adding-new-classes).)

---

## Project Structure

```
ai-camera-detector/
├── dataset/            # Per-class image folders (auto-created)
│   ├── human/
│   ├── apple/
│   └── …
├── models/             # Saved model weights (.pt, .onnx)
├── checkpoints/        # Epoch checkpoints
├── logs/               # TensorBoard logs + loguru logs
├── src/
│   ├── config.py           # Centralised config & hyperparameters
│   ├── camera.py           # OpenCV webcam wrapper
│   ├── dataset_capture.py  # Interactive capture tool
│   ├── dataset_loader.py   # DataLoaders + YOLO YAML generator
│   ├── train.py            # Training pipeline (YOLOv8 + PyTorch)
│   ├── detector.py         # Inference engine (YOLO / classifier / ONNX)
│   ├── realtime_detection.py # Live detection loop
│   └── utils.py            # Logging, FPS, drawing helpers, ONNX export
├── main.py             # Unified CLI entry point
├── requirements.txt
└── README.md
```

---

## 1 · Linux Setup

### System dependencies

```bash
sudo apt update && sudo apt install -y \
    python3-pip python3-venv git \
    libgl1-mesa-glx libglib2.0-0 \
    v4l-utils      # webcam diagnostics
```

### Python environment

```bash
git clone https://github.com/yourname/ai-camera-detector.git
cd ai-camera-detector

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2 · GPU Setup (CUDA)

Install the CUDA-enabled PyTorch build **before** running `pip install -r requirements.txt`:

```bash
# PyTorch with CUDA 12.1
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Then install the rest
pip install -r requirements.txt

# Verify
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

For ONNX GPU inference, swap `onnxruntime` for `onnxruntime-gpu`:

```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

---

## 3 · Collect Dataset Images

```bash
python main.py capture
```

| Key | Action |
|-----|--------|
| `S` | Save current frame to active class folder |
| `C` | Cycle to next class |
| `A` | Toggle auto-labelling via pretrained YOLOv8n |
| `Q` / `ESC` | Quit |

Aim for **≥ 50 images per class** for good results.  
Vary distance, angle, background, and lighting.

Check what you've collected:

```bash
python main.py info
```

---

## 4 · Train the Model

### Recommended — YOLOv8 fine-tuning

```bash
python main.py train \
    --epochs 50 \
    --batch  16 \
    --lr     0.0001
```

### Fallback — lightweight PyTorch classifier

```bash
python main.py train --no-yolo --epochs 30
```

Monitor training in TensorBoard:

```bash
tensorboard --logdir logs/tensorboard
# open http://localhost:6006
```

Outputs saved to:
- `models/best_yolo.pt`  (or `best_classifier.pt`)
- `checkpoints/epoch_XXXX.pt`

---

## 5 · Real-Time Detection

```bash
python main.py detect
```

**Advanced options:**

```bash
python main.py detect \
    --model  models/best_yolo.pt \
    --conf   0.45 \
    --skip   1        # skip 1 frame between inferences (doubles display FPS)
```

| Key | Action |
|-----|--------|
| `S` | Save annotated frame |
| `R` | Reload model (pick up retrained weights) |
| `Q` / `ESC` | Quit |

---

## 6 · Export to ONNX

```bash
python main.py export                          # auto-detects best_yolo.pt
python main.py export --model models/best_yolo.pt

# Then run detection with ONNX
python main.py detect --onnx
```

---

## 7 · Interactive App (all modes in one window)

```bash
python main.py app
```

Launches a menu-driven interface — press `C`, `D`, or `T` from the menu.

---

## 8 · Configuration

Edit `config.yaml` (auto-generated on first run) or `src/config.py` directly.

Key hyperparameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 50 | Training epochs |
| `batch_size` | 16 | Images per batch |
| `learning_rate` | 1e-4 | Initial LR |
| `image_size` | 640 | Input resolution (px) |
| `confidence_threshold` | 0.25 | Detection threshold |
| `frame_skip` | 0 | Frames skipped between inference calls |
| `use_amp` | True | Automatic Mixed Precision |

---

## Adding New Classes

1. Add the class name to `classes` in `src/config.py`:

   ```python
   classes: List[str] = [..., "scissors"]
   ```

2. Create the folder:

   ```bash
   mkdir dataset/scissors
   ```

3. Re-run capture → training → detection.

---

## Command Reference

```bash
python main.py capture                    # Collect images
python main.py train  [--epochs N] [--batch N] [--lr F] [--no-yolo]
python main.py detect [--model PATH] [--conf F] [--skip N] [--onnx]
python main.py export [--model PATH]
python main.py info                       # Dataset + GPU summary
python main.py app                        # Interactive all-in-one UI
```

---

## Performance Tips

| Technique | How |
|-----------|-----|
| GPU acceleration | Install CUDA PyTorch (§2) |
| Higher display FPS | `--skip 1` or `--skip 2` |
| Fastest inference | Export to ONNX + `--onnx` |
| Smaller model | Change `model_name = "yolov8n.pt"` (nano) |
| More accuracy | Change to `"yolov8s.pt"` or `"yolov8m.pt"` |

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Primary training & inference | PyTorch + Ultralytics YOLOv8 |
| Camera capture | OpenCV |
| Data augmentation | torchvision transforms + albumentations |
| ONNX export | ultralytics export + onnxruntime |
| Logging | loguru + TensorBoard |
| Config | PyYAML + Python dataclasses |

---

## License

MIT
