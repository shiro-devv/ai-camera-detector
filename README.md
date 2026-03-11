# 🎯 AI Camera Detector

A **production-grade, real-time object detection system** that lets you:

1. **Collect** labelled images straight from your webcam
2. **Train** a YOLOv8 model (or a lightweight PyTorch classifier) on your own data
3. **Detect** objects in real time with bounding boxes, confidence scores, and an FPS counter

Default detectable classes:
`human · apple · banana · orange · cat · dog · bottle · phone · keyboard · laptop`

You can add any class you want — see [Adding New Classes](#-adding-new-classes).

---

## Table of Contents

- [Project Structure](#-project-structure)
- [Arch Linux Setup](#-arch-linux-setup)
- [Linux Ubuntu / Debian Setup](#-linux-ubuntu--debian-setup)
- [Windows Setup](#-windows-setup)
- [GPU Setup CUDA](#-gpu-setup-cuda)
- [Known Issues and Fixes](#-known-issues--fixes)
- [Collecting Your Dataset](#-collecting-your-dataset)
- [Training the Model](#-training-the-model)
- [Real-Time Detection](#-real-time-detection)
- [Export to ONNX](#-export-to-onnx)
- [Configuration](#-configuration)
- [Adding New Classes](#-adding-new-classes)
- [Command Reference](#-command-reference)
- [Performance Tips](#-performance-tips)
- [Tech Stack](#-tech-stack)

---

## 📁 Project Structure

```
ai-camera-detector/
├── dataset/                  # Per-class image folders (auto-created)
│   ├── human/
│   ├── apple/
│   ├── banana/
│   └── …
├── models/                   # Saved model weights (.pt, .onnx)
├── checkpoints/              # Epoch-level checkpoints
├── logs/                     # TensorBoard + loguru log files
├── src/
│   ├── config.py             # All hyperparameters and settings
│   ├── camera.py             # OpenCV webcam wrapper
│   ├── dataset_capture.py    # Interactive image capture tool
│   ├── dataset_loader.py     # PyTorch DataLoaders + YOLO YAML
│   ├── train.py              # Training pipeline
│   ├── detector.py           # Inference engine (YOLO / classifier / ONNX)
│   ├── realtime_detection.py # Live detection loop
│   └── utils.py              # Helpers: logging, FPS, drawing, ONNX export
├── main.py                   # Unified CLI entry point
├── config.yaml               # User-editable config file
├── requirements.txt
└── README.md
```

---

## 🐧 Arch Linux Setup

### Python version requirement

> ⚠️ PyTorch does **not** support Python 3.14 yet. You must use **Python 3.12**.

```bash
# Check your version
python --version

# If it shows 3.14, install 3.12 from AUR
yay -S python312

# Verify
python3.12 --version
```

### System dependencies

```bash
# Core packages
sudo pacman -S git opencv python-opencv v4l-utils

# Qt / display support
sudo pacman -S qt6-wayland qt5-wayland

# Fonts (fixes QFontDatabase warnings)
sudo pacman -S ttf-dejavu

# XWayland (needed for Wayland compositors: Niri, Sway, Hyprland)
sudo pacman -S xorg-xwayland
```

### Python virtual environment

```bash
git clone https://github.com/shiro-devv/ai-camera-detector.git
cd ai-camera-detector

# Use --system-site-packages so the venv can see system opencv
python -m venv --system-site-packages environment
source environment/bin/activate

pip install --upgrade pip

# CPU-only PyTorch (safe default — add CUDA later if you have an NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (TensorFlow is optional and very large — skip it)
pip install ultralytics numpy pandas matplotlib tqdm \
    scikit-learn scipy albumentations onnx onnxruntime \
    tensorboard pyyaml pillow rich loguru
```

### Wayland / Display setup

Arch users on Wayland compositors need to tell OpenCV which display backend to use.
Add these to your shell config (`~/.bashrc`, `~/.zshrc`, or `~/.profile`):

```bash
export DISPLAY=:1              # check yours with: ps aux | grep Xwayland
export QT_QPA_PLATFORM=xcb
```

Or set them inline each time:

```bash
DISPLAY=:1 QT_QPA_PLATFORM=xcb python main.py capture
```

#### How to find your XWayland display number

```bash
ps aux | grep Xwayland
# Look for: Xwayland :1  (the :1 is your display number)
```

#### Compositor-specific notes

| Compositor | Notes |
|------------|-------|
| **Niri** | XWayland runs automatically. Check display with `ps aux \| grep Xwayland` |
| **Sway** | Add `xwayland enable` to `~/.config/sway/config` |
| **Hyprland** | Add `xwayland { enabled = true }` to `hyprland.conf` |
| **KDE Plasma Wayland** | Usually works with `QT_QPA_PLATFORM=wayland` |
| **GNOME Wayland** | Usually works out of the box with `DISPLAY=:0` |

### Verify everything works

```bash
source environment/bin/activate
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "from ultralytics import YOLO; print('Ultralytics OK')"
```

---

## 🐧 Linux (Ubuntu / Debian) Setup

### System dependencies

```bash
sudo apt update && sudo apt install -y \
    python3.12 python3.12-venv python3-pip \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 libxrender1 libxext6 \
    v4l-utils \
    fonts-dejavu
```

### Python virtual environment

```bash
git clone https://github.com/yourname/ai-camera-detector.git
cd ai-camera-detector

python3.12 -m venv environment
source environment/bin/activate

pip install --upgrade pip

# CPU-only PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Remaining dependencies
pip install -r requirements.txt
```

### Display setup (Ubuntu with Wayland)

Ubuntu 22.04+ defaults to Wayland. If the webcam window fails to open:

```bash
# Option 1 — force X11 at the login screen (most reliable)
# At the login screen click the gear icon → select "Ubuntu on Xorg"

# Option 2 — set env vars in terminal
export QT_QPA_PLATFORM=xcb
export DISPLAY=:0
python main.py capture
```

### Verify

```bash
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__)"
```

---

## 🪟 Windows Setup

### Requirements

- Windows 10 or 11 (64-bit)
- Python 3.11 or 3.12 from [python.org](https://www.python.org/downloads/)
  - During install, check **"Add Python to PATH"**
- Git from [git-scm.com](https://git-scm.com)
- A working webcam

### Install steps

Open **Command Prompt** or **PowerShell**:

```powershell
git clone https://github.com/yourname/ai-camera-detector.git
cd ai-camera-detector

# Create virtual environment
python -m venv environment
environment\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# CPU PyTorch
pip install torch torchvision torchaudio

# Remaining dependencies
pip install ultralytics numpy pandas matplotlib tqdm scikit-learn scipy `
    albumentations onnx onnxruntime tensorboard pyyaml pillow rich loguru
```

### Running on Windows

No display environment variables needed — OpenCV windows work natively:

```powershell
# Activate environment (required every new terminal session)
environment\Scripts\activate

# Collect images
python main.py capture

# Train
python main.py train --epochs 50 --batch 16 --lr 0.0001 --no-yolo

# Detect
python main.py detect
```

### Windows webcam troubleshooting

If the camera does not open, try a different index:

```powershell
# List available cameras
python -c "
import cv2
for i in range(5):
    c = cv2.VideoCapture(i)
    if c.isOpened():
        print(f'Camera found at index {i}')
        c.release()
"
```

Then set the correct index in `config.yaml`:

```yaml
camera:
  device_index: 1   # change from 0 to whichever index worked
```

---

## ⚡ GPU Setup (CUDA)

Only needed if you have an NVIDIA GPU. Integrated graphics (Intel / AMD) use CPU automatically.

### Check if you have a CUDA GPU

```bash
nvidia-smi
```

### Install CUDA PyTorch

```bash
# Activate your environment first
source environment/bin/activate    # Linux / Arch
environment\Scripts\activate       # Windows

# Remove the CPU version
pip uninstall torch torchvision torchaudio -y

# CUDA 12.1 (recommended for RTX 30xx / 40xx cards)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 (for older cards)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU is detected
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### Arch Linux — system CUDA

```bash
sudo pacman -S cuda cudnn nvidia nvidia-utils
# Reboot after installing the driver
```

### ONNX GPU inference

```bash
pip uninstall onnxruntime -y
pip install onnxruntime-gpu
```

---

## 🐛 Known Issues & Fixes

### `_GeneratorContextManager` is not iterable

```
TypeError: '_GeneratorContextManager' object is not iterable
```

The `@contextmanager` decorator must be removed from `stream()` in `src/camera.py`.

Open `src/camera.py` and find:

```python
@contextmanager          # ← DELETE this line
def stream(self, ...):
```

Also remove `from contextlib import contextmanager` from the imports at the top of that file.

---

### Qt platform plugin not found (Wayland / Arch / Niri)

```
qt.qpa.plugin: Could not find the Qt platform plugin "wayland"
Available platform plugins are: xcb.
[1] IOT instruction (core dumped)
```

**Fix — set these before running:**

```bash
export DISPLAY=:1           # find yours with: ps aux | grep Xwayland
export QT_QPA_PLATFORM=xcb
python main.py capture
```

---

### `QFontDatabase: Cannot find font directory`

Harmless warning. Fix by installing DejaVu fonts:

```bash
sudo pacman -S ttf-dejavu       # Arch
sudo apt install fonts-dejavu   # Ubuntu / Debian
```

---

### `Disk quota exceeded` during pip install

The CUDA PyTorch wheels are very large (~3 GB). Fix:

```bash
# Clear pip cache
pip cache purge

# Install CPU-only version instead
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

### Python 3.14 — PyTorch import crash

PyTorch does not support Python 3.14. Install Python 3.12:

```bash
yay -S python312                                # Arch
sudo apt install python3.12 python3.12-venv     # Ubuntu
```

Then recreate your venv:

```bash
rm -rf environment
python3.12 -m venv --system-site-packages environment
source environment/bin/activate
```

---

### Dataset validation failed

```
RuntimeError: Dataset validation failed. Capture more images.
```

Either capture more images (20+ per class), or lower the minimum and trim your class list in `src/config.py`:

```python
min_images_per_class: int = 5   # lower from 20

classes: List[str] = [
    "human", "cat", "dog", "bottle"   # only classes you have images for
]
```

---

### TensorFlow warnings on startup

TensorFlow is completely optional. Silence the warnings with:

```bash
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=3
```

Or uninstall it entirely:

```bash
pip uninstall tensorflow -y
```

---

## 📸 Collecting Your Dataset

```bash
# Arch Linux / Wayland
export DISPLAY=:1
export QT_QPA_PLATFORM=xcb
source environment/bin/activate
python main.py capture

# Ubuntu / Debian
source environment/bin/activate
python main.py capture

# Windows
environment\Scripts\activate
python main.py capture
```

A webcam window opens. The current class label is shown at the bottom of the window.

| Key | Action |
|-----|--------|
| `S` | Save current frame into the active class folder |
| `C` | Cycle to the next class |
| `A` | Toggle auto-labelling using a pretrained YOLOv8n |
| `Q` / `ESC` | Quit |

### Tips for a good dataset

- Aim for **30–50 images per class minimum**
- Vary **distance** — close, medium, far
- Vary **angles** — straight on, left, right, tilted
- Vary **backgrounds** — different rooms and lighting conditions
- Include **partial views** — half the object visible
- For `human` — move your face around, try different expressions, with/without glasses

### Check what you have collected

```bash
python main.py info
```

Output example:

```
── Dataset ───────────────────────────────────────
  human        [████████████████████████████████████████] 45
  apple        [████████████████████████████            ] 32
  bottle       [████████████████████████████████████████] 45
  cat          [████████                                ] 8  ← needs more
```

---

## 🏋️ Training the Model

### Option A — PyTorch classifier (recommended for CPU / integrated graphics)

Fast to train, works well on laptops without a dedicated GPU:

```bash
python main.py train --epochs 50 --batch 16 --lr 0.0001 --no-yolo
```

### Option B — YOLOv8 fine-tuning (recommended if you have a GPU)

Produces full bounding-box detection:

```bash
python main.py train --epochs 50 --batch 16 --lr 0.0001
```

### Monitor training with TensorBoard

```bash
tensorboard --logdir logs/tensorboard
# Open http://localhost:6006 in your browser
```

### Training output files

| File | Description |
|------|-------------|
| `models/best_classifier.pt` | Best classifier weights |
| `models/best_yolo.pt` | Best YOLO weights |
| `checkpoints/epoch_XXXX.pt` | Periodic epoch checkpoints |
| `logs/tensorboard/` | TensorBoard event files |

### Recommended image counts

| Images per class | Expected accuracy |
|-----------------|-------------------|
| 10–20 | ~60–70% (proof of concept) |
| 30–50 | ~80–85% (usable) |
| 100+ | ~90%+ (good) |
| 500+ | Production quality |

---

## 🎥 Real-Time Detection

```bash
# Arch Linux / Wayland
export DISPLAY=:1
export QT_QPA_PLATFORM=xcb
python main.py detect

# Ubuntu / Windows
python main.py detect
```

### Controls during detection

| Key | Action |
|-----|--------|
| `S` | Save the current annotated frame to `detections/` |
| `R` | Reload model weights (after retraining without restarting) |
| `Q` / `ESC` | Quit |

### Advanced detection options

```bash
python main.py detect --model models/best_classifier.pt --conf 0.45 --skip 1
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | auto | Path to `.pt` or `.onnx` model file |
| `--conf` | `0.4` | Confidence threshold — raise to reduce false positives |
| `--skip` | `0` | Frames skipped between inferences — increase for higher display FPS |
| `--onnx` | off | Use ONNX runtime for faster CPU inference |

---

## 📦 Export to ONNX

ONNX gives significantly faster CPU inference (~2× speedup):

```bash
# Export
python main.py export

# Run detection using the ONNX model
python main.py detect --onnx
```

---

## ⚙️ Configuration

Edit `config.yaml` in the project root. Changes take effect on the next run.

```yaml
camera:
  device_index: 0      # webcam index — try 1 if 0 doesn't work
  width: 1280
  height: 720
  fps: 30

dataset:
  classes:
    - human
    - apple
    - banana
    - orange
    - cat
    - dog
    - bottle
    - phone
    - keyboard
    - laptop
  min_images_per_class: 20   # lower to 5 if just testing
  augmentation_enabled: true

training:
  epochs: 50
  batch_size: 16             # lower to 8 if you run out of memory
  learning_rate: 0.0001
  image_size: 640
  confidence_threshold: 0.25
  patience: 10               # early stop after N epochs with no improvement
  use_amp: true              # automatic mixed precision

inference:
  confidence_threshold: 0.4  # raise to reduce false positives
  frame_skip: 0              # increase for higher FPS on slow hardware
  use_onnx: false
  device: auto               # auto | cuda | cpu
```

---

## ➕ Adding New Classes

1. Add the class name to `config.yaml`:

```yaml
dataset:
  classes:
    - human
    - apple
    - scissors    # new class
```

2. Create the image folder:

```bash
mkdir dataset/scissors        # Linux / Arch
mkdir dataset\scissors        # Windows
```

3. Collect images, train, and detect:

```bash
python main.py capture
python main.py train --no-yolo
python main.py detect
```

---

## 📋 Command Reference

```bash
# Activate environment first (every new terminal session)
source environment/bin/activate      # Linux / Arch
environment\Scripts\activate         # Windows

# Main commands
python main.py capture               # Collect webcam images
python main.py train                 # Train with YOLOv8 (GPU recommended)
python main.py train --no-yolo       # Train PyTorch classifier (CPU friendly)
python main.py detect                # Run real-time detection
python main.py export                # Export model to ONNX
python main.py info                  # Show dataset stats and GPU info
python main.py app                   # Interactive all-in-one menu

# Training options
python main.py train --epochs 50 --batch 16 --lr 0.0001
python main.py train --no-yolo --epochs 30 --batch 8

# Detection options
python main.py detect --conf 0.45
python main.py detect --model models/best_classifier.pt
python main.py detect --skip 1
python main.py detect --onnx

# With display env vars (Arch / Wayland)
DISPLAY=:1 QT_QPA_PLATFORM=xcb python main.py capture
DISPLAY=:1 QT_QPA_PLATFORM=xcb python main.py detect
```

---

## 🚀 Performance Tips

| Goal | Solution |
|------|----------|
| Higher display FPS | `--skip 1` or `--skip 2` |
| Fastest CPU inference | Export to ONNX then use `--onnx` |
| Less memory usage | Lower `batch_size` to `4` or `8` in `config.yaml` |
| Better accuracy | Collect 100+ images per class |
| Smaller faster model | Set `model_name: yolov8n.pt` in `config.yaml` |
| More accurate model | Set `model_name: yolov8s.pt` or `yolov8m.pt` |
| GPU training | Install CUDA PyTorch — see GPU Setup section |
| Reduce false positives | Raise `confidence_threshold` to `0.5` or `0.6` |

---

## 🛠️ Tech Stack

| Component | Library |
|-----------|---------|
| Object detection | Ultralytics YOLOv8 |
| Training framework | PyTorch |
| Camera capture | OpenCV |
| Data augmentation | torchvision + albumentations |
| ONNX export / inference | onnx + onnxruntime |
| Training monitoring | TensorBoard |
| Logging | loguru |
| Config | PyYAML + Python dataclasses |

---

## 📄 License

MIT
