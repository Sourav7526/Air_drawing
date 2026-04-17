# ✏️ Air Drawing System

**Stack:** Python 3.10+ · MediaPipe · PyTorch · OpenCV · JavaScript

## Architecture
```
Camera → HandTracker (MediaPipe) → GestureRecognizer (PyTorch MLP + rules)
       → ActionDispatcher → Canvas (Vector-based) → HUD Overlay → Display
```

## Gestures
The system uses rule-based logic with a neural fallback. Below are the default mappings:

| Action | Python App (Desktop) | Web Edition (Browser) |
| :--- | :--- | :--- |
| **Draw** | Index finger only | Index finger only |
| **Erase** | Open palm (4+ fingers) | Open palm (4+ fingers) |
| **Clear** | Peace sign (✌) | Peace sign (✌) |
| **Cycle Color** | Pinch (Thumb+Index) | Pinch (Thumb+Index) |
| **Undo / Redo** | Thumb / Thumb+Pinky | Thumb / Thumb+Pinky |
| **Brush Size** | Index+Middle+Ring (±) | Index+Middle+Ring (±) |
| **Save PNG** | 4 Fingers (no thumb) | 4 Fingers (no thumb) |
| **Rectangle** | L-shape (Thumb+Index) | Ring + Pinky |
| **Circle** | Tripod (Thumb+Index+Middle) | Thumb + Index + Middle |

## Quick Start (Desktop)
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python air_draw.py
```

## Quick Start (Web)
Simply open `index.html` in a modern browser. The web edition uses MediaPipe via CDN and does not require a local server for basic use.

## Project Structure
```
.
├── air_draw.py           # Main Python application
├── app.js               # Main Web application logic
├── config.py             # Central configuration system
├── hand_tracker.py       # MediaPipe wrapper & smoothing
├── gesture_recognizer.py # GestureNet (MLP) & rule engine
├── canvas.py             # Vector canvas & shape logic
├── ui.py                 # HUD & UI rendering
├── index.html           # Web UI entry point
├── style.css            # Web styling
├── requirements.txt      # Python dependencies
├── models/               # Pre-trained weights (.pt / .onnx)
├── utils/                # Training & data collection scripts
└── tests/                # Pytest suite
```

## Features
- **Real-time Smoothing**: Robust hand-tracking with jitter reduction.
- **Auto-Snap**: Freehand strokes can automatically snap to perfect geometric shapes.
- **Vector Engine**: Supports high-quality undo/redo history.
- **Dual Engine**: High-performance Python backend and a zero-install Web version.

## Training Pipeline
If you wish to retrain the gesture model:
```bash
python utils/collect_data.py   # Collect training samples
python utils/train_model.py     # Train GestureNet (MLP)
python utils/export_onnx.py    # (Optional) Export for faster inference
```
