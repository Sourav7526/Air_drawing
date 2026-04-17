# ✏️ Air Drawing System — Production Edition

**Stack:** Python 3.10+ · MediaPipe · PyTorch · OpenCV

## Architecture
```
Camera → HandTracker (MediaPipe) → GestureRecognizer (PyTorch MLP + rules)
       → ActionDispatcher → Canvas (BGRA vector) → UI (HUD) → Display
```

## 12 Gestures
| Pose | Action | | Pose | Action |
|------|--------|-|------|--------|
| Index only | Draw | | Thumb only | Undo |
| Open palm | Erase | | Hang-loose 🤙 | Redo |
| Peace ✌ | Clear | | Pinky only | Save |
| Pinch | Colour + | | 3-finger | Brush + |
| Horns 🤘 | Brush − | | L-shape | Rectangle |
| Fist/O | Circle | | | |

## Quick Start
```bash
pip install -r requirements.txt
python air_draw.py [--camera 0] [--gpu] [--debug] [--frame-skip 2]
```

## Training Pipeline
```bash
python utils/collect_data.py          # ≥200 samples per gesture
python utils/augment_data.py          # 4× dataset (optional)
python utils/train_model.py --augment --epochs 120
python utils/evaluate_model.py --save-cm
python utils/export_onnx.py --verify  # optional ONNX speedup
```

## Tests
```bash
pytest tests/ -v
```

## Project Structure
```
air_drawing/
├── air_draw.py           Main loop + error handling
├── config.py             All settings + CLI overrides
├── logger.py             Logging (console + file)
├── hand_tracker.py       MediaPipe + tip smoother
├── gesture_recognizer.py GestureNet (12 classes) + rule fallback
├── canvas.py             Vector canvas, shapes, undo/redo
├── ui.py                 HUD, FPS graph, confidence bar
├── requirements.txt
├── models/               gesture_net.pt / .onnx
├── data/                 gesture_data.npz / _aug.npz
├── saved/                drawing_*.png
├── logs/                 runtime.log
├── utils/                collect, train, augment, evaluate, visualize, benchmark, export
└── tests/                pytest suite (gesture, canvas, tracker)
```

## GestureNet
```
63 → Linear(128)+BN+ReLU+Dropout → Linear(64)+BN+ReLU+Dropout
   → Linear(32)+BN+ReLU+Dropout → Linear(12) → softmax → class
```
~0.5 ms inference on CPU. Adam + ReduceLROnPlateau + early stopping.

## Troubleshooting
| Problem | Fix |
|---------|-----|
| Camera not found | --camera 1 or check permissions |
| Low accuracy | Collect more data + augment + retrain |
| Low FPS | --frame-skip 2 or --gpu |
