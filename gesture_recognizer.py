"""
gesture_recognizer.py
---------------------
GestureNet (PyTorch MLP) + rule-based fallback + optional ONNX inference.

Gestures
--------
  0  IDLE
  1  DRAW             index finger only
  2  ERASE            open palm (4+ fingers)
  3  CLEAR            peace sign (index + middle)
  4  COLOR_PICK       pinch (thumb + index close)
  5  UNDO             thumb only
  6  REDO             thumb + pinky
  7  SAVE_CANVAS      four fingers (no thumb)
  8  BRUSH_SIZE_UP    index + middle + ring
  9  BRUSH_SIZE_DOWN  index + pinky only
 10  DRAW_RECTANGLE   index + thumb (L-shape, no middle)
 11  DRAW_CIRCLE      index + middle + thumb (tripod)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from enum import IntEnum
from typing import Optional

from config import get_config
from logger import get_logger

log = get_logger(__name__)


# ── Gesture enum ───────────────────────────────────────────────────────────────

class Gesture(IntEnum):
    IDLE            = 0
    DRAW            = 1
    ERASE           = 2
    CLEAR           = 3
    COLOR_PICK      = 4
    UNDO            = 5
    REDO            = 6
    SAVE_CANVAS     = 7
    BRUSH_SIZE_UP   = 8
    BRUSH_SIZE_DOWN = 9
    DRAW_RECTANGLE  = 10
    DRAW_CIRCLE     = 11

GESTURE_NAMES  = {g: g.name for g in Gesture}
NUM_CLASSES    = len(Gesture)
INPUT_DIM      = 63    # 21 landmarks × 3


# ── Model ──────────────────────────────────────────────────────────────────────

class GestureNet(nn.Module):
    """
    Configurable MLP for real-time gesture classification.

    Parameters
    ----------
    input_dim     : feature dimension (default 63)
    num_classes   : number of gesture classes
    hidden_layers : list of hidden unit counts per layer
    dropout_rate  : dropout probability applied after each hidden layer
    """

    def __init__(
        self,
        input_dim:     int       = INPUT_DIM,
        num_classes:   int       = NUM_CLASSES,
        hidden_layers: list[int] = None,
        dropout_rate:  float     = 0.3,
    ) -> None:
        super().__init__()
        cfg = get_config().model
        hidden = hidden_layers or cfg.hidden_layers

        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
            ]
            prev = h
        layers.append(nn.Linear(prev, num_classes))

        self.net = nn.Sequential(*layers)
        log.debug("GestureNet built: %s → %s", input_dim, hidden + [num_classes])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def predict(self, landmarks: np.ndarray) -> tuple[Gesture, float]:
        """
        landmarks : ndarray (21, 3) normalised
        Returns   : (Gesture, confidence 0-1)
        """
        self.eval()
        flat   = landmarks.flatten().astype(np.float32)
        x      = torch.from_numpy(flat).unsqueeze(0).to(next(self.parameters()).device)
        logits = self(x)
        probs  = F.softmax(logits, dim=1)
        idx    = int(probs.argmax(dim=1).item())
        conf   = float(probs[0, idx].item())
        return Gesture(idx), conf


# ── Gesture Recognizer ─────────────────────────────────────────────────────────

class GestureRecognizer:
    """
    High-level gesture recognition interface.

    Strategy
    --------
    1. Try neural model if weights exist and confidence ≥ threshold.
    2. Fall back to deterministic rule-based logic.
    3. Optionally use ONNX runtime for faster CPU inference.
    """

    def __init__(self) -> None:
        cfg = get_config()
        self._threshold  = cfg.model.neural_threshold
        self._pinch_thr  = cfg.model.pinch_dist_thresh
        self._use_onnx   = cfg.model.use_onnx
        self._device     = self._select_device(cfg.model.use_gpu)

        self._model: Optional[GestureNet]  = None
        self._onnx_session                 = None
        self._last_confidence: float       = 0.0

        if self._use_onnx:
            self._load_onnx()
        else:
            self._load_pytorch()

    # ── Public ─────────────────────────────────────────────────────────────────

    def recognize(
        self,
        landmarks:  np.ndarray,    # (21, 3)
        fingers_up: list[bool],    # [thumb, index, middle, ring, pinky]
    ) -> Gesture:
        """Return the predicted Gesture for the current hand pose."""
        gesture = Gesture.IDLE

        if self._model is not None:
            try:
                gesture, conf = self._model.predict(landmarks)
                self._last_confidence = conf
                if conf >= self._threshold:
                    return gesture
            except Exception as exc:
                log.warning("Neural predict failed: %s – using rule-based", exc)

        if self._onnx_session is not None:
            try:
                return self._onnx_predict(landmarks)
            except Exception as exc:
                log.warning("ONNX predict failed: %s – using rule-based", exc)

        self._last_confidence = 1.0   # rule-based = deterministic
        return self._rule_based(landmarks, fingers_up)

    @property
    def last_confidence(self) -> float:
        return self._last_confidence

    # ── Loading ────────────────────────────────────────────────────────────────

    def _load_pytorch(self) -> None:
        weights = get_config().paths.gesture_weights
        if not weights.exists():
            log.info("No weights at %s – rule-based mode active.", weights)
            return
        try:
            model = GestureNet()
            model.load_state_dict(torch.load(weights, map_location=self._device))
            model.to(self._device).eval()
            self._model = model
            log.info("GestureNet loaded from %s (device=%s)", weights, self._device)
        except Exception as exc:
            log.error("Failed to load GestureNet: %s", exc)

    def _load_onnx(self) -> None:
        try:
            import onnxruntime as ort
            onnx_path = get_config().paths.gesture_onnx
            if not onnx_path.exists():
                log.warning("ONNX file not found at %s", onnx_path)
                return
            self._onnx_session = ort.InferenceSession(str(onnx_path))
            log.info("ONNX session created from %s", onnx_path)
        except ImportError:
            log.warning("onnxruntime not installed – falling back to PyTorch.")
            self._load_pytorch()

    def _onnx_predict(self, landmarks: np.ndarray) -> Gesture:
        flat  = landmarks.flatten().astype(np.float32)[np.newaxis]
        out   = self._onnx_session.run(None, {"input": flat})[0]
        probs = np.exp(out) / np.exp(out).sum()
        idx   = int(np.argmax(probs))
        self._last_confidence = float(probs[0, idx])
        return Gesture(idx)

    # ── Device selection ───────────────────────────────────────────────────────

    @staticmethod
    def _select_device(prefer_gpu: bool) -> torch.device:
        if prefer_gpu and torch.cuda.is_available():
            log.info("Using CUDA for GestureNet inference.")
            return torch.device("cuda")
        log.info("Using CPU for GestureNet inference.")
        return torch.device("cpu")

    # ── Rule-based fallback ────────────────────────────────────────────────────

    def _rule_based(self, lm: np.ndarray, fingers: list[bool]) -> Gesture:
        thumb, index, middle, ring, pinky = fingers

        if self._pinch_detected(lm):
            return Gesture.COLOR_PICK

        count = sum(fingers)

        if fingers == [False, True, False, False, False]:   return Gesture.DRAW
        if fingers == [False, True, True,  False, False]:   return Gesture.CLEAR
        if fingers == [True,  False, False, False, True]:   return Gesture.REDO
        if fingers == [False, True, True,  True,  False]:   return Gesture.BRUSH_SIZE_UP
        if fingers == [False, True, False, False, True]:    return Gesture.BRUSH_SIZE_DOWN
        if fingers == [False, False, False, True,  True]:   return Gesture.DRAW_RECTANGLE
        if fingers == [True,  True, True,  False, False]:   return Gesture.DRAW_CIRCLE
        if fingers == [False, True, True,  True,  True]:    return Gesture.SAVE_CANVAS
        if fingers == [True,  False, False, False, False]:  return Gesture.UNDO
        if count >= 4:                                       return Gesture.ERASE

        return Gesture.IDLE

    def _pinch_detected(self, lm: np.ndarray) -> bool:
        return float(np.linalg.norm(lm[4, :2] - lm[8, :2])) < self._pinch_thr


# ── Dataset / Training helpers ─────────────────────────────────────────────────

class GestureDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
    def __len__(self) -> int:          return len(self.y)
    def __getitem__(self, i):          return self.X[i], self.y[i]


def train_gesture_net(
    X_train:    np.ndarray,
    y_train:    np.ndarray,
    X_val:      np.ndarray | None = None,
    y_val:      np.ndarray | None = None,
    epochs:     int   = 80,
    lr:         float = 1e-3,
    batch_size: int   = 32,
    patience:   int   = 10,
    save_path:  Path  = None,
) -> GestureNet:
    """
    Train GestureNet with early stopping on validation loss.

    Returns the best model (also saved to save_path).
    """
    if save_path is None:
        save_path = get_config().paths.gesture_weights

    cfg    = get_config().model
    device = torch.device("cuda" if cfg.use_gpu and torch.cuda.is_available() else "cpu")

    dataset   = GestureDataset(X_train, y_train)
    loader    = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model     = GestureNet().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0

    log.info("Training GestureNet for up to %d epochs on %s …", epochs, device)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = correct = total = 0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimiser.zero_grad()
            logits = model(X_b)
            loss   = criterion(logits, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total_loss += loss.item() * len(y_b)
            correct    += (logits.argmax(1) == y_b).sum().item()
            total      += len(y_b)

        train_loss = total_loss / total
        train_acc  = 100 * correct / total

        # ── Validation ───────────────────────────────────────────────────────
        val_loss = train_loss   # fallback if no val set
        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                Xv = torch.from_numpy(X_val.astype(np.float32)).to(device)
                yv = torch.from_numpy(y_val.astype(np.int64)).to(device)
                val_loss = criterion(model(Xv), yv).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            log.info("  Epoch %3d/%d  train_loss=%.4f  acc=%.1f%%  val_loss=%.4f",
                     epoch, epochs, train_loss, train_acc, val_loss)

        if no_improve >= patience:
            log.info("Early stopping at epoch %d (no val improvement for %d epochs).",
                     epoch, patience)
            break

    # Save best weights
    if best_state:
        model.load_state_dict(best_state)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    log.info("GestureNet saved to %s", save_path)
    return model
