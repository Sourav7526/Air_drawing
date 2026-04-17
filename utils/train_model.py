"""utils/train_model.py – Train GestureNet"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse, numpy as np
from sklearn.model_selection import train_test_split
from gesture_recognizer import train_gesture_net, Gesture
from config import cfg
from logger import get_logger
log = get_logger(__name__)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       default=str(cfg.DATA_PATH))
    p.add_argument("--weights",    default=str(cfg.MODEL_WEIGHTS))
    p.add_argument("--epochs",     type=int,   default=cfg.EPOCHS)
    p.add_argument("--lr",         type=float, default=cfg.LEARNING_RATE)
    p.add_argument("--batch",      type=int,   default=cfg.BATCH_SIZE)
    p.add_argument("--dropout",    type=float, default=cfg.DROPOUT_RATE)
    p.add_argument("--early-stop", type=int,   default=cfg.EARLY_STOP)
    p.add_argument("--gpu",        action="store_true")
    p.add_argument("--augment",    action="store_true")
    a = p.parse_args()

    dp = Path(a.data)
    if not dp.exists():
        log.error("Dataset not found: %s — run utils/collect_data.py first.", dp)
        sys.exit(1)

    data = np.load(dp)
    X, y = data["X"], data["y"]
    log.info("Loaded %d samples", len(y))

    if a.augment:
        from utils.augment_data import augment_dataset
        X, y = augment_dataset(X, y, factor=4)
        log.info("Augmented → %d samples", len(y))

    for g in Gesture:
        log.info("  %-18s %d", g.name, (y==g.value).sum())

    X_tr, X_v, y_tr, y_v = train_test_split(X, y, test_size=cfg.VAL_SPLIT, stratify=y, random_state=42)
    log.info("Train: %d  Val: %d", len(y_tr), len(y_v))

    model = train_gesture_net(
        X_tr, y_tr, X_v, y_v,
        epochs=a.epochs, lr=a.lr, batch_size=a.batch,
        hidden_layers=cfg.HIDDEN_LAYERS, dropout_rate=a.dropout,
        early_stop=a.early_stop, save_path=Path(a.weights), use_gpu=a.gpu,
    )
    from utils.evaluate_model import evaluate
    evaluate(model, X_v, y_v)

if __name__ == "__main__": main()
