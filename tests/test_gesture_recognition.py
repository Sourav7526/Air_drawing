"""tests/test_gesture_recognition.py"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest, numpy as np, torch
from gesture_recognizer import (
    GestureNet, GestureRecognizer, GestureDataset,
    Gesture, NUM_CLASSES, INPUT_DIM,
)

@pytest.fixture
def lm():
    m = np.zeros((21,3), np.float32)
    m[4,:2] = [0.1, 0.1]; m[8,:2] = [0.9, 0.9]
    return m

@pytest.fixture
def rec(tmp_path):
    return GestureRecognizer(weights_path=tmp_path/"none.pt", neural_threshold=0.80, pinch_distance_threshold=0.06)

class TestGestureNet:
    def test_shape(self):
        m = GestureNet(); x = torch.randn(1, INPUT_DIM)
        assert m(x).shape == (1, NUM_CLASSES)

    def test_predict_valid(self):
        m = GestureNet(); lm = np.random.rand(21,3).astype(np.float32)
        g, c = m.predict(lm)
        assert isinstance(g, Gesture) and 0 <= c <= 1

    def test_custom_hidden(self):
        m = GestureNet(hidden_layers=[64,32]); x = torch.randn(1, INPUT_DIM)
        assert m(x).shape == (1, NUM_CLASSES)

    def test_save_load(self, tmp_path):
        m = GestureNet(); p = tmp_path/"w.pt"
        torch.save(m.state_dict(), p)
        m2 = GestureNet(); m2.load_state_dict(torch.load(p, map_location="cpu"))
        rl = np.random.rand(21,3).astype(np.float32)
        g1,c1 = m.predict(rl); g2,c2 = m2.predict(rl)
        assert g1 == g2 and abs(c1-c2) < 1e-5

class TestDataset:
    def test_len_item(self):
        X = np.random.rand(20, INPUT_DIM).astype(np.float32)
        y = np.zeros(20, np.int64)
        ds = GestureDataset(X, y)
        assert len(ds)==20 and ds[0][0].shape==(INPUT_DIM,)

class TestRules:
    def test_draw(self, rec, lm):      assert rec.recognize(lm,[False,True,False,False,False])[0] == Gesture.DRAW
    def test_erase(self, rec, lm):     assert rec.recognize(lm,[True]*5)[0] == Gesture.ERASE
    def test_clear(self, rec, lm):     assert rec.recognize(lm,[False,True,True,False,False])[0] == Gesture.CLEAR
    def test_undo(self, rec, lm):      assert rec.recognize(lm,[True,False,False,False,False])[0] == Gesture.UNDO
    def test_redo(self, rec, lm):      assert rec.recognize(lm,[True,False,False,False,True])[0] == Gesture.REDO
    def test_save(self, rec, lm):      assert rec.recognize(lm,[False,False,False,False,True])[0] == Gesture.SAVE_CANVAS
    def test_brush_up(self, rec, lm):  assert rec.recognize(lm,[False,True,True,True,False])[0] == Gesture.BRUSH_SIZE_UP
    def test_pinch(self, rec):
        m=np.zeros((21,3),np.float32); m[4,:2]=[0.5,0.5]; m[8,:2]=[0.53,0.53]
        assert rec.recognize(m,[True,True,False,False,False])[0] == Gesture.COLOR_PICK
    def test_confidence_float(self, rec, lm):
        _,c = rec.recognize(lm,[False]*5)
        assert isinstance(c, float) and 0 <= c <= 1
