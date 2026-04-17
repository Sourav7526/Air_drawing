"""tests/test_hand_tracker.py"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pytest, numpy as np
from hand_tracker import HandTracker, TipSmoother, HandData

class TestTipSmoother:
    def test_single(self):
        s=TipSmoother(5); assert s.smooth((100,200))==(100,200)
    def test_average(self):
        s=TipSmoother(5); s.smooth((0,0)); assert s.smooth((100,100))==(50,50)
    def test_window(self):
        s=TipSmoother(2)
        for _ in range(10): s.smooth((10,20))
        assert len(s._buf)==2
    def test_reset(self):
        s=TipSmoother(5); s.smooth((100,100)); s.reset()
        assert len(s._buf)==0 and s.smooth((50,50))==(50,50)
    def test_ints(self):
        s=TipSmoother(3); s.smooth((1,1)); s.smooth((2,2))
        x,y=s.smooth((3,3)); assert isinstance(x,int) and isinstance(y,int)

class TestTracker:
    def test_is_ready(self):
        t=HandTracker(); assert t.is_ready; t.release()
    def test_double_release(self):
        t=HandTracker(); t.release(); t.release()
    def test_blank_returns_empty(self):
        t=HandTracker()
        r=t.process(np.zeros((480,640,3),np.uint8))
        assert isinstance(r,list) and len(r)==0; t.release()
    def test_process_returns_list(self):
        t=HandTracker()
        r=t.process(np.random.randint(0,255,(480,640,3),np.uint8))
        assert isinstance(r,list); t.release()
    def test_items_are_handdata(self):
        t=HandTracker()
        for item in t.process(np.zeros((480,640,3),np.uint8)):
            assert isinstance(item,HandData)
        t.release()

class TestHandData:
    def _hd(self):
        return HandData(
            landmarks=np.zeros((21,3),np.float32),
            landmarks_px=np.zeros((21,2),int),
            handedness="Right",
            fingers_up=[False,True,False,False,False],
            index_tip_px=(320,240),
            raw_tip_px=(318,242),
        )
    def test_shape(self):
        hd=self._hd(); assert hd.landmarks.shape==(21,3) and hd.landmarks_px.shape==(21,2)
    def test_fingers_bool(self):
        for f in self._hd().fingers_up: assert isinstance(f,bool)
