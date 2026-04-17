"""tests/test_canvas.py"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pytest, numpy as np
from canvas import Canvas, DrawMode, PALETTE

@pytest.fixture
def c(): return Canvas(640, 480, max_undo=10)

class TestBasics:
    def test_init(self, c):      assert c.color_idx==0 and c.thickness==6
    def test_color(self, c):     assert c.current_color==PALETTE[0]
    def test_set_idx(self, c):   c.set_color_idx(3); assert c.color_idx==3
    def test_wrap(self, c):      c.set_color_idx(100); assert 0<=c.color_idx<len(PALETTE)
    def test_next_prev(self, c): c.next_color(); assert c.color_idx==1; c.prev_color(); assert c.color_idx==0
    def test_thickness(self, c):
        orig=c.thickness; c.increase_thickness(); assert c.thickness>orig
        c.decrease_thickness(); assert c.thickness==orig

class TestDraw:
    def test_stroke(self, c):
        c.begin_stroke(); c.add_point(100,100); c.add_point(200,200); c.end_stroke()
        assert len(c._strokes)==1 and len(c._strokes[0].points)==2

    def test_composite_shape(self, c):
        f = np.zeros((480,640,3),np.uint8)
        r = c.composite(f)
        assert r.shape==(480,640,3) and r.dtype==np.uint8

    def test_clear(self, c):
        c.begin_stroke(); c.add_point(10,10); c.end_stroke()
        c.clear(); assert len(c._strokes)==0

    def test_to_png(self, c, tmp_path):
        c.begin_stroke(); c.add_point(50,50); c.end_stroke()
        p = tmp_path/"t.png"; c.to_png(p)
        assert p.exists() and p.stat().st_size>0

class TestUndoRedo:
    def test_undo(self, c):
        c.begin_stroke(); c.add_point(1,1); c.end_stroke()
        assert len(c._strokes)==1; c.undo(); assert len(c._strokes)==0
    def test_redo(self, c):
        c.begin_stroke(); c.add_point(1,1); c.end_stroke()
        c.undo(); c.redo(); assert len(c._strokes)==1
    def test_undo_empty(self, c): c.undo()
    def test_redo_empty(self, c): c.redo()

class TestShapes:
    def test_rect(self, c):
        c.mode=DrawMode.RECTANGLE; c.set_shape_start(10,10); c.update_shape_end(200,200); c.end_stroke()
        assert c._strokes[0].mode==DrawMode.RECTANGLE
    def test_circle(self, c):
        c.mode=DrawMode.CIRCLE; c.set_shape_start(100,100); c.update_shape_end(200,200); c.end_stroke()
        assert c._strokes[0].mode==DrawMode.CIRCLE
