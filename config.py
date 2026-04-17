"""
config.py  –  Central configuration for the Air Drawing System.
All tunable parameters live here; CLI flags in air_draw.py can override them.
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field

ROOT = Path(__file__).resolve().parent

@dataclass
class Paths:
    models_dir:      Path = ROOT / "models"
    data_dir:        Path = ROOT / "data"
    saved_dir:       Path = ROOT / "saved"
    logs_dir:        Path = ROOT / "logs"
    gesture_weights: Path = ROOT / "models" / "gesture_net.pt"
    gesture_onnx:    Path = ROOT / "models" / "gesture_net.onnx"
    dataset:         Path = ROOT / "data"   / "gesture_data.npz"
    runtime_log:     Path = ROOT / "logs"   / "runtime.log"

@dataclass
class CameraConfig:
    index:       int = 0
    width:       int = 1280
    height:      int = 720
    fps:         int = 30
    skip_frames: int = 0   # process every N+1 frame (0 = never skip)

@dataclass
class ModelConfig:
    hidden_layers:     list = field(default_factory=lambda: [128, 64, 32])
    dropout_rate:      float = 0.3
    neural_threshold:  float = 0.80
    pinch_dist_thresh: float = 0.06
    use_gpu:           bool  = True
    use_onnx:          bool  = False

@dataclass
class GestureConfig:
    debounce_frames:  int = 4
    jitter_threshold: int = 6
    smoothing_window: int = 5

@dataclass
class BrushConfig:
    default_thickness: int = 6
    min_thickness:     int = 2
    max_thickness:     int = 40
    thickness_step:    int = 2
    default_eraser_sz: int = 40
    eraser_step:       int = 5
    max_undo_steps:    int = 30

@dataclass
class UIConfig:
    show_fps_graph:    bool  = True
    show_confidence:   bool  = True
    show_finger_debug: bool  = False
    fps_graph_length:  int   = 60
    hud_alpha:         float = 0.55
    swatch_size:       int   = 36
    swatch_margin:     int   = 8

@dataclass
class LogConfig:
    level:        str  = "INFO"
    to_file:      bool = True
    to_console:   bool = True
    max_bytes:    int  = 5_242_880
    backup_count: int  = 3

@dataclass
class Config:
    paths:   Paths         = field(default_factory=Paths)
    camera:  CameraConfig  = field(default_factory=CameraConfig)
    model:   ModelConfig   = field(default_factory=ModelConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    brush:   BrushConfig   = field(default_factory=BrushConfig)
    ui:      UIConfig      = field(default_factory=UIConfig)
    log:     LogConfig     = field(default_factory=LogConfig)

_cfg: Config | None = None

def get_config() -> Config:
    global _cfg
    if _cfg is None:
        _cfg = Config()
        _ensure_dirs(_cfg)
    return _cfg

def _ensure_dirs(cfg: Config) -> None:
    for d in (cfg.paths.models_dir, cfg.paths.data_dir,
              cfg.paths.saved_dir,  cfg.paths.logs_dir):
        d.mkdir(parents=True, exist_ok=True)
