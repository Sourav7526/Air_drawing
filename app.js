/**
 * app.js — Air Drawing Web Edition
 *
 * Pipeline: Camera → MediaPipe Hands → GestureRecognizer → Canvas → UI
 *
 * All 12 gestures from the original desktop app are preserved via rule-based
 * recognition ported from gesture_recognizer.py.
 */

"use strict";

// ── Constants ──────────────────────────────────────────────────────────────────

const PALETTE = [
  { rgb: [255, 0, 0], name: "Red" },
  { rgb: [255, 165, 0], name: "Orange" },
  { rgb: [255, 255, 0], name: "Yellow" },
  { rgb: [0, 255, 0], name: "Green" },
  { rgb: [0, 0, 255], name: "Blue" },
  { rgb: [255, 0, 255], name: "Magenta" },
  { rgb: [255, 255, 255], name: "White" },
  { rgb: [128, 128, 128], name: "Grey" },
];

const Gesture = Object.freeze({
  IDLE: 0,
  DRAW: 1,
  ERASE: 2,
  CLEAR: 3,
  COLOR_PICK: 4,
  UNDO: 5,
  REDO: 6,
  SAVE_CANVAS: 7,
  BRUSH_SIZE_UP: 8,
  BRUSH_SIZE_DOWN: 9,
  DRAW_RECTANGLE: 10,
  DRAW_CIRCLE: 11,
});

const GESTURE_NAMES = [
  "Idle", "Draw", "Erase", "Clear", "Color Pick",
  "Undo", "Redo", "Save", "Brush +", "Brush −",
  "Rectangle", "Circle"
];

const DrawMode = Object.freeze({
  FREEHAND: "freehand",
  RECTANGLE: "rectangle",
  CIRCLE: "circle",
});

// Landmark indices
const THUMB_TIP = 4, INDEX_TIP = 8, MIDDLE_TIP = 12, RING_TIP = 16, PINKY_TIP = 20;
const THUMB_IP = 3, INDEX_PIP = 6, MIDDLE_PIP = 10, RING_PIP = 14, PINKY_PIP = 18;

// ── Config ─────────────────────────────────────────────────────────────────────

const CFG = {
  brush: { default: 6, min: 2, max: 40, step: 2 },
  eraser: { default: 40, step: 5, min: 10, max: 100 },
  maxUndo: 30,
  debounceFrames: 4,
  jitterThreshold: 6,
  smoothingWindow: 5,
  pinchDistThresh: 0.06,
};

// ── State ──────────────────────────────────────────────────────────────────────

const state = {
  colorIdx: 0,
  thickness: CFG.brush.default,
  eraserSz: CFG.eraser.default,
  drawMode: DrawMode.FREEHAND,

  // Drawing state
  strokes: [],
  undoStack: [],
  redoStack: [],
  currentStroke: null,
  shapeStart: null,

  // Gesture state
  prevGesture: Gesture.IDLE,
  gestureBuf: [],
  prevTip: null,
  lastKnownTip: null,  // track last known tip for shape endpoints

  // Smoother
  smoothBuf: [],

  // UI
  showHelp: false,
  statusTimeout: null,

  // FPS
  fpsTimes: [],
};

// ── DOM refs ───────────────────────────────────────────────────────────────────

let video, overlayCanvas, overlayCtx, drawCanvas, drawCtx;
let elGestureLabel, elConfFill, elBrushInfo, elModeInfo, elFpsBadge, elStatusToast;
let elHelpPanel, elPaletteBar, elColorName;

// ── Initialization ─────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  video = document.getElementById("webcam");
  overlayCanvas = document.getElementById("overlay");
  overlayCtx = overlayCanvas.getContext("2d");
  drawCanvas = document.getElementById("drawing");
  drawCtx = drawCanvas.getContext("2d");

  elGestureLabel = document.getElementById("gesture-label");
  elConfFill = document.getElementById("conf-fill");
  elBrushInfo = document.getElementById("brush-info");
  elModeInfo = document.getElementById("mode-info");
  elFpsBadge = document.getElementById("fps-badge");
  elStatusToast = document.getElementById("status-toast");
  elHelpPanel = document.getElementById("help-panel");
  elPaletteBar = document.getElementById("palette-bar");
  elColorName = document.getElementById("color-name");

  buildPalette();
  bindToolbar();
  bindKeyboard();

  document.getElementById("start-btn").addEventListener("click", startCamera);
  document.getElementById("help-close").addEventListener("click", () => toggleHelp(false));
});


// ── Camera ─────────────────────────────────────────────────────────────────────

async function startCamera() {
  const errorEl = document.getElementById("camera-error");
  errorEl.style.display = "none";

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();

    // Wait for video dimensions
    await new Promise(resolve => {
      if (video.videoWidth > 0) return resolve();
      video.addEventListener("loadedmetadata", resolve, { once: true });
    });

    const W = video.videoWidth, H = video.videoHeight;
    overlayCanvas.width = W; overlayCanvas.height = H;
    drawCanvas.width = W; drawCanvas.height = H;
    video.width = W; video.height = H;

    // Hide prompt, show UI
    document.getElementById("camera-prompt").style.display = "none";
    document.getElementById("loading").classList.add("visible");

    initMediaPipe(W, H);
  } catch (err) {
    console.error("Camera error:", err);
    errorEl.style.display = "block";
  }
}


// ── MediaPipe Hands ────────────────────────────────────────────────────────────

function initMediaPipe(W, H) {
  const hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`,
  });

  hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.75,
    minTrackingConfidence: 0.75,
  });

  hands.onResults((results) => onHandResults(results, W, H));

  // Use Camera utility for frame loop
  const camera = new Camera(video, {
    onFrame: async () => { await hands.send({ image: video }); },
    width: W,
    height: H,
  });

  camera.start().then(() => {
    document.getElementById("loading").classList.remove("visible");
    document.getElementById("canvas-stack").style.display = "block";
    document.getElementById("palette-bar").style.display = "flex";
    document.getElementById("info-panel").style.display = "block";
    document.getElementById("toolbar").style.display = "flex";
    document.getElementById("fps-badge").style.display = "block";
  });
}


// ── Main frame callback ────────────────────────────────────────────────────────

function onHandResults(results, W, H) {
  // FPS tracking
  const now = performance.now();
  state.fpsTimes.push(now);
  while (state.fpsTimes.length > 0 && now - state.fpsTimes[0] > 1000) state.fpsTimes.shift();
  const fps = state.fpsTimes.length;
  elFpsBadge.textContent = `FPS ${fps}`;
  elFpsBadge.className = "glass-panel" + (fps >= 24 ? " good" : fps >= 15 ? " ok" : " bad");

  // Clear overlay
  overlayCtx.clearRect(0, 0, W, H);

  let gesture = Gesture.IDLE;
  let confidence = 0;
  let fingersUp = null;
  let tipRaw = null;

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    const landmarks = results.multiHandLandmarks[0];
    const handedness = results.multiHandedness?.[0]?.label || "Right";

    // Draw landmarks on overlay (mirrored)
    overlayCtx.save();
    overlayCtx.translate(W, 0);
    overlayCtx.scale(-1, 1);
    drawConnectors(overlayCtx, landmarks, HAND_CONNECTIONS, { color: "rgba(0,200,255,0.4)", lineWidth: 2 });
    drawLandmarks(overlayCtx, landmarks, { color: "rgba(0,200,255,0.8)", lineWidth: 1, radius: 3 });
    overlayCtx.restore();

    // Compute fingersUp (mirrored coordinates — video is flipped)
    const lm = landmarks.map(l => [l.x, l.y, l.z]);
    fingersUp = computeFingersUp(lm, handedness);

    // Get index tip in pixel coords (mirrored)
    tipRaw = [Math.round((1 - landmarks[INDEX_TIP].x) * W), Math.round(landmarks[INDEX_TIP].y * H)];

    // Gesture recognition
    gesture = recognizeGesture(lm, fingersUp);
    confidence = 1.0; // rule-based = deterministic
  }

  // Debounce
  state.gestureBuf.push(gesture);
  if (state.gestureBuf.length > CFG.debounceFrames) state.gestureBuf.shift();
  const stable = mostCommon(state.gestureBuf);

  // Smooth tip
  const tip = tipRaw ? smoothTip(tipRaw) : null;

  // Track last known tip so shape endpoints aren't lost
  if (tip) state.lastKnownTip = tip;

  // Canvas actions
  processGesture(stable, tip, W, H);

  // Update HUD
  updateHUD(stable, confidence);
}


// ── Finger detection (ported from hand_tracker.py) ─────────────────────────────

function computeFingersUp(lm, handedness) {
  const fingers = [];

  // Thumb: compare x of tip vs IP joint
  // MediaPipe mirrored: "Right" label = user's right hand but mirrored in image
  if (handedness === "Right") {
    fingers.push(lm[THUMB_TIP][0] > lm[THUMB_IP][0]); // mirrored, so > instead of <
  } else {
    fingers.push(lm[THUMB_TIP][0] < lm[THUMB_IP][0]);
  }

  // Other fingers: tip y < PIP y means finger is up
  fingers.push(lm[INDEX_TIP][1] < lm[INDEX_PIP][1]);
  fingers.push(lm[MIDDLE_TIP][1] < lm[MIDDLE_PIP][1]);
  fingers.push(lm[RING_TIP][1] < lm[RING_PIP][1]);
  fingers.push(lm[PINKY_TIP][1] < lm[PINKY_PIP][1]);

  return fingers;
}


// ── Gesture recognition (ported from gesture_recognizer.py _rule_based) ────────

function recognizeGesture(lm, fingers) {
  const [thumb, index, middle, ring, pinky] = fingers;

  // Pinch check
  const dx = lm[4][0] - lm[8][0];
  const dy = lm[4][1] - lm[8][1];
  if (Math.sqrt(dx * dx + dy * dy) < CFG.pinchDistThresh) return Gesture.COLOR_PICK;

  const count = fingers.filter(Boolean).length;

  // Open palm (4+ fingers) = ERASE — must be checked BEFORE exact 4-finger patterns
  // so that unreliable thumb detection doesn't accidentally trigger Save
  if (count >= 4) return Gesture.ERASE;

  // Exact pattern matches
  if (eq(fingers, [false, true, false, false, false])) return Gesture.DRAW;
  if (eq(fingers, [false, true, true, false, false])) return Gesture.CLEAR;
  if (eq(fingers, [true, false, false, false, true])) return Gesture.REDO;
  if (eq(fingers, [false, true, true, true, false])) return Gesture.BRUSH_SIZE_UP;
  if (eq(fingers, [false, true, false, false, true])) return Gesture.BRUSH_SIZE_DOWN;
  if (eq(fingers, [false, false, false, true, true])) return Gesture.DRAW_RECTANGLE;
  if (eq(fingers, [true, true, true, false, false])) return Gesture.DRAW_CIRCLE;
  if (eq(fingers, [true, false, false, false, false])) return Gesture.UNDO;

  return Gesture.IDLE;
}

function eq(a, b) {
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}


// ── Tip smoother (ported from smoother.py) ─────────────────────────────────────

function smoothTip(pt) {
  state.smoothBuf.push(pt);
  if (state.smoothBuf.length > CFG.smoothingWindow) state.smoothBuf.shift();
  const xs = state.smoothBuf.map(p => p[0]);
  const ys = state.smoothBuf.map(p => p[1]);
  return [
    Math.round(xs.reduce((a, b) => a + b, 0) / xs.length),
    Math.round(ys.reduce((a, b) => a + b, 0) / ys.length),
  ];
}

function resetSmoother() { state.smoothBuf = []; }


// ── Canvas action processing ───────────────────────────────────────────────────

function processGesture(stable, tip, W, H) {
  const justChanged = (stable !== state.prevGesture);

  if (stable === Gesture.DRAW && tip) {
    if (state.prevGesture !== Gesture.DRAW) {
      beginStroke(tip);
    } else {
      const prev = state.prevTip || tip;
      const dx = tip[0] - prev[0], dy = tip[1] - prev[1];
      if (dx * dx + dy * dy >= CFG.jitterThreshold * CFG.jitterThreshold) {
        addPoint(tip);
      }
    }
  } else if ((stable === Gesture.DRAW_RECTANGLE || stable === Gesture.DRAW_CIRCLE) && tip) {
    state.drawMode = stable === Gesture.DRAW_RECTANGLE ? DrawMode.RECTANGLE : DrawMode.CIRCLE;
    updateModeButtons();
    if (state.prevGesture !== stable) {
      beginStroke(tip);
    } else {
      renderShapePreview(tip);
    }
  } else {
    // Finish any active stroke
    if ([Gesture.DRAW, Gesture.DRAW_RECTANGLE, Gesture.DRAW_CIRCLE].includes(state.prevGesture)) {
      endStroke(tip || state.lastKnownTip || [0, 0]);
      state.drawMode = DrawMode.FREEHAND;
      updateModeButtons();
    }
    resetSmoother();
    state.prevTip = null;

    if (stable === Gesture.ERASE && tip) {
      erase(tip);
    } else if (stable === Gesture.CLEAR && justChanged) {
      clearCanvas();
      showStatus("Canvas cleared");
    } else if (stable === Gesture.UNDO && justChanged) {
      undo();
      showStatus("Undo");
    } else if (stable === Gesture.REDO && justChanged) {
      redo();
      showStatus("Redo");
    } else if (stable === Gesture.COLOR_PICK && justChanged) {
      nextColor();
      showStatus(`Color: ${PALETTE[state.colorIdx].name}`);
    } else if (stable === Gesture.SAVE_CANVAS && justChanged) {
      saveDrawing();
      showStatus("Drawing saved!");
    } else if (stable === Gesture.BRUSH_SIZE_UP && justChanged) {
      state.thickness = Math.min(state.thickness + CFG.brush.step, CFG.brush.max);
      showStatus(`Brush: ${state.thickness}px`);
    } else if (stable === Gesture.BRUSH_SIZE_DOWN && justChanged) {
      state.thickness = Math.max(state.thickness - CFG.brush.step, CFG.brush.min);
      showStatus(`Brush: ${state.thickness}px`);
    }
  }

  if (stable === Gesture.DRAW && tip) state.prevTip = tip;
  state.prevGesture = stable;

  // Draw brush/eraser cursor on overlay
  if (tip) {
    const isEraser = (stable === Gesture.ERASE);
    const sz = isEraser ? state.eraserSz : state.thickness;
    drawCursor(tip, sz, isEraser);
  }
}


// ── Drawing primitives ─────────────────────────────────────────────────────────

function beginStroke(pt) {
  if (state.currentStroke) commitStroke();
  // Save canvas snapshot before this stroke (so we can restore if shape-snapping)
  state.preStrokeSnapshot = drawCtx.getImageData(0, 0, drawCanvas.width, drawCanvas.height);
  state.currentStroke = {
    points: [],
    color: currentColorCSS(),
    thickness: state.thickness,
    mode: state.drawMode,
  };
  if (state.drawMode === DrawMode.FREEHAND) {
    state.currentStroke.points.push(pt);
  } else {
    state.shapeStart = pt;
  }
}

function addPoint(pt) {
  if (!state.currentStroke) { beginStroke(pt); return; }
  if (state.drawMode !== DrawMode.FREEHAND) return;

  const pts = state.currentStroke.points;
  pts.push(pt);

  // Draw segment on canvas
  if (pts.length >= 2) {
    const p1 = pts[pts.length - 2], p2 = pts[pts.length - 1];
    drawCtx.strokeStyle = state.currentStroke.color;
    drawCtx.lineWidth = state.currentStroke.thickness;
    drawCtx.lineCap = "round";
    drawCtx.lineJoin = "round";
    drawCtx.beginPath();
    drawCtx.moveTo(p1[0], p1[1]);
    drawCtx.lineTo(p2[0], p2[1]);
    drawCtx.stroke();
  }
}

function endStroke(pt) {
  if (!state.currentStroke) return;

  // Use the stroke's own mode (not state.drawMode which may have been reset)
  const mode = state.currentStroke.mode;

  if (mode === DrawMode.RECTANGLE && state.shapeStart) {
    state.currentStroke.points = [state.shapeStart, pt];
    drawRectOnCanvas(state.shapeStart, pt, state.currentStroke.color, state.currentStroke.thickness);
  } else if (mode === DrawMode.CIRCLE && state.shapeStart) {
    const dx = pt[0] - state.shapeStart[0], dy = pt[1] - state.shapeStart[1];
    const r = Math.sqrt(dx * dx + dy * dy);
    state.currentStroke.points = [state.shapeStart, pt];
    state.currentStroke.radius = r;
    drawCircleOnCanvas(state.shapeStart, r, state.currentStroke.color, state.currentStroke.thickness);
  }

  state.shapeStart = null;
  commitStroke();
}

function commitStroke() {
  const s = state.currentStroke;
  if (s && s.points.length > 0) {
    // Try to auto-snap freehand strokes into perfect shapes
    if (s.mode === DrawMode.FREEHAND && s.points.length >= 10) {
      const snapped = trySnapShape(s);
      if (snapped) {
        // Restore canvas to pre-stroke state (preserves erased areas)
        if (state.preStrokeSnapshot) {
          drawCtx.putImageData(state.preStrokeSnapshot, 0, 0);
        }
        // Draw the snapped shape on top
        drawSnappedShape(snapped);
        // Save to stroke history
        state.undoStack.push([...state.strokes]);
        if (state.undoStack.length > CFG.maxUndo) state.undoStack.shift();
        state.redoStack = [];
        state.strokes.push(snapped);
        state.currentStroke = null;
        state.preStrokeSnapshot = null;
        return;
      }
    }
    state.undoStack.push([...state.strokes]);
    if (state.undoStack.length > CFG.maxUndo) state.undoStack.shift();
    state.redoStack = [];
    state.strokes.push(s);
  }
  state.currentStroke = null;
  state.preStrokeSnapshot = null;
}

// Draw a single snapped shape onto the canvas (avoids redrawAll which would lose erases)
function drawSnappedShape(s) {
  drawCtx.strokeStyle = s.color;
  drawCtx.lineWidth = s.thickness;
  drawCtx.lineCap = "round";
  drawCtx.lineJoin = "round";

  if (s.mode === DrawMode.CIRCLE && s.points.length === 2) {
    drawCircleOnCanvas(s.points[0], s.radius, s.color, s.thickness);
  } else if (s.mode === DrawMode.RECTANGLE && s.points.length === 2) {
    drawRectOnCanvas(s.points[0], s.points[1], s.color, s.thickness);
  } else if (s.mode === "ellipse" && s.points.length >= 1) {
    drawEllipseOnCanvas(s.points[0], s.rx, s.ry, s.color, s.thickness);
  } else if (s.mode === "triangle" && s.points.length === 3) {
    drawPolygonOnCanvas(s.points, s.color, s.thickness);
  } else if (s.mode === "diamond" && s.points.length === 4) {
    drawPolygonOnCanvas(s.points, s.color, s.thickness);
  } else if (s.mode === "line" && s.points.length === 2) {
    drawLineOnCanvas(s.points[0], s.points[1], s.color, s.thickness);
  }
}
// ── Shape recognition (auto-snap freehand to perfect shapes) ──────────────────
// Supports: Circle, Triangle

function trySnapShape(stroke) {
  const pts = stroke.points;
  if (pts.length < 5) return null;

  const first = pts[0], last = pts[pts.length - 1];
  const closeDist = dist(first, last);
  const bbox = getBoundingBox(pts);
  const diagonal = Math.sqrt(bbox.w * bbox.w + bbox.h * bbox.h);

  // Minimum size to avoid false positives
  if (diagonal < 35) return null;

  // Only closed shapes (start ≈ end)
  if (closeDist > diagonal * 0.35) return null;

  // ── 1. Circle (closed, low radial variance) ───────────────────────────
  const circleResult = tryDetectCircle(pts, bbox);
  if (circleResult) {
    showStatus("✨ Snapped to Circle!");
    return makeStroke(stroke, "circle", circleResult);
  }

  // ── 2. Triangle (3 dominant corners) ──────────────────────────────────
  const triResult = tryDetectTriangle(pts, bbox);
  if (triResult) {
    showStatus("✨ Snapped to Triangle!");
    return makeStroke(stroke, "triangle", triResult);
  }

  return null;
}

function makeStroke(original, shapeType, data) {
  const s = {
    color: original.color,
    thickness: original.thickness,
    shapeType: shapeType,
  };

  switch (shapeType) {
    case "circle":
      s.mode = DrawMode.CIRCLE;
      s.points = [data.center, [data.center[0] + data.radius, data.center[1]]];
      s.radius = data.radius;
      break;
    case "ellipse":
      s.mode = "ellipse";
      s.points = [data.center];
      s.rx = data.rx;
      s.ry = data.ry;
      break;
    case "rectangle":
      s.mode = DrawMode.RECTANGLE;
      s.points = [data.p1, data.p2];
      break;
    case "triangle":
      s.mode = "triangle";
      s.points = data.vertices;
      break;
    case "diamond":
      s.mode = "diamond";
      s.points = data.vertices;
      break;
    case "line":
      s.mode = "line";
      s.points = [data.p1, data.p2];
      break;
  }
  return s;
}


// ── Individual shape detectors ────────────────────────────────────────────────

function tryDetectLine(pts, first, last) {
  // Check if all points are close to the line from first to last
  const len = dist(first, last);
  if (len < 30) return null;

  let maxDev = 0;
  for (const p of pts) {
    const d = pointToLineDist(p, first, last);
    if (d > maxDev) maxDev = d;
  }

  // Points should stay close to the line (< 12% of length)
  if (maxDev < len * 0.12) {
    return { p1: [...first], p2: [...last] };
  }
  return null;
}

function tryDetectCircle(pts, bbox) {
  let cx = 0, cy = 0;
  for (const p of pts) { cx += p[0]; cy += p[1]; }
  cx /= pts.length; cy /= pts.length;
  const center = [Math.round(cx), Math.round(cy)];

  const radii = pts.map(p => dist(p, center));
  const meanR = radii.reduce((a, b) => a + b, 0) / radii.length;
  if (meanR < 15) return null;

  const variance = radii.reduce((a, r) => a + (r - meanR) ** 2, 0) / radii.length;
  const cv = Math.sqrt(variance) / meanR;
  const aspect = bbox.w / (bbox.h || 1);

  // Circle: low radial variance AND roughly square bbox
  if (cv < 0.18 && aspect > 0.75 && aspect < 1.33) {
    return { center, radius: Math.round(meanR) };
  }
  return null;
}

function tryDetectEllipse(pts, bbox) {
  let cx = 0, cy = 0;
  for (const p of pts) { cx += p[0]; cy += p[1]; }
  cx /= pts.length; cy /= pts.length;
  const center = [Math.round(cx), Math.round(cy)];

  const rx = bbox.w / 2, ry = bbox.h / 2;
  if (rx < 15 || ry < 15) return null;

  // Check how well points fit on the ellipse (x-cx)²/rx² + (y-cy)²/ry² ≈ 1
  let totalError = 0;
  for (const p of pts) {
    const ex = (p[0] - cx) / rx, ey = (p[1] - cy) / ry;
    const ellipseVal = ex * ex + ey * ey;
    totalError += Math.abs(ellipseVal - 1);
  }
  const avgError = totalError / pts.length;

  // Good ellipse fit if average error < 0.35
  if (avgError < 0.35) {
    return { center, rx: Math.round(rx), ry: Math.round(ry) };
  }
  return null;
}

function tryDetectTriangle(pts, bbox) {
  // Find corners using angle-based approach
  const corners = findCorners(pts, 3);
  if (corners.length !== 3) return null;

  // Verify the triangle covers a reasonable area
  const area = triangleArea(corners[0], corners[1], corners[2]);
  const bboxArea = bbox.w * bbox.h;
  if (area < bboxArea * 0.2) return null; // too thin

  // Check that points follow the triangle edges reasonably
  let nearEdge = 0;
  const margin = Math.max(bbox.w, bbox.h) * 0.15;
  for (const p of pts) {
    const d01 = pointToSegmentDist(p, corners[0], corners[1]);
    const d12 = pointToSegmentDist(p, corners[1], corners[2]);
    const d20 = pointToSegmentDist(p, corners[2], corners[0]);
    if (Math.min(d01, d12, d20) < margin) nearEdge++;
  }

  if (nearEdge / pts.length > 0.7) {
    return { vertices: corners.map(c => [Math.round(c[0]), Math.round(c[1])]) };
  }
  return null;
}

function tryDetectDiamond(pts, bbox) {
  // Diamond has 4 corners near the midpoints of each bbox edge
  const cx = bbox.x + bbox.w / 2, cy = bbox.y + bbox.h / 2;
  const expectedCorners = [
    [cx, bbox.y],              // top
    [bbox.x + bbox.w, cy],     // right
    [cx, bbox.y + bbox.h],     // bottom
    [bbox.x, cy],              // left
  ];

  // Check aspect ratio — diamond should not be too elongated
  const aspect = bbox.w / (bbox.h || 1);
  if (aspect < 0.3 || aspect > 3.5) return null;
  if (bbox.w < 30 || bbox.h < 30) return null;

  // Check points lie along the diamond edges
  let nearEdge = 0;
  const margin = Math.max(bbox.w, bbox.h) * 0.15;
  for (const p of pts) {
    let minD = Infinity;
    for (let i = 0; i < 4; i++) {
      const d = pointToSegmentDist(p, expectedCorners[i], expectedCorners[(i + 1) % 4]);
      if (d < minD) minD = d;
    }
    if (minD < margin) nearEdge++;
  }

  if (nearEdge / pts.length > 0.72) {
    return { vertices: expectedCorners.map(c => [Math.round(c[0]), Math.round(c[1])]) };
  }
  return null;
}

function tryDetectRectangle(pts, bbox) {
  const margin = Math.max(bbox.w, bbox.h) * 0.2;
  let onEdge = 0;

  for (const p of pts) {
    const nearLeft = Math.abs(p[0] - bbox.x) < margin;
    const nearRight = Math.abs(p[0] - (bbox.x + bbox.w)) < margin;
    const nearTop = Math.abs(p[1] - bbox.y) < margin;
    const nearBottom = Math.abs(p[1] - (bbox.y + bbox.h)) < margin;
    if (nearLeft || nearRight || nearTop || nearBottom) onEdge++;
  }

  const edgeRatio = onEdge / pts.length;
  const midX = bbox.x + bbox.w / 2, midY = bbox.y + bbox.h / 2;
  let q = [false, false, false, false];
  for (const p of pts) {
    if (p[0] < midX && p[1] < midY) q[0] = true;
    if (p[0] >= midX && p[1] < midY) q[1] = true;
    if (p[0] < midX && p[1] >= midY) q[2] = true;
    if (p[0] >= midX && p[1] >= midY) q[3] = true;
  }

  if (bbox.w < 30 || bbox.h < 30) return null;

  if (edgeRatio > 0.7 && q.every(Boolean)) {
    return { p1: [bbox.x, bbox.y], p2: [bbox.x + bbox.w, bbox.y + bbox.h] };
  }
  return null;
}


// ── Geometry helpers ──────────────────────────────────────────────────────────

function getBoundingBox(pts) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of pts) {
    if (p[0] < minX) minX = p[0];
    if (p[1] < minY) minY = p[1];
    if (p[0] > maxX) maxX = p[0];
    if (p[1] > maxY) maxY = p[1];
  }
  return { x: minX, y: minY, w: maxX - minX, h: maxY - minY };
}

function dist(a, b) {
  const dx = a[0] - b[0], dy = a[1] - b[1];
  return Math.sqrt(dx * dx + dy * dy);
}

function pointToLineDist(p, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const len = Math.sqrt(dx * dx + dy * dy);
  if (len === 0) return dist(p, a);
  return Math.abs(dy * p[0] - dx * p[1] + b[0] * a[1] - b[1] * a[0]) / len;
}

function pointToSegmentDist(p, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const lenSq = dx * dx + dy * dy;
  if (lenSq === 0) return dist(p, a);
  let t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / lenSq;
  t = Math.max(0, Math.min(1, t));
  return dist(p, [a[0] + t * dx, a[1] + t * dy]);
}

function triangleArea(a, b, c) {
  return Math.abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2;
}

function findCorners(pts, n) {
  // Resample to equally spaced points, then find corners by max angular change
  const total = pts.length;
  if (total < n * 3) return [];

  const step = Math.max(Math.floor(total / 40), 2);
  const angles = [];

  for (let i = step; i < total - step; i++) {
    const prev = pts[i - step], curr = pts[i], next = pts[i + step];
    const a1 = Math.atan2(curr[1] - prev[1], curr[0] - prev[0]);
    const a2 = Math.atan2(next[1] - curr[1], next[0] - curr[0]);
    let diff = Math.abs(a2 - a1);
    if (diff > Math.PI) diff = 2 * Math.PI - diff;
    angles.push({ idx: i, angle: diff });
  }

  // Sort by sharpest angle
  angles.sort((a, b) => b.angle - a.angle);

  // Pick top N corners that are far enough apart
  const minDist = Math.max(total / (n * 2), 5);
  const corners = [];
  for (const a of angles) {
    if (corners.length >= n) break;
    const tooClose = corners.some(c => Math.abs(c.idx - a.idx) < minDist);
    if (!tooClose && a.angle > 0.3) {
      corners.push(a);
    }
  }

  if (corners.length !== n) return [];
  corners.sort((a, b) => a.idx - b.idx);
  return corners.map(c => pts[c.idx]);
}

function erase(pt) {
  endStroke(pt);
  drawCtx.save();
  drawCtx.globalCompositeOperation = "destination-out";
  drawCtx.beginPath();
  drawCtx.arc(pt[0], pt[1], state.eraserSz, 0, Math.PI * 2);
  drawCtx.fill();
  drawCtx.restore();
}

function clearCanvas() {
  state.undoStack.push([...state.strokes]);
  if (state.undoStack.length > CFG.maxUndo) state.undoStack.shift();
  state.redoStack = [];
  state.strokes = [];
  state.currentStroke = null;
  drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
}

function undo() {
  if (state.undoStack.length === 0) return;
  state.redoStack.push([...state.strokes]);
  state.strokes = state.undoStack.pop();
  state.currentStroke = null;
  redrawAll();
}

function redo() {
  if (state.redoStack.length === 0) return;
  state.undoStack.push([...state.strokes]);
  state.strokes = state.redoStack.pop();
  state.currentStroke = null;
  redrawAll();
}

function redrawAll() {
  drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
  for (const s of state.strokes) {
    drawCtx.strokeStyle = s.color;
    drawCtx.lineWidth = s.thickness;
    drawCtx.lineCap = "round";
    drawCtx.lineJoin = "round";

    if (s.mode === DrawMode.FREEHAND) {
      for (let i = 1; i < s.points.length; i++) {
        drawCtx.beginPath();
        drawCtx.moveTo(s.points[i - 1][0], s.points[i - 1][1]);
        drawCtx.lineTo(s.points[i][0], s.points[i][1]);
        drawCtx.stroke();
      }
    } else if (s.mode === DrawMode.RECTANGLE && s.points.length === 2) {
      drawRectOnCanvas(s.points[0], s.points[1], s.color, s.thickness);
    } else if (s.mode === DrawMode.CIRCLE && s.points.length === 2) {
      const dx = s.points[1][0] - s.points[0][0], dy = s.points[1][1] - s.points[0][1];
      const r = s.radius || Math.sqrt(dx * dx + dy * dy);
      drawCircleOnCanvas(s.points[0], r, s.color, s.thickness);
    } else if (s.mode === "ellipse" && s.points.length >= 1) {
      drawEllipseOnCanvas(s.points[0], s.rx, s.ry, s.color, s.thickness);
    } else if (s.mode === "triangle" && s.points.length === 3) {
      drawPolygonOnCanvas(s.points, s.color, s.thickness);
    } else if (s.mode === "diamond" && s.points.length === 4) {
      drawPolygonOnCanvas(s.points, s.color, s.thickness);
    } else if (s.mode === "line" && s.points.length === 2) {
      drawLineOnCanvas(s.points[0], s.points[1], s.color, s.thickness);
    }
  }
}

function drawRectOnCanvas(p1, p2, color, thickness) {
  drawCtx.strokeStyle = color;
  drawCtx.lineWidth = thickness;
  drawCtx.lineCap = "round";
  drawCtx.lineJoin = "round";
  drawCtx.strokeRect(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1]);
}

function drawCircleOnCanvas(center, r, color, thickness) {
  drawCtx.strokeStyle = color;
  drawCtx.lineWidth = thickness;
  drawCtx.beginPath();
  drawCtx.arc(center[0], center[1], r, 0, Math.PI * 2);
  drawCtx.stroke();
}

function drawEllipseOnCanvas(center, rx, ry, color, thickness) {
  drawCtx.strokeStyle = color;
  drawCtx.lineWidth = thickness;
  drawCtx.beginPath();
  drawCtx.ellipse(center[0], center[1], rx, ry, 0, 0, Math.PI * 2);
  drawCtx.stroke();
}

function drawPolygonOnCanvas(vertices, color, thickness) {
  if (vertices.length < 3) return;
  drawCtx.strokeStyle = color;
  drawCtx.lineWidth = thickness;
  drawCtx.lineCap = "round";
  drawCtx.lineJoin = "round";
  drawCtx.beginPath();
  drawCtx.moveTo(vertices[0][0], vertices[0][1]);
  for (let i = 1; i < vertices.length; i++) {
    drawCtx.lineTo(vertices[i][0], vertices[i][1]);
  }
  drawCtx.closePath();
  drawCtx.stroke();
}

function drawLineOnCanvas(p1, p2, color, thickness) {
  drawCtx.strokeStyle = color;
  drawCtx.lineWidth = thickness;
  drawCtx.lineCap = "round";
  drawCtx.beginPath();
  drawCtx.moveTo(p1[0], p1[1]);
  drawCtx.lineTo(p2[0], p2[1]);
  drawCtx.stroke();
}

function renderShapePreview(pt) {
  // Preview is drawn on overlay canvas (non-destructive)
  if (!state.shapeStart) return;
  const c = currentColorCSS();
  const t = state.thickness;
  const ctx = overlayCtx;

  ctx.strokeStyle = c;
  ctx.lineWidth = t;
  ctx.setLineDash([6, 4]);

  if (state.drawMode === DrawMode.RECTANGLE) {
    ctx.strokeRect(state.shapeStart[0], state.shapeStart[1],
      pt[0] - state.shapeStart[0], pt[1] - state.shapeStart[1]);
  } else if (state.drawMode === DrawMode.CIRCLE) {
    const dx = pt[0] - state.shapeStart[0], dy = pt[1] - state.shapeStart[1];
    const r = Math.sqrt(dx * dx + dy * dy);
    ctx.beginPath();
    ctx.arc(state.shapeStart[0], state.shapeStart[1], r, 0, Math.PI * 2);
    ctx.stroke();
  }

  ctx.setLineDash([]);
}

function drawCursor(tip, size, isEraser) {
  const ctx = overlayCtx;
  if (isEraser) {
    ctx.strokeStyle = "rgba(180,180,180,0.7)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(tip[0], tip[1], size, 0, Math.PI * 2);
    ctx.stroke();
    // Crosshair
    ctx.beginPath();
    ctx.moveTo(tip[0] - size, tip[1]); ctx.lineTo(tip[0] + size, tip[1]);
    ctx.moveTo(tip[0], tip[1] - size); ctx.lineTo(tip[0], tip[1] + size);
    ctx.strokeStyle = "rgba(180,180,180,0.4)";
    ctx.lineWidth = 1;
    ctx.stroke();
  } else {
    ctx.fillStyle = currentColorCSS();
    ctx.beginPath();
    ctx.arc(tip[0], tip[1], Math.max(size / 2, 3), 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,0.6)";
    ctx.lineWidth = 1;
    ctx.stroke();
  }
}


// ── Color helpers ──────────────────────────────────────────────────────────────

function currentColorCSS() {
  const c = PALETTE[state.colorIdx].rgb;
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

function nextColor() {
  state.colorIdx = (state.colorIdx + 1) % PALETTE.length;
  updatePaletteUI();
}

function setColorIdx(i) {
  state.colorIdx = i % PALETTE.length;
  updatePaletteUI();
}


// ── Save ───────────────────────────────────────────────────────────────────────

function saveDrawing() {
  // Composite: white background + drawing
  const tmpCanvas = document.createElement("canvas");
  tmpCanvas.width = drawCanvas.width;
  tmpCanvas.height = drawCanvas.height;
  const tmpCtx = tmpCanvas.getContext("2d");
  tmpCtx.fillStyle = "#ffffff";
  tmpCtx.fillRect(0, 0, tmpCanvas.width, tmpCanvas.height);
  tmpCtx.drawImage(drawCanvas, 0, 0);

  const link = document.createElement("a");
  const ts = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  link.download = `air_drawing_${ts}.png`;
  link.href = tmpCanvas.toDataURL("image/png");
  link.click();
}


// ── Utility ────────────────────────────────────────────────────────────────────

function mostCommon(arr) {
  if (arr.length === 0) return Gesture.IDLE;
  const counts = {};
  for (const v of arr) counts[v] = (counts[v] || 0) + 1;
  let maxVal = arr[0], maxCount = 0;
  for (const [v, c] of Object.entries(counts)) {
    if (c > maxCount) { maxCount = c; maxVal = Number(v); }
  }
  return maxVal;
}


// ── UI Updates ─────────────────────────────────────────────────────────────────

function updateHUD(gesture, confidence) {
  elGestureLabel.textContent = GESTURE_NAMES[gesture] || "Idle";
  const pct = Math.round(confidence * 100);
  elConfFill.style.width = `${pct}%`;
  elConfFill.style.background = confidence >= 0.8
    ? "var(--success)"
    : "var(--warning)";
  elBrushInfo.textContent = `${state.thickness}px`;
  elModeInfo.textContent = state.drawMode.charAt(0).toUpperCase() + state.drawMode.slice(1);
}

function showStatus(msg) {
  elStatusToast.textContent = msg;
  elStatusToast.classList.add("visible");
  clearTimeout(state.statusTimeout);
  state.statusTimeout = setTimeout(() => {
    elStatusToast.classList.remove("visible");
  }, 1500);
}

function toggleHelp(force) {
  state.showHelp = force !== undefined ? force : !state.showHelp;
  elHelpPanel.classList.toggle("visible", state.showHelp);
}


// ── Palette UI ─────────────────────────────────────────────────────────────────

function buildPalette() {
  const bar = document.getElementById("palette-bar");
  PALETTE.forEach((c, i) => {
    const el = document.createElement("div");
    el.className = "swatch" + (i === state.colorIdx ? " active" : "");
    el.style.background = `rgb(${c.rgb.join(",")})`;
    el.dataset.idx = i;
    el.innerHTML = `<span class="key-hint">${i + 1}</span>`;
    el.addEventListener("click", () => { setColorIdx(i); });
    bar.appendChild(el);
  });

  const nameEl = document.createElement("span");
  nameEl.id = "color-name";
  nameEl.textContent = PALETTE[state.colorIdx].name;
  bar.appendChild(nameEl);
  elColorName = nameEl;
}

function updatePaletteUI() {
  const swatches = document.querySelectorAll(".swatch");
  swatches.forEach((el, i) => {
    el.classList.toggle("active", i === state.colorIdx);
  });
  if (elColorName) elColorName.textContent = PALETTE[state.colorIdx].name;
}


// ── Toolbar ────────────────────────────────────────────────────────────────────

function bindToolbar() {
  document.getElementById("btn-freehand").addEventListener("click", () => { state.drawMode = DrawMode.FREEHAND; updateModeButtons(); });
  document.getElementById("btn-rect").addEventListener("click", () => { state.drawMode = DrawMode.RECTANGLE; updateModeButtons(); });
  document.getElementById("btn-circle").addEventListener("click", () => { state.drawMode = DrawMode.CIRCLE; updateModeButtons(); });
  document.getElementById("btn-undo").addEventListener("click", () => { undo(); showStatus("Undo"); });
  document.getElementById("btn-redo").addEventListener("click", () => { redo(); showStatus("Redo"); });
  document.getElementById("btn-brush-up").addEventListener("click", () => {
    state.thickness = Math.min(state.thickness + CFG.brush.step, CFG.brush.max);
    showStatus(`Brush: ${state.thickness}px`);
  });
  document.getElementById("btn-brush-down").addEventListener("click", () => {
    state.thickness = Math.max(state.thickness - CFG.brush.step, CFG.brush.min);
    showStatus(`Brush: ${state.thickness}px`);
  });
  document.getElementById("btn-clear").addEventListener("click", () => { clearCanvas(); showStatus("Canvas cleared"); });
  document.getElementById("btn-save").addEventListener("click", () => { saveDrawing(); showStatus("Drawing saved!"); });
  document.getElementById("btn-help").addEventListener("click", () => toggleHelp());
}

function updateModeButtons() {
  document.getElementById("btn-freehand").classList.toggle("active", state.drawMode === DrawMode.FREEHAND);
  document.getElementById("btn-rect").classList.toggle("active", state.drawMode === DrawMode.RECTANGLE);
  document.getElementById("btn-circle").classList.toggle("active", state.drawMode === DrawMode.CIRCLE);
  elModeInfo.textContent = state.drawMode.charAt(0).toUpperCase() + state.drawMode.slice(1);
}


// ── Keyboard shortcuts ─────────────────────────────────────────────────────────

function bindKeyboard() {
  document.addEventListener("keydown", (e) => {
    const key = e.key.toLowerCase();
    if (key === "s") { saveDrawing(); showStatus("Drawing saved!"); }
    else if (key === "h") { toggleHelp(); }
    else if (key === "+" || key === "=") {
      state.thickness = Math.min(state.thickness + CFG.brush.step, CFG.brush.max);
      showStatus(`Brush: ${state.thickness}px`);
    }
    else if (key === "-") {
      state.thickness = Math.max(state.thickness - CFG.brush.step, CFG.brush.min);
      showStatus(`Brush: ${state.thickness}px`);
    }
    else if (key === "[") {
      state.eraserSz = Math.max(state.eraserSz - CFG.eraser.step, CFG.eraser.min);
      showStatus(`Eraser: ${state.eraserSz}px`);
    }
    else if (key === "]") {
      state.eraserSz = Math.min(state.eraserSz + CFG.eraser.step, CFG.eraser.max);
      showStatus(`Eraser: ${state.eraserSz}px`);
    }
    else if (key === "u") { undo(); showStatus("Undo"); }
    else if (key === "r") { redo(); showStatus("Redo"); }
    else if (key === "f") { state.drawMode = DrawMode.FREEHAND; updateModeButtons(); showStatus("Freehand"); }
    else if (key === "x") { state.drawMode = DrawMode.RECTANGLE; updateModeButtons(); showStatus("Rectangle"); }
    else if (key === "c") { state.drawMode = DrawMode.CIRCLE; updateModeButtons(); showStatus("Circle"); }
    else if (key >= "1" && key <= "8") {
      setColorIdx(parseInt(key) - 1);
      showStatus(`Color: ${PALETTE[state.colorIdx].name}`);
    }
  });
}
