#!/usr/bin/env python3
"""Web-based natural language robot navigator.

Open http://localhost:5000 in your browser.
Type commands like:
  - "go to the person"
  - "find a chair"
  - "navigate to the bottle on the table"
  - "stop"

The system uses a local LLM-free approach:
1. Parses your command to extract the target object
2. Matches it to YOLO's 80 object classes using fuzzy matching
3. YOLO detects the object in the camera feed
4. DA2 estimates depth/distance
5. MPPI plans a collision-free trajectory
6. Robot drives there

Runs on http://localhost:5000
"""

import sys
import os
import base64
import io
import json
import time
import threading
from difflib import get_close_matches

import numpy as np
import cv2
import requests
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'third_party', 'Depth-Anything-V2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'third_party', 'Depth-Anything-V2', 'metric_depth'))

SDK_URL = "http://localhost:8000"
FLIP_LINEAR = False
FOV_H_DEG = 90.0

# ---------------------------------------------------------------------------
# Robot control
# ---------------------------------------------------------------------------

def send_control(linear, angular):
    try:
        if FLIP_LINEAR:
            linear = -linear
        requests.post(f"{SDK_URL}/control-legacy",
                      json={"command": {"linear": linear, "angular": angular, "lamp": 0}},
                      timeout=1.0)
    except Exception:
        pass


def get_frame():
    try:
        resp = requests.get(f"{SDK_URL}/v2/front", timeout=3)
        data = resp.json()
        b64 = data.get("front_frame")
        if b64:
            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            if img.size[0] > 10:
                return np.array(img, dtype=np.uint8)
    except Exception:
        pass
    return None


def pixel_to_angle(px, img_width, fov_deg=FOV_H_DEG):
    center = img_width / 2.0
    fov_rad = np.radians(fov_deg)
    return float(np.arctan((px - center) / (img_width / (2 * np.tan(fov_rad / 2)))))


# ---------------------------------------------------------------------------
# YOLO class names (COCO 80)
# ---------------------------------------------------------------------------

YOLO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# Common aliases
ALIASES = {
    'human': 'person', 'people': 'person', 'man': 'person', 'woman': 'person',
    'guy': 'person', 'someone': 'person', 'somebody': 'person',
    'sofa': 'couch', 'monitor': 'tv', 'screen': 'tv', 'television': 'tv',
    'phone': 'cell phone', 'mobile': 'cell phone', 'cellphone': 'cell phone',
    'pc': 'laptop', 'computer': 'laptop', 'notebook': 'laptop',
    'bag': 'backpack', 'rucksack': 'backpack',
    'mug': 'cup', 'glass': 'wine glass',
    'desk': 'dining table', 'table': 'dining table',
    'plant': 'potted plant', 'flower': 'potted plant',
    'fridge': 'refrigerator', 'bike': 'bicycle', 'motorbike': 'motorcycle',
    'auto': 'car', 'vehicle': 'car', 'van': 'truck',
    'puppy': 'dog', 'kitten': 'cat',
    'ball': 'sports ball', 'toy': 'teddy bear',
}


def parse_target(command: str) -> tuple[str, str]:
    """Parse natural language command into a YOLO class name.

    Returns (matched_class, explanation).
    """
    cmd = command.lower().strip()

    if not cmd or cmd in ('stop', 'halt', 'freeze', 'wait'):
        return '', 'Stopping robot.'

    # Remove common prefixes
    for prefix in ['go to the ', 'go to ', 'navigate to the ', 'navigate to ',
                   'find the ', 'find a ', 'find ', 'drive to the ', 'drive to ',
                   'move to the ', 'move to ', 'head to the ', 'head to ',
                   'approach the ', 'approach ', 'get to the ', 'get to ',
                   'look for the ', 'look for a ', 'look for ',
                   'go towards the ', 'go towards ']:
        if cmd.startswith(prefix):
            cmd = cmd[len(prefix):]
            break

    # Remove trailing words
    for suffix in [' please', ' now', ' quickly', ' slowly', ' over there',
                   ' near me', ' on the left', ' on the right', ' ahead']:
        if cmd.endswith(suffix):
            cmd = cmd[:-len(suffix)]

    target = cmd.strip()

    # Direct match
    if target in YOLO_CLASSES:
        return target, f'Found exact match: {target}'

    # Alias match
    if target in ALIASES:
        matched = ALIASES[target]
        return matched, f'"{target}" → {matched}'

    # Fuzzy match against YOLO classes
    matches = get_close_matches(target, YOLO_CLASSES, n=1, cutoff=0.5)
    if matches:
        return matches[0], f'Best match for "{target}": {matches[0]}'

    # Fuzzy match against aliases
    matches = get_close_matches(target, list(ALIASES.keys()), n=1, cutoff=0.5)
    if matches:
        matched = ALIASES[matches[0]]
        return matched, f'"{target}" ≈ "{matches[0]}" → {matched}'

    # Substring match
    for cls in YOLO_CLASSES:
        if target in cls or cls in target:
            return cls, f'Partial match: "{target}" → {cls}'

    return '', f'Cannot find "{target}" in YOLO classes. Try: {", ".join(YOLO_CLASSES[:10])}...'


# ---------------------------------------------------------------------------
# Navigation loop (runs in background)
# ---------------------------------------------------------------------------

class Navigator:
    def __init__(self):
        self.target_class = ''
        self.auto_mode = True
        self.running = True
        self.linear = 0.0
        self.angular = 0.0
        self.lock = threading.Lock()

        # State for web UI
        self.last_frame_b64 = ''
        self.last_annotated_b64 = ''
        self.last_depth_b64 = ''
        self.last_mppi_b64 = ''
        self.detected_objects = []
        self.target_info = {}
        self.status = 'Initializing...'
        self.fps = 0.0
        # Persistence: remember last known target position
        self.last_target_angle = None  # last known angle to target
        self.target_lost_frames = 0    # how many frames since last seen

        # Load models
        print("Loading YOLO...")
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        from ultralytics import YOLO
        self.yolo = YOLO("yolo11m.pt")
        self.yolo.to(device)
        print(f"YOLO on {device}")
        print("Loading DA2...")
        from depth_estimator import DepthEstimator
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.estimator = DepthEstimator(model_size='small', max_depth=10.0, device=device)
        print(f"DA2 on {device}")
        print("Loading MPPI...")
        from T_mppi_planner import MPPIPlanner, MPPIConfig
        self.mppi = MPPIPlanner(MPPIConfig(
            num_samples=512, horizon=12, dt=0.15,
            max_linear=0.30, max_angular=0.45,
        ))
        self.status = 'Ready — type a command!'
        print("All models loaded.")

    def set_target(self, target_class: str):
        with self.lock:
            self.target_class = target_class
            print(f"[NAV] Target set to: '{target_class}'")
            if not target_class:
                self.linear = 0.0
                self.angular = 0.0
                send_control(0, 0)

    def run(self):
        frame_count = 0
        while self.running:
            t0 = time.time()

            frame = get_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            h, w = frame.shape[:2]

            # YOLO
            results = self.yolo(frame, verbose=False, conf=0.2, device='cuda')

            # Collect detections
            detections = []
            for r in results:
                if r.boxes is None:
                    continue
                for i in range(len(r.boxes)):
                    cls_id = int(r.boxes.cls[i])
                    detections.append({
                        'class': r.names[cls_id],
                        'confidence': float(r.boxes.conf[i]),
                        'bbox': r.boxes.xyxy[i].tolist(),
                    })
            self.detected_objects = list(set(d['class'] for d in detections))

            # DA2 depth
            depth = self.estimator.estimate(frame)

            # Find target
            with self.lock:
                current_target = self.target_class

            if frame_count % 20 == 0:
                print(f"[LOOP] target_class='{current_target}' detections={[d['class'] for d in detections]}")

            target_det = None
            goal_dir = 0.0
            target_dist = None
            target_angle = None

            if current_target:
                best_conf = 0
                for det in detections:
                    if det['class'] == current_target and det['confidence'] > best_conf:
                        x1, y1, x2, y2 = det['bbox']
                        target_det = {
                            **det,
                            'center_x': (x1 + x2) / 2,
                            'center_y': (y1 + y2) / 2,
                        }
                        best_conf = det['confidence']

                if target_det:
                    cx = (target_det['bbox'][0] + target_det['bbox'][2]) / 2
                    cy = (target_det['bbox'][1] + target_det['bbox'][3]) / 2
                    target_angle = pixel_to_angle(cx, w)
                    goal_dir = target_angle

                    dh, dw = depth.shape
                    dx = min(max(int(cx * dw / w), 0), dw - 1)
                    dy = min(max(int(cy * dh / h), 0), dh - 1)
                    r = 5
                    region = depth[max(0, dy-r):min(dh, dy+r), max(0, dx-r):min(dw, dx+r)]
                    if region.size > 0:
                        target_dist = float(np.median(region))

            # Visual servoing: keep target bounding box centered in frame
            cmd_lin = 0.0
            cmd_ang = 0.0
            STEER_GAIN = 0.35
            MAX_LOST_FRAMES = 15  # keep heading toward last known position for this many frames

            if current_target and target_det:
                # Target found — reset lost counter, remember angle
                self.target_lost_frames = 0
                cx = target_det['center_x']
                frame_center = w / 2.0
                pixel_error = cx - frame_center
                normalized_error = pixel_error / (w / 2.0)
                self.last_target_angle = normalized_error

                cmd_ang = -STEER_GAIN * normalized_error

                if target_dist is not None and target_dist < 0.8:
                    cmd_lin = 0.0
                    cmd_ang = 0.0
                    self.status = f'ARRIVED at {current_target}! ({target_dist:.1f}m)'
                elif abs(normalized_error) > 0.4:
                    cmd_lin = 0.05  # creep forward while centering
                    self.status = f'Centering {current_target}...'
                else:
                    cmd_lin = 0.30
                    if target_dist is not None and target_dist < 2.0:
                        cmd_lin = max(0.10, 0.15 * target_dist)
                    dist_str = f'{target_dist:.1f}m' if target_dist else '?m'
                    self.status = f'Navigating → {current_target} ({dist_str}, err={normalized_error:+.2f})'

            elif current_target:
                self.target_lost_frames += 1

                if self.target_lost_frames < MAX_LOST_FRAMES and self.last_target_angle is not None:
                    # Recently lost — keep going in last known direction
                    cmd_ang = -STEER_GAIN * self.last_target_angle * 0.5
                    cmd_lin = 0.15
                    self.status = f'Lost {current_target} — heading to last position ({self.target_lost_frames}/{MAX_LOST_FRAMES})'
                else:
                    # Lost for too long — slow search spin
                    cmd_lin = 0.0
                    cmd_ang = 0.15  # gentle spin
                    self.status = f'Searching for {current_target}...'
            else:
                self.status = 'Idle — type a command!'
                self.last_target_angle = None
                self.target_lost_frames = 0

            if self.auto_mode:
                self.linear = cmd_lin
                self.angular = cmd_ang

            send_control(self.linear, self.angular)

            # --- Build annotated frame ---
            frame_draw = frame.copy()
            for det in detections:
                x1, y1, x2, y2 = [int(v) for v in det['bbox']]
                is_target = (target_det and det is target_det)
                color = (0, 255, 0) if is_target else (100, 100, 255)
                thick = 3 if is_target else 1
                cv2.rectangle(frame_draw, (x1, y1), (x2, y2), color, thick)
                label = f"{det['class']} {det['confidence']:.0%}"
                if is_target and target_dist:
                    label += f" {target_dist:.1f}m"
                cv2.putText(frame_draw, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw center line + target tracking indicator
            center_x = w // 2
            cv2.line(frame_draw, (center_x, 0), (center_x, h), (255, 255, 0), 1)
            if target_det:
                tcx = int(target_det['center_x'])
                tcy = int(target_det['center_y'])
                # Line from center to target
                cv2.line(frame_draw, (center_x, h // 2), (tcx, tcy), (0, 255, 0), 2)
                # Crosshair on target center
                cv2.circle(frame_draw, (tcx, tcy), 10, (0, 255, 0), 2)
                cv2.drawMarker(frame_draw, (tcx, tcy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

            # Encode images for web
            _, buf = cv2.imencode('.jpg', cv2.cvtColor(frame_draw, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
            self.last_annotated_b64 = base64.b64encode(buf).decode()

            # Depth colormap
            depth_norm = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-6) * 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_PLASMA)
            _, buf2 = cv2.imencode('.jpg', depth_color, [cv2.IMWRITE_JPEG_QUALITY, 70])
            self.last_depth_b64 = base64.b64encode(buf2).decode()

            # MPPI trajectory — render with OpenCV (fast, no matplotlib)
            _, _, mppi_debug = self.mppi.plan(depth, goal_direction_rad=goal_dir)
            mppi_img = np.zeros((300, 400, 3), dtype=np.uint8)
            mppi_img[:] = (10, 10, 10)
            # Grid
            cv2.line(mppi_img, (200, 0), (200, 300), (40, 40, 40), 1)
            cv2.line(mppi_img, (0, 150), (400, 150), (40, 40, 40), 1)
            # Scale: 1m = 100px, origin at (50, 150)
            ox, oy, scale = 50, 250, 100
            def to_px(x, y):
                return int(ox + x * scale), int(oy - y * scale)
            # Top trajectories
            if "top_x" in mppi_debug:
                for i in range(len(mppi_debug["top_x"])):
                    pts = [to_px(mppi_debug["top_x"][i][j], mppi_debug["top_y"][i][j])
                           for j in range(len(mppi_debug["top_x"][i]))]
                    for k in range(len(pts) - 1):
                        cv2.line(mppi_img, pts[k], pts[k+1], (60, 60, 60), 1)
            # Best trajectory
            best_x = mppi_debug["best_x"]
            best_y = mppi_debug["best_y"]
            cost = mppi_debug["best_cost"]
            clr = (0, 255, 100) if cost < 5 else (0, 170, 255) if cost < 15 else (0, 0, 255)
            pts = [to_px(best_x[j], best_y[j]) for j in range(len(best_x))]
            for k in range(len(pts) - 1):
                cv2.line(mppi_img, pts[k], pts[k+1], clr, 3)
            cv2.circle(mppi_img, pts[-1], 6, clr, -1)
            # Robot
            cv2.rectangle(mppi_img, (ox-5, oy-5), (ox+5, oy+5), (255, 255, 255), -1)
            # Goal direction
            if target_angle is not None:
                gx, gy = to_px(1.5 * np.cos(target_angle), 1.5 * np.sin(target_angle))
                cv2.line(mppi_img, (ox, oy), (gx, gy), (255, 0, 255), 2)
                cv2.drawMarker(mppi_img, (gx, gy), (255, 0, 255), cv2.MARKER_STAR, 15, 2)
            cv2.putText(mppi_img, f'MPPI cost={cost:.1f}', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (233, 69, 96), 2)
            _, buf3 = cv2.imencode('.jpg', mppi_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            self.last_mppi_b64 = base64.b64encode(buf3).decode()

            self.target_info = {
                'target': current_target,
                'found': target_det is not None,
                'distance': target_dist,
                'angle': float(np.degrees(target_angle)) if target_angle else None,
                'linear': self.linear,
                'angular': self.angular,
            }

            dt = time.time() - t0
            self.fps = 1.0 / max(dt, 0.01)
            frame_count += 1

            if frame_count % 20 == 0:
                print(f"[{frame_count}] {self.status} | fps={self.fps:.1f}")

            elapsed = time.time() - t0
            if elapsed < 0.1:
                time.sleep(0.1 - elapsed)


# ---------------------------------------------------------------------------
# Web server (Flask-free, pure stdlib)
# ---------------------------------------------------------------------------

from http.server import HTTPServer, BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

navigator = None

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Robot Navigator</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0a0a0a; color: #e0e0e0; }
.header { background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 20px; text-align: center; border-bottom: 2px solid #0f3460; }
.header h1 { font-size: 28px; color: #e94560; }
.header p { color: #888; margin-top: 5px; }
.main { display: flex; flex-wrap: wrap; gap: 15px; padding: 15px; justify-content: center; }
.panel { background: #1a1a2e; border-radius: 12px; overflow: hidden; border: 1px solid #2a2a4a; }
.panel-title { background: #16213e; padding: 8px 15px; font-size: 14px; color: #0f3460; font-weight: 600; color: #e94560; }
.video-panel { width: 31%; min-width: 300px; }
.video-panel img { width: 100%; display: block; }
.command-panel { width: 96%; min-width: 400px; padding: 20px; }
.command-panel:first-of-type { order: -1; }
.input-row { display: flex; gap: 10px; margin-bottom: 15px; }
.input-row input { flex: 1; padding: 14px 20px; font-size: 18px; border: 2px solid #2a2a4a; border-radius: 8px; background: #0a0a0a; color: #fff; outline: none; }
.input-row input:focus { border-color: #e94560; }
.input-row button { padding: 14px 30px; font-size: 18px; background: #e94560; color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: 600; }
.input-row button:hover { background: #c73450; }
.status-bar { display: flex; gap: 20px; flex-wrap: wrap; align-items: center; }
.status-item { background: #0a0a0a; padding: 10px 18px; border-radius: 8px; font-size: 14px; }
.status-item.active { border: 1px solid #e94560; }
.status-item .label { color: #888; font-size: 11px; text-transform: uppercase; }
.status-item .value { font-size: 18px; font-weight: 600; margin-top: 2px; }
.objects-list { margin-top: 15px; display: flex; gap: 8px; flex-wrap: wrap; }
.obj-tag { background: #16213e; padding: 6px 14px; border-radius: 20px; font-size: 13px; cursor: pointer; border: 1px solid #2a2a4a; }
.obj-tag:hover { border-color: #e94560; color: #e94560; }
.obj-tag.target { background: #e94560; color: white; border-color: #e94560; }
.quick-cmds { margin-top: 10px; display: flex; gap: 8px; flex-wrap: wrap; }
.quick-btn { background: #0f3460; padding: 8px 16px; border-radius: 6px; font-size: 13px; cursor: pointer; border: none; color: #ddd; }
.quick-btn:hover { background: #1a4a80; }
.quick-btn.stop { background: #c73450; }
</style>
</head>
<body>
<div class="header">
    <h1>ROBOT NAVIGATOR</h1>
    <p>YOLO + Depth Anything V2 + MPPI Trajectory Planning</p>
</div>

<div class="main">
    <div class="panel command-panel">
        <div class="input-row">
            <input type="text" id="cmd-input" placeholder="Tell the robot where to go... (e.g. 'go to the chair')" autofocus>
            <button onclick="sendCommand()">GO</button>
        </div>

        <div class="quick-cmds">
            <button class="quick-btn" onclick="quickCmd('person')">Person</button>
            <button class="quick-btn" onclick="quickCmd('chair')">Chair</button>
            <button class="quick-btn" onclick="quickCmd('bottle')">Bottle</button>
            <button class="quick-btn" onclick="quickCmd('laptop')">Laptop</button>
            <button class="quick-btn" onclick="quickCmd('backpack')">Backpack</button>
            <button class="quick-btn" onclick="quickCmd('cup')">Cup</button>
            <button class="quick-btn" onclick="quickCmd('tv')">TV</button>
            <button class="quick-btn" onclick="quickCmd('book')">Book</button>
            <button class="quick-btn stop" onclick="quickCmd('stop')">STOP</button>
        </div>

        <div class="status-bar" style="margin-top:15px;">
            <div class="status-item active" id="status-box">
                <div class="label">Status</div>
                <div class="value" id="status-text">Initializing...</div>
            </div>
            <div class="status-item">
                <div class="label">Target</div>
                <div class="value" id="target-text">—</div>
            </div>
            <div class="status-item">
                <div class="label">Distance</div>
                <div class="value" id="dist-text">—</div>
            </div>
            <div class="status-item">
                <div class="label">Angle</div>
                <div class="value" id="angle-text">—</div>
            </div>
            <div class="status-item">
                <div class="label">Command</div>
                <div class="value" id="cmd-text">—</div>
            </div>
            <div class="status-item">
                <div class="label">FPS</div>
                <div class="value" id="fps-text">—</div>
            </div>
        </div>

        <div class="objects-list" id="objects-list"></div>
    </div>

    <div class="panel video-panel">
        <div class="panel-title">YOLO Object Detection</div>
        <img id="yolo-img" src="" alt="YOLO feed">
    </div>
    <div class="panel video-panel">
        <div class="panel-title">DA2 Depth Map</div>
        <img id="depth-img" src="" alt="Depth feed">
    </div>
    <div class="panel video-panel">
        <div class="panel-title">MPPI Trajectory</div>
        <img id="mppi-img" src="" alt="MPPI trajectories">
    </div>
</div>

<script>
function sendCommand() {
    const input = document.getElementById('cmd-input');
    const cmd = input.value.trim();
    if (!cmd) return;
    fetch('/command', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({command: cmd})})
    .then(r => r.json())
    .then(d => {
        document.getElementById('status-text').textContent = d.message || d.status;
        input.value = '';
    });
}

function quickCmd(obj) {
    if (obj === 'stop') {
        fetch('/command', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({command: 'stop'})})
        .then(r => r.json()).then(d => { document.getElementById('status-text').textContent = 'Stopped'; });
    } else {
        document.getElementById('cmd-input').value = 'go to the ' + obj;
        sendCommand();
    }
}

document.getElementById('cmd-input').addEventListener('keypress', e => { if (e.key === 'Enter') sendCommand(); });

function updateFeed() {
    fetch('/state').then(r => r.json()).then(d => {
        if (d.annotated) document.getElementById('yolo-img').src = 'data:image/jpeg;base64,' + d.annotated;
        if (d.depth) document.getElementById('depth-img').src = 'data:image/jpeg;base64,' + d.depth;
        if (d.mppi) document.getElementById('mppi-img').src = 'data:image/jpeg;base64,' + d.mppi;
        document.getElementById('status-text').textContent = d.status || '—';
        document.getElementById('target-text').textContent = d.target || '—';
        document.getElementById('dist-text').textContent = d.distance ? d.distance.toFixed(1) + 'm' : '—';
        document.getElementById('angle-text').textContent = d.angle ? d.angle.toFixed(0) + '°' : '—';
        document.getElementById('cmd-text').textContent = '(' + (d.linear||0).toFixed(2) + ', ' + (d.angular||0).toFixed(2) + ')';
        document.getElementById('fps-text').textContent = (d.fps||0).toFixed(1);

        const objList = document.getElementById('objects-list');
        objList.innerHTML = '';
        (d.objects || []).sort().forEach(obj => {
            const tag = document.createElement('span');
            tag.className = 'obj-tag' + (obj === d.target ? ' target' : '');
            tag.textContent = obj;
            tag.onclick = () => { document.getElementById('cmd-input').value = 'go to the ' + obj; sendCommand(); };
            objList.appendChild(tag);
        });
    }).catch(() => {});
}

setInterval(updateFeed, 500);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress logs

    def do_GET(self):
        path = urlparse(self.path).path

        if path == '/' or path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode())

        elif path == '/state':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            state = {
                'annotated': navigator.last_annotated_b64 if navigator else '',
                'depth': navigator.last_depth_b64 if navigator else '',
                'mppi': navigator.last_mppi_b64 if navigator else '',
                'status': navigator.status if navigator else '',
                'target': navigator.target_info.get('target', ''),
                'distance': navigator.target_info.get('distance'),
                'angle': navigator.target_info.get('angle'),
                'linear': navigator.target_info.get('linear', 0),
                'angular': navigator.target_info.get('angular', 0),
                'objects': navigator.detected_objects if navigator else [],
                'fps': navigator.fps if navigator else 0,
            }
            self.wfile.write(json.dumps(state).encode())
        else:
            self.send_error(404)

    def do_POST(self):
        if urlparse(self.path).path == '/command':
            length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(length))
            command = body.get('command', '')

            matched, explanation = parse_target(command)
            nav_exists = navigator is not None
            if nav_exists:
                navigator.set_target(matched)
                readback = navigator.target_class

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'matched': matched,
                'message': explanation,
                'status': 'ok',
                'nav_exists': nav_exists,
                'readback': readback if nav_exists else None,
            }).encode())
        else:
            self.send_error(404)


def main():
    global navigator

    navigator = Navigator()

    # Start nav loop in background
    nav_thread = threading.Thread(target=navigator.run, daemon=True)
    nav_thread.start()

    # Start web server
    port = 5000
    server = ThreadingHTTPServer(('0.0.0.0', port), Handler)
    print(f"\n{'='*50}")
    print(f"  ROBOT NAVIGATOR ready!")
    print(f"  Open http://localhost:{port}")
    print(f"{'='*50}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        navigator.running = False
        send_control(0, 0)
        server.server_close()
        print("\nStopped.")


if __name__ == '__main__':
    main()
