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

import numpy as np
import cv2
import requests
from PIL import Image

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(_ROOT, 'frodo_ai', 'perception'))
sys.path.insert(0, os.path.join(_ROOT, 'frodo_ai', 'planning'))
sys.path.insert(0, os.path.join(_ROOT, 'third_party', 'Depth-Anything-V2'))
sys.path.insert(0, os.path.join(_ROOT, 'third_party', 'Depth-Anything-V2', 'metric_depth'))

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
        else:
            print(f"[get_frame] No front_frame in response: {data}")
    except Exception as e:
        print(f"[get_frame] Error: {e}")
    return None


def pixel_to_angle(px, img_width, fov_deg=FOV_H_DEG):
    center = img_width / 2.0
    fov_rad = np.radians(fov_deg)
    return float(np.arctan((px - center) / (img_width / (2 * np.tan(fov_rad / 2)))))


# ---------------------------------------------------------------------------
# NLP parsing — single source of truth in object_detector.py
# ---------------------------------------------------------------------------
from object_detector import parse_target


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
        self.last_target_cx = None     # last known centroid x (for instance lock)
        self.last_target_cy = None     # last known centroid y

        # Smooth control state
        self._smooth_ang = 0.0
        self._smooth_lin = 0.0

        # Search state
        self.search_start_time = None  # set when search spin begins

        # Load models
        print("Loading YOLO 26m...")
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        from ultralytics import YOLO
        self.yolo = YOLO("yolo26m.pt")
        self.yolo.to(device)
        print(f"YOLO on {device}")
        print("Loading DA2...")
        from depth_estimator import DepthEstimator
        self.estimator = DepthEstimator(model_size='small', max_depth=10.0, device=device)
        print(f"DA2 on {device}")
        print("Loading MPPI...")
        from mppi_planner import MPPIPlanner, MPPIConfig
        self.mppi = MPPIPlanner(MPPIConfig(
            num_samples=512, horizon=12, dt=0.15,
            max_linear=0.30, max_angular=0.45,
        ))
        self.status = 'Ready — type a command!'
        print("All models loaded.")

    def set_target(self, target_class: str):
        with self.lock:
            self.target_class = target_class
            self.search_start_time = None  # reset search on new command
            print(f"[NAV] Target set to: '{target_class}'")
            if not target_class:
                self.linear = 0.0
                self.angular = 0.0
                send_control(0, 0)

    def run(self):
        frame_count = 0
        while self.running:
            try:
                t0 = time.time()

                frame = get_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue

                h, w = frame.shape[:2]

                # YOLO — runs every frame, detects all 80 COCO classes
                results = self.yolo(frame, verbose=False, conf=0.2, device=self.device)
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

                # Find target (read before depth so we can skip DA2 when idle)
                with self.lock:
                    current_target = self.target_class

                # DA2 depth — skip when idle to avoid bottlenecking at ~7fps for nothing
                depth = self.estimator.estimate(frame) if current_target else None

                if frame_count % 20 == 0:
                    print(f"[LOOP] target_class='{current_target}' detections={[d['class'] for d in detections]}")

                target_det = None
                goal_dir = 0.0
                target_dist = None
                target_angle = None

                if current_target:
                    candidates = []
                    for det in detections:
                        if det['class'] == current_target:
                            x1, y1, x2, y2 = det['bbox']
                            cx_ = (x1 + x2) / 2
                            cy_ = (y1 + y2) / 2
                            candidates.append({**det, 'center_x': cx_, 'center_y': cy_})

                    if candidates:
                        if self.last_target_cx is not None:
                            # Always lock onto the instance closest to last known position
                            target_det = min(
                                candidates,
                                key=lambda d: (d['center_x'] - self.last_target_cx) ** 2
                                            + (d['center_y'] - self.last_target_cy) ** 2
                            )
                        else:
                            # First detection ever: pick highest confidence
                            target_det = max(candidates, key=lambda d: d['confidence'])

                    if target_det:
                        cx = (target_det['bbox'][0] + target_det['bbox'][2]) / 2
                        cy = (target_det['bbox'][1] + target_det['bbox'][3]) / 2
                        target_angle = pixel_to_angle(cx, w)
                        goal_dir = target_angle

                        dh, dw = depth.shape
                        dx = min(max(int(cx * dw / w), 0), dw - 1)
                        dy = min(max(int(cy * dh / h), 0), dh - 1)
                        rr = 5
                        region = depth[max(0, dy-rr):min(dh, dy+rr), max(0, dx-rr):min(dw, dx+rr)]
                        if region.size > 0:
                            target_dist = float(np.median(region))

                # -------------------------------------------------------
                # Visual servo + MPPI path planning
                # MPPI controls linear speed (obstacle-aware).
                # Visual servo controls angular (pixel-accurate target tracking).
                # -------------------------------------------------------
                STEER_GAIN    = 0.25
                EMA_ANG       = 0.50
                LIN_RAMP      = 0.05
                DEADBAND      = 0.05
                MAX_LOST      = 15
                ARRIVE_DIST   = 1.2    # metres — stop this far from the target
                ARRIVE_BBOX   = 0.55   # fraction of frame height — stop if bbox this tall

                raw_cmd_ang = 0.0
                raw_cmd_lin = 0.0
                mppi_debug   = None

                if current_target and target_det:
                    self.target_lost_frames = 0
                    self.search_start_time = None  # target re-acquired — cancel search
                    cx = target_det['center_x']
                    cy = target_det['center_y']
                    self.last_target_cx = cx
                    self.last_target_cy = cy

                    frame_center = w / 2.0
                    normalized_error = (cx - frame_center) / (w / 2.0)
                    self.last_target_angle = normalized_error
                    goal_dir = pixel_to_angle(cx, w)

                    if abs(normalized_error) < DEADBAND:
                        normalized_error = 0.0

                    # Arrival check: depth threshold OR bbox-fill guard (for when
                    # depth fails at very close range and chair fills the frame)
                    x1, y1, x2, y2 = target_det['bbox']
                    bbox_height_frac = (y2 - y1) / h
                    dist_arrived = (target_dist is not None and target_dist < ARRIVE_DIST)
                    bbox_arrived = (bbox_height_frac > ARRIVE_BBOX)

                    if dist_arrived or bbox_arrived:
                        # Hard-zero EMA — robot stops immediately, no angular bleed
                        self._smooth_ang = 0.0
                        self._smooth_lin = 0.0
                        dist_str = f'{target_dist:.2f}m' if target_dist else f'bbox {bbox_height_frac:.0%}'
                        self.status = f'ARRIVED at {current_target}! ({dist_str})'
                    else:
                        raw_cmd_ang = -STEER_GAIN * normalized_error

                        if depth is not None:
                            # MPPI controls linear speed only — slows down near obstacles.
                            # Visual servo has sole control of angular — keeps target centred.
                            mppi_lin, _, mppi_debug = self.mppi.plan(depth, goal_direction_rad=goal_dir)
                            raw_cmd_lin = mppi_lin
                        else:
                            speed_scale = max(0.25, 1.0 - abs(normalized_error))
                            raw_cmd_lin = 0.30 * speed_scale

                        dist_str = f'{target_dist:.1f}m' if target_dist else '?m'
                        self.status = f'Navigating → {current_target} ({dist_str}, err={normalized_error:+.2f})'

                elif current_target:
                    self.target_lost_frames += 1
                    if self.target_lost_frames < MAX_LOST and self.last_target_angle is not None:
                        raw_cmd_ang = -STEER_GAIN * self.last_target_angle * 0.5
                        raw_cmd_lin = 0.10
                        self.status = f'Lost {current_target} — heading to last position ({self.target_lost_frames}/{MAX_LOST})'
                    else:
                        # Timed 360° search — one full rotation at 0.22 rad/s ≈ 28s, then give up.
                        # Direction: toward where target was last seen to re-acquire faster.
                        if self.search_start_time is None:
                            self.search_start_time = time.time()
                        elapsed = time.time() - self.search_start_time
                        SEARCH_TIMEOUT = 30.0  # seconds
                        if elapsed < SEARCH_TIMEOUT:
                            search_dir = -1.0 if (self.last_target_angle and self.last_target_angle > 0) else 1.0
                            self._smooth_ang = 0.22 * search_dir  # bypass EMA for immediate spin
                            raw_cmd_ang = 0.22 * search_dir
                            raw_cmd_lin = 0.0
                            pct = int(elapsed / SEARCH_TIMEOUT * 100)
                            self.status = f'Searching for {current_target}... ({pct}%)'
                        else:
                            # Full rotation done — give up and clear target
                            self.search_start_time = None
                            with self.lock:
                                self.target_class = ''
                            self._smooth_ang = 0.0
                            self._smooth_lin = 0.0
                            self.status = f'Could not find {current_target}. Try a new command.'
                else:
                    self.status = 'Idle — type a command!'
                    self.last_target_angle = None
                    self.last_target_cx = None
                    self.last_target_cy = None
                    self.target_lost_frames = 0

                # EMA on angular, ramp on linear
                self._smooth_ang = EMA_ANG * raw_cmd_ang + (1.0 - EMA_ANG) * self._smooth_ang
                lin_delta = raw_cmd_lin - self._smooth_lin
                lin_delta = max(-LIN_RAMP, min(LIN_RAMP, lin_delta))
                self._smooth_lin += lin_delta

                cmd_lin = self._smooth_lin
                cmd_ang = self._smooth_ang

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

                center_x = w // 2
                cv2.line(frame_draw, (center_x, 0), (center_x, h), (255, 255, 0), 1)
                if target_det:
                    tcx = int(target_det['center_x'])
                    tcy = int(target_det['center_y'])
                    cv2.line(frame_draw, (center_x, h // 2), (tcx, tcy), (0, 255, 0), 2)
                    cv2.circle(frame_draw, (tcx, tcy), 10, (0, 255, 0), 2)
                    cv2.drawMarker(frame_draw, (tcx, tcy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

                # Encode images for web
                _, buf = cv2.imencode('.jpg', cv2.cvtColor(frame_draw, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
                self.last_annotated_b64 = base64.b64encode(buf).decode()

                # Depth colormap (only when depth was computed)
                if depth is not None:
                    depth_norm = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-6) * 255).astype(np.uint8)
                    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_PLASMA)
                    _, buf2 = cv2.imencode('.jpg', depth_color, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    self.last_depth_b64 = base64.b64encode(buf2).decode()

                # MPPI trajectory viz (only when MPPI was run during navigation)
                if mppi_debug is not None:
                    mppi_img = np.zeros((300, 400, 3), dtype=np.uint8)
                    mppi_img[:] = (10, 10, 10)
                    cv2.line(mppi_img, (200, 0), (200, 300), (40, 40, 40), 1)
                    cv2.line(mppi_img, (0, 150), (400, 150), (40, 40, 40), 1)
                    ox, oy, scale = 50, 250, 100
                    def to_px(x, y):
                        return int(ox + x * scale), int(oy - y * scale)
                    if "top_x" in mppi_debug:
                        for i in range(len(mppi_debug["top_x"])):
                            pts = [to_px(mppi_debug["top_x"][i][j], mppi_debug["top_y"][i][j])
                                   for j in range(len(mppi_debug["top_x"][i]))]
                            for k in range(len(pts) - 1):
                                cv2.line(mppi_img, pts[k], pts[k+1], (60, 60, 60), 1)
                    best_x = mppi_debug["best_x"]
                    best_y = mppi_debug["best_y"]
                    cost = mppi_debug["best_cost"]
                    clr = (0, 255, 100) if cost < 5 else (0, 170, 255) if cost < 15 else (0, 0, 255)
                    pts = [to_px(best_x[j], best_y[j]) for j in range(len(best_x))]
                    for k in range(len(pts) - 1):
                        cv2.line(mppi_img, pts[k], pts[k+1], clr, 3)
                    cv2.circle(mppi_img, pts[-1], 6, clr, -1)
                    cv2.rectangle(mppi_img, (ox-5, oy-5), (ox+5, oy+5), (255, 255, 255), -1)
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

            except Exception as e:
                print(f"[ERROR] Navigator loop: {e}")
                import traceback; traceback.print_exc()
                time.sleep(0.5)


# ---------------------------------------------------------------------------
# Web server (Flask-free, pure stdlib)
# ---------------------------------------------------------------------------

from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from urllib.parse import urlparse

navigator = None

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Frodo Follower</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: #111;
  color: #e0e0e0;
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 18px;
  height: 42px;
  background: #181818;
  border-bottom: 1px solid #252525;
  flex-shrink: 0;
}
.logo { font-size: 13px; font-weight: 600; letter-spacing: 2px; color: #fff; text-transform: uppercase; }
.topbar-right { display: flex; align-items: center; gap: 22px; }
.stat { display: flex; align-items: baseline; gap: 5px; }
.stat-key { font-size: 10px; color: #555; text-transform: uppercase; letter-spacing: 0.5px; }
.stat-val { font-size: 13px; font-weight: 500; }
.dev-badge { font-size: 10px; padding: 2px 7px; border-radius: 3px; font-weight: 600; letter-spacing: 0.5px; }
.dev-gpu { background: #192819; color: #4caf50; }
.dev-cpu { background: #2a1818; color: #ef5350; }

.workspace {
  display: grid;
  grid-template-columns: 230px 1fr 190px;
  flex: 1;
  overflow: hidden;
  min-height: 0;
}

/* ── Sidebar ── */
.sidebar {
  background: #161616;
  border-right: 1px solid #242424;
  display: flex;
  flex-direction: column;
  padding: 14px;
  gap: 18px;
  overflow-y: auto;
}
.sec-label {
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: #484848;
  margin-bottom: 7px;
  font-weight: 600;
}
.cmd-input {
  width: 100%;
  background: #111;
  border: 1px solid #242424;
  border-radius: 4px;
  padding: 8px 11px;
  color: #e0e0e0;
  font-size: 13px;
  outline: none;
  transition: border-color .15s;
}
.cmd-input:focus { border-color: #3b82f6; }
.btn {
  width: 100%;
  padding: 8px 12px;
  border-radius: 4px;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  margin-top: 6px;
  border: none;
  transition: opacity .15s;
}
.btn:hover { opacity: .85; }
.btn-go   { background: #3b82f6; color: #fff; }
.btn-stop { background: #1e1e1e; border: 1px solid #2e2e2e; color: #ef5350; margin-top: 5px; }

.status-box {
  background: #111;
  border: 1px solid #242424;
  border-radius: 4px;
  padding: 9px 11px;
  font-size: 12px;
  color: #888;
  line-height: 1.5;
  min-height: 36px;
  transition: border-color .2s;
}
.status-box.navigating { border-color: #3b82f6; color: #93c5fd; }
.status-box.arrived    { border-color: #4caf50; color: #86efac; }
.status-box.searching  { border-color: #d97706; color: #fcd34d; }
.status-box.notfound   { border-color: #ef5350; color: #fca5a5; }

.target-val { font-size: 13px; font-weight: 500; color: #3b82f6; }
.target-val.none { color: #444; }

.obj-cloud { display: flex; flex-wrap: wrap; gap: 5px; }
.obj-tag {
  font-size: 11px;
  padding: 3px 8px;
  border-radius: 3px;
  background: #1c1c1c;
  border: 1px solid #282828;
  color: #888;
  cursor: pointer;
  transition: border-color .12s, color .12s;
}
.obj-tag:hover { border-color: #3b82f6; color: #93c5fd; }
.obj-tag.active { background: #1a2e4a; border-color: #3b82f6; color: #93c5fd; }

.hist-list { display: flex; flex-direction: column; gap: 3px; }
.hist-item { font-size: 11px; color: #484848; padding: 1px 0; }

/* ── Center ── */
.center {
  display: flex;
  flex-direction: column;
  overflow: hidden;
  background: #0a0a0a;
}
.feed-wrap {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  background: #000;
}
.feed-wrap.main-feed  { flex: 3; border-bottom: 1px solid #1e1e1e; }
.feed-wrap.depth-feed { flex: 2; }
.feed-img { max-width: 100%; max-height: 100%; object-fit: contain; display: block; }
.feed-tag {
  position: absolute;
  top: 8px; left: 8px;
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: #444;
  background: rgba(0,0,0,.55);
  padding: 2px 7px;
  border-radius: 3px;
}
.lock-tag {
  position: absolute;
  bottom: 9px;
  left: 50%;
  transform: translateX(-50%);
  font-size: 11px;
  color: #4caf50;
  background: rgba(0,0,0,.65);
  padding: 3px 10px;
  border-radius: 3px;
  border: 1px solid #2a3e2a;
  display: none;
}
.lock-tag.on { display: block; }

/* ── Right panel ── */
.right {
  background: #161616;
  border-left: 1px solid #242424;
  display: flex;
  flex-direction: column;
  padding: 14px;
  gap: 18px;
  overflow-y: auto;
}
.metric-list { display: flex; flex-direction: column; gap: 10px; }
.metric-row  { display: flex; justify-content: space-between; align-items: center; }
.metric-key  { font-size: 11px; color: #555; }
.metric-val  { font-size: 13px; font-weight: 500; color: #e0e0e0; }
.vel-track   { height: 3px; background: #1c1c1c; border-radius: 2px; margin-top: 5px; overflow: hidden; }
.vel-fill    { height: 100%; border-radius: 2px; transition: width .1s; }
.vel-fwd  { background: #4caf50; }
.vel-rev  { background: #ef5350; }
.vel-turn { background: #3b82f6; }
.mppi-img { width: 100%; border-radius: 3px; display: block; background: #111; }
</style>
</head>
<body>

<div class="topbar">
  <div class="logo">Frodo Follower</div>
  <div class="topbar-right">
    <div class="stat"><span class="stat-key">FPS</span><span class="stat-val" id="fps-val">—</span></div>
    <div class="stat"><span class="stat-key">Dist</span><span class="stat-val" id="dist-val">—</span></div>
    <div class="stat"><span class="stat-key">Angle</span><span class="stat-val" id="angle-val">—</span></div>
    <div class="stat"><span class="stat-key">Lin / Ang</span><span class="stat-val" id="cmd-val">—</span></div>
    <div id="dev-badge" class="dev-badge dev-gpu">GPU</div>
  </div>
</div>

<div class="workspace">

  <div class="sidebar">
    <div>
      <div class="sec-label">Command</div>
      <input class="cmd-input" id="cmd-input" placeholder="go to the chair..." autofocus>
      <button class="btn btn-go" onclick="sendCmd()">Send</button>
      <button class="btn btn-stop" onclick="stopRobot()">&#9632; Stop</button>
    </div>

    <div>
      <div class="sec-label">Status</div>
      <div class="status-box" id="status-box">Initializing...</div>
    </div>

    <div>
      <div class="sec-label">Target</div>
      <div class="target-val none" id="target-val">None</div>
    </div>

    <div>
      <div class="sec-label">Detected</div>
      <div class="obj-cloud" id="obj-cloud"></div>
    </div>

    <div>
      <div class="sec-label">History</div>
      <div class="hist-list" id="hist-list"></div>
    </div>
  </div>

  <div class="center">
    <div class="feed-wrap main-feed">
      <img class="feed-img" id="main-feed" src="" alt="">
      <div class="feed-tag">Detection</div>
      <div class="lock-tag" id="lock-tag">Target locked</div>
    </div>
    <div class="feed-wrap depth-feed">
      <img class="feed-img" id="depth-feed" src="" alt="">
      <div class="feed-tag">Depth</div>
    </div>
  </div>

  <div class="right">
    <div>
      <div class="sec-label">Velocity</div>
      <div class="metric-list">
        <div>
          <div class="metric-row">
            <span class="metric-key">Linear</span>
            <span class="metric-val" id="lin-val">0.00</span>
          </div>
          <div class="vel-track"><div class="vel-fill vel-fwd" id="lin-bar" style="width:0%"></div></div>
        </div>
        <div>
          <div class="metric-row">
            <span class="metric-key">Angular</span>
            <span class="metric-val" id="ang-val">0.00</span>
          </div>
          <div class="vel-track"><div class="vel-fill vel-turn" id="ang-bar" style="width:0%"></div></div>
        </div>
      </div>
    </div>

    <div>
      <div class="sec-label">Metrics</div>
      <div class="metric-list" id="metrics-list">
        <div class="metric-row"><span class="metric-key">Distance</span><span class="metric-val" id="m-dist">—</span></div>
        <div class="metric-row"><span class="metric-key">Angle</span><span class="metric-val" id="m-angle">—</span></div>
        <div class="metric-row"><span class="metric-key">FPS</span><span class="metric-val" id="m-fps">—</span></div>
      </div>
    </div>

    <div>
      <div class="sec-label">MPPI</div>
      <img class="mppi-img" id="mppi-img" src="" alt="">
    </div>
  </div>

</div>

<script>
let history_ = [];

document.getElementById('cmd-input').addEventListener('keypress', e => {
  if (e.key === 'Enter') sendCmd();
});

function sendCmd() {
  const inp = document.getElementById('cmd-input');
  const cmd = inp.value.trim();
  if (!cmd) return;
  addHist(cmd);
  inp.value = '';
  fetch('/command', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({command: cmd})})
    .then(r => r.json()).then(d => setStatus(d.message || d.status, 'navigating'));
}

function stopRobot() {
  addHist('stop');
  fetch('/command', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({command: 'stop'})})
    .then(() => setStatus('Stopped.', ''));
}

function addHist(cmd) {
  history_.unshift(cmd);
  if (history_.length > 6) history_.pop();
  document.getElementById('hist-list').innerHTML =
    history_.map(c => `<div class="hist-item">› ${c}</div>`).join('');
}

function setStatus(msg, cls) {
  const el = document.getElementById('status-box');
  el.className = 'status-box ' + (cls || '');
  el.textContent = msg;
}

function updateFeed() {
  fetch('/state').then(r => r.json()).then(d => {
    if (d.annotated) document.getElementById('main-feed').src  = 'data:image/jpeg;base64,' + d.annotated;
    if (d.depth)     document.getElementById('depth-feed').src = 'data:image/jpeg;base64,' + d.depth;
    if (d.mppi)      document.getElementById('mppi-img').src   = 'data:image/jpeg;base64,' + d.mppi;

    document.getElementById('fps-val').textContent   = (d.fps || 0).toFixed(1);
    document.getElementById('dist-val').textContent  = d.distance ? d.distance.toFixed(2) + 'm' : '—';
    document.getElementById('angle-val').textContent = d.angle ? d.angle.toFixed(1) + '°' : '—';
    document.getElementById('cmd-val').textContent   = (d.linear||0).toFixed(2) + ' / ' + (d.angular||0).toFixed(2);

    document.getElementById('m-dist').textContent  = d.distance ? d.distance.toFixed(2) + 'm' : '—';
    document.getElementById('m-angle').textContent = d.angle ? d.angle.toFixed(1) + '°' : '—';
    document.getElementById('m-fps').textContent   = (d.fps || 0).toFixed(1);

    const badge = document.getElementById('dev-badge');
    const gpu = (d.gpu || '').toLowerCase() === 'cuda';
    badge.textContent = gpu ? 'GPU' : 'CPU';
    badge.className = 'dev-badge ' + (gpu ? 'dev-gpu' : 'dev-cpu');

    const st = (d.status || '').toLowerCase();
    let cls = '';
    if (st.includes('navigating'))      cls = 'navigating';
    else if (st.includes('arrived'))    cls = 'arrived';
    else if (st.includes('search'))     cls = 'searching';
    else if (st.includes('could not'))  cls = 'notfound';
    setStatus(d.status || '—', cls);

    const tv = document.getElementById('target-val');
    if (d.target) { tv.textContent = d.target; tv.className = 'target-val'; }
    else           { tv.textContent = 'None';   tv.className = 'target-val none'; }

    document.getElementById('lock-tag').className = 'lock-tag' + (d.target && d.distance ? ' on' : '');

    const cloud = document.getElementById('obj-cloud');
    cloud.innerHTML = '';
    (d.objects || []).sort().forEach(obj => {
      const t = document.createElement('span');
      t.className = 'obj-tag' + (obj === d.target ? ' active' : '');
      t.textContent = obj;
      t.onclick = () => {
        addHist('go to the ' + obj);
        fetch('/command', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({command: 'go to the ' + obj})});
      };
      cloud.appendChild(t);
    });

    const linPct = Math.min(100, Math.abs(d.linear  || 0) / 0.35 * 100);
    const angPct = Math.min(100, Math.abs(d.angular || 0) / 0.45 * 100);
    const lb = document.getElementById('lin-bar');
    lb.style.width = linPct + '%';
    lb.className = 'vel-fill ' + ((d.linear || 0) >= 0 ? 'vel-fwd' : 'vel-rev');
    document.getElementById('lin-val').textContent = (d.linear  || 0).toFixed(2);
    document.getElementById('ang-bar').style.width = angPct + '%';
    document.getElementById('ang-val').textContent = (d.angular || 0).toFixed(2);

  }).catch(() => {});
}

setInterval(updateFeed, 300);
</script>
</body>
</html>
"""



class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress logs

    def do_GET(self):
        try:
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
                    'gpu': navigator.device if navigator else 'unknown',
                }
                self.wfile.write(json.dumps(state).encode())
            else:
                self.send_error(404)
        except BrokenPipeError:
            pass

    def do_POST(self):
        if urlparse(self.path).path == '/command':
            try:
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
            except BrokenPipeError:
                pass
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
