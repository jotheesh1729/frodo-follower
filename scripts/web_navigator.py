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
                        servo_ang = -STEER_GAIN * normalized_error

                        if depth is not None:
                            mppi_lin, mppi_ang, mppi_debug = self.mppi.plan(depth, goal_direction_rad=goal_dir)
                            raw_cmd_lin = mppi_lin

                            # Obstacle steering blend: check forward corridor for obstacles.
                            # When clear → pure visual servo (accurate tracking).
                            # When obstacle ahead → blend in MPPI angular (steers around it).
                            dh_, dw_ = depth.shape
                            cy_ = int(dh_ * 0.55)
                            cw_ = dw_ // 5
                            fwd = depth[cy_:, max(0, dw_//2 - cw_):dw_//2 + cw_]
                            nearest_obs = float(np.percentile(fwd, 10)) if fwd.size > 0 else 10.0
                            # blend 0=all servo, 1=all MPPI, capped at 0.8 so target never lost
                            BLEND_START = 1.5
                            blend = float(np.clip(1.0 - nearest_obs / BLEND_START, 0.0, 0.8))
                            raw_cmd_ang = (1.0 - blend) * servo_ang + blend * mppi_ang
                        else:
                            speed_scale = max(0.25, 1.0 - abs(normalized_error))
                            raw_cmd_lin = 0.30 * speed_scale
                            raw_cmd_ang = servo_ang

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
<title>Frodo-Follower</title>
<style>
:root {
  --bg: #0d0d0f;
  --surface: #16181d;
  --surface2: #1e2028;
  --border: #2a2d38;
  --accent: #6c63ff;
  --accent2: #ff4d6d;
  --green: #00e5a0;
  --yellow: #ffd166;
  --text: #e2e4ed;
  --muted: #6b7280;
  --radius: 10px;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); height: 100vh; display: flex; flex-direction: column; overflow: hidden; }

/* ── Top bar ── */
.topbar { display: flex; align-items: center; justify-content: space-between; padding: 10px 20px; background: var(--surface); border-bottom: 1px solid var(--border); flex-shrink: 0; }
.topbar-left { display: flex; align-items: center; gap: 12px; }
.logo { font-size: 18px; font-weight: 700; letter-spacing: 1px; color: var(--accent); }
.logo span { color: var(--accent2); }
.pill { font-size: 11px; padding: 3px 10px; border-radius: 20px; font-weight: 600; letter-spacing: .5px; }
.pill-gpu { background: #1a2a1a; color: var(--green); border: 1px solid #1e4d1e; }
.pill-cpu { background: #2a1a1a; color: var(--yellow); border: 1px solid #4d1e1e; }
.topbar-right { display: flex; gap: 20px; }
.stat { text-align: right; }
.stat-label { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: .5px; }
.stat-val { font-size: 15px; font-weight: 600; }
.stat-val.green { color: var(--green); }
.stat-val.red { color: var(--accent2); }
.stat-val.yellow { color: var(--yellow); }

/* ── Layout ── */
.workspace { display: grid; grid-template-columns: 300px 1fr 220px; grid-template-rows: 1fr; gap: 0; flex: 1; overflow: hidden; }

/* ── Left sidebar ── */
.sidebar { background: var(--surface); border-right: 1px solid var(--border); display: flex; flex-direction: column; padding: 14px; gap: 12px; overflow-y: auto; }
.sidebar-section { display: flex; flex-direction: column; gap: 8px; }
.section-title { font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); font-weight: 600; }
.cmd-row { display: flex; gap: 8px; }
.cmd-input { flex: 1; background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius); padding: 10px 14px; color: var(--text); font-size: 14px; outline: none; transition: border .2s; }
.cmd-input:focus { border-color: var(--accent); }
.btn-go { background: var(--accent); color: white; border: none; border-radius: var(--radius); padding: 10px 18px; font-weight: 700; font-size: 14px; cursor: pointer; transition: opacity .15s; }
.btn-go:hover { opacity: .85; }
.btn-stop { background: var(--accent2); color: white; border: none; border-radius: var(--radius); padding: 10px 18px; font-weight: 700; font-size: 14px; cursor: pointer; width: 100%; transition: opacity .15s; }
.btn-stop:hover { opacity: .85; }
.quick-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
.qbtn { background: var(--surface2); border: 1px solid var(--border); border-radius: 8px; padding: 8px 6px; font-size: 12px; color: var(--text); cursor: pointer; text-align: center; transition: border-color .15s, color .15s; }
.qbtn:hover { border-color: var(--accent); color: var(--accent); }
.status-card { background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius); padding: 10px 14px; }
.status-card.navigating { border-color: var(--accent); }
.status-card.arrived { border-color: var(--green); }
.status-card.searching { border-color: var(--yellow); }
.status-card.stopped { border-color: var(--border); }
#status-msg { font-size: 13px; line-height: 1.4; }
.metric-row { display: flex; justify-content: space-between; align-items: center; }
.metric-label { font-size: 11px; color: var(--muted); }
.metric-val { font-size: 13px; font-weight: 600; }
.target-badge { display: inline-flex; align-items: center; gap: 6px; background: var(--accent); color: white; border-radius: 6px; padding: 4px 10px; font-size: 12px; font-weight: 700; }
.target-badge.empty { background: var(--surface2); color: var(--muted); }
.obj-cloud { display: flex; flex-wrap: wrap; gap: 5px; }
.obj-chip { background: var(--surface2); border: 1px solid var(--border); border-radius: 20px; padding: 4px 10px; font-size: 11px; cursor: pointer; transition: all .15s; }
.obj-chip:hover { border-color: var(--accent); color: var(--accent); }
.obj-chip.active { background: var(--accent); border-color: var(--accent); color: white; }
.cmd-hist { display: flex; flex-direction: column; gap: 4px; max-height: 100px; overflow-y: auto; }
.hist-item { font-size: 11px; color: var(--muted); padding: 2px 0; border-bottom: 1px solid var(--border); }

/* ── Center canvas ── */
.canvas-area { position: relative; background: #000; display: flex; align-items: center; justify-content: center; overflow: hidden; }
#main-feed { max-width: 100%; max-height: 100%; object-fit: contain; display: block; }
.canvas-overlay { position: absolute; top: 10px; left: 10px; display: flex; gap: 6px; }
.feed-badge { background: rgba(0,0,0,.65); border: 1px solid var(--border); border-radius: 6px; padding: 4px 10px; font-size: 11px; cursor: pointer; transition: border-color .15s; }
.feed-badge.active { border-color: var(--accent); color: var(--accent); }
.lock-indicator { position: absolute; bottom: 12px; left: 50%; transform: translateX(-50%); background: rgba(0,0,0,.7); border: 1px solid var(--green); border-radius: 8px; padding: 5px 14px; font-size: 12px; color: var(--green); font-weight: 600; display: none; }
.lock-indicator.visible { display: block; }

/* ── Right panel ── */
.right-panel { background: var(--surface); border-left: 1px solid var(--border); display: flex; flex-direction: column; overflow: hidden; }
.rpanel-block { flex: 1; display: flex; flex-direction: column; border-bottom: 1px solid var(--border); overflow: hidden; min-height: 0; }
.rpanel-block:last-child { border-bottom: none; }
.rpanel-title { font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); padding: 8px 12px; font-weight: 600; background: var(--surface2); flex-shrink: 0; }
.rpanel-img { width: 100%; flex: 1; object-fit: contain; display: block; background: #000; min-height: 0; }
.vel-bars { padding: 10px 14px; display: flex; flex-direction: column; gap: 8px; }
.vel-row { display: flex; flex-direction: column; gap: 4px; }
.vel-label { font-size: 10px; color: var(--muted); display: flex; justify-content: space-between; }
.vel-track { background: var(--bg); border-radius: 4px; height: 8px; overflow: hidden; }
.vel-fill { height: 100%; border-radius: 4px; transition: width .1s, background .1s; }
.vel-fill.fwd { background: var(--green); }
.vel-fill.turn { background: var(--accent); }
.vel-fill.rev { background: var(--accent2); }
</style>
</head>
<body>

<!-- Top bar -->
<div class="topbar">
  <div class="topbar-left">
    <div class="logo">FRODO<span>FOLLOWER</span></div>
    <div id="gpu-pill" class="pill pill-gpu">GPU</div>
  </div>
  <div class="topbar-right">
    <div class="stat"><div class="stat-label">FPS</div><div class="stat-val green" id="fps-val">—</div></div>
    <div class="stat"><div class="stat-label">Distance</div><div class="stat-val" id="dist-val">—</div></div>
    <div class="stat"><div class="stat-label">Angle</div><div class="stat-val" id="angle-val">—</div></div>
    <div class="stat"><div class="stat-label">Lin / Ang</div><div class="stat-val" id="cmd-val">—</div></div>
  </div>
</div>

<!-- Main workspace -->
<div class="workspace">

  <!-- Left sidebar -->
  <div class="sidebar">

    <div class="sidebar-section">
      <div class="section-title">Command</div>
      <div class="cmd-row">
        <input class="cmd-input" id="cmd-input" placeholder="go to the chair..." autofocus>
        <button class="btn-go" onclick="sendCommand()">GO</button>
      </div>
      <button class="btn-stop" onclick="stopRobot()">&#9632; STOP</button>
    </div>

    <div class="sidebar-section">
      <div class="section-title">Quick targets</div>
      <div class="quick-grid">
        <div class="qbtn" onclick="quickNav('person')">Person</div>
        <div class="qbtn" onclick="quickNav('chair')">Chair</div>
        <div class="qbtn" onclick="quickNav('bottle')">Bottle</div>
        <div class="qbtn" onclick="quickNav('laptop')">Laptop</div>
        <div class="qbtn" onclick="quickNav('backpack')">Backpack</div>
        <div class="qbtn" onclick="quickNav('cup')">Cup</div>
        <div class="qbtn" onclick="quickNav('tv')">TV</div>
        <div class="qbtn" onclick="quickNav('book')">Book</div>
      </div>
    </div>

    <div class="sidebar-section">
      <div class="section-title">Status</div>
      <div class="status-card stopped" id="status-card">
        <div id="status-msg">Initializing...</div>
      </div>
      <div class="metric-row">
        <span class="metric-label">Target locked</span>
        <span class="target-badge empty" id="target-badge">None</span>
      </div>
    </div>

    <div class="sidebar-section">
      <div class="section-title">Detected objects</div>
      <div class="obj-cloud" id="obj-cloud"></div>
    </div>

    <div class="sidebar-section">
      <div class="section-title">History</div>
      <div class="cmd-hist" id="cmd-hist"></div>
    </div>

  </div>

  <!-- Center canvas -->
  <div class="canvas-area">
    <img id="main-feed" src="" alt="">
    <div class="canvas-overlay">
      <div class="feed-badge active" onclick="switchFeed('yolo')">Detection</div>
      <div class="feed-badge" onclick="switchFeed('depth')">Depth</div>
      <div class="feed-badge" onclick="switchFeed('mppi')">MPPI</div>
    </div>
    <div class="lock-indicator" id="lock-ind">&#9679; TARGET LOCKED</div>
  </div>

  <!-- Right panel -->
  <div class="right-panel">
    <div class="rpanel-block">
      <div class="rpanel-title">Depth map</div>
      <img class="rpanel-img" id="depth-thumb" src="" alt="">
    </div>
    <div class="rpanel-block">
      <div class="rpanel-title">MPPI planner</div>
      <img class="rpanel-img" id="mppi-thumb" src="" alt="">
    </div>
    <div class="rpanel-block" style="flex: 0 0 auto;">
      <div class="rpanel-title">Velocity</div>
      <div class="vel-bars">
        <div class="vel-row">
          <div class="vel-label"><span>Linear</span><span id="lin-num">0.00</span></div>
          <div class="vel-track"><div class="vel-fill fwd" id="lin-bar" style="width:0%"></div></div>
        </div>
        <div class="vel-row">
          <div class="vel-label"><span>Angular</span><span id="ang-num">0.00</span></div>
          <div class="vel-track"><div class="vel-fill turn" id="ang-bar" style="width:0%"></div></div>
        </div>
      </div>
    </div>
  </div>

</div>

<script>
let activeFeed = 'yolo';
let lastData = {};
let cmdHistory = [];

function switchFeed(type) {
  activeFeed = type;
  document.querySelectorAll('.feed-badge').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  updateMainFeed(lastData);
}

function updateMainFeed(d) {
  const feeds = { yolo: d.annotated, depth: d.depth, mppi: d.mppi };
  const src = feeds[activeFeed];
  if (src) document.getElementById('main-feed').src = 'data:image/jpeg;base64,' + src;
}

function sendCommand() {
  const inp = document.getElementById('cmd-input');
  const cmd = inp.value.trim();
  if (!cmd) return;
  addHistory(cmd);
  inp.value = '';
  fetch('/command', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({command: cmd})})
    .then(r => r.json()).then(d => setStatus(d.message || d.status, 'navigating'));
}

function quickNav(obj) {
  addHistory('go to the ' + obj);
  fetch('/command', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({command: 'go to the ' + obj})})
    .then(r => r.json()).then(d => setStatus(d.message, 'navigating'));
}

function stopRobot() {
  addHistory('stop');
  fetch('/command', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({command: 'stop'})})
    .then(() => setStatus('Stopped.', 'stopped'));
}

function addHistory(cmd) {
  cmdHistory.unshift(cmd);
  if (cmdHistory.length > 8) cmdHistory.pop();
  const el = document.getElementById('cmd-hist');
  el.innerHTML = cmdHistory.map(c => `<div class="hist-item">&rsaquo; ${c}</div>`).join('');
}

function setStatus(msg, cls) {
  const card = document.getElementById('status-card');
  card.className = 'status-card ' + (cls || 'stopped');
  document.getElementById('status-msg').textContent = msg;
}

document.getElementById('cmd-input').addEventListener('keypress', e => { if (e.key === 'Enter') sendCommand(); });

function updateFeed() {
  fetch('/state').then(r => r.json()).then(d => {
    lastData = d;

    // Main feed
    updateMainFeed(d);

    // Right panel thumbnails
    if (d.depth) document.getElementById('depth-thumb').src = 'data:image/jpeg;base64,' + d.depth;
    if (d.mppi)  document.getElementById('mppi-thumb').src  = 'data:image/jpeg;base64,' + d.mppi;

    // Top bar stats
    document.getElementById('fps-val').textContent   = (d.fps||0).toFixed(1);
    document.getElementById('dist-val').textContent  = d.distance ? d.distance.toFixed(2)+'m' : '—';
    document.getElementById('angle-val').textContent = d.angle ? d.angle.toFixed(1)+'°' : '—';
    document.getElementById('cmd-val').textContent   = (d.linear||0).toFixed(2)+' / '+(d.angular||0).toFixed(2);

    // GPU pill
    const pill = document.getElementById('gpu-pill');
    const onGpu = (d.gpu||'').toLowerCase() === 'cuda';
    pill.textContent = onGpu ? 'GPU' : 'CPU';
    pill.className = 'pill ' + (onGpu ? 'pill-gpu' : 'pill-cpu');

    // Status card
    const st = (d.status||'').toLowerCase();
    let cls = 'stopped';
    if (st.includes('navigating')) cls = 'navigating';
    else if (st.includes('arrived')) cls = 'arrived';
    else if (st.includes('search') || st.includes('lost')) cls = 'searching';
    setStatus(d.status||'—', cls);

    // Target badge
    const badge = document.getElementById('target-badge');
    if (d.target) { badge.textContent = d.target; badge.className = 'target-badge'; }
    else           { badge.textContent = 'None';   badge.className = 'target-badge empty'; }

    // Lock indicator
    const lockEl = document.getElementById('lock-ind');
    lockEl.className = 'lock-indicator' + (d.target && d.distance ? ' visible' : '');

    // Object cloud
    const cloud = document.getElementById('obj-cloud');
    cloud.innerHTML = '';
    (d.objects||[]).sort().forEach(obj => {
      const c = document.createElement('span');
      c.className = 'obj-chip' + (obj === d.target ? ' active' : '');
      c.textContent = obj;
      c.onclick = () => quickNav(obj);
      cloud.appendChild(c);
    });

    // Velocity bars (max linear ~0.35, max angular ~0.45)
    const linPct = Math.min(100, Math.abs(d.linear||0) / 0.35 * 100);
    const angPct = Math.min(100, Math.abs(d.angular||0) / 0.45 * 100);
    const linBar = document.getElementById('lin-bar');
    linBar.style.width = linPct + '%';
    linBar.className = 'vel-fill ' + ((d.linear||0) >= 0 ? 'fwd' : 'rev');
    document.getElementById('lin-num').textContent = (d.linear||0).toFixed(2);
    document.getElementById('ang-bar').style.width = angPct + '%';
    document.getElementById('ang-num').textContent = (d.angular||0).toFixed(2);

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
