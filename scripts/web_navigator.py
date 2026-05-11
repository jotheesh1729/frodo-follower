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
4. DA3/DA2 estimates depth/distance
5. MPPI plans a collision-free trajectory
6. Kalman filter tracks the target through occlusions
7. PID visual servo keeps the target centred

Runs on http://localhost:5000
"""

import sys
import os
import base64
import io
import json
import time
import threading
import queue

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
    except Exception as e:
        print(f"[ERROR] send_control failed: {e}")


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
# NLP parsing
# ---------------------------------------------------------------------------
from object_detector import parse_target


# ---------------------------------------------------------------------------
# Async depth worker
# ---------------------------------------------------------------------------

class DepthWorker(threading.Thread):
    def __init__(self, device: str, model_size: str = 'base', max_depth: float = 10.0, version: int = 3):
        super().__init__(daemon=True)
        self._device = device
        self._model_size = model_size
        self._max_depth = max_depth
        self._version = version
        self._in: queue.Queue = queue.Queue(maxsize=1)
        self._depth = None
        self._confidence = None
        self._lock = threading.Lock()
        self._ready = threading.Event()

    def run(self):
        import torch
        if self._device == 'cuda':
            torch.cuda.set_device(0)
        from depth_estimator import DepthEstimator
        estimator = DepthEstimator(
            model_size=self._model_size,
            max_depth=self._max_depth,
            device=self._device,
            version=self._version
        )
        self._ready.set()
        while True:
            frame = self._in.get()
            if frame is None:
                break
            depth, confidence = estimator.estimate(frame)
            with self._lock:
                self._depth = depth
                self._confidence = confidence

    def submit(self, frame: np.ndarray):
        try:
            self._in.put_nowait(frame.copy())
        except queue.Full:
            pass

    def get(self):
        with self._lock:
            return self._depth, self._confidence

    def wait_ready(self, timeout: float = 180.0):
        self._ready.wait(timeout)


# ---------------------------------------------------------------------------
# MPPI Visualization
# ---------------------------------------------------------------------------

def render_mppi_viz(mppi_debug, cost_columns, target_angle, img_w=400, img_h=220):
    """Clearance bar chart — taller bar = more room in that direction.

    Left edge = left edge of camera FOV, right edge = right edge.
    Green = clear, amber = tight, red = blocked.
    A magenta triangle marks the target direction.
    A white tick marks the planned MPPI steering direction.
    """
    emergency = mppi_debug.get("emergency", False) if mppi_debug else False
    best_cost  = mppi_debug.get("best_cost", 0.0) if mppi_debug else 0.0

    bg = (25, 20, 30)
    img = np.full((img_h, img_w, 3), bg, dtype=np.uint8)

    cols = cost_columns if cost_columns is not None else np.zeros(img_w)
    n_cols = len(cols)

    # Reserve 30px header + 28px footer for text
    header_h = 30
    footer_h = 28
    bar_area_top    = header_h
    bar_area_bottom = img_h - footer_h
    bar_max_h = bar_area_bottom - bar_area_top

    # --- Draw clearance bars ---
    for i in range(img_w):
        col_idx = min(int(i * n_cols / img_w), n_cols - 1)
        cost_val  = float(cols[col_idx])
        clearance = 1.0 - cost_val
        bar_h = int(clearance * bar_max_h)

        if clearance > 0.55:
            color = (50, 200, 80)   # green
        elif clearance > 0.25:
            color = (40, 160, 220)  # amber/blue
        else:
            color = (60, 60, 210)   # red

        y_bot = bar_area_bottom
        y_top = bar_area_bottom - bar_h
        cv2.line(img, (i, y_bot), (i, y_top), color, 1)

    # Baseline
    cv2.line(img, (0, bar_area_bottom), (img_w, bar_area_bottom), (80, 80, 80), 1)

    # Center tick (forward direction)
    cx = img_w // 2
    cv2.line(img, (cx, bar_area_bottom), (cx, bar_area_bottom - bar_max_h),
             (60, 60, 60), 1)
    cv2.putText(img, "FWD", (cx - 12, bar_area_bottom - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)

    fov_rad = np.radians(FOV_H_DEG)

    def angle_to_px(ang_rad):
        # angle 0 = forward = center; positive = left = smaller x (matches image)
        norm = -ang_rad / (fov_rad / 2)   # -1 (left) .. +1 (right)
        return int((norm * 0.5 + 0.5) * img_w)

    # Target direction triangle
    if target_angle is not None:
        tx = angle_to_px(target_angle)
        tx = max(6, min(img_w - 6, tx))
        pts_tri = np.array([
            [tx,      bar_area_top + 4],
            [tx - 6,  bar_area_top + 14],
            [tx + 6,  bar_area_top + 14],
        ], np.int32)
        cv2.fillPoly(img, [pts_tri], (220, 60, 220))
        cv2.putText(img, "T", (tx - 3, bar_area_top + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 60, 220), 1)

    # Planned angular direction tick (from best trajectory endpoint)
    if mppi_debug and not emergency:
        best_y_arr = mppi_debug.get("best_y", None)
        best_x_arr = mppi_debug.get("best_x", None)
        if best_y_arr is not None and best_x_arr is not None and len(best_x_arr) > 1:
            plan_ang = float(np.arctan2(best_y_arr[-1], max(best_x_arr[-1], 0.01)))
            px = angle_to_px(plan_ang)
            px = max(4, min(img_w - 4, px))
            cv2.line(img, (px, bar_area_bottom), (px, bar_area_bottom - 20),
                     (240, 240, 60), 2)

    # --- Header text ---
    if emergency:
        cv2.rectangle(img, (0, 0), (img_w, header_h), (0, 0, 120), -1)
        cv2.putText(img, "! OBSTACLE STOP !", (img_w // 2 - 68, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 255), 2)
    else:
        label = f"Clearance  cost={best_cost:.1f}"
        cv2.putText(img, label, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    # --- Footer: direction labels ---
    cv2.putText(img, "LEFT", (4, img_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (90, 90, 90), 1)
    cv2.putText(img, "RIGHT", (img_w - 42, img_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (90, 90, 90), 1)
    cv2.putText(img, "magenta=target  yellow=plan",
                (img_w // 2 - 70, img_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (80, 80, 80), 1)

    return img


# ---------------------------------------------------------------------------
# Navigator
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
        self.last_annotated_b64 = ''
        self.last_depth_b64 = ''
        self.last_mppi_b64 = ''
        self.detected_objects = []
        self.target_info = {}
        self.status = 'Initializing...'
        self.fps = 0.0
        self.tracker_state = ''

        # PID state
        self._pid_integral = 0.0
        self._pid_prev_error = 0.0

        # Smooth control state
        self._smooth_ang = 0.0
        self._smooth_lin = 0.0
        self._last_time = time.time()

        # Obstacle escape sequence state
        # Phase 0 = normal, 1 = backing up, 2 = turning clear
        self._escape_phase = 0
        self._escape_until = 0.0
        self._escape_ang = 0.0

        # Load models
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        # YOLO
        _scripts_dir = os.path.dirname(os.path.abspath(__file__))
        _engine = os.path.join(_scripts_dir, 'yolo26m.engine')
        _weights = 'yolo26m.pt'
        yolo_path = _engine if os.path.exists(_engine) else _weights
        print(f"Loading YOLO ({os.path.basename(yolo_path)})...")
        from ultralytics import YOLO
        self.yolo = YOLO(yolo_path, task='detect')
        if not yolo_path.endswith('.engine'):
            self.yolo.to(device)
        print(f"YOLO ready on {device}")

        # Depth (DA3 with DA2 fallback)
        print("Loading depth model (background thread)...")
        self._depth_worker = DepthWorker(device=device, model_size='base', max_depth=10.0, version=2)
        self._depth_worker.start()
        self._depth_worker.wait_ready()
        print("Depth model ready")

        # MPPI
        print("Loading MPPI...")
        from mppi_planner import MPPIPlanner, MPPIConfig
        self.mppi = MPPIPlanner(MPPIConfig(
            num_samples=512, horizon=12, dt=0.15,
            max_linear=0.30, max_angular=0.45,
        ))

        # Target tracker (Kalman + Re-ID)
        from target_tracker import TargetTracker
        self.tracker = TargetTracker(
            coast_timeout=2.0,
            search_timeout=30.0,
            reid_threshold=0.3,
            fov_h_deg=FOV_H_DEG,
        )

        self.status = 'Ready — type a command!'
        print("All models loaded.")

    def set_target(self, target_class: str):
        with self.lock:
            self.target_class = target_class
            self.tracker.set_target(target_class)
            self._pid_integral = 0.0
            self._pid_prev_error = 0.0
            print(f"[NAV] Target set to: '{target_class}'")
            if not target_class:
                self.linear = 0.0
                self.angular = 0.0
                send_control(0, 0)

    def run(self):
        from target_tracker import TrackerState
        frame_count = 0

        # PID gains
        KP = 0.30
        KI = 0.02
        KD = 0.08

        # Control params
        EMA_ANG = 0.50
        LIN_RAMP = 0.05
        ARRIVE_DIST = 1.2
        ARRIVE_BBOX = 0.85
        SEARCH_ANG_VEL = 0.22

        while self.running:
            try:
                t0 = time.time()
                dt = t0 - self._last_time
                self._last_time = t0
                dt = max(0.01, min(dt, 0.5))

                frame = get_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue

                h, w = frame.shape[:2]

                # YOLO detection
                results = self.yolo(frame, verbose=False, conf=0.2, device=self.device)
                detections = []
                for r in results:
                    if r.boxes is None:
                        continue
                    for i in range(len(r.boxes)):
                        cls_id = int(r.boxes.cls[i])
                        x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                        detections.append({
                            'class': r.names[cls_id],
                            'confidence': float(r.boxes.conf[i]),
                            'bbox': [x1, y1, x2, y2],
                            'center_x': (x1 + x2) / 2,
                            'center_y': (y1 + y2) / 2,
                        })
                self.detected_objects = list(set(d['class'] for d in detections))

                with self.lock:
                    current_target = self.target_class

                # Depth
                if current_target:
                    self._depth_worker.submit(frame)
                depth, confidence = self._depth_worker.get() if current_target else (None, None)

                # --- Tracker update ---
                tracker_result = self.tracker.update(
                    frame=frame,
                    detections=detections,
                    depth_map=depth,
                    dt=dt,
                    robot_angular_vel=self.angular,
                    frame_width=w,
                    frame_height=h,
                )
                self.tracker_state = tracker_result.state.name.lower()

                # --- Control ---
                raw_cmd_ang = 0.0
                raw_cmd_lin = 0.0
                mppi_debug = None
                mppi_ang = 0.0
                target_det = tracker_result.detection
                target_angle = tracker_result.angle if tracker_result.state != TrackerState.IDLE else None
                target_dist = tracker_result.distance if tracker_result.distance > 0 else None

                if current_target and tracker_result.state == TrackerState.TRACKING:
                    # Target visible — PID steering + MPPI speed
                    normalized_error = tracker_result.angle / (np.radians(FOV_H_DEG) / 2)

                    # Arrival check
                    arrived = False
                    if target_det:
                        x1, y1, x2, y2 = target_det['bbox']
                        bbox_height_frac = (y2 - y1) / h
                        dist_arrived = (target_dist is not None and target_dist < ARRIVE_DIST)
                        bbox_arrived = (bbox_height_frac > ARRIVE_BBOX)
                        arrived = dist_arrived or bbox_arrived

                    if arrived:
                        self._smooth_ang = 0.0
                        self._smooth_lin = 0.0
                        self._pid_integral = 0.0
                        dist_str = f'{target_dist:.2f}m' if target_dist else f'bbox {bbox_height_frac:.0%}'
                        self.status = f'ARRIVED at {current_target}! ({dist_str})'
                    else:
                        # PID angular control
                        self._pid_integral += normalized_error * dt
                        self._pid_integral = max(-1.0, min(1.0, self._pid_integral))
                        derivative = (normalized_error - self._pid_prev_error) / dt
                        self._pid_prev_error = normalized_error

                        raw_cmd_ang = -(KP * normalized_error
                                       + KI * self._pid_integral
                                       + KD * derivative)

                        # MPPI for linear speed AND obstacle-aware steering blend
                        if depth is not None:
                            goal_dir = tracker_result.angle
                            mppi_lin, mppi_ang, mppi_debug = self.mppi.plan(
                                depth, goal_direction_rad=goal_dir,
                                confidence_map=confidence)
                            raw_cmd_lin = mppi_lin

                            # Blend PID steering with MPPI steering.
                            # When the forward path is clear, PID dominates (smooth tracking).
                            # When obstacles block the center, MPPI takes over to steer around.
                            cost_cols = mppi_debug.get('cost_columns', np.zeros(10))
                            nc = len(cost_cols)
                            center_cost = float(np.max(
                                cost_cols[nc // 3: 2 * nc // 3]))
                            # 0 = no obstacle → pure PID; 1 = fully blocked → mostly MPPI
                            mppi_blend = float(np.clip(center_cost * 1.8, 0.0, 0.75))
                            raw_cmd_ang = (1.0 - mppi_blend) * raw_cmd_ang + mppi_blend * mppi_ang
                            
                            # Turn-before-move: If the target is far to the side, slow down linear speed
                            # so the robot turns to center the target first instead of driving past it.
                            turn_slowdown = max(0.1, 1.0 - abs(normalized_error))
                            raw_cmd_lin = raw_cmd_lin * turn_slowdown
                        else:
                            mppi_debug = None
                            speed_scale = max(0.0, 1.0 - abs(normalized_error) * 1.5)
                            raw_cmd_lin = 0.30 * speed_scale

                        # Trigger backup-escape when MPPI sees a hard block
                        if (mppi_debug and mppi_debug.get('emergency')
                                and self._escape_phase == 0):
                            self._escape_phase = 1
                            self._escape_until = time.time() + 1.0
                            # mppi_ang already holds the escape rotation MPPI computed
                            self._escape_ang = mppi_ang

                        dist_str = f'{target_dist:.1f}m' if target_dist else '?m'
                        self.status = f'Navigating → {current_target} ({dist_str})'

                elif current_target and tracker_result.state == TrackerState.PREDICTING:
                    # Coasting on Kalman prediction.
                    # If the last known distance is within arrival range the target
                    # probably just slipped out of YOLO's view because it was too
                    # close — stop instead of driving into it.
                    normalized_error = tracker_result.angle / (np.radians(FOV_H_DEG) / 2)
                    raw_cmd_ang = -KP * normalized_error * 0.5
                    if tracker_result.distance > 0 and tracker_result.distance < ARRIVE_DIST + 0.5:
                        raw_cmd_lin = 0.0
                        self.status = f'Holding — {current_target} very close, re-acquiring...'
                    else:
                        raw_cmd_lin = 0.08
                        self.status = f'Predicting {current_target} position...'

                elif current_target and tracker_result.state == TrackerState.SEARCHING:
                    # Search spin
                    direction = tracker_result.search_direction
                    self._smooth_ang = SEARCH_ANG_VEL * direction
                    raw_cmd_ang = SEARCH_ANG_VEL * direction
                    raw_cmd_lin = 0.0
                    pct = int(tracker_result.search_progress * 100)
                    self.status = f'Searching for {current_target}... ({pct}%)'

                elif current_target and tracker_result.state == TrackerState.LOST:
                    self._smooth_ang = 0.0
                    self._smooth_lin = 0.0
                    with self.lock:
                        self.target_class = ''
                    self.status = f'Could not find {current_target}. Try a new command.'

                else:
                    self.status = 'Idle — type a command!'
                    self._pid_integral = 0.0
                    self._pid_prev_error = 0.0

                # --- Obstacle escape sequence (overrides all tracker commands) ---
                now_t = time.time()
                if self._escape_phase == 1:
                    if now_t < self._escape_until:
                        raw_cmd_lin = -0.15   # reverse
                        raw_cmd_ang = 0.0
                        self.status = 'Obstacle! Backing up...'
                    else:
                        # Back-up done → start turning
                        self._escape_phase = 2
                        self._escape_until = now_t + 0.8
                        raw_cmd_lin = 0.0
                        raw_cmd_ang = self._escape_ang
                        self.status = 'Obstacle! Turning clear...'
                elif self._escape_phase == 2:
                    if now_t < self._escape_until:
                        raw_cmd_lin = 0.0
                        raw_cmd_ang = self._escape_ang
                        self.status = 'Obstacle! Turning clear...'
                    else:
                        self._escape_phase = 0   # resume normal tracking

                # EMA smoothing
                self._smooth_ang = EMA_ANG * raw_cmd_ang + (1.0 - EMA_ANG) * self._smooth_ang
                lin_delta = raw_cmd_lin - self._smooth_lin
                lin_delta = max(-LIN_RAMP, min(LIN_RAMP, lin_delta))
                self._smooth_lin += lin_delta

                if self.auto_mode:
                    self.linear = self._smooth_lin
                    self.angular = self._smooth_ang

                send_control(self.linear, self.angular)

                # --- Annotated frame ---
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

                _, buf = cv2.imencode('.jpg', cv2.cvtColor(frame_draw, cv2.COLOR_RGB2BGR),
                                      [cv2.IMWRITE_JPEG_QUALITY, 70])
                self.last_annotated_b64 = base64.b64encode(buf).decode()

                # Depth colormap
                if depth is not None:
                    d_min, d_max = depth.min(), depth.max()
                    depth_norm = ((depth - d_min) / (d_max - d_min + 1e-6) * 255).astype(np.uint8)
                    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_PLASMA)
                    _, buf2 = cv2.imencode('.jpg', depth_color, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    self.last_depth_b64 = base64.b64encode(buf2).decode()

                # MPPI visualization
                if mppi_debug is not None:
                    cost_cols = mppi_debug.get("cost_columns", np.zeros(10))
                    mppi_img = render_mppi_viz(mppi_debug, cost_cols, target_angle)
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

                elapsed = time.time() - t0
                self.fps = 1.0 / max(elapsed, 0.01)
                frame_count += 1

                if frame_count % 20 == 0:
                    print(f"[{frame_count}] {self.status} | fps={self.fps:.1f} | tracker={self.tracker_state}")

                if elapsed < 0.1:
                    time.sleep(0.1 - elapsed)

            except Exception as e:
                print(f"[ERROR] Navigator loop: {e}")
                import traceback; traceback.print_exc()
                time.sleep(0.5)


# ---------------------------------------------------------------------------
# Web server
# ---------------------------------------------------------------------------

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

navigator = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        try:
            path = urlparse(self.path).path

            if path == '/' or path == '/index.html':
                # Serve web/index.html from disk
                html_path = os.path.join(_ROOT, 'web', 'index.html')
                if os.path.exists(html_path):
                    with open(html_path, 'r') as f:
                        html = f.read()
                else:
                    html = '<h1>web/index.html not found</h1>'
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                self.wfile.write(html.encode())

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
                    'tracker_state': navigator.tracker_state if navigator else '',
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

    nav_thread = threading.Thread(target=navigator.run, daemon=True)
    nav_thread.start()

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
