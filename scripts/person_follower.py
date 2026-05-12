#!/usr/bin/env python3
# Person Follower — YOLO-World + VLM search guidance + EKF tracker + DA3 depth
# Click a person in the web UI to lock on and follow.
# Run:  python3 scripts/person_follower.py
# UI:   http://localhost:5001

import sys, os, time, threading, base64, io, argparse
import numpy as np
import cv2
import requests
from PIL import Image
from flask import Flask, request, jsonify
import logging

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in [
    os.path.join(_ROOT, "frodo_ai", "perception"),
    os.path.join(_ROOT, "third_party", "Depth-Anything-V2"),
    os.path.join(_ROOT, "third_party", "Depth-Anything-V2", "metric_depth"),
]:
    sys.path.insert(0, _p)

from target_tracker import TargetTracker, TrackerState
from depth_estimator import DepthEstimator

SDK_URL     = "http://localhost:8000"
FOV_H_DEG   = 90.0
CONF_LOW    = 0.35
TARGET_DIST = 2.5   # metres — stop at this distance from the person


def send_cmd(linear, angular):
    try:
        requests.post(f"{SDK_URL}/control-legacy",
                      json={"command": {"linear": linear, "angular": angular, "lamp": 0}},
                      timeout=1.0)
    except Exception:
        pass


def get_frame():
    try:
        b64 = requests.get(f"{SDK_URL}/v2/front", timeout=3).json().get("front_frame")
        if b64:
            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            if img.size[0] > 10:
                return np.array(img, dtype=np.uint8)
    except Exception:
        pass
    return None


def jpg_b64(frame_rgb, q=60):
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                          [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf).decode()


def depth_b64(depth):
    d8 = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-6) * 255).astype(np.uint8)
    _, buf = cv2.imencode(".jpg", cv2.applyColorMap(d8, cv2.COLORMAP_PLASMA),
                          [cv2.IMWRITE_JPEG_QUALITY, 50])
    return base64.b64encode(buf).decode()


class VLMGuide(threading.Thread):
    """Background VLM thread — search guidance and obstacle bypass direction."""

    def __init__(self, model_name="qwen"):
        super().__init__(daemon=True)
        self._lock        = threading.Lock()
        self._frame       = None
        self._mode        = "search"   # "search" | "stuck"
        self._model_name  = model_name
        self.hint         = None       # "left" | "center" | "right" | None
        self.stuck_dir    = None       # "left" | "right" | None
        self.status       = "VLM loading…"

    def submit(self, frame_rgb, mode="search"):
        with self._lock:
            self._frame = frame_rgb.copy()
            self._mode  = mode

    def run(self):
        try:
            from vlm_backend import make_backend
        except ImportError:
            self.status = "VLM unavailable — vlm_backend.py not found"
            return

        self.status = f"Loading {self._model_name}…"
        try:
            backend = make_backend(self._model_name)
            backend.load()
            self.status = f"{backend.name} ready"
        except Exception as e:
            self.status = f"VLM load failed: {e}"
            return

        while True:
            with self._lock:
                frame = self._frame
                mode  = self._mode
                self._frame = None
            if frame is None:
                time.sleep(0.2)
                continue

            try:
                if mode == "stuck":
                    prompt  = ("There is an obstacle directly ahead of a robot. "
                               "Should it turn LEFT or RIGHT to go around? "
                               "Reply with one word: LEFT or RIGHT.")
                    max_tok = 5
                else:
                    prompt  = ("Do you see a person in this image? "
                               "Reply with exactly one word: LEFT, RIGHT, CENTER, or NO.")
                    max_tok = 8

                ans = backend.infer(Image.fromarray(frame), prompt, max_tok)

                with self._lock:
                    if mode == "stuck":
                        self.stuck_dir = "left" if "left" in ans else ("right" if "right" in ans else None)
                        self.status = f"Bypass → {self.stuck_dir or '?'}"
                    else:
                        if   "left"   in ans: self.hint = "left"
                        elif "right"  in ans: self.hint = "right"
                        elif "center" in ans or "middle" in ans: self.hint = "center"
                        else: self.hint = None
                        self.status = f"Person → {self.hint or 'not found'}"
            except Exception as e:
                self.status = f"VLM error: {e}"


class PersonFollower:

    def __init__(self, vlm_model="qwen"):
        self.running     = True
        self.frame_b64   = ""
        self.depth_frame = ""
        self.status      = "Loading models…"
        self.lin         = 0.0
        self.ang         = 0.0
        self.locked      = False
        self.fps         = 0.0

        self._click  = None   # (nx, ny) normalised, set by UI click
        self._clock  = threading.Lock()
        self._ready  = threading.Event()

        self._yolo    = None
        self._depth   = None
        self._tracker = TargetTracker(
            coast_timeout=2.0,
            search_timeout=30.0,
            reid_threshold=0.45,
            fov_h_deg=FOV_H_DEG,
        )
        self._vlm = VLMGuide(model_name=vlm_model)
        self._vlm.start()

        threading.Thread(target=self._load_models, daemon=True).start()

    def _load_models(self):
        try:
            self.status = "Loading YOLO-World…"
            from ultralytics import YOLOWorld
            self._yolo = YOLOWorld(os.path.join(_ROOT, "yolov8s-worldv2.pt"))

            # CLIP tokenize returns CPU tensors; model lives on CUDA — patch it.
            try:
                import clip.model as _cm
                _orig = _cm.CLIP.encode_text
                def _enc(self, text):
                    return _orig(self, text.to(next(self.parameters()).device))
                _cm.CLIP.encode_text = _enc
            except Exception:
                pass

            self._yolo.set_classes(["person"])
            self._yolo(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False)

            self.status = "Loading DepthEstimator…"
            self._depth = DepthEstimator(model_size="small", max_depth=10.0)

            self._ready.set()
            self.status = "Ready — click a person to follow"
            print("Models ready.")
        except Exception as e:
            self.status = f"Load error: {e}"

    def click_lock(self, nx, ny):
        with self._clock:
            self._click = (nx, ny)

    def stop(self):
        self.locked = False
        self._tracker.reset()
        self.lin = self.ang = 0.0
        send_cmd(0, 0)
        self.status = "Stopped"

    def _detect(self, frame):
        dets = []
        for r in self._yolo(frame, verbose=False, conf=CONF_LOW):
            if r.boxes is None:
                continue
            for i in range(len(r.boxes)):
                cls_id = int(r.boxes.cls[i])
                x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                dets.append({
                    "class":      r.names[cls_id],
                    "confidence": float(r.boxes.conf[i]),
                    "bbox":       (x1, y1, x2, y2),
                    "center_x":  (x1 + x2) / 2,
                    "center_y":  (y1 + y2) / 2,
                })
        return dets

    def _annotate(self, frame, dets, result):
        out = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        for d in dets:
            x1, y1, x2, y2 = (int(v) for v in d["bbox"])
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 180, 180), 1)
        if result.state == TrackerState.TRACKING and result.detection:
            bx1, by1, bx2, by2 = (int(v) for v in result.detection["bbox"])
            cv2.rectangle(out, (bx1, by1), (bx2, by2), (0, 255, 255), 3)
            tx, ty = (bx1 + bx2) // 2, (by1 + by2) // 2
            cv2.line(out, (out.shape[1] // 2, out.shape[0]), (tx, ty), (0, 255, 255), 2)
            cv2.circle(out, (tx, ty), 6, (0, 255, 255), -1)
        cv2.putText(out, self.status[:70], (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, self.status[:70], (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
        fps_txt = f"{self.fps:.0f} FPS"
        cv2.putText(out, fps_txt, (out.shape[1] - 85, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, fps_txt, (out.shape[1] - 85, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 100), 1, cv2.LINE_AA)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    def run_loop(self):
        self._ready.wait()
        prev_t = time.time()
        KP, KD = 0.25, 0.08
        _prev  = 0.0

        _prev_state          = None
        _bypass_active       = False
        _bypass_dir          = 1.0
        _bypass_until        = 0.0
        _bypass_cooldown     = 0.0
        _retry_bypass        = False
        _last_search_vlm     = 0.0
        _last_stuck_vlm      = 0.0
        _last_tracked_dist   = float("inf")
        _reacquire_active    = False
        _reacquire_until     = 0.0
        _reacquire_tried     = False
        _reacquire_srch_end  = 0.0

        while self.running:
            t0     = time.time()
            dt     = max(0.01, min(t0 - prev_t, 0.5))
            prev_t = t0

            frame = get_frame()
            if frame is None:
                self.status = "SDK offline — check localhost:8000"
                time.sleep(0.2)
                continue

            h, w = frame.shape[:2]
            dets = self._detect(frame)

            # Handle UI click — lock onto the clicked person
            with self._clock:
                click = self._click
                self._click = None
            if click is not None:
                nx, ny = click
                lx, ly = nx * w, ny * h
                hit = next((d for d in dets
                            if d["bbox"][0] <= lx <= d["bbox"][2]
                            and d["bbox"][1] <= ly <= d["bbox"][3]), None)
                if hit:
                    self._tracker.set_target("person")
                    self._tracker.update(frame, [hit], None, dt, self.ang, w, h)
                    self.locked          = True
                    _prev                = 0.0
                    _bypass_active       = False
                    _bypass_cooldown     = 0.0
                    self._vlm.hint       = None
                    self._vlm.stuck_dir  = None
                    _last_search_vlm     = 0.0
                    _last_stuck_vlm      = 0.0
                    _last_tracked_dist   = float("inf")
                    _reacquire_active    = False
                    _reacquire_tried     = False
                    _reacquire_srch_end  = 0.0

            try:
                depth_raw, _ = self._depth.estimate(frame)
            except Exception:
                time.sleep(0.05)
                continue
            depth = cv2.GaussianBlur(depth_raw, (5, 5), 1.5)

            # Spatial lock: always pass the single detection nearest to EKF prediction.
            # Prevents confusion when multiple people are visible.
            tracked = [d for d in dets if d["class"] == "person"] if self.locked else []
            if self._tracker._initialized and len(tracked) > 0:
                _fov  = np.radians(FOV_H_DEG)
                _pred = float(self._tracker._x[0])
                def _angle(d):
                    return float(np.arctan(
                        (d["center_x"] - w / 2) / (w / (2 * np.tan(_fov / 2)))))
                tracked = [min(tracked, key=lambda d: abs(_angle(d) - _pred))]

            result = self._tracker.update(
                frame=frame, detections=tracked, depth_map=depth,
                dt=dt, robot_angular_vel=self.ang,
                frame_width=w, frame_height=h,
            )

            raw_lin, raw_ang = 0.0, 0.0

            # Obstacle depths — left / centre / right thirds of middle band
            dh, dw = depth.shape
            d_L = float(np.median(depth[int(dh*0.3):int(dh*0.8), :int(dw*0.33)]))
            d_C = float(np.median(depth[int(dh*0.3):int(dh*0.8), int(dw*0.33):int(dw*0.66)]))
            d_R = float(np.median(depth[int(dh*0.3):int(dh*0.8), int(dw*0.66):]))
            obs = 0.0
            if d_L < 1.2: obs -= (1.2 - d_L) * 0.3
            if d_R < 1.2: obs += (1.2 - d_R) * 0.3
            if d_C < 1.5: obs += (1.5 - d_C) * 0.2 * (1.0 if d_L > d_R else -1.0)
            obs = float(np.clip(obs, -1.0, 1.0))

            if not self.locked:
                self.status = "Ready — click a person to follow"
            else:
                # Bypass trigger
                if (not _bypass_active and t0 >= _bypass_cooldown
                        and (result.state in (TrackerState.TRACKING, TrackerState.PREDICTING)
                             or _retry_bypass)
                        and 0.6 <= d_C < 1.1):
                    _bypass_active = True
                    _bypass_until  = t0 + 3.5
                    if not _retry_bypass:
                        # Prefer VLM direction if fresh, else fall back to depth asymmetry
                        vlm_stuck = self._vlm.stuck_dir
                        if vlm_stuck == "left":
                            _bypass_dir = 1.0
                        elif vlm_stuck == "right":
                            _bypass_dir = -1.0
                        elif abs(d_L - d_R) > 0.25:
                            _bypass_dir = 1.0 if d_L >= d_R else -1.0
                        else:
                            _bypass_dir = -1.0 if float(self._tracker._x[0]) >= 0 else 1.0
                    _retry_bypass = False

                # Bypass end
                _just_ended = False
                if _bypass_active and (t0 >= _bypass_until or d_C >= 1.2):
                    _bypass_active  = False
                    _just_ended     = True
                    if d_C >= 1.2:
                        _bypass_cooldown = t0 + 3.0
                    else:
                        _bypass_dir      = -_bypass_dir
                        _retry_bypass    = True
                        _bypass_cooldown = t0 + 0.3

                if d_C < 0.6:
                    # Emergency backup — pick gap direction now and turn while reversing
                    _bypass_active   = False
                    _bypass_cooldown = t0 + 0.3
                    _retry_bypass    = True
                    if abs(d_L - d_R) > 0.15:
                        _bypass_dir = 1.0 if d_L >= d_R else -1.0
                    if (t0 - _last_stuck_vlm) >= 4.0:
                        self._vlm.submit(frame, mode="stuck")
                        _last_stuck_vlm = t0
                    raw_lin = -0.15
                    raw_ang = _bypass_dir * 0.45   # decisive turn while reversing
                    self.status = "Obstacle! Backing up…"

                elif _bypass_active:
                    raw_ang = _bypass_dir * 0.45 + obs * 0.20
                    raw_lin = 0.10
                    self.status = f"Going around… {'←' if _bypass_dir > 0 else '→'}"

                elif result.state in (TrackerState.TRACKING, TrackerState.PREDICTING):
                    _last_tracked_dist = result.distance
                    _reacquire_tried   = False   # successful track — allow one future re-find
                    norm_err = result.angle / (np.radians(FOV_H_DEG) / 2)
                    if _just_ended or _prev_state not in (
                            TrackerState.TRACKING, TrackerState.PREDICTING):
                        _prev     = norm_err
                        self.ang  = 0.0
                    deriv    = float(np.clip((norm_err - _prev) / dt, -2.0, 2.0))
                    _prev    = norm_err
                    raw_ang  = float(np.clip(-(KP * norm_err + KD * deriv) + obs * 0.30, -0.40, 0.40))
                    slowdown = max(0.1, 1.0 - abs(norm_err))
                    dist     = result.distance
                    if dist < TARGET_DIST - 0.4:
                        # Overshot — back up gently, clear linear EMA momentum
                        raw_lin  = -0.08
                        self.lin = 0.0
                        self.status = f"Too close — backing up ({dist:.1f} m)"
                    elif dist <= TARGET_DIST:
                        # At stop distance — hard stop to clear EMA coasting
                        raw_lin  = 0.0
                        self.lin = 0.0
                        self.status = f"Following — holding {dist:.1f} m"
                    else:
                        raw_lin = min(0.25, (dist - TARGET_DIST) * 0.3) * slowdown
                        self.status = f"Following → {dist:.1f} m  {np.degrees(result.angle):+.0f}°"

                elif result.state == TrackerState.SEARCHING:
                    # Post-reacquire search window expired
                    if _reacquire_tried and t0 > _reacquire_srch_end > 0:
                        self.locked = False
                        self.status = "Person lost — click to re-lock"
                    else:
                        if (t0 - _last_search_vlm) >= 4.5 and not dets:
                            self._vlm.submit(frame)
                            _last_search_vlm = t0
                        vlm_hint = self._vlm.hint
                        if dets:
                            raw_ang = 0.05 * result.search_direction + obs * 0.3
                            self.status = "Possible person — confirming…"
                        elif vlm_hint == "left":
                            raw_ang = 0.20 + obs * 0.3
                            self.status = "VLM: person is left — turning ←"
                        elif vlm_hint == "right":
                            raw_ang = -0.20 + obs * 0.3
                            self.status = "VLM: person is right — turning →"
                        elif vlm_hint == "center":
                            raw_lin = 0.12; raw_ang = obs * 0.3
                            self.status = "VLM: person ahead — moving forward"
                        else:
                            raw_ang = 0.18 * result.search_direction + obs * 0.3
                            self.status = f"Searching… {int(result.search_progress * 100)}%"

                elif result.state == TrackerState.LOST:
                    # Lost at close range — back up briefly and retry once
                    if not _reacquire_tried and _last_tracked_dist < 3.5 and not _reacquire_active:
                        _reacquire_active = True
                        _reacquire_until  = t0 + 2.0
                        raw_lin = -0.12
                        self.status = "Person lost close — backing up to re-find…"
                    elif _reacquire_active:
                        if t0 < _reacquire_until:
                            raw_lin = -0.12
                            self.status = "Backing up to re-find person…"
                        else:
                            _reacquire_active   = False
                            _reacquire_tried    = True
                            _reacquire_srch_end = t0 + 8.0
                            self._tracker.set_target("person")
                            _last_search_vlm = 0.0
                            self.status = "Re-searching for person…"
                    else:
                        self.locked = False
                        self.status = "Person lost — click to re-lock"

            _prev_state = result.state

            raw_ang  = float(np.clip(raw_ang, -0.50, 0.50))
            self.lin = self.lin * 0.55 + raw_lin * 0.45
            self.ang = float(np.clip(self.ang * 0.45 + raw_ang * 0.55, -0.50, 0.50))
            send_cmd(self.lin, self.ang)

            self.frame_b64   = jpg_b64(self._annotate(frame, dets, result))
            self.depth_frame = depth_b64(depth)

            elapsed = time.time() - t0
            self.fps = 0.9 * self.fps + 0.1 / max(elapsed, 0.01)
            if elapsed < 0.10:
                time.sleep(0.10 - elapsed)


# ── Flask ─────────────────────────────────────────────────────────────────────

follower = None
app = Flask(__name__)
logging.getLogger("werkzeug").setLevel(logging.ERROR)


@app.route("/")
def index():
    with open(os.path.join(_ROOT, "web", "follower.html")) as f:
        return f.read()


@app.route("/state")
def state():
    return jsonify({
        "frame":       follower.frame_b64,
        "depth_frame": follower.depth_frame,
        "status":      follower.status,
        "locked":      follower.locked,
        "linear":      round(follower.lin, 2),
        "angular":     round(follower.ang, 2),
        "fps":         round(follower.fps, 1),
        "vlm":         follower._vlm.status,
    })


@app.route("/lock", methods=["POST"])
def lock():
    data = request.json or {}
    follower.click_lock(data.get("x", 0.5), data.get("y", 0.5))
    return jsonify({"ok": True})


@app.route("/stop", methods=["POST"])
def stop():
    follower.stop()
    send_cmd(0, 0)
    return jsonify({"ok": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Person Follower")
    parser.add_argument("--vlm-model", default="qwen",
                        choices=["qwen", "internvl"],
                        help="VLM backend: qwen (Qwen2-VL-2B) or internvl (InternVL2-2B)")
    args = parser.parse_args()

    follower = PersonFollower(vlm_model=args.vlm_model)
    threading.Thread(target=follower.run_loop, daemon=True).start()
    print(f"Open http://localhost:5001  [VLM: {args.vlm_model}]")
    app.run(host="0.0.0.0", port=5001)
