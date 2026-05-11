#!/usr/bin/env python3
# Smart Navigator — YOLO-World + VLM verify + EKF tracker + DA3 depth
# Run:  python3 scripts/smart_navigator.py [--vlm-model qwen|internvl]
# UI:   http://localhost:5002

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
CONF_HIGH   = 0.55
CONF_LOW    = 0.28
ARRIVE_DIST = 2.0
VLM_EVERY   = 3.0

# Spatial direction phrases — stripped from the query before YOLO/VLM
_SPATIAL_PHRASES = [
    (("on your left",  "to your left",  "on the left",  "to the left",  "left side"),  "left"),
    (("on your right", "to your right", "on the right", "to the right", "right side"), "right"),
    (("in front of you", "directly ahead", "straight ahead"),                           "center"),
]


# ── helpers ───────────────────────────────────────────────────────────────────

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


# ── VLM verifier ──────────────────────────────────────────────────────────────

class VLMVerifier(threading.Thread):
    """Verifies YOLO crops (yes/no), guides search direction, and advises stuck recovery."""

    def __init__(self, model_name="qwen"):
        super().__init__(daemon=True)
        self._lock           = threading.Lock()
        self._crop           = None
        self._crop_query     = ""
        self._mode           = "verify"   # "verify" | "search" | "stuck"
        self._model_name     = model_name
        self.verified        = False
        self.verified_for    = ""
        self.search_hint     = None       # "left" | "center" | "right" | None
        self.search_hint_for = ""
        self.stuck_hint      = None       # "left" | "right" | None
        self.status          = "VLM loading…"

    def submit(self, crop_rgb, query, mode="verify"):
        with self._lock:
            self._crop, self._crop_query, self._mode = crop_rgb.copy(), query, mode

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
                crop, q, mode = self._crop, self._crop_query, self._mode
                self._crop = None
            if crop is None:
                time.sleep(0.2)
                continue

            self.status = f"VLM [{mode}] '{q}'…"
            try:
                if mode == "verify":
                    prompt  = f"Does this image show a {q}? Reply only: yes or no."
                    max_tok = 5
                elif mode == "search":
                    prompt  = (f"Do you see a {q} in this image? "
                               f"Reply with exactly one word: LEFT, RIGHT, CENTER, or NO.")
                    max_tok = 8
                else:  # stuck
                    prompt  = ("There is an obstacle directly ahead of a robot. "
                               "Should it turn LEFT or RIGHT to go around? "
                               "Reply with one word: LEFT or RIGHT.")
                    max_tok = 5

                ans = backend.infer(Image.fromarray(crop), prompt, max_tok)

                if mode == "verify":
                    ok = ans.startswith("yes")
                    with self._lock:
                        self.verified, self.verified_for = ok, q
                    self.status = f"{'✓' if ok else '✗'} '{q}' → {ans}"

                elif mode == "search":
                    if   "left"   in ans: hint = "left"
                    elif "right"  in ans: hint = "right"
                    elif "center" in ans or "middle" in ans: hint = "center"
                    else: hint = None
                    with self._lock:
                        self.search_hint, self.search_hint_for = hint, q
                    self.status = f"Search: '{q}' → {hint or 'not found'}"

                else:  # stuck
                    hint = "left" if "left" in ans else ("right" if "right" in ans else None)
                    with self._lock:
                        self.stuck_hint = hint
                    self.status = f"Stuck guidance: → {hint or '?'}"

            except Exception as e:
                self.status = f"VLM error: {e}"


# ── navigator ─────────────────────────────────────────────────────────────────

class SmartNavigator:

    def __init__(self, vlm_model="qwen"):
        self.running   = True
        self.frame_b64 = ""
        self.depth_b64 = ""
        self.status    = "Loading models…"
        self.lin       = 0.0
        self.ang       = 0.0
        self.distance  = None
        self.angle_deg = None
        self.fps       = 0.0

        self._pend_q        = ""
        self._active_q      = ""
        self._active_base_q = ""
        self._active_vlm_q  = ""       # query sent to VLM (spatial phrases stripped)
        self._active_spatial = None    # "left" | "right" | "center" | None
        self._qlock         = threading.Lock()
        self._last_vlm      = 0.0

        self._yolo    = None
        self._depth   = None
        self._tracker = None
        self._vlm     = VLMVerifier(model_name=vlm_model)
        self._vlm.start()
        self._ready   = threading.Event()
        threading.Thread(target=self._load_models, daemon=True).start()

    def _load_models(self):
        try:
            self.status = "Loading YOLO-World…"
            from ultralytics import YOLOWorld
            self._yolo = YOLOWorld(os.path.join(_ROOT, "yolov8s-worldv2.pt"))
            try:
                import clip.model as _cm
                _orig_enc = _cm.CLIP.encode_text
                def _enc(self, text):
                    return _orig_enc(self, text.to(next(self.parameters()).device))
                _cm.CLIP.encode_text = _enc
            except Exception:
                pass

            self.status = "Warming up YOLO-World…"
            self._yolo.set_classes(["person"])
            self._yolo(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False)

            self.status = "Loading DepthEstimator…"
            self._depth = DepthEstimator(model_size="small", max_depth=10.0)

            self._tracker = TargetTracker(fov_h_deg=FOV_H_DEG, reid_threshold=0.45)
            self._ready.set()
            self.status = "Ready — set a target"
            print("All models ready.")
        except Exception as e:
            self.status = f"Model load error: {e}"

    def set_query(self, q):
        with self._qlock:
            self._pend_q = q

    def get_query(self):
        with self._qlock:
            return self._pend_q

    @staticmethod
    def _parse_query(q: str):
        """Split a full query into (vlm_desc, yolo_noun, spatial_dir).

        "tv on your left"     → ("tv",                  "tv",      "left")
        "red chair on right"  → ("red chair",            "chair",   "right")
        "person on the chair" → ("person on the chair",  "person",  None)
        "monitor"             → ("monitor",              "monitor", None)
        """
        low = q.lower().strip()
        spatial_dir = None

        for phrases, direction in _SPATIAL_PHRASES:
            for phrase in phrases:
                if phrase in low:
                    cleaned = low.replace(phrase, "").strip().strip(",").strip()
                    if cleaned:
                        low = cleaned
                    spatial_dir = direction
                    break
            if spatial_dir:
                break

        yolo_noun = SmartNavigator._base_query(low)
        return low, yolo_noun, spatial_dir

    @staticmethod
    def _base_query(q: str) -> str:
        """Extract a simple YOLO-compatible noun from a descriptive query."""
        low = q.lower().strip()
        for sep in (' with ', ' in ', ' wearing ', ' holding ', ' near ',
                    ' next to ', ' that ', ' having ', ' by ', ' beside ',
                    ' on ',  ' at '):
            if sep in low:
                low = low.split(sep)[0].strip()
                break
        words = low.split()
        if len(words) > 1:
            return words[-1]
        return words[0] if words else q

    def _detect(self, frame, q):
        if not q:
            return []
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

    def _trust(self, det):
        """Return True if this detection should be considered reliable."""
        vlm_q   = self._active_vlm_q
        base_q  = self._active_base_q
        is_desc = vlm_q.lower().strip() != base_q.lower().strip()
        if is_desc:
            return self._vlm.verified and self._vlm.verified_for == vlm_q
        if det["confidence"] >= CONF_HIGH:
            return True
        return self._vlm.verified and self._vlm.verified_for == vlm_q

    def _annotate(self, frame, dets, result):
        out = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        for d in dets:
            x1, y1, x2, y2 = (int(v) for v in d["bbox"])
            col = (0, 220, 80) if self._trust(d) else (0, 200, 200)
            cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
            cv2.putText(out, f"{d['class']} {d['confidence']:.0%}",
                        (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        if result.state == TrackerState.TRACKING and result.detection:
            bx1, by1, bx2, by2 = (int(v) for v in result.detection["bbox"])
            cv2.rectangle(out, (bx1, by1), (bx2, by2), (0, 255, 255), 3)
            tx, ty = (bx1 + bx2) // 2, (by1 + by2) // 2
            cv2.line(out, (out.shape[1] // 2, out.shape[0]), (tx, ty), (0, 255, 255), 2)
            cv2.circle(out, (tx, ty), 6, (0, 255, 255), -1)
        cv2.putText(out, result.state.name, (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, result.state.name, (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, self._vlm.status[:70], (8, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 255), 1, cv2.LINE_AA)
        fps_txt = f"{self.fps:.0f} FPS"
        cv2.putText(out, fps_txt, (out.shape[1] - 80, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, fps_txt, (out.shape[1] - 80, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 100), 1, cv2.LINE_AA)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    def run_loop(self):
        self._ready.wait()
        prev_t = time.time()
        KP, KD         = 0.25, 0.08
        TARGET_DIST    = 1.5
        _pid_prev            = 0.0
        _prev_state          = None
        _bypass_active       = False
        _bypass_dir          = 1.0
        _bypass_until        = 0.0
        _bypass_cooldown     = 0.0
        _retry_bypass        = False
        _last_search_vlm     = 0.0
        _last_tracked_dist   = float("inf")
        _reacquire_active    = False
        _reacquire_until     = 0.0
        _reacquire_tried     = False
        _reacquire_srch_end  = 0.0     # deadline for the post-backup search window

        while self.running:
            t0  = time.time()
            dt  = max(0.01, min(t0 - prev_t, 0.5))
            prev_t = t0

            with self._qlock:
                pend = self._pend_q
            if pend != self._active_q:
                self._active_q = pend
                if pend:
                    vlm_q, base_q, spatial = self._parse_query(pend)
                else:
                    vlm_q, base_q, spatial = "", "", None
                self._active_base_q  = base_q
                self._active_vlm_q   = vlm_q
                self._active_spatial = spatial
                if pend:
                    self._yolo.set_classes([base_q])
                    self._tracker.set_target(base_q)
                    self._last_vlm = 0.0
                    if base_q != pend:
                        print(f"[NAV] '{pend}' → YOLO='{base_q}' vlm='{vlm_q}'"
                              + (f" spatial={spatial}" if spatial else ""))
                else:
                    self._tracker.reset()
                self._vlm.verified        = False
                self._vlm.search_hint     = None
                self._vlm.stuck_hint      = None
                _pid_prev            = 0.0
                _bypass_active       = False
                _bypass_cooldown     = 0.0
                _retry_bypass        = False
                _last_search_vlm     = 0.0
                _last_tracked_dist   = float("inf")
                _reacquire_active    = False
                _reacquire_tried     = False
                _reacquire_srch_end  = 0.0

            q     = self._active_q
            frame = get_frame()
            if frame is None:
                self.status = "SDK offline — check localhost:8000"
                time.sleep(0.2)
                continue

            h, w = frame.shape[:2]

            if not q:
                self.frame_b64 = jpg_b64(frame)
                try:
                    _d, _ = self._depth.estimate(frame)
                    self.depth_b64 = depth_b64(_d)
                except Exception:
                    pass
                self.status = "Ready — set a target"
                time.sleep(0.10)
                continue

            dets = self._detect(frame, self._active_base_q)

            # Spatial filter — keep only detections on the requested side
            if self._active_spatial == "left":
                dets = [d for d in dets if d["center_x"] < w * 0.50]
            elif self._active_spatial == "right":
                dets = [d for d in dets if d["center_x"] > w * 0.50]
            elif self._active_spatial == "center":
                dets = [d for d in dets if w * 0.25 < d["center_x"] < w * 0.75]

            # Periodic VLM verification on best detection
            if dets and (t0 - self._last_vlm) >= VLM_EVERY:
                best = max(dets, key=lambda d: d["confidence"])
                x1, y1, x2, y2 = (int(v) for v in best["bbox"])
                crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                if crop.size > 0:
                    self._vlm.submit(crop, self._active_vlm_q)
                    self._last_vlm = t0

            trusted = [d for d in dets if self._trust(d)]

            # Spatial commitment: once EKF is initialised, lock to the detection
            # nearest the predicted bearing — prevents jumping to a second instance.
            if self._tracker._initialized and len(trusted) > 0:
                _fov  = np.radians(FOV_H_DEG)
                _pred = float(self._tracker._x[0])
                def _ang(d):
                    return float(np.arctan(
                        (d['center_x'] - w / 2) / (w / (2 * np.tan(_fov / 2)))))
                trusted = [min(trusted, key=lambda d: abs(_ang(d) - _pred))]

            try:
                depth_raw, conf_map = self._depth.estimate(frame)
            except Exception:
                time.sleep(0.05)
                continue
            depth = cv2.GaussianBlur(depth_raw, (5, 5), 1.5)

            result = self._tracker.update(
                frame=frame, detections=trusted, depth_map=depth,
                dt=dt, robot_angular_vel=self.ang,
                frame_width=w, frame_height=h,
            )

            raw_lin, raw_ang = 0.0, 0.0
            self.distance = self.angle_deg = None

            # 5-band horizontal depth scan
            dh, dw = depth.shape
            _strip  = depth[int(dh*0.3):int(dh*0.8), :]
            _bw     = dw // 5
            d_bands = [float(np.median(_strip[:, i*_bw:(i+1)*_bw])) for i in range(5)]
            d_C = d_bands[2]
            obs = 0.0
            for _i, _d in enumerate(d_bands):
                if _d < 1.2:
                    _push = (1.2 - _d) / 1.2
                    _side = (_i - 2) / 2.0
                    obs  -= _push * _side * 0.35
            obs = float(np.clip(obs, -1.0, 1.0))

            # Apply any VLM stuck hint for bypass direction
            if _retry_bypass and self._vlm.stuck_hint is not None:
                _bypass_dir = 1.0 if self._vlm.stuck_hint == "left" else -1.0
                self._vlm.stuck_hint = None

            # Bypass trigger
            if (not _bypass_active
                    and t0 >= _bypass_cooldown
                    and (result.state in (TrackerState.TRACKING, TrackerState.PREDICTING)
                         or _retry_bypass)
                    and 0.6 <= d_C < 1.1):
                _bypass_active = True
                _bypass_until  = t0 + 3.5
                if not _retry_bypass:
                    left_gap  = min(d_bands[0], d_bands[1])
                    right_gap = min(d_bands[3], d_bands[4])
                    if abs(left_gap - right_gap) > 0.2:
                        _bypass_dir = 1.0 if left_gap > right_gap else -1.0
                    else:
                        _bypass_dir = -1.0 if float(self._tracker._x[0]) >= 0 else 1.0
                _retry_bypass = False
                print(f"[NAV] Bypass → {'L' if _bypass_dir>0 else 'R'}  "
                      f"bands={[f'{d:.1f}' for d in d_bands]}")

            # Bypass end
            _bypass_just_ended = False
            if _bypass_active and (t0 >= _bypass_until or d_C >= 1.2):
                path_cleared       = d_C >= 1.2
                _bypass_active     = False
                _bypass_just_ended = True
                if path_cleared:
                    _bypass_cooldown = t0 + 3.0
                else:
                    _bypass_dir      = -_bypass_dir
                    _retry_bypass    = True
                    _bypass_cooldown = t0 + 0.3
                    self._vlm.submit(frame, self._active_vlm_q, mode="stuck")
                    print(f"[NAV] Bypass timed out — retrying {'L' if _bypass_dir>0 else 'R'}")

            if d_C < 0.6:
                # Emergency backup — pick gap direction now and turn while reversing
                _bypass_active   = False
                _bypass_cooldown = t0 + 0.3
                _retry_bypass    = True
                left_gap  = min(d_bands[0], d_bands[1])
                right_gap = min(d_bands[3], d_bands[4])
                if abs(left_gap - right_gap) > 0.15:
                    _bypass_dir = 1.0 if left_gap > right_gap else -1.0
                self._vlm.submit(frame, self._active_vlm_q, mode="stuck")
                raw_lin = -0.15
                raw_ang = _bypass_dir * 0.45   # decisive turn while reversing
                self.status = "Obstacle! Backing up…"

            elif _bypass_active:
                raw_ang = _bypass_dir * 0.45 + obs * 0.20
                raw_lin = 0.10
                self.distance  = result.distance
                self.angle_deg = float(np.degrees(self._tracker._x[0]))
                self.status = f"Going around… {'←' if _bypass_dir > 0 else '→'}"

            elif result.state in (TrackerState.TRACKING, TrackerState.PREDICTING):
                _last_tracked_dist = result.distance
                _reacquire_tried   = False   # successful track — allow one future re-find
                norm_err = result.angle / (np.radians(FOV_H_DEG) / 2)
                if _bypass_just_ended or _prev_state not in (
                        TrackerState.TRACKING, TrackerState.PREDICTING):
                    _pid_prev = norm_err
                    self.ang  = 0.0
                _bypass_active = False
                self.distance  = result.distance
                self.angle_deg = float(np.degrees(result.angle))
                if result.distance <= ARRIVE_DIST:
                    self.status = f"Arrived at '{q}' ({result.distance:.1f} m)"
                else:
                    deriv     = float(np.clip((norm_err - _pid_prev) / dt, -2.0, 2.0))
                    _pid_prev = norm_err
                    raw_ang   = float(np.clip(
                        -(KP * norm_err + KD * deriv) + obs * 0.30, -0.40, 0.40))
                    slowdown  = max(0.1, 1.0 - abs(norm_err))
                    raw_lin   = min(0.30, (result.distance - TARGET_DIST) * 0.3) * slowdown
                    self.status = f"→ '{q}'  {result.distance:.1f} m  {self.angle_deg:+.0f}°"

            elif result.state == TrackerState.SEARCHING:
                _bypass_active = False

                # Post-reacquire search window expired — declare final result
                if _reacquire_tried and t0 > _reacquire_srch_end > 0:
                    if _last_tracked_dist <= ARRIVE_DIST:
                        self.status = f"Arrived at '{q}' (~{_last_tracked_dist:.1f} m)"
                    else:
                        self.status = f"Lost '{q}' — set a new target"
                else:
                    # Periodically ask VLM where the target is
                    if (t0 - _last_search_vlm) >= VLM_EVERY * 1.5 and not dets:
                        self._vlm.submit(frame, self._active_vlm_q, mode="search")
                        _last_search_vlm = t0

                    vlm_dir = (self._vlm.search_hint
                               if self._vlm.search_hint_for == self._active_vlm_q else None)

                    if dets:
                        raw_ang = 0.05 * result.search_direction + obs * 0.3
                        self.status = f"Possible '{q}' — verifying…"
                    elif vlm_dir == "left":
                        raw_ang = 0.20 + obs * 0.3
                        self.status = f"VLM: '{q}' is left — turning ←"
                    elif vlm_dir == "right":
                        raw_ang = -0.20 + obs * 0.3
                        self.status = f"VLM: '{q}' is right — turning →"
                    elif vlm_dir == "center":
                        raw_lin = 0.12
                        raw_ang = obs * 0.3
                        self.status = f"VLM: '{q}' ahead — moving forward"
                    else:
                        raw_ang = 0.18 * result.search_direction + obs * 0.3
                        if abs(obs) > 0.1:
                            raw_lin = 0.10
                        self.status = f"Searching '{q}'…  {int(result.search_progress * 100)}%"

            elif result.state == TrackerState.LOST:
                # Lost at close range — back up briefly and retry before giving up
                if not _reacquire_tried and _last_tracked_dist < 3.5 and not _reacquire_active:
                    _reacquire_active = True
                    _reacquire_until  = t0 + 2.0
                    raw_lin = -0.12
                    self.status = f"Lost '{q}' close — backing up to re-find…"
                elif _reacquire_active:
                    if t0 < _reacquire_until:
                        raw_lin = -0.12
                        self.status = f"Backing up to re-find '{q}'…"
                    else:
                        # Backup done — restart the tracker for a short search window
                        _reacquire_active   = False
                        _reacquire_tried    = True
                        _reacquire_srch_end = t0 + 8.0
                        self._tracker.set_target(self._active_base_q)
                        _last_search_vlm = 0.0
                        self.status = f"Re-searching for '{q}'…"
                else:
                    self.status = f"Lost '{q}' — set a new target"

            else:
                self.status = f"Starting search for '{q}'…"

            _prev_state = result.state

            raw_ang  = float(np.clip(raw_ang, -0.50, 0.50))
            self.lin = self.lin * 0.55 + raw_lin * 0.45
            self.ang = float(np.clip(self.ang * 0.45 + raw_ang * 0.55, -0.50, 0.50))
            send_cmd(self.lin, self.ang)

            self.frame_b64 = jpg_b64(self._annotate(frame, dets, result))
            self.depth_b64 = depth_b64(depth)

            elapsed = time.time() - t0
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(elapsed, 0.01))
            if elapsed < 0.10:
                time.sleep(0.10 - elapsed)


# ── Flask ─────────────────────────────────────────────────────────────────────

nav = None
app = Flask(__name__)
logging.getLogger("werkzeug").setLevel(logging.ERROR)


@app.route("/")
def index():
    with open(os.path.join(_ROOT, "web", "smart_nav.html")) as f:
        return f.read()


@app.route("/state")
def state():
    return jsonify({
        "frame":    nav.frame_b64,
        "depth":    nav.depth_b64,
        "status":   nav.status,
        "vlm":      nav._vlm.status,
        "query":    nav.get_query(),
        "distance": nav.distance,
        "angle":    nav.angle_deg,
        "linear":   nav.lin,
        "angular":  nav.ang,
        "verified": nav._vlm.verified and nav._vlm.verified_for == nav._active_vlm_q,
        "fps":      round(nav.fps, 1),
    })


@app.route("/query", methods=["POST"])
def set_query():
    q = (request.json or {}).get("query", "").strip()
    nav.set_query(q)
    return jsonify({"ok": True, "query": q})


@app.route("/stop", methods=["POST"])
def stop_robot():
    nav.set_query("")
    nav.lin = nav.ang = 0.0
    send_cmd(0, 0)
    return jsonify({"ok": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Navigator")
    parser.add_argument("--vlm-model", default="qwen",
                        choices=["qwen", "internvl"],
                        help="VLM backend: qwen (Qwen2-VL-2B) or internvl (InternVL2-2B)")
    args = parser.parse_args()

    nav = SmartNavigator(vlm_model=args.vlm_model)
    threading.Thread(target=nav.run_loop, daemon=True).start()
    print(f"Open http://localhost:5002  [VLM: {args.vlm_model}]")
    app.run(host="0.0.0.0", port=5002)
