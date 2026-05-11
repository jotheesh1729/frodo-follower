#!/usr/bin/env python3
# Smart Navigator — YOLO-World + VLM verify + EKF tracker + DA3 depth
# Run:  python3 scripts/smart_navigator.py
# UI:   http://localhost:5002

import sys, os, time, threading, base64, io
import numpy as np
import cv2
import requests
from PIL import Image
from flask import Flask, request, jsonify
import logging
import torch

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
    """Verifies YOLO crops (yes/no) and guides search rotation direction."""

    def __init__(self):
        super().__init__(daemon=True)
        self._lock           = threading.Lock()
        self._crop           = None
        self._crop_query     = ""
        self._mode           = "verify"   # "verify" | "search"
        self.verified        = False
        self.verified_for    = ""
        self.search_hint     = None       # "left" | "center" | "right" | None
        self.search_hint_for = ""
        self.status          = "VLM loading…"

    def submit(self, crop_rgb, query, mode="verify"):
        with self._lock:
            self._crop, self._crop_query, self._mode = crop_rgb.copy(), query, mode

    def run(self):
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            from qwen_vl_utils import process_vision_info
            import torch
        except ImportError:
            self.status = "VLM unavailable (pip install transformers qwen-vl-utils)"
            return

        self.status = "Loading Qwen2-VL-2B…"
        try:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.float16, device_map="auto")
            proc  = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
            self.status = "VLM ready"
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
                from qwen_vl_utils import process_vision_info
                import torch

                if mode == "verify":
                    prompt = f"Does this image show a {q}? Reply only: yes or no."
                    max_tok = 5
                else:  # search
                    prompt = (f"Do you see a {q} in this image? "
                              f"Reply with exactly one word: LEFT, RIGHT, CENTER, or NO.")
                    max_tok = 8

                msgs = [{"role": "user", "content": [
                    {"type": "image", "image": Image.fromarray(crop)},
                    {"type": "text",  "text": prompt},
                ]}]
                text    = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                imgs, _ = process_vision_info(msgs)
                inp     = proc(text=[text], images=imgs, return_tensors="pt").to("cuda")
                ids     = model.generate(**inp, max_new_tokens=max_tok)
                ans     = proc.batch_decode([ids[0][len(inp.input_ids[0]):]],
                                            skip_special_tokens=True)[0].strip().lower()

                if mode == "verify":
                    ok = ans.startswith("yes")
                    with self._lock:
                        self.verified, self.verified_for = ok, q
                    self.status = f"{'ok' if ok else 'no'} '{q}'"

                else:  # search
                    if "left"   in ans: hint = "left"
                    elif "right" in ans: hint = "right"
                    elif "center" in ans or "middle" in ans: hint = "center"
                    else: hint = None
                    with self._lock:
                        self.search_hint, self.search_hint_for = hint, q
                    self.status = f"Search: '{q}' → {hint or 'not found'}"

            except Exception as e:
                self.status = f"VLM error: {e}"


# ── MPPI controller ───────────────────────────────────────────────────────────

class MPPIController:
    """
    Mapless Model Predictive Path Integral controller.

    Uses the live depth frame as a one-shot local obstacle map — no persistent
    map required.  Jointly optimises target tracking and obstacle avoidance in
    a single cost function, replacing the PD + bypass-arc approach.

    All trajectory rollouts are batched as tensor operations so the hot path
    runs entirely on GPU (or CPU if unavailable).
    """

    N       = 512    # trajectory samples
    T       = 15     # horizon steps
    DT      = 0.10   # seconds per step  →  1.5 s total horizon

    SIGMA_V = 0.15   # linear velocity perturbation std  (m/s)
    SIGMA_W = 0.25   # angular velocity perturbation std (rad/s)
    LAM     = 0.8    # MPPI temperature — low enough to commit, high enough not to be jerky

    # Cost weights
    W_OBS_HIT   = 60.0    # base cost when trajectory hits an obstacle
    W_OBS_DEPTH = 25.0    # extra cost per metre of penetration into obstacle
    W_EMERGENCY = 300.0   # cost for any waypoint within 0.45 m of an obstacle
    W_GOAL_BEAR =  6.0    # terminal bearing error cost
    W_GOAL_DIST =  4.0    # terminal distance-to-goal cost — drives forward motion
    W_RUN_BEAR  =  0.3    # per-step bearing cost — prevents "swing wide then correct" plans
    W_EFFORT_V  =  0.05   # linear velocity effort regularisation
    W_EFFORT_W  =  0.08   # angular velocity effort regularisation

    def __init__(self, fov_h_deg: float = 90.0):
        self.fov    = np.radians(fov_h_deg)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._U: torch.Tensor = torch.zeros(self.T, 2, device=self.device)
        print(f"[MPPI] running on {self.device}  N={self.N}  T={self.T}  dt={self.DT}s")

    def reset(self):
        """Clear nominal control sequence — call when target changes or robot is stuck."""
        self._U = torch.zeros(self.T, 2, device=self.device)

    @torch.no_grad()
    def step(
        self,
        depth_np: np.ndarray,
        target_angle: float,
        target_dist: float,
    ) -> tuple[float, float]:
        """
        Compute optimal (v, w) for this timestep.

        depth_np:     H×W float32 depth map in metres
        target_angle: EKF bearing to target, radians  (+ve = camera-right, same as pixel_to_angle)
        target_dist:  EKF range to target, metres

        Returns (v_cmd, w_cmd) as Python floats.
        """
        dev = self.device
        N, T, dt = self.N, self.T, self.DT

        depth = torch.as_tensor(depth_np, dtype=torch.float32, device=dev)
        dh, dw = depth.shape

        # Per-column minimum depth in the obstacle band (30 %–70 % of frame).
        # Min gives the tightest obstacle at each bearing — conservative and correct.
        r0, r1    = int(dh * 0.30), int(dh * 0.70)
        depth_col = depth[r0:r1, :].min(dim=0).values   # (dw,)

        # Focal length: pixels per radian horizontally
        focal = float(dw) / (2.0 * np.tan(self.fov / 2.0))

        # ── Sample noise perturbations [N, T, 2] ─────────────────────────────
        eps = torch.randn(N, T, 2, device=dev)
        eps[:, :, 0].mul_(self.SIGMA_V)
        eps[:, :, 1].mul_(self.SIGMA_W)

        # Perturbed control sequences — broadcast nominal [1,T,2] + noise [N,T,2]
        V = self._U.unsqueeze(0) + eps          # [N, T, 2]
        V[:, :, 0].clamp_(-0.15, 0.35)         # linear:  reverse → max forward
        V[:, :, 1].clamp_(-0.50, 0.50)         # angular: hard cap matches PD limit

        # ── Simulate N trajectories in parallel ───────────────────────────────
        # All start at the robot origin facing forward (ego-centric frame)
        x  = torch.zeros(N, device=dev)
        y  = torch.zeros(N, device=dev)
        th = torch.zeros(N, device=dev)
        costs = torch.zeros(N, device=dev)

        # Target position in ego frame (held fixed over horizon — valid for 3 s).
        # pixel_to_angle returns +ve for camera-right, but simulation uses standard
        # math where +y = left, so we negate the lateral component.
        tx = float(target_dist * np.cos(target_angle))
        ty = float(-target_dist * np.sin(target_angle))

        for t in range(T):
            v  = V[:, t, 0]
            w  = V[:, t, 1]

            # Unicycle kinematics
            x  = x  + v * torch.cos(th) * dt
            y  = y  + v * torch.sin(th) * dt
            th = th + w * dt

            # Only penalise waypoints in front of the camera
            forward = (x > 0.15).float()

            # Project waypoint into depth image column.
            # Standard math: +y = left, so rightward paths have bear_wp < 0.
            # Image convention: right = px > dw/2, so negate to match.
            bear_wp = torch.atan2(y, x)                          # horiz bearing
            dist_wp = torch.hypot(x, y)                          # range to waypoint
            px = (dw / 2.0 - bear_wp * focal).long().clamp_(0, dw - 1)
            d_at = depth_col[px]                                  # obstacle depth there

            # Collision: obstacle is closer than the waypoint
            hit     = ((d_at < dist_wp).float()) * forward
            penetr  = (dist_wp - d_at).clamp(min=0.0)
            costs  += hit * (self.W_OBS_HIT + penetr * self.W_OBS_DEPTH)

            # Emergency: obstacle very close in this direction regardless
            emerg  = ((d_at < 0.45).float()) * forward
            costs += emerg * self.W_EMERGENCY

            # Per-step bearing cost — discourages "swing hard then correct" plans
            s_dx   = tx - x
            s_dy   = ty - y
            s_bear = torch.atan2(s_dy, s_dx) - th
            s_bear = torch.atan2(torch.sin(s_bear), torch.cos(s_bear))
            costs += s_bear.abs() * self.W_RUN_BEAR

        # ── Terminal (horizon-end) goal cost ─────────────────────────────────
        dx = tx - x
        dy = ty - y
        g_dist  = torch.hypot(dx, dy)
        g_bear  = torch.atan2(dy, dx) - th
        g_bear  = torch.atan2(torch.sin(g_bear), torch.cos(g_bear))   # wrap to [-π, π]
        costs  += g_bear.abs() * self.W_GOAL_BEAR + g_dist * self.W_GOAL_DIST

        # ── Control effort penalty ────────────────────────────────────────────
        costs += V[:, :, 0].pow(2).sum(1) * self.W_EFFORT_V
        costs += V[:, :, 1].pow(2).sum(1) * self.W_EFFORT_W

        # ── MPPI weight update ────────────────────────────────────────────────
        beta   = costs.min()
        w_mppi = torch.exp(-(costs - beta) / self.LAM)
        w_mppi = w_mppi / (w_mppi.sum() + 1e-8)

        # Weighted noise → update nominal sequence
        delta   = (w_mppi.view(N, 1, 1) * eps).sum(0)    # [T, 2]
        self._U = self._U + delta
        self._U[:, 0].clamp_(-0.15, 0.35)
        self._U[:, 1].clamp_(-0.50, 0.50)

        v_out = float(self._U[0, 0])
        w_out = float(self._U[0, 1])

        # Receding horizon — shift sequence forward, zero-pad tail
        self._U = torch.roll(self._U, -1, dims=0)
        self._U[-1] = 0.0

        return v_out, w_out


# ── navigator ─────────────────────────────────────────────────────────────────

class SmartNavigator:

    def __init__(self):
        self.running   = True
        self.frame_b64 = ""
        self.depth_b64 = ""
        self.status    = "Loading models…"
        self.lin       = 0.0
        self.ang       = 0.0
        self.distance  = None
        self.angle_deg = None
        self.fps       = 0.0

        self._pend_q      = ""
        self._active_q    = ""
        self._active_base_q = ""
        self._qlock       = threading.Lock()
        self._last_vlm    = 0.0

        self._yolo    = None
        self._depth   = None
        self._tracker = None
        self._vlm     = VLMVerifier()
        self._vlm.start()
        self._mppi    = MPPIController(fov_h_deg=FOV_H_DEG)
        self._ready   = threading.Event()
        threading.Thread(target=self._load_models, daemon=True).start()

    def _load_models(self):
        try:
            self.status = "Loading YOLO-World…"
            from ultralytics import YOLOWorld
            self._yolo = YOLOWorld(os.path.join(_ROOT, "yolov8s-worldv2.pt"))
            # CLIP tokenize() always returns CPU tensors but the model is on CUDA.
            # Patch encode_text to move tokens to the right device before indexing.
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
    def _base_query(q: str) -> str:
        """Extract a simple YOLO-compatible noun from a descriptive query.

        'person with brown shirt' → 'person'
        'red chair'               → 'chair'
        'blue bottle'             → 'bottle'
        'tv'                      → 'tv'
        """
        low = q.lower().strip()
        for sep in (' with ', ' in ', ' wearing ', ' holding ', ' near ',
                    ' next to ', ' that ', ' having ', ' by ', ' beside '):
            if sep in low:
                low = low.split(sep)[0].strip()
                break
        words = low.split()
        if len(words) > 1:
            return words[-1]   # last word is typically the noun
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

    def _trust(self, det, q):
        is_descriptive = q.lower().strip() != self._active_base_q.lower().strip()
        if is_descriptive:
            # Full description must be confirmed by VLM — high YOLO conf alone isn't enough
            return self._vlm.verified and self._vlm.verified_for == q
        if det["confidence"] >= CONF_HIGH:
            return True
        return self._vlm.verified and self._vlm.verified_for == q

    def _annotate(self, frame, dets, q, result):
        out = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        for d in dets:
            x1, y1, x2, y2 = (int(v) for v in d["bbox"])
            col = (0, 220, 80) if self._trust(d, q) else (0, 200, 200)
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
        _last_search_vlm = 0.0

        while self.running:
            t0  = time.time()
            dt  = max(0.01, min(t0 - prev_t, 0.5))
            prev_t = t0

            with self._qlock:
                pend = self._pend_q
            if pend != self._active_q:
                self._active_q = pend
                base_q = self._base_query(pend) if pend else ""
                self._active_base_q = base_q
                if pend:
                    self._yolo.set_classes([base_q])
                    self._tracker.set_target(base_q)
                    self._last_vlm = 0.0
                    if base_q != pend:
                        print(f"[NAV] Descriptive query '{pend}' → YOLO class '{base_q}', VLM verifies full description")
                else:
                    self._tracker.reset()
                self._vlm.verified    = False
                self._vlm.search_hint = None
                self._mppi.reset()
                _last_search_vlm = 0.0

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

            dets = self._detect(frame, q)

            if dets and (t0 - self._last_vlm) >= VLM_EVERY:
                best = max(dets, key=lambda d: d["confidence"])
                x1, y1, x2, y2 = (int(v) for v in best["bbox"])
                crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                if crop.size > 0:
                    self._vlm.submit(crop, q)
                    self._last_vlm = t0

            trusted = [d for d in dets if self._trust(d, q)]

            # Once we have a prior EKF estimate, commit to exactly ONE detection —
            # whichever is closest to the predicted angle — in ALL states.
            # Identical objects (2 TVs, 2 chairs) have the same appearance histogram
            # so spatial commitment is the only reliable discriminator.
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

            # Centre depth for emergency detection only
            dh, dw = depth.shape
            d_C = float(np.median(depth[int(dh*0.3):int(dh*0.7),
                                        int(dw*0.33):int(dw*0.66)]))

            if d_C < 0.5:
                # Hard safety override — MPPI plan is stale after reversing
                self._mppi.reset()
                raw_lin = -0.15
                self.status = "Obstacle! Backing up…"

            elif result.state in (TrackerState.TRACKING, TrackerState.PREDICTING):
                self.distance  = result.distance
                self.angle_deg = float(np.degrees(result.angle))
                if result.distance <= ARRIVE_DIST:
                    self._mppi.reset()
                    self.status = f"Arrived at '{q}' ({result.distance:.1f} m)"
                else:
                    raw_lin, raw_ang = self._mppi.step(depth, result.angle, result.distance)
                    self.status = f"→ '{q}'  {result.distance:.1f} m  {self.angle_deg:+.0f}°"

            elif result.state == TrackerState.SEARCHING:
                self._mppi.reset()

                if (t0 - _last_search_vlm) >= VLM_EVERY * 1.5 and not dets:
                    self._vlm.submit(frame, q, mode="search")
                    _last_search_vlm = t0

                vlm_dir = (self._vlm.search_hint
                           if self._vlm.search_hint_for == q else None)

                if dets:
                    raw_ang = 0.05 * result.search_direction
                    self.status = f"Possible '{q}' — verifying…"
                elif vlm_dir == "left":
                    raw_ang = 0.20
                    self.status = f"VLM: '{q}' is left — turning ←"
                elif vlm_dir == "right":
                    raw_ang = -0.20
                    self.status = f"VLM: '{q}' is right — turning →"
                elif vlm_dir == "center":
                    raw_lin = 0.12
                    self.status = f"VLM: '{q}' is ahead — moving forward"
                else:
                    raw_ang = 0.18 * result.search_direction
                    self.status = f"Searching '{q}'…  {int(result.search_progress * 100)}%"

            elif result.state == TrackerState.LOST:
                self._mppi.reset()
                self.status = f"Lost '{q}' — set a new target"
            else:
                self.status = f"Starting search for '{q}'…"

            raw_ang = float(np.clip(raw_ang, -0.50, 0.50))

            self.lin = self.lin * 0.5 + raw_lin * 0.5
            self.ang = float(np.clip(self.ang * 0.55 + raw_ang * 0.45, -0.50, 0.50))
            send_cmd(self.lin, self.ang)

            self.frame_b64 = jpg_b64(self._annotate(frame, dets, q, result))
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
        "verified": nav._vlm.verified and nav._vlm.verified_for == nav.get_query(),
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
    nav = SmartNavigator()
    threading.Thread(target=nav.run_loop, daemon=True).start()
    print("Open http://localhost:5002")
    app.run(host="0.0.0.0", port=5002)
