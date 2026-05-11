"""Target tracker with Extended Kalman Filter and lightweight Re-ID.

Maintains a smooth state estimate of the target object using an EKF
and provides appearance-based re-identification to avoid locking onto
the wrong instance after occlusion.

State vector: [angle, angular_velocity, distance, approach_velocity]
- angle: bearing to target in robot frame (radians, 0=forward, positive=left)
- angular_velocity: rate of change of bearing (rad/s)
- distance: depth to target (metres)
- approach_velocity: closing speed (m/s, negative = approaching)

Author: Jotheesh Reddy Kummathi
"""

from __future__ import annotations

import time
from enum import Enum, auto
from typing import Optional

import cv2
import numpy as np


class TrackerState(Enum):
    """Target tracker state machine states."""
    IDLE = auto()        # No target set
    TRACKING = auto()    # Target visible, Kalman updating
    PREDICTING = auto()  # Target lost briefly, Kalman predict only
    SEARCHING = auto()   # Lost too long, spinning to re-acquire
    LOST = auto()        # Search timed out, give up


class TargetTracker:
    """EKF-based target tracker with appearance re-identification.

    Usage:
        tracker = TargetTracker()
        tracker.set_target("tv")

        # Every frame:
        result = tracker.update(
            frame=rgb_frame,
            detections=[...],    # list of dicts with 'class', 'bbox', etc.
            depth_map=depth,
            dt=0.1,
            robot_angular_vel=0.05,
            frame_width=640,
            frame_height=480,
        )
        # result.state, result.angle, result.distance, result.search_direction
    """

    def __init__(
        self,
        coast_timeout: float = 2.0,
        search_timeout: float = 30.0,
        reid_threshold: float = 0.3,
        process_noise: float = 0.1,
        measurement_noise_angle: float = 0.05,
        measurement_noise_distance: float = 0.3,
        fov_h_deg: float = 90.0,
    ):
        self.coast_timeout = coast_timeout
        self.search_timeout = search_timeout
        self.reid_threshold = reid_threshold
        self.fov_h_deg = fov_h_deg

        # EKF matrices
        # State: [angle, ang_vel, distance, approach_vel]
        self._x = np.zeros(4)  # state
        self._P = np.eye(4) * 1.0  # covariance

        # Process noise
        self._Q = np.diag([
            process_noise * 0.5,     # angle
            process_noise * 1.0,     # angular velocity
            process_noise * 0.5,     # distance
            process_noise * 0.5,     # approach velocity
        ])

        # Measurement noise
        self._R = np.diag([
            measurement_noise_angle,
            measurement_noise_distance,
        ])

        # Measurement matrix: we observe [angle, distance]
        self._H = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
        ])

        # State
        self.state = TrackerState.IDLE
        self.target_class: str = ''
        self._appearance: Optional[np.ndarray] = None
        self._appearance_aspect: float = 0.0
        self._appearance_size: float = 0.0
        self._last_seen_time: float = 0.0
        self._search_start_time: float = 0.0
        self._initialized = False
        self._last_search_direction = 1.0  # +1 = left, -1 = right

    def set_target(self, target_class: str):
        """Set or clear the target object class."""
        self.target_class = target_class
        self.state = TrackerState.IDLE if not target_class else TrackerState.SEARCHING
        self._initialized = False
        self._appearance = None
        self._search_start_time = time.time()
        self._x = np.zeros(4)
        self._P = np.eye(4) * 1.0

    def reset(self):
        """Clear all state."""
        self.set_target('')

    def update(
        self,
        frame: np.ndarray,
        detections: list[dict],
        depth_map: Optional[np.ndarray],
        dt: float,
        robot_angular_vel: float,
        frame_width: int,
        frame_height: int,
    ) -> TrackerResult:
        """Run one tracker update step.

        Args:
            frame: RGB image (H, W, 3)
            detections: list of dicts with keys: 'class', 'bbox' [x1,y1,x2,y2],
                        'confidence', 'center_x', 'center_y'
            depth_map: (H, W) depth in metres, or None
            dt: time since last frame (seconds)
            robot_angular_vel: current commanded angular velocity (rad/s)
            frame_width: image width in pixels
            frame_height: image height in pixels

        Returns:
            TrackerResult with state, angle, distance, search_direction, match_score
        """
        if not self.target_class:
            self.state = TrackerState.IDLE
            return TrackerResult(state=TrackerState.IDLE)

        now = time.time()

        # --- EKF Predict step (always runs) ---
        self._predict(dt, robot_angular_vel)

        # --- Find candidate detections ---
        candidates = [d for d in detections if d['class'] == self.target_class]

        # --- Select best candidate ---
        best_det = None
        best_score = 0.0

        if candidates:
            for det in candidates:
                score = self._score_candidate(det, frame, frame_width, frame_height)
                if score > best_score:
                    best_score = score
                    best_det = det

        # --- State transitions ---
        if best_det is not None:
            # Check re-ID threshold if we had a previous appearance
            if self._appearance is not None and self.state in (TrackerState.SEARCHING, TrackerState.PREDICTING):
                if best_score < self.reid_threshold:
                    # Reject — doesn't look like our target
                    best_det = None

        if best_det is not None:
            # Measurement available — update EKF
            angle_meas = self._pixel_to_angle(best_det['center_x'], frame_width)
            dist_meas = self._get_depth_at(
                best_det['center_x'], best_det['center_y'],
                depth_map, frame_width, frame_height
            )

            self._update_measurement(angle_meas, dist_meas)

            # Update appearance signature
            self._update_appearance(frame, best_det['bbox'], frame_width, frame_height)

            self._last_seen_time = now
            self.state = TrackerState.TRACKING
            self._initialized = True

            # Remember which side the target is on for search
            if angle_meas > 0:
                self._last_search_direction = 1.0
            elif angle_meas < 0:
                self._last_search_direction = -1.0

            return TrackerResult(
                state=TrackerState.TRACKING,
                angle=float(self._x[0]),
                angular_velocity=float(self._x[1]),
                distance=float(self._x[2]),
                approach_velocity=float(self._x[3]),
                match_score=best_score,
                detection=best_det,
            )

        # No valid detection found
        if not self._initialized:
            # Never saw the target — keep searching
            self.state = TrackerState.SEARCHING
            elapsed = now - self._search_start_time
            if elapsed > self.search_timeout:
                self.state = TrackerState.LOST
                self.target_class = ''
                return TrackerResult(state=TrackerState.LOST)
            return TrackerResult(
                state=TrackerState.SEARCHING,
                search_direction=self._last_search_direction,
                search_progress=elapsed / self.search_timeout,
            )

        # Was tracking, now lost
        time_since_seen = now - self._last_seen_time

        if time_since_seen < self.coast_timeout:
            # Coast on Kalman prediction
            self.state = TrackerState.PREDICTING
            return TrackerResult(
                state=TrackerState.PREDICTING,
                angle=float(self._x[0]),
                angular_velocity=float(self._x[1]),
                distance=float(self._x[2]),
                approach_velocity=float(self._x[3]),
            )

        # Lost for too long — switch to search
        if self.state != TrackerState.SEARCHING:
            self._search_start_time = now
            # Search in the direction of the last known target position
            self._last_search_direction = 1.0 if self._x[0] > 0 else -1.0
        self.state = TrackerState.SEARCHING

        search_elapsed = now - self._search_start_time
        if search_elapsed > self.search_timeout:
            self.state = TrackerState.LOST
            self.target_class = ''
            return TrackerResult(state=TrackerState.LOST)

        return TrackerResult(
            state=TrackerState.SEARCHING,
            search_direction=self._last_search_direction,
            search_progress=search_elapsed / self.search_timeout,
            angle=float(self._x[0]),
            distance=float(self._x[2]),
        )

    # ------------------------------------------------------------------
    # EKF internals
    # ------------------------------------------------------------------

    def _predict(self, dt: float, robot_angular_vel: float):
        """EKF predict step with constant-velocity model + robot odometry."""
        # State transition:
        # angle' = angle + ang_vel * dt - robot_ang_vel * dt
        # ang_vel' = ang_vel  (constant velocity assumption)
        # dist' = dist + approach_vel * dt
        # approach_vel' = approach_vel

        F = np.eye(4)
        F[0, 1] = dt  # angle ← angular_velocity * dt
        F[2, 3] = dt  # distance ← approach_velocity * dt

        # Robot odometry compensation
        self._x[0] -= robot_angular_vel * dt

        # Predict state
        self._x = F @ self._x

        # Predict covariance
        self._P = F @ self._P @ F.T + self._Q * dt

    def _update_measurement(self, angle: float, distance: float):
        """EKF update step with [angle, distance] measurement."""
        z = np.array([angle, distance])
        y = z - self._H @ self._x  # innovation

        S = self._H @ self._P @ self._H.T + self._R  # innovation covariance
        K = self._P @ self._H.T @ np.linalg.inv(S)  # Kalman gain

        self._x = self._x + K @ y
        I = np.eye(4)
        self._P = (I - K @ self._H) @ self._P

    # ------------------------------------------------------------------
    # Appearance Re-ID
    # ------------------------------------------------------------------

    def _update_appearance(self, frame: np.ndarray, bbox: list, frame_w: int, frame_h: int):
        """Update the appearance signature from the current detection."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_w, x2)
        y2 = min(frame_h, y2)

        if x2 <= x1 or y2 <= y1:
            return

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return

        # HSV histogram (16 bins for H, 8 for S)
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        # EMA update of appearance (smooth, doesn't jump)
        if self._appearance is None:
            self._appearance = hist
        else:
            self._appearance = 0.8 * self._appearance + 0.2 * hist

        self._appearance_aspect = (x2 - x1) / max(y2 - y1, 1)
        self._appearance_size = (x2 - x1) * (y2 - y1) / max(frame_w * frame_h, 1)

    def _compute_appearance_similarity(self, frame: np.ndarray, bbox: list, frame_w: int, frame_h: int) -> float:
        """Compute appearance similarity between a candidate and the stored signature."""
        if self._appearance is None:
            return 1.0  # no prior appearance, accept anything

        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_w, x2)
        y2 = min(frame_h, y2)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return 0.0

        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        # Cosine similarity
        dot = np.dot(self._appearance, hist)
        norm_a = np.linalg.norm(self._appearance) + 1e-8
        norm_b = np.linalg.norm(hist) + 1e-8
        cosine_sim = dot / (norm_a * norm_b)

        # Size consistency
        aspect = (x2 - x1) / max(y2 - y1, 1)
        size = (x2 - x1) * (y2 - y1) / max(frame_w * frame_h, 1)
        aspect_sim = 1.0 - min(abs(aspect - self._appearance_aspect) / max(self._appearance_aspect, 0.1), 1.0)
        size_sim = 1.0 - min(abs(size - self._appearance_size) / max(self._appearance_size, 0.001), 1.0)

        return float(0.6 * cosine_sim + 0.25 * aspect_sim + 0.15 * size_sim)

    def _score_candidate(self, det: dict, frame: np.ndarray, frame_w: int, frame_h: int) -> float:
        """Score a candidate detection combining appearance, spatial proximity, and confidence."""
        # Appearance similarity
        app_sim = self._compute_appearance_similarity(frame, det['bbox'], frame_w, frame_h)

        # Spatial proximity to Kalman prediction
        if self._initialized:
            det_angle = self._pixel_to_angle(det['center_x'], frame_w)
            predicted_angle = self._x[0]
            angle_diff = abs(det_angle - predicted_angle)
            spatial_sim = max(0.0, 1.0 - angle_diff / (np.radians(self.fov_h_deg) / 2))
        else:
            spatial_sim = 1.0  # first detection, no prior

        # Detection confidence
        conf = det.get('confidence', 0.5)

        # Heavily weight spatial proximity (60%) so it doesn't jump to identical objects
        # just because their YOLO confidence is slightly higher.
        return float(0.3 * app_sim + 0.6 * spatial_sim + 0.1 * conf)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pixel_to_angle(self, px: float, img_width: int) -> float:
        """Convert pixel x-coordinate to angle from center."""
        center = img_width / 2.0
        fov_rad = np.radians(self.fov_h_deg)
        return float(np.arctan((px - center) / (img_width / (2 * np.tan(fov_rad / 2)))))

    def _get_depth_at(
        self, cx: float, cy: float,
        depth_map: Optional[np.ndarray],
        frame_w: int, frame_h: int,
    ) -> float:
        """Get median depth at a detection center point."""
        if depth_map is None:
            return self._x[2] if self._initialized else 5.0  # fallback

        dh, dw = depth_map.shape
        dx = min(max(int(cx * dw / frame_w), 0), dw - 1)
        dy = min(max(int(cy * dh / frame_h), 0), dh - 1)
        r = 5
        region = depth_map[max(0, dy - r):min(dh, dy + r), max(0, dx - r):min(dw, dx + r)]
        if region.size > 0:
            return float(np.median(region))
        return self._x[2] if self._initialized else 5.0


class TrackerResult:
    """Result from a tracker update step."""

    def __init__(
        self,
        state: TrackerState,
        angle: float = 0.0,
        angular_velocity: float = 0.0,
        distance: float = 0.0,
        approach_velocity: float = 0.0,
        match_score: float = 0.0,
        search_direction: float = 0.0,
        search_progress: float = 0.0,
        detection: Optional[dict] = None,
    ):
        self.state = state
        self.angle = angle
        self.angular_velocity = angular_velocity
        self.distance = distance
        self.approach_velocity = approach_velocity
        self.match_score = match_score
        self.search_direction = search_direction
        self.search_progress = search_progress
        self.detection = detection
