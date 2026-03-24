"""Visual servoing controller — keeps target object centered in frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ServoConfig:
    steer_gain: float = 0.35
    max_linear: float = 0.30
    centering_linear: float = 0.05
    centering_threshold: float = 0.4      # normalized error above this = turn in place
    arrival_distance: float = 0.8         # stop when this close
    slowdown_distance: float = 2.0        # start slowing below this
    max_lost_frames: int = 15             # keep heading toward last position
    search_angular: float = 0.15          # spin speed when searching
    lost_linear: float = 0.15             # creep forward when recently lost


class VisualServo:
    """Centers a target object in the camera frame and drives toward it."""

    def __init__(self, config: ServoConfig = ServoConfig()):
        self.config = config
        self.last_target_angle: Optional[float] = None
        self.target_lost_frames: int = 0

    def reset(self):
        self.last_target_angle = None
        self.target_lost_frames = 0

    def compute(
        self,
        target_det: Optional[dict],
        target_dist: Optional[float],
        frame_width: int,
        target_name: str = "",
    ) -> tuple[float, float, str]:
        """Compute (linear, angular, status) from detection state.

        Args:
            target_det: detection dict with 'center_x' or None if not found
            target_dist: depth to target in meters or None
            frame_width: image width in pixels
            target_name: name of target for status messages
        """
        cfg = self.config

        if not target_name:
            self.reset()
            return 0.0, 0.0, "Idle"

        if target_det is not None:
            self.target_lost_frames = 0
            cx = target_det['center_x']
            frame_center = frame_width / 2.0
            normalized_error = (cx - frame_center) / (frame_width / 2.0)
            self.last_target_angle = normalized_error

            angular = -cfg.steer_gain * normalized_error

            if target_dist is not None and target_dist < cfg.arrival_distance:
                return 0.0, 0.0, f"ARRIVED at {target_name}! ({target_dist:.1f}m)"

            if abs(normalized_error) > cfg.centering_threshold:
                return cfg.centering_linear, angular, f"Centering {target_name}..."

            linear = cfg.max_linear
            if target_dist is not None and target_dist < cfg.slowdown_distance:
                linear = max(0.10, 0.15 * target_dist)
            dist_str = f'{target_dist:.1f}m' if target_dist else '?m'
            return linear, angular, f"Navigating -> {target_name} ({dist_str})"

        # Target not found
        self.target_lost_frames += 1

        if self.target_lost_frames < cfg.max_lost_frames and self.last_target_angle is not None:
            angular = -cfg.steer_gain * self.last_target_angle * 0.5
            return cfg.lost_linear, angular, \
                f"Lost {target_name} - last heading ({self.target_lost_frames}/{cfg.max_lost_frames})"

        return 0.0, cfg.search_angular, f"Searching for {target_name}..."
