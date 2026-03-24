"""Outdoor GPS-based controller for ERC missions.

Simple proportional controller that steers toward GPS waypoints.
Optionally integrates monocular depth for obstacle avoidance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ControlCommand:
    linear: float
    angular: float
    reason: str
    debug: Optional[dict] = None


@dataclass
class OutdoorControllerConfig:
    # Speed limits (SDK range is -1 to 1)
    max_linear: float = 0.35
    min_linear: float = 0.08
    max_angular: float = 0.45
    # Heading P-gain: angular = -gain * heading_error_deg
    heading_p_gain: float = 0.008
    # Slow down when heading error is large
    heading_slow_threshold_deg: float = 30.0
    heading_slow_scale: float = 0.4
    # Stop and turn in place when heading error is very large
    turn_in_place_threshold_deg: float = 60.0
    turn_in_place_angular: float = 0.30
    # Slowdown near checkpoint
    slowdown_radius_m: float = 20.0
    slowdown_min_scale: float = 0.35
    # Arrival — creep slowly
    arrival_radius_m: float = 8.0
    arrival_linear: float = 0.10
    # Depth obstacle avoidance
    depth_stop_m: float = 0.5           # full stop if obstacle closer than this
    depth_slow_m: float = 1.2           # slow down if obstacle closer than this
    depth_override_angular: float = 0.30  # steer away angular when obstacle detected
    # Stall detection: if commanding forward but RPM is near zero
    rpm_stall_threshold: float = 1.5
    rpm_stall_max_ticks: int = 8
    # Angular rate limiting for smooth turns
    angular_rate_limit: float = 0.15


class OutdoorController:
    """P-controller with obstacle avoidance for outdoor GPS waypoint following."""

    def __init__(self, config: OutdoorControllerConfig = OutdoorControllerConfig()):
        self.config = config
        self._prev_angular: float = 0.0
        self._rpm_stall_count: int = 0

    def reset(self) -> None:
        self._prev_angular = 0.0
        self._rpm_stall_count = 0

    def _rate_limit_angular(self, desired: float) -> float:
        delta = desired - self._prev_angular
        limit = self.config.angular_rate_limit
        if delta > limit:
            desired = self._prev_angular + limit
        elif delta < -limit:
            desired = self._prev_angular - limit
        self._prev_angular = desired
        return desired

    def compute_command(
        self,
        nav_state: dict,
        depth_clearance_m: Optional[float] = None,
        safe_direction_rad: Optional[float] = None,
        rpm_mean: Optional[float] = None,
    ) -> ControlCommand:
        """Compute motor command from GPS nav state and optional depth info.

        Args:
            nav_state: dict from GPSWaypointManager.compute_nav_state()
            depth_clearance_m: forward clearance in meters (None = no depth data)
            safe_direction_rad: if obstacle detected, safest direction in radians
                                (0=forward, positive=left, negative=right)
            rpm_mean: average wheel RPM (for stall detection)
        """
        cfg = self.config

        if nav_state.get("mission_complete"):
            return ControlCommand(0.0, 0.0, "mission_complete")

        distance_m = nav_state["distance_m"]
        h_error = nav_state["heading_error_deg"]
        within_arrival = nav_state["within_arrival"]
        within_slowdown = nav_state["within_slowdown"]

        # --- Depth obstacle check (highest priority) ---
        if depth_clearance_m is not None:
            if depth_clearance_m < cfg.depth_stop_m:
                # Very close obstacle — steer away or stop
                if safe_direction_rad is not None:
                    angular = cfg.depth_override_angular * (1.0 if safe_direction_rad > 0 else -1.0)
                    angular = self._rate_limit_angular(angular)
                    return ControlCommand(0.0, angular, "depth_obstacle_turn",
                                          debug={"clearance": depth_clearance_m})
                return ControlCommand(0.0, 0.0, "depth_obstacle_stop",
                                      debug={"clearance": depth_clearance_m})
            elif depth_clearance_m < cfg.depth_slow_m:
                # Moderate obstacle — reduce speed
                depth_scale = (depth_clearance_m - cfg.depth_stop_m) / (cfg.depth_slow_m - cfg.depth_stop_m)
                depth_scale = max(0.2, min(1.0, depth_scale))
            else:
                depth_scale = 1.0
        else:
            depth_scale = 1.0

        # --- RPM stall detection ---
        if rpm_mean is not None and rpm_mean < cfg.rpm_stall_threshold:
            self._rpm_stall_count += 1
        else:
            self._rpm_stall_count = 0

        if self._rpm_stall_count >= cfg.rpm_stall_max_ticks:
            self._rpm_stall_count = 0
            return ControlCommand(-0.12, 0.0, "rpm_stall_backup")

        # --- Within arrival radius: creep forward ---
        if within_arrival:
            angular = -cfg.heading_p_gain * h_error
            angular = max(-cfg.max_angular, min(cfg.max_angular, angular))
            angular = self._rate_limit_angular(angular)
            linear = cfg.arrival_linear * depth_scale
            return ControlCommand(linear, angular, "arrival_creep",
                                  debug={"dist": distance_m, "h_err": h_error})

        # --- Turn in place if heading error is very large ---
        abs_error = abs(h_error)
        if abs_error > cfg.turn_in_place_threshold_deg:
            sign = -1.0 if h_error > 0 else 1.0  # negative error → target is right → turn right (neg angular)
            angular = sign * cfg.turn_in_place_angular
            angular = self._rate_limit_angular(angular)
            return ControlCommand(0.0, angular, "turn_in_place",
                                  debug={"h_err": h_error})

        # --- Normal driving: P-control on heading, speed from distance ---
        angular = -cfg.heading_p_gain * h_error
        angular = max(-cfg.max_angular, min(cfg.max_angular, angular))
        angular = self._rate_limit_angular(angular)

        linear = cfg.max_linear

        # Slow down for large heading error
        if abs_error > cfg.heading_slow_threshold_deg:
            heading_scale = cfg.heading_slow_scale
        else:
            heading_scale = 1.0

        # Slow down near checkpoint
        if within_slowdown:
            proximity_scale = max(cfg.slowdown_min_scale,
                                  distance_m / cfg.slowdown_radius_m)
        else:
            proximity_scale = 1.0

        linear *= heading_scale * proximity_scale * depth_scale
        linear = max(cfg.min_linear, min(cfg.max_linear, linear))

        return ControlCommand(linear, angular, "drive_to_checkpoint",
                              debug={"dist": distance_m, "h_err": h_error,
                                     "depth_scale": depth_scale})
