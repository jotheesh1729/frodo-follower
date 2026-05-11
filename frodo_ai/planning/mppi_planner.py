"""MPPI (Model Predictive Path Integral) planner for obstacle avoidance.

Uses DA2/DA3 depth map as the cost landscape. Samples many candidate
trajectories, scores them against obstacle proximity + goal direction,
and returns the optimal (linear, angular) command.

Vehicle model: unicycle (x, y, theta) with (v, omega) controls.
Cost: depth-based obstacle penalty + goal heading penalty + smoothness.

Improvements over v1:
- Absolute depth cost (no frame-relative normalization)
- Gaussian obstacle inflation for thin obstacles
- Multi-row bottom-weighted depth sampling
- Heading-corrected trajectory projection
- Emergency stop when center of frame is blocked
- Optional confidence weighting from DA3
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

try:
    from scipy.ndimage import gaussian_filter1d
except ImportError:
    # Fallback if scipy not installed
    def gaussian_filter1d(x, sigma, **kwargs):
        """Simple Gaussian blur approximation using numpy."""
        kernel_size = int(6 * sigma + 1) | 1
        k = np.exp(-0.5 * (np.arange(kernel_size) - kernel_size // 2) ** 2 / sigma ** 2)
        k /= k.sum()
        return np.convolve(x, k, mode='same')


@dataclass
class MPPIConfig:
    # Trajectory sampling
    num_samples: int = 512          # number of candidate trajectories
    horizon: int = 12               # planning horizon (timesteps)
    dt: float = 0.15                # timestep duration (seconds)

    # Control limits
    max_linear: float = 0.35
    min_linear: float = 0.05
    max_angular: float = 0.50

    # Sampling noise (std dev for perturbations)
    linear_noise_std: float = 0.10
    angular_noise_std: float = 0.25

    # Cost weights
    obstacle_weight: float = 60.0    # penalty for being near obstacles
    goal_weight: float = 5.0         # reward for heading toward goal
    smooth_weight: float = 2.0       # penalty for jerky controls
    backward_penalty: float = 20.0   # penalty for going backward
    clearance_bonus: float = 5.0     # bonus for maintaining side clearance

    # MPPI temperature (lower = more greedy)
    temperature: float = 0.5

    # Depth map params
    fov_horizontal_deg: float = 90.0  # camera FOV

    # Image crop for depth (top cut removes sky; keep as much ground as possible)
    crop_top: float = 0.20
    crop_bottom: float = 0.85

    # Emergency stop
    emergency_depth: float = 0.6     # metres — stop if center is this close
    emergency_width_frac: float = 0.33  # fraction of frame width for emergency check

    # Obstacle inflation
    inflation_sigma: float = 5.0     # Gaussian blur sigma for obstacle inflation (pixels)

    # Absolute depth thresholds (metres → cost)
    # Cost = clip(1/(depth + depth_cost_offset) - depth_cost_bias, 0, 1)
    depth_cost_offset: float = 0.1
    depth_cost_bias: float = 0.1


class MPPIPlanner:
    """MPPI trajectory planner using depth maps for obstacle avoidance."""

    def __init__(self, config: MPPIConfig = MPPIConfig()):
        self.config = config
        self._prev_linear = np.full(config.horizon, config.max_linear * 0.5)
        self._prev_angular = np.zeros(config.horizon)

    def reset(self):
        cfg = self.config
        self._prev_linear = np.full(cfg.horizon, cfg.max_linear * 0.5)
        self._prev_angular = np.zeros(cfg.horizon)

    def plan(
        self,
        depth_map: np.ndarray,
        goal_direction_rad: Optional[float] = None,
        confidence_map: Optional[np.ndarray] = None,
    ) -> tuple[float, float, dict]:
        """Plan the next control command given a depth map.

        Args:
            depth_map: (H, W) depth in meters from DA2/DA3
            goal_direction_rad: direction to goal in robot frame
                                (0 = forward, positive = left, negative = right)
                                None if no goal (pure obstacle avoidance)
            confidence_map: optional (H, W) confidence from DA3 (0-1).
                           Low confidence regions have reduced obstacle cost.

        Returns:
            linear: forward speed
            angular: turn rate
            debug: dict with trajectory info for visualization
        """
        cfg = self.config
        N = cfg.num_samples
        H = cfg.horizon

        # --- Emergency stop check ---
        emergency, escape_ang = self._check_emergency(depth_map)
        if emergency:
            self.reset()
            debug = {
                "emergency": True,
                "best_x": np.zeros(H + 1),
                "best_y": np.zeros(H + 1),
                "best_cost": 999.0,
                "mean_cost": 999.0,
                "top_x": np.zeros((1, H + 1)),
                "top_y": np.zeros((1, H + 1)),
                "top_costs": np.array([999.0]),
                "weights_entropy": 0.0,
                "cost_columns": np.zeros(10),
            }
            # Back up slowly while turning
            backup_speed = -max(cfg.min_linear, 0.2)
            return float(backup_speed), float(escape_ang), debug

        # --- Build obstacle cost map from depth ---
        cost_columns, fov_rad = self._build_cost_columns(depth_map, confidence_map)

        # --- Sample control sequences ---
        lin_noise = np.random.randn(N, H) * cfg.linear_noise_std
        ang_noise = np.random.randn(N, H) * cfg.angular_noise_std

        lin_samples = self._prev_linear[None, :] + lin_noise  # (N, H)
        ang_samples = self._prev_angular[None, :] + ang_noise  # (N, H)

        # Clamp
        lin_samples = np.clip(lin_samples, -cfg.min_linear, cfg.max_linear)
        ang_samples = np.clip(ang_samples, -cfg.max_angular, cfg.max_angular)

        # --- Rollout trajectories ---
        x = np.zeros((N, H + 1))
        y = np.zeros((N, H + 1))
        theta = np.zeros((N, H + 1))

        for t in range(H):
            theta[:, t + 1] = theta[:, t] + ang_samples[:, t] * cfg.dt
            x[:, t + 1] = x[:, t] + lin_samples[:, t] * np.cos(theta[:, t + 1]) * cfg.dt
            y[:, t + 1] = y[:, t] + lin_samples[:, t] * np.sin(theta[:, t + 1]) * cfg.dt

        # --- Compute costs ---
        costs = np.zeros(N)

        # 1. Obstacle cost: project trajectory points into cost columns
        obstacle_costs = self._compute_obstacle_cost(
            x[:, 1:], y[:, 1:], theta[:, 1:], cost_columns, fov_rad)
        costs += cfg.obstacle_weight * obstacle_costs

        # 2. Goal heading cost
        if goal_direction_rad is not None:
            heading_error = np.abs(theta[:, 1:] - goal_direction_rad)
            heading_error = np.minimum(heading_error, 2 * np.pi - heading_error)
            costs += cfg.goal_weight * np.mean(heading_error, axis=1)

        # 3. Smoothness cost
        ang_diff = np.diff(ang_samples, axis=1)
        costs += cfg.smooth_weight * np.sum(ang_diff ** 2, axis=1)

        # 4. Backward penalty
        backward_mask = lin_samples < 0
        costs += cfg.backward_penalty * np.sum(backward_mask, axis=1)

        # 5. Clearance corridor bonus (reward trajectories staying in clear zones)
        clearance_reward = self._compute_clearance_bonus(
            x[:, 1:], y[:, 1:], theta[:, 1:], cost_columns, fov_rad)
        costs -= cfg.clearance_bonus * clearance_reward

        # --- MPPI weighting ---
        min_cost = np.min(costs)
        weights = np.exp(-(costs - min_cost) / cfg.temperature)
        weights /= np.sum(weights) + 1e-10

        # Weighted average of control sequences
        opt_linear = np.sum(weights[:, None] * lin_samples, axis=0)
        opt_angular = np.sum(weights[:, None] * ang_samples, axis=0)

        # Clamp final result
        opt_linear = np.clip(opt_linear, cfg.min_linear, cfg.max_linear)
        opt_angular = np.clip(opt_angular, -cfg.max_angular, cfg.max_angular)

        # Save for next iteration (warm start)
        self._prev_linear = np.roll(opt_linear, -1)
        self._prev_linear[-1] = cfg.max_linear * 0.5
        self._prev_angular = np.roll(opt_angular, -1)
        self._prev_angular[-1] = 0.0

        # --- Visualization data ---
        best_idx = np.argmin(costs)
        top_indices = np.argsort(costs)[:10]

        debug = {
            "emergency": False,
            "best_x": x[best_idx],
            "best_y": y[best_idx],
            "best_cost": float(costs[best_idx]),
            "mean_cost": float(np.mean(costs)),
            "top_x": x[top_indices],
            "top_y": y[top_indices],
            "top_costs": costs[top_indices],
            "weights_entropy": float(-np.sum(weights * np.log(weights + 1e-10))),
            "cost_columns": cost_columns,
        }

        return float(opt_linear[0]), float(opt_angular[0]), debug

    def _build_cost_columns(
        self,
        depth_map: np.ndarray,
        confidence_map: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, float]:
        """Convert depth map to 1D obstacle cost vector per angular column.

        Uses absolute depth thresholds (not frame-relative normalization),
        bottom-weighted multi-row averaging, and Gaussian inflation.

        Returns:
            cost_columns: (W,) array where high values = obstacle
            fov_rad: field of view in radians
        """
        cfg = self.config
        h, w = depth_map.shape

        # Crop to middle band
        row_start = int(h * cfg.crop_top)
        row_end = int(h * cfg.crop_bottom)
        depth_crop = depth_map[row_start:row_end, :]
        h_crop = depth_crop.shape[0]

        # Absolute depth cost: close = high cost, far = zero cost
        # cost = clip(1/(depth + 0.1) - 0.1, 0, 1)
        # At 0.5m → 1.0, at 1m → 0.8, at 2m → 0.38, at 5m → 0.10, at 10m → 0.0
        cost_2d = np.clip(
            1.0 / (depth_crop + cfg.depth_cost_offset) - cfg.depth_cost_bias,
            0.0, 1.0
        )

        # Apply confidence weighting if available (DA3)
        if confidence_map is not None:
            conf_crop = confidence_map[row_start:row_end, :]
            cost_2d *= conf_crop

        # Bottom-weighted row averaging
        # Ground-level obstacles (bottom rows) matter more than upper rows
        row_weights = np.exp(np.linspace(0.0, 2.0, h_crop))
        row_weights /= row_weights.sum()
        cost_columns = np.average(cost_2d, axis=0, weights=row_weights)

        # Gaussian inflation — thin obstacles get a safety buffer
        cost_columns = gaussian_filter1d(cost_columns, sigma=cfg.inflation_sigma)

        # Re-normalize to [0, 1] after inflation
        col_max = cost_columns.max()
        if col_max > 1e-6:
            cost_columns = cost_columns / col_max

        return cost_columns, np.radians(cfg.fov_horizontal_deg)

    def _compute_obstacle_cost(
        self,
        x: np.ndarray,      # (N, H)
        y: np.ndarray,      # (N, H)
        theta: np.ndarray,  # (N, H)
        cost_columns: np.ndarray,  # (W,)
        fov_rad: float,
    ) -> np.ndarray:
        """Project trajectory points onto cost columns with heading correction."""
        N, H = x.shape
        W = len(cost_columns)
        costs = np.zeros(N)

        for t in range(H):
            # Heading-corrected projection:
            # Transform trajectory point (x, y) into the view from the robot's
            # current heading at this timestep
            cos_t = np.cos(theta[:, t])
            sin_t = np.sin(theta[:, t])

            # Robot-frame coordinates at this step
            dx = x[:, t] * cos_t + y[:, t] * sin_t
            dy = -x[:, t] * sin_t + y[:, t] * cos_t

            # Angle relative to robot heading
            angles = np.arctan2(dy, np.maximum(dx, 0.01))

            # Map angle to column index
            col = ((angles / (fov_rad / 2)) * 0.5 + 0.5) * W
            col = np.clip(col.astype(int), 0, W - 1)

            # Distance-based weighting (closer trajectory points matter more)
            dist = np.sqrt(x[:, t] ** 2 + y[:, t] ** 2)
            dist_weight = 1.0 / (1.0 + dist * 2.0)

            # Time weighting (later steps slightly more important)
            time_weight = 1.0 + t * 0.15

            costs += cost_columns[col] * dist_weight * time_weight

        return costs / H

    def _compute_clearance_bonus(
        self,
        x: np.ndarray,      # (N, H)
        y: np.ndarray,      # (N, H)
        theta: np.ndarray,  # (N, H)
        cost_columns: np.ndarray,  # (W,)
        fov_rad: float,
    ) -> np.ndarray:
        """Reward trajectories that maintain clearance on both sides."""
        N, H = x.shape
        W = len(cost_columns)
        bonus = np.zeros(N)

        for t in range(0, H, 3):  # sample every 3rd step for efficiency
            cos_t = np.cos(theta[:, t])
            sin_t = np.sin(theta[:, t])
            dx = x[:, t] * cos_t + y[:, t] * sin_t
            dy = -x[:, t] * sin_t + y[:, t] * cos_t
            angles = np.arctan2(dy, np.maximum(dx, 0.01))
            col = ((angles / (fov_rad / 2)) * 0.5 + 0.5) * W
            col = np.clip(col.astype(int), 0, W - 1)

            # Check 3 columns left and right for clearance
            spread = 3
            for offset in range(-spread, spread + 1):
                neighbor_col = np.clip(col + offset, 0, W - 1)
                clear = cost_columns[neighbor_col] < 0.2  # low cost = clear
                bonus += clear.astype(float)

        return bonus / (H // 3 + 1)

    def _check_emergency(self, depth_map: np.ndarray) -> tuple[bool, float]:
        """Check if the center of the frame is dangerously close.

        Returns:
            (is_emergency, escape_angular_velocity)
        """
        cfg = self.config
        h, w = depth_map.shape

        # Check center strip
        col_start = int(w * (0.5 - cfg.emergency_width_frac / 2))
        col_end = int(w * (0.5 + cfg.emergency_width_frac / 2))
        center_strip = depth_map[int(h * 0.3):int(h * 0.7), col_start:col_end]

        median_depth = np.median(center_strip)

        if median_depth < cfg.emergency_depth:
            # Find safest direction (left or right half with more clearance)
            left_depth = np.median(depth_map[:, :w // 2])
            right_depth = np.median(depth_map[:, w // 2:])
            escape_dir = cfg.max_angular if left_depth > right_depth else -cfg.max_angular
            return True, escape_dir

        return False, 0.0
