"""MPPI (Model Predictive Path Integral) planner for obstacle avoidance.

Uses DA2 depth map as the cost landscape. Samples many candidate
trajectories, scores them against obstacle proximity + goal direction,
and returns the optimal (linear, angular) command.

Vehicle model: unicycle (x, y, theta) with (v, omega) controls.
Cost: depth-based obstacle penalty + goal heading penalty + smoothness.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


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
    obstacle_weight: float = 50.0    # penalty for being near obstacles
    goal_weight: float = 5.0         # reward for heading toward goal
    smooth_weight: float = 2.0       # penalty for jerky controls
    backward_penalty: float = 20.0   # penalty for going backward

    # MPPI temperature (lower = more greedy)
    temperature: float = 0.5

    # Depth map params
    fov_horizontal_deg: float = 90.0  # camera FOV
    obstacle_threshold_rel: float = 0.30  # relative depth below this = obstacle

    # Image crop for depth (middle band)
    crop_top: float = 0.20
    crop_bottom: float = 0.65


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
    ) -> tuple[float, float, dict]:
        """Plan the next control command given a depth map.

        Args:
            depth_map: (H, W) depth in meters from DA2
            goal_direction_rad: direction to goal in robot frame
                                (0 = forward, positive = left, negative = right)
                                None if no goal (pure obstacle avoidance)

        Returns:
            linear: forward speed
            angular: turn rate
            debug: dict with trajectory info for visualization
        """
        cfg = self.config
        N = cfg.num_samples
        H = cfg.horizon

        # --- Build obstacle cost map from depth ---
        cost_map, fov_rad = self._build_cost_map(depth_map)

        # --- Sample control sequences ---
        # Perturb around previous optimal sequence
        lin_noise = np.random.randn(N, H) * cfg.linear_noise_std
        ang_noise = np.random.randn(N, H) * cfg.angular_noise_std

        lin_samples = self._prev_linear[None, :] + lin_noise  # (N, H)
        ang_samples = self._prev_angular[None, :] + ang_noise  # (N, H)

        # Clamp
        lin_samples = np.clip(lin_samples, -cfg.min_linear, cfg.max_linear)
        ang_samples = np.clip(ang_samples, -cfg.max_angular, cfg.max_angular)

        # --- Rollout trajectories ---
        # State: (x, y, theta) starting at origin facing forward
        x = np.zeros((N, H + 1))
        y = np.zeros((N, H + 1))
        theta = np.zeros((N, H + 1))

        for t in range(H):
            theta[:, t + 1] = theta[:, t] + ang_samples[:, t] * cfg.dt
            x[:, t + 1] = x[:, t] + lin_samples[:, t] * np.cos(theta[:, t + 1]) * cfg.dt
            y[:, t + 1] = y[:, t] + lin_samples[:, t] * np.sin(theta[:, t + 1]) * cfg.dt

        # --- Compute costs ---
        costs = np.zeros(N)

        # 1. Obstacle cost: project trajectory points into depth map
        obstacle_costs = self._compute_obstacle_cost(
            x[:, 1:], y[:, 1:], theta[:, 1:], cost_map, fov_rad, depth_map.shape)
        costs += cfg.obstacle_weight * obstacle_costs

        # 2. Goal heading cost: reward trajectories heading toward goal
        if goal_direction_rad is not None:
            # Average heading error across trajectory
            heading_error = np.abs(theta[:, 1:] - goal_direction_rad)
            heading_error = np.minimum(heading_error, 2 * np.pi - heading_error)
            costs += cfg.goal_weight * np.mean(heading_error, axis=1)

        # 3. Smoothness cost: penalize angular acceleration
        ang_diff = np.diff(ang_samples, axis=1)
        costs += cfg.smooth_weight * np.sum(ang_diff ** 2, axis=1)

        # 4. Backward penalty
        backward_mask = lin_samples < 0
        costs += cfg.backward_penalty * np.sum(backward_mask, axis=1)

        # --- MPPI weighting ---
        # Shift costs for numerical stability
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

        # --- Best trajectory for visualization ---
        best_idx = np.argmin(costs)
        best_x = x[best_idx]
        best_y = y[best_idx]

        # Top 10 trajectories for visualization
        top_indices = np.argsort(costs)[:10]

        debug = {
            "best_x": best_x,
            "best_y": best_y,
            "best_cost": float(costs[best_idx]),
            "mean_cost": float(np.mean(costs)),
            "top_x": x[top_indices],
            "top_y": y[top_indices],
            "top_costs": costs[top_indices],
            "weights_entropy": float(-np.sum(weights * np.log(weights + 1e-10))),
        }

        return float(opt_linear[0]), float(opt_angular[0]), debug

    def _build_cost_map(self, depth_map: np.ndarray) -> tuple[np.ndarray, float]:
        """Convert depth map to obstacle cost map.

        Returns cost_map (H_crop, W) where high values = obstacle,
        and the FOV in radians.
        """
        cfg = self.config
        h, w = depth_map.shape

        # Crop to middle band
        row_start = int(h * cfg.crop_top)
        row_end = int(h * cfg.crop_bottom)
        depth_crop = depth_map[row_start:row_end, :]

        # Normalize: closer = higher cost
        max_depth = np.max(depth_crop)
        if max_depth < 1e-6:
            return np.zeros_like(depth_crop), np.radians(cfg.fov_horizontal_deg)

        # Cost = 1 - normalized_depth (close objects have high cost)
        normalized = depth_crop / max_depth
        cost_map = np.clip(1.0 - normalized, 0.0, 1.0)

        # Boost cost for very close objects
        close_mask = normalized < cfg.obstacle_threshold_rel
        cost_map[close_mask] *= 3.0

        return cost_map, np.radians(cfg.fov_horizontal_deg)

    def _compute_obstacle_cost(
        self,
        x: np.ndarray,      # (N, H)
        y: np.ndarray,       # (N, H)
        theta: np.ndarray,   # (N, H)
        cost_map: np.ndarray,  # (H_crop, W)
        fov_rad: float,
        depth_shape: tuple,
    ) -> np.ndarray:
        """Project trajectory points onto cost map and compute obstacle cost."""
        N, H = x.shape
        h_cost, w_cost = cost_map.shape
        costs = np.zeros(N)

        # For each trajectory point, compute the angle from robot origin
        # and look up the cost in the depth map at that angle
        for t in range(H):
            # Direction of each trajectory point relative to robot at origin
            angles = np.arctan2(y[:, t], np.maximum(x[:, t], 0.01))  # (N,)

            # Map angle to pixel column
            # angle=0 → center, angle=+fov/2 → left edge, angle=-fov/2 → right edge
            col = ((angles / (fov_rad / 2)) * 0.5 + 0.5) * w_cost
            col = np.clip(col.astype(int), 0, w_cost - 1)

            # Distance along trajectory (closer points matter more)
            dist = np.sqrt(x[:, t] ** 2 + y[:, t] ** 2)
            dist_weight = 1.0 / (1.0 + dist * 2.0)  # closer = higher weight

            # Look up cost at center row of cost map for each column
            center_row = h_cost // 2
            # Average a few rows for robustness
            row_range = slice(max(0, center_row - 3), min(h_cost, center_row + 3))
            col_costs = np.mean(cost_map[row_range, :], axis=0)  # (W,)

            costs += col_costs[col] * dist_weight * (1.0 + t * 0.1)  # later steps weighted more

        return costs / H
