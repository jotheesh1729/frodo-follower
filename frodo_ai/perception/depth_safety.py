"""
Depth Safety Layer

Runtime safety layer using monocular depth estimation.
Can be plugged into the Navigator to verify/override unsafe waypoints.

This is the depth equivalent of how navigator.py orchestrates CityWalker.
It wraps depth_estimator.py into a reusable safety check that can be
composed with any waypoint-producing model.

Usage:
    from depth_safety import DepthSafetyLayer

    safety = DepthSafetyLayer(margin=0.5)

    # In your control loop:
    frame = rover.get_camera_frame()
    safe_wp, was_overridden = safety.check_waypoint(frame, predicted_waypoint)

Author: Vivek Mattam
"""

import numpy as np
import time


class DepthSafetyLayer:
    """
    Runtime depth safety checking for waypoint navigation.

    Takes a predicted waypoint and the current camera frame,
    checks if the direction is safe using depth estimation,
    and either passes through or overrides with a safe direction.
    """

    def __init__(self, model_size='small', margin=0.5, num_bins=32,
                 fov_horizontal=90.0, speed_scale=0.5, device=None):
        """
        Initialize depth safety layer.

        Args:
            model_size: Depth model size ('small', 'base', 'large')
            margin: Safety margin in meters (waypoints with less clearance are unsafe)
            num_bins: Number of angular bins for polar clearance
            fov_horizontal: Camera horizontal FOV in degrees
            speed_scale: Scale factor for safe override waypoints (0-1)
            device: 'cuda' or 'cpu' (auto if None)
        """
        from depth_estimator import DepthEstimator

        self.estimator = DepthEstimator(model_size=model_size, device=device)
        self.margin = margin
        self.num_bins = num_bins
        self.fov_horizontal = fov_horizontal
        self.speed_scale = speed_scale

        # Stats
        self.total_checks = 0
        self.total_overrides = 0
        self.last_clearance = None
        self.last_bin_centers = None
        self.last_inference_ms = 0.0

    def check_waypoint(self, frame, waypoint):
        """
        Check if a waypoint is safe given the current camera frame.

        Args:
            frame: RGB image (H, W, 3) numpy array, values 0-255
            waypoint: (x, y) predicted waypoint in robot frame (meters)
                      x = forward, y = left

        Returns:
            safe_waypoint: (x, y) - either the original or a safe override
            was_overridden: True if the waypoint was replaced
        """
        self.total_checks += 1

        # Estimate depth
        start = time.time()
        depth = self.estimator.estimate(frame)
        self.last_inference_ms = (time.time() - start) * 1000

        # Compute polar clearance
        clearance, bin_centers = self.estimator.get_polar_clearance(
            depth, num_bins=self.num_bins, fov_horizontal=self.fov_horizontal
        )
        self.last_clearance = clearance
        self.last_bin_centers = bin_centers

        # Check if waypoint direction is safe
        safe, wp_clearance = self.estimator.is_waypoint_safe(
            np.array(waypoint), clearance, bin_centers, margin=self.margin
        )

        if safe:
            return waypoint, False

        # Unsafe - find safe direction and override
        self.total_overrides += 1
        best_angle, best_cl = self.estimator.get_safe_direction(
            clearance, bin_centers, margin=self.margin
        )

        # Create safe waypoint in best direction
        safe_wp = np.array([
            np.cos(best_angle) * self.speed_scale,
            np.sin(best_angle) * self.speed_scale
        ])

        return safe_wp, True

    def check_waypoints(self, frame, waypoints):
        """
        Check multiple waypoints (e.g., all 5 CityWalker predictions).

        Only checks the first waypoint (immediate next action).

        Args:
            frame: RGB image (H, W, 3)
            waypoints: (T, 2) array of predicted waypoints

        Returns:
            safe_waypoints: (T, 2) - first may be overridden, rest unchanged
            was_overridden: True if first waypoint was replaced
        """
        safe_wp, was_overridden = self.check_waypoint(frame, waypoints[0])

        result = waypoints.copy()
        if was_overridden:
            result[0] = safe_wp

        return result, was_overridden

    def get_clearance_at_direction(self, angle_rad):
        """
        Get clearance at a specific direction (uses last computed clearance).

        Args:
            angle_rad: Direction in radians (0 = forward)

        Returns:
            clearance_meters: Clearance in that direction, or None if no data
        """
        if self.last_clearance is None or self.last_bin_centers is None:
            return None

        bin_idx = np.argmin(np.abs(self.last_bin_centers - angle_rad))
        return self.last_clearance[bin_idx]

    def get_min_clearance(self):
        """Get minimum clearance across all directions (from last frame)."""
        if self.last_clearance is None:
            return None
        return float(np.min(self.last_clearance))

    def get_forward_clearance(self):
        """Get clearance directly ahead (from last frame)."""
        return self.get_clearance_at_direction(0.0)

    def get_stats(self):
        """Get safety layer statistics."""
        override_rate = (self.total_overrides / max(self.total_checks, 1)) * 100
        return {
            'total_checks': self.total_checks,
            'total_overrides': self.total_overrides,
            'override_rate_pct': override_rate,
            'last_inference_ms': self.last_inference_ms,
            'min_clearance': self.get_min_clearance(),
            'forward_clearance': self.get_forward_clearance(),
        }

    def reset_stats(self):
        """Reset safety statistics."""
        self.total_checks = 0
        self.total_overrides = 0


class DummyDepthSafety:
    """
    No-op safety layer for when depth checking is disabled.

    Matches DepthSafetyLayer interface but always passes through.
    Use this when you want to disable safety without changing code.
    """

    def __init__(self):
        self.total_checks = 0
        self.total_overrides = 0
        self.last_inference_ms = 0.0

    def check_waypoint(self, frame, waypoint):
        self.total_checks += 1
        return waypoint, False

    def check_waypoints(self, frame, waypoints):
        self.total_checks += 1
        return waypoints, False

    def get_stats(self):
        return {
            'total_checks': self.total_checks,
            'total_overrides': 0,
            'override_rate_pct': 0.0,
            'last_inference_ms': 0.0,
            'min_clearance': None,
            'forward_clearance': None,
        }

    def reset_stats(self):
        self.total_checks = 0


# Quick test
if __name__ == "__main__":
    print("=" * 60)
    print("DEPTH SAFETY LAYER TEST")
    print("=" * 60)

    print("\n[1] Creating safety layer...")
    try:
        safety = DepthSafetyLayer(model_size='small', margin=0.5)
        print("  Safety layer created!")

        # Test with dummy frame
        print("\n[2] Testing with dummy frame...")
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dummy_wp = np.array([1.0, 0.0])  # Straight ahead

        safe_wp, overridden = safety.check_waypoint(dummy_frame, dummy_wp)
        print(f"  Original: ({dummy_wp[0]:.2f}, {dummy_wp[1]:.2f})")
        print(f"  Result:   ({safe_wp[0]:.2f}, {safe_wp[1]:.2f})")
        print(f"  Overridden: {overridden}")
        print(f"  Inference: {safety.last_inference_ms:.1f}ms")

        # Test with multiple waypoints
        print("\n[3] Testing with multiple waypoints...")
        dummy_wps = np.random.randn(5, 2) * 0.5
        safe_wps, overridden = safety.check_waypoints(dummy_frame, dummy_wps)
        print(f"  Overridden: {overridden}")

        # Stats
        print(f"\n[4] Stats: {safety.get_stats()}")

        print("\nDepth safety layer test passed!")

    except Exception as e:
        print(f"\n  Error: {e}")
        print("  (Need Depth Anything V2 checkpoint for full test)")
        print("  Module structure is ready.")
