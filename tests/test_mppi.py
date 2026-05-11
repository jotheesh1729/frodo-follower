"""Tests for the improved MPPI planner."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'frodo_ai', 'planning'))
from mppi_planner import MPPIPlanner, MPPIConfig


@pytest.fixture
def planner():
    return MPPIPlanner(MPPIConfig(num_samples=256, horizon=8, dt=0.1))


def make_depth(h=480, w=640, default_depth=5.0):
    """Create a uniform depth map."""
    return np.full((h, w), default_depth, dtype=np.float32)


class TestAbsoluteDepthCost:
    """Test that cost is based on absolute depth, not relative."""

    def test_wall_at_2m_same_cost_regardless_of_max(self, planner):
        """A wall at 2m should have the same cost whether the max is 3m or 10m."""
        d1 = make_depth(default_depth=3.0)
        d1[:, 300:340] = 2.0  # wall at 2m

        d2 = make_depth(default_depth=10.0)
        d2[:, 300:340] = 2.0  # same wall at 2m, but background is 10m

        _, _, debug1 = planner.plan(d1, goal_direction_rad=0.0)
        planner.reset()
        _, _, debug2 = planner.plan(d2, goal_direction_rad=0.0)

        # Costs should be similar (not wildly different)
        cost_ratio = debug1["best_cost"] / max(debug2["best_cost"], 0.01)
        assert 0.3 < cost_ratio < 3.0, f"Cost ratio {cost_ratio} too extreme"

    def test_close_obstacle_high_cost(self, planner):
        """An obstacle at 0.5m should produce higher cost than one at 5m."""
        d_close = make_depth(default_depth=5.0)
        d_close[:, 280:360] = 0.5

        d_far = make_depth(default_depth=5.0)
        d_far[:, 280:360] = 4.0

        _, _, debug_close = planner.plan(d_close, goal_direction_rad=0.0)
        planner.reset()
        _, _, debug_far = planner.plan(d_far, goal_direction_rad=0.0)

        assert debug_close["best_cost"] > debug_far["best_cost"]


class TestEmergencyStop:
    """Test emergency stop when center is blocked."""

    def test_emergency_triggers(self, planner):
        """Should return zero linear when center is blocked at 0.3m."""
        d = make_depth(default_depth=5.0)
        d[:, 200:440] = 0.3  # block center

        lin, ang, debug = planner.plan(d, goal_direction_rad=0.0)
        assert lin == 0.0
        assert debug.get("emergency", False)

    def test_no_emergency_when_clear(self, planner):
        """Should not trigger emergency when center is clear."""
        d = make_depth(default_depth=5.0)
        lin, ang, debug = planner.plan(d, goal_direction_rad=0.0)
        assert lin > 0.0
        assert not debug.get("emergency", False)


class TestGoalDirection:
    """Test that the planner steers toward the goal."""

    def test_steers_toward_goal(self, planner):
        """In open space, planner should produce angular toward goal."""
        d = make_depth(default_depth=8.0)

        # Goal to the left
        _, ang_left, _ = planner.plan(d, goal_direction_rad=0.3)
        planner.reset()
        _, ang_right, _ = planner.plan(d, goal_direction_rad=-0.3)

        # Angular should differ (left goal → positive ang, right → negative)
        assert ang_left > ang_right


class TestCostColumns:
    """Test that cost_columns are returned in debug."""

    def test_cost_columns_in_debug(self, planner):
        d = make_depth(default_depth=5.0)
        _, _, debug = planner.plan(d, goal_direction_rad=0.0)
        assert "cost_columns" in debug
        assert len(debug["cost_columns"]) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
