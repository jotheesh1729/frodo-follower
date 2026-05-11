"""Tests for the Kalman filter target tracker."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'frodo_ai', 'perception'))
from target_tracker import TargetTracker, TrackerState


@pytest.fixture
def tracker():
    return TargetTracker(coast_timeout=1.0, search_timeout=5.0, reid_threshold=0.3)


def make_frame(h=480, w=640):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def make_detection(cls='tv', cx=320, cy=240, w=100, h=80, conf=0.9):
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    return {
        'class': cls,
        'confidence': conf,
        'bbox': [x1, y1, x2, y2],
        'center_x': cx,
        'center_y': cy,
    }


def make_depth(h=480, w=640, val=3.0):
    return np.full((h, w), val, dtype=np.float32)


class TestTrackerStateTransitions:
    def test_idle_to_tracking(self, tracker):
        tracker.set_target('tv')
        frame = make_frame()
        det = make_detection('tv', cx=320)
        result = tracker.update(frame, [det], make_depth(), 0.1, 0.0, 640, 480)
        assert result.state == TrackerState.TRACKING

    def test_tracking_to_predicting(self, tracker):
        tracker.set_target('tv')
        frame = make_frame()
        det = make_detection('tv', cx=320)

        # First: acquire target
        tracker.update(frame, [det], make_depth(), 0.1, 0.0, 640, 480)

        # Second: target disappears — should predict
        result = tracker.update(frame, [], make_depth(), 0.1, 0.0, 640, 480)
        assert result.state == TrackerState.PREDICTING

    def test_predicting_to_searching(self, tracker):
        tracker.set_target('tv')
        frame = make_frame()
        det = make_detection('tv', cx=320)

        tracker.update(frame, [det], make_depth(), 0.1, 0.0, 640, 480)

        # Simulate time passing (coast_timeout=1.0)
        import time
        tracker._last_seen_time = time.time() - 2.0  # 2s ago

        result = tracker.update(frame, [], make_depth(), 0.1, 0.0, 640, 480)
        assert result.state == TrackerState.SEARCHING


class TestKalmanFilter:
    def test_angle_prediction(self, tracker):
        """Kalman should predict target angle during occlusion."""
        tracker.set_target('tv')
        frame = make_frame()

        # Target on the right side
        det = make_detection('tv', cx=480)  # right of center
        r1 = tracker.update(frame, [det], make_depth(), 0.1, 0.0, 640, 480)
        initial_angle = r1.angle

        # Target disappears
        r2 = tracker.update(frame, [], make_depth(), 0.1, 0.0, 640, 480)

        # Predicted angle should be close to initial
        assert abs(r2.angle - initial_angle) < 0.2

    def test_odometry_compensation(self, tracker):
        """Kalman should compensate for robot rotation."""
        tracker.set_target('tv')
        frame = make_frame()

        det = make_detection('tv', cx=320)  # center
        tracker.update(frame, [det], make_depth(), 0.1, 0.0, 640, 480)

        # Robot rotates left (positive angular), target should appear more to the right
        r = tracker.update(frame, [], make_depth(), 0.1, 0.5, 640, 480)
        # Angle should shift negative (target moves right in robot frame)
        assert r.angle < 0


class TestReID:
    def test_selects_closest_to_prediction(self, tracker):
        """With two TVs, should select the one closer to Kalman prediction."""
        tracker.set_target('tv')
        frame = make_frame()

        # Lock onto left TV
        det_left = make_detection('tv', cx=200)
        tracker.update(frame, [det_left], make_depth(), 0.1, 0.0, 640, 480)

        # Both TVs visible — should still prefer left one
        det_right = make_detection('tv', cx=500)
        result = tracker.update(frame, [det_left, det_right], make_depth(), 0.1, 0.0, 640, 480)

        assert result.detection is not None
        assert abs(result.detection['center_x'] - 200) < 50


class TestSearchDirection:
    def test_search_direction_follows_last_known(self, tracker):
        """Search should spin toward where target was last seen."""
        tracker.set_target('tv')
        frame = make_frame()

        # Target on the right
        det = make_detection('tv', cx=500)
        tracker.update(frame, [det], make_depth(), 0.1, 0.0, 640, 480)

        # Force into search
        import time
        tracker._last_seen_time = time.time() - 5.0

        result = tracker.update(frame, [], make_depth(), 0.1, 0.0, 640, 480)
        assert result.state == TrackerState.SEARCHING
        # Search direction should be negative (right)
        assert result.search_direction == -1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
