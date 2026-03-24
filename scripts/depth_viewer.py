#!/usr/bin/env python3
"""Live depth visualization + autonomous obstacle avoidance.

Shows camera feed, DA2 depth map, and obstacle clearance in real-time.
Two modes:
  AUTO mode (default): Robot drives forward and automatically avoids obstacles.
  MANUAL mode: You control with WASD.

Controls:
    m       - toggle MANUAL/AUTO mode
    w/s     - forward/backward (manual mode)
    a/d     - turn left/right (manual mode)
    x/space - stop
    q       - quit
"""

import sys
import os
import base64
import io
import time
import threading

import numpy as np
import requests
from PIL import Image

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'third_party', 'Depth-Anything-V2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'third_party', 'Depth-Anything-V2', 'metric_depth'))

SDK_URL = "http://localhost:8000"

# Control state
current_linear = 0.0
current_angular = 0.0
running = True
auto_mode = True  # Start in autonomous mode

# Obstacle avoidance params
AUTO_LINEAR = 0.35          # cruise speed
TURN_ANGULAR = 0.40         # turn speed when avoiding
# Relative thresholds (fraction of max depth in scene)
DEPTH_STOP_REL = 0.15       # obstacle is really close relative to scene
DEPTH_SLOW_REL = 0.25       # getting close
DEPTH_TURN_REL = 0.35       # start gentle correction
# Absolute fallback — if the ENTIRE scene is close, stop no matter what
ABS_STOP_M = 0.5            # if max depth in scene < this, everything is close → STOP
ABS_SLOW_M = 1.0            # if max depth < this → SLOW


# Set True if robot drives backward when given positive linear
FLIP_LINEAR = False

def send_control(linear, angular):
    try:
        if FLIP_LINEAR:
            linear = -linear
        requests.post(f"{SDK_URL}/control-legacy",
                      json={"command": {"linear": linear, "angular": angular, "lamp": 0}},
                      timeout=1.0)
    except Exception:
        pass


def get_frame():
    try:
        resp = requests.get(f"{SDK_URL}/v2/front", timeout=3)
        data = resp.json()
        b64 = data.get("front_frame")
        if b64:
            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            if img.size[0] > 10:
                return np.array(img, dtype=np.uint8)
    except Exception:
        pass
    return None


def on_key(event):
    global current_linear, current_angular, running, auto_mode
    key = event.key
    if key == 'q':
        running = False
        send_control(0, 0)
        plt.close('all')
        return
    elif key == 'm':
        auto_mode = not auto_mode
        current_linear = 0.0
        current_angular = 0.0
        send_control(0, 0)
        print(f"Mode: {'AUTO' if auto_mode else 'MANUAL'}")
        return

    if not auto_mode:
        if key == 'w':
            current_linear = 0.35
            current_angular = 0.0
        elif key == 's':
            current_linear = -0.25
            current_angular = 0.0
        elif key == 'a':
            current_linear = 0.0
            current_angular = 0.4
        elif key == 'd':
            current_linear = 0.0
            current_angular = -0.4
        elif key in ('x', ' '):
            current_linear = 0.0
            current_angular = 0.0
        send_control(current_linear, current_angular)
    elif key in ('x', ' '):
        current_linear = 0.0
        current_angular = 0.0
        send_control(0, 0)


def compute_auto_command(clearance, bin_centers, fwd_clearance, max_clearance):
    """Autonomous obstacle avoidance using relative + absolute depth."""
    if max_clearance < 1e-6:
        return 0.0, 0.0, "NO_DEPTH"

    # Compare left half vs right half to decide turn direction
    left_clear = float(np.mean(clearance[:num_bins // 2]))
    right_clear = float(np.mean(clearance[num_bins // 2:]))
    turn_sign = 1.0 if left_clear > right_clear else -1.0

    # ABSOLUTE check first — if the whole scene is close, stop
    # This catches the case where obstacle fills entire frame
    if max_clearance < ABS_STOP_M:
        return 0.0, turn_sign * TURN_ANGULAR, "ABS_STOP"
    if max_clearance < ABS_SLOW_M:
        return AUTO_LINEAR * 0.3, turn_sign * TURN_ANGULAR * 0.7, "ABS_SLOW"

    # RELATIVE check — obstacle closer than rest of scene
    fwd_rel = fwd_clearance / max_clearance
    safe_rel = float(clearance.max()) / max_clearance

    if fwd_rel < DEPTH_STOP_REL:
        if safe_rel < DEPTH_STOP_REL:
            return -0.12, 0.0, "BACKUP"
        return 0.0, turn_sign * TURN_ANGULAR, "AVOID"

    elif fwd_rel < DEPTH_SLOW_REL:
        scale = (fwd_rel - DEPTH_STOP_REL) / (DEPTH_SLOW_REL - DEPTH_STOP_REL)
        linear = AUTO_LINEAR * max(0.3, scale)
        return linear, turn_sign * TURN_ANGULAR * 0.7, "SLOW"

    elif fwd_rel < DEPTH_TURN_REL:
        scale = (fwd_rel - DEPTH_SLOW_REL) / (DEPTH_TURN_REL - DEPTH_SLOW_REL)
        linear = AUTO_LINEAR * max(0.6, scale)
        return linear, turn_sign * TURN_ANGULAR * 0.4, "STEER"

    else:
        return AUTO_LINEAR, 0.0, "CLEAR"


# Global for bin count (used in compute_auto_command)
num_bins = 16


def main():
    global running

    # Load DA2
    print("Loading DA2 on GPU...")
    from depth_estimator import DepthEstimator
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    estimator = DepthEstimator(model_size='small', max_depth=10.0, device=device)
    print(f"DA2 ready on {device}")

    # Setup plot
    fig = plt.figure(figsize=(16, 5))
    fig.canvas.manager.set_window_title('DA2 Live Depth Viewer — WASD to drive, Q to quit')
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 0.8])

    ax_cam = fig.add_subplot(gs[0])
    ax_depth = fig.add_subplot(gs[1])
    ax_clear = fig.add_subplot(gs[2])

    # Initial dummy data
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    im_cam = ax_cam.imshow(dummy)
    ax_cam.set_title("Camera")
    ax_cam.axis('off')

    im_depth = ax_depth.imshow(np.zeros((100, 100)), cmap='plasma', vmin=0, vmax=10)
    ax_depth.set_title("DA2 Depth")
    ax_depth.axis('off')

    ax_clear.set_xlim(-50, 50)
    ax_clear.set_ylim(0, 5)
    ax_clear.set_xlabel('Angle (°)')
    ax_clear.set_ylabel('Clearance (m)')
    ax_clear.set_title('Obstacle Clearance')

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout()
    plt.ion()
    plt.show()

    print("\nAUTO MODE — robot drives and avoids obstacles automatically")
    print("Press M to switch to manual (WASD), Q to quit")
    print("Click on the figure window first to capture keys!\n")

    frame_count = 0
    try:
        while running:
            t0 = time.time()

            frame = get_frame()
            if frame is None:
                plt.pause(0.1)
                continue

            # Run DA2
            depth = estimator.estimate(frame)

            # Custom clearance: use MIDDLE band (20%-65% of image height)
            # This ignores ceiling/sky (top) and ground plane (bottom)
            h, w = depth.shape
            row_start = int(h * 0.20)
            row_end = int(h * 0.65)
            depth_mid = depth[row_start:row_end, :]

            # Compute clearance per angular bin using MEDIAN (robust to ground pixels)
            num_bins = 16
            fov_rad = np.radians(90.0)
            fx = w / (2.0 * np.tan(fov_rad / 2.0))
            cx = w / 2.0
            u = np.arange(w)
            yaw_per_col = np.arctan((u - cx) / fx)
            bin_edges = np.linspace(-fov_rad / 2, fov_rad / 2, num_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            clearance = np.full(num_bins, 10.0)
            for b in range(num_bins):
                mask = (yaw_per_col >= bin_edges[b]) & (yaw_per_col < bin_edges[b + 1])
                cols = np.where(mask)[0]
                if len(cols) > 0:
                    vals = depth_mid[:, cols].flatten()
                    valid = vals[vals > 0.05]
                    if len(valid) > 0:
                        clearance[b] = float(np.median(valid))

            fwd = float(np.min(clearance[6:10]))

            # Update camera
            im_cam.set_data(frame)

            # Update depth
            im_depth.set_data(depth)
            im_depth.set_clim(vmin=depth.min(), vmax=min(depth.max(), 10))
            ax_depth.set_title(f"DA2 Depth ({depth.min():.1f}-{depth.max():.1f}m)")

            # Update clearance
            ax_clear.clear()
            angles_deg = np.degrees(bin_centers)
            max_cl = float(clearance.max())
            stop_thresh = max_cl * DEPTH_STOP_REL
            slow_thresh = max_cl * DEPTH_SLOW_REL
            colors = ['red' if c < stop_thresh else 'orange' if c < slow_thresh else 'green' for c in clearance]
            ax_clear.bar(angles_deg, clearance, width=5, color=colors,
                         edgecolor='black', linewidth=0.5)
            ax_clear.axhline(y=stop_thresh, color='red', linestyle='--', linewidth=1.5, label=f'STOP ({DEPTH_STOP_REL:.0%})')
            ax_clear.axhline(y=slow_thresh, color='orange', linestyle='--', linewidth=1.5, label=f'SLOW ({DEPTH_SLOW_REL:.0%})')
            ax_clear.set_xlim(-50, 50)
            ax_clear.set_ylim(0, max(5, clearance.max() * 1.1))
            ax_clear.set_xlabel('Angle (°)')
            ax_clear.set_ylabel('Clearance (m)')

            # Auto obstacle avoidance
            if auto_mode:
                current_linear, current_angular, auto_status = compute_auto_command(
                    clearance, bin_centers, fwd, max_cl)
            else:
                auto_status = "MANUAL"

            dt = (time.time() - t0) * 1000
            fwd_rel = fwd / max_cl if max_cl > 0 else 0
            cmd_str = f"cmd=({current_linear:.2f}, {current_angular:.2f})"
            mode_str = f"AUTO:{auto_status}" if auto_mode else "MANUAL"
            ax_clear.set_title(f'Fwd={fwd_rel:.0%} [{mode_str}] {dt:.0f}ms {cmd_str}')
            ax_clear.legend(loc='upper right', fontsize=8)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            left_cl = float(np.mean(clearance[:num_bins // 2]))
            right_cl = float(np.mean(clearance[num_bins // 2:]))

            frame_count += 1
            if frame_count % 10 == 0:
                print(f"[{frame_count}] fwd={fwd:.1f}m ({fwd_rel:.0%}) max={max_cl:.1f}m L={left_cl:.1f} R={right_cl:.1f} {mode_str} {dt:.0f}ms {cmd_str}")
            # Save debug snapshot every 50 frames
            if frame_count % 50 == 1:
                fig.savefig('/tmp/da2_live_debug.png', dpi=80)

            # Send command
            send_control(current_linear, current_angular)

            elapsed = time.time() - t0
            if elapsed < 0.15:
                plt.pause(0.15 - elapsed)

    except KeyboardInterrupt:
        pass
    finally:
        running = False
        send_control(0, 0)
        print("\nStopped.")


if __name__ == "__main__":
    main()
