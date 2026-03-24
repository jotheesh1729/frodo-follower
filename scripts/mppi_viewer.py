#!/usr/bin/env python3
"""Live MPPI trajectory planner with DA2 depth obstacle avoidance.

Shows camera, depth map, obstacle cost map, and MPPI planned trajectories
in real-time while the robot drives autonomously.

Controls:
    m       - toggle MANUAL/AUTO mode
    w/s/a/d - manual drive
    x/space - stop
    q       - quit
"""

import sys
import os
import base64
import io
import time

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
auto_mode = True

# Set True if robot drives backward for positive linear
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
        print(f"Mode: {'AUTO (MPPI)' if auto_mode else 'MANUAL'}")
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


def main():
    global running, current_linear, current_angular

    # Load DA2
    print("Loading DA2 on GPU...")
    from depth_estimator import DepthEstimator
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    estimator = DepthEstimator(model_size='small', max_depth=10.0, device=device)
    print(f"DA2 ready on {device}")

    # Load MPPI
    from T_mppi_planner import MPPIPlanner, MPPIConfig
    mppi = MPPIPlanner(MPPIConfig(
        num_samples=512,
        horizon=12,
        dt=0.15,
        max_linear=0.35,
        max_angular=0.50,
    ))
    print("MPPI planner ready")

    # Setup plot: 2x2 grid
    fig = plt.figure(figsize=(16, 9))
    fig.canvas.manager.set_window_title('MPPI Depth Planner — M=toggle mode, Q=quit')
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

    ax_cam = fig.add_subplot(gs[0, 0])
    ax_depth = fig.add_subplot(gs[0, 1])
    ax_traj = fig.add_subplot(gs[1, 0])
    ax_cost = fig.add_subplot(gs[1, 1])

    # Initial dummy
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    im_cam = ax_cam.imshow(dummy)
    ax_cam.set_title("Camera")
    ax_cam.axis('off')

    im_depth = ax_depth.imshow(np.zeros((100, 100)), cmap='plasma', vmin=0, vmax=10)
    ax_depth.set_title("DA2 Depth")
    ax_depth.axis('off')

    ax_traj.set_xlim(-1.0, 2.5)
    ax_traj.set_ylim(-1.5, 1.5)
    ax_traj.set_aspect('equal')
    ax_traj.set_xlabel('Forward (m)')
    ax_traj.set_ylabel('Left (m)')
    ax_traj.set_title('MPPI Trajectories')
    ax_traj.grid(True, alpha=0.3)

    im_cost = ax_cost.imshow(np.zeros((100, 100)), cmap='Reds', vmin=0, vmax=1)
    ax_cost.set_title("Obstacle Cost Map")
    ax_cost.axis('off')

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout()
    plt.ion()
    plt.show()

    print("\nAUTO MODE (MPPI) — robot plans trajectory around obstacles")
    print("Press M for manual, Q to quit")
    print("Click on the figure window first!\n")

    frame_count = 0
    try:
        while running:
            t0 = time.time()

            frame = get_frame()
            if frame is None:
                plt.pause(0.1)
                continue

            # DA2 depth
            depth = estimator.estimate(frame)

            # MPPI plan
            t_mppi = time.time()
            linear, angular, debug = mppi.plan(depth, goal_direction_rad=0.0)
            mppi_ms = (time.time() - t_mppi) * 1000

            if auto_mode:
                current_linear = linear
                current_angular = angular

            # --- Update plots ---
            # Camera
            im_cam.set_data(frame)

            # Depth
            im_depth.set_data(depth)
            im_depth.set_clim(vmin=depth.min(), vmax=min(depth.max(), 10))
            ax_depth.set_title(f"DA2 Depth ({depth.min():.1f}-{depth.max():.1f}m)")

            # Trajectories
            ax_traj.clear()
            ax_traj.set_xlim(-0.5, 2.0)
            ax_traj.set_ylim(-1.2, 1.2)
            ax_traj.set_aspect('equal')
            ax_traj.grid(True, alpha=0.3)

            # Draw top candidate trajectories (light gray)
            if "top_x" in debug:
                for i in range(len(debug["top_x"])):
                    alpha = 0.15 + 0.05 * (len(debug["top_x"]) - i)
                    ax_traj.plot(debug["top_x"][i], debug["top_y"][i],
                                 'gray', alpha=min(alpha, 0.5), linewidth=0.8)

            # Draw best trajectory (green/red based on cost)
            best_x = debug["best_x"]
            best_y = debug["best_y"]
            best_cost = debug["best_cost"]
            color = 'green' if best_cost < 5.0 else 'orange' if best_cost < 15.0 else 'red'
            ax_traj.plot(best_x, best_y, color=color, linewidth=3, label=f'Best (cost={best_cost:.1f})')
            ax_traj.plot(best_x[-1], best_y[-1], 'o', color=color, markersize=8)

            # Robot position
            ax_traj.plot(0, 0, 'ks', markersize=10, label='Robot')
            # Robot heading arrow
            ax_traj.arrow(0, 0, 0.15, 0, head_width=0.05, head_length=0.03, fc='black', ec='black')

            mode_str = "MPPI" if auto_mode else "MANUAL"
            ax_traj.set_title(f'Trajectories [{mode_str}] cmd=({current_linear:.2f}, {current_angular:.2f})')
            ax_traj.set_xlabel('Forward (m)')
            ax_traj.set_ylabel('Left (m)')
            ax_traj.legend(loc='upper right', fontsize=8)

            # Cost map
            cfg = mppi.config
            h, w = depth.shape
            row_start = int(h * cfg.crop_top)
            row_end = int(h * cfg.crop_bottom)
            depth_crop = depth[row_start:row_end, :]
            max_d = np.max(depth_crop)
            if max_d > 0:
                cost_vis = np.clip(1.0 - depth_crop / max_d, 0, 1)
            else:
                cost_vis = np.zeros_like(depth_crop)
            im_cost.set_data(cost_vis)
            im_cost.set_clim(0, 1)
            ax_cost.set_title(f"Cost Map (MPPI: {mppi_ms:.0f}ms, {cfg.num_samples} samples)")

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            dt = (time.time() - t0) * 1000
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"[{frame_count}] lin={current_linear:.2f} ang={current_angular:.2f} "
                      f"cost={debug['best_cost']:.1f} mppi={mppi_ms:.0f}ms total={dt:.0f}ms "
                      f"[{mode_str}]")

            # Send command
            send_control(current_linear, current_angular)

            elapsed = time.time() - t0
            if elapsed < 0.12:
                plt.pause(0.12 - elapsed)

    except KeyboardInterrupt:
        pass
    finally:
        running = False
        send_control(0, 0)
        print("\nStopped.")


if __name__ == "__main__":
    main()
