#!/usr/bin/env python3
"""Real-time 3D point cloud mapper with MPPI autonomous obstacle avoidance.

MPPI plans trajectories around obstacles using DA2 depth while
simultaneously building a 3D point cloud map of the environment.

Controls:
    m       - toggle AUTO(MPPI) / MANUAL mode
    w/s/a/d - manual drive
    x/space - stop
    r       - reset map
    q       - quit
"""

import sys
import os
import base64
import io
import time

import numpy as np
import cv2
import requests
from PIL import Image

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'third_party', 'Depth-Anything-V2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'third_party', 'Depth-Anything-V2', 'metric_depth'))

SDK_URL = "http://localhost:8000"
FLIP_LINEAR = False

# Camera intrinsics (approximate for FrodoBot wide-angle)
FOV_H_DEG = 90.0
IMG_W = 640
IMG_H = 480
FX = IMG_W / (2.0 * np.tan(np.radians(FOV_H_DEG / 2.0)))
FY = FX
CX = IMG_W / 2.0
CY = IMG_H / 2.0

# Point cloud params
MAX_DEPTH = 8.0           # ignore points beyond this
DOWNSAMPLE = 8            # take every Nth pixel (speed vs density)
MAX_MAP_POINTS = 200_000  # cap total map points
VOXEL_SIZE = 0.05         # voxel grid filter size (meters)

running = True
current_linear = 0.0
current_angular = 0.0
auto_mode = True


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


def depth_to_pointcloud(depth: np.ndarray, rgb: np.ndarray,
                         downsample: int = DOWNSAMPLE) -> tuple[np.ndarray, np.ndarray]:
    """Convert depth map + RGB to colored 3D point cloud.

    Returns:
        points: (N, 3) xyz in camera frame (x=right, y=down, z=forward)
        colors: (N, 3) RGB normalized 0-1
    """
    h, w = depth.shape
    rgb_resized = cv2.resize(rgb, (w, h))

    # Subsample
    rows = np.arange(0, h, downsample)
    cols = np.arange(0, w, downsample)
    rr, cc = np.meshgrid(rows, cols, indexing='ij')
    rr = rr.flatten()
    cc = cc.flatten()

    z = depth[rr, cc]
    valid = (z > 0.1) & (z < MAX_DEPTH)
    rr, cc, z = rr[valid], cc[valid], z[valid]

    # Back-project to 3D
    # Scale intrinsics to depth map resolution
    fx = FX * w / IMG_W
    fy = FY * h / IMG_H
    cx = CX * w / IMG_W
    cy = CY * h / IMG_H

    x = (cc - cx) * z / fx
    y = (rr - cy) * z / fy

    points = np.stack([x, y, z], axis=1)  # (N, 3)
    colors = rgb_resized[rr, cc].astype(np.float32) / 255.0  # (N, 3)

    return points, colors


def estimate_motion(prev_gray: np.ndarray, curr_gray: np.ndarray,
                     prev_depth: np.ndarray) -> np.ndarray:
    """Estimate camera motion between frames using optical flow + PnP.

    Returns 4x4 transformation matrix (camera_new_from_camera_old).
    """
    # Detect features
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return np.eye(4)

    # Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < 10:
        return np.eye(4)

    matches = sorted(matches, key=lambda m: m.distance)[:200]

    # Get 3D points from previous frame
    h, w = prev_depth.shape
    fx = FX * w / IMG_W
    fy = FY * h / IMG_H
    cx = CX * w / IMG_W
    cy = CY * h / IMG_H
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    pts_3d = []
    pts_2d = []
    for m in matches:
        p1 = kp1[m.queryIdx].pt
        p2 = kp2[m.trainIdx].pt
        col, row = int(p1[0] * w / IMG_W), int(p1[1] * h / IMG_H)
        col = min(max(col, 0), w - 1)
        row = min(max(row, 0), h - 1)
        z = prev_depth[row, col]
        if 0.1 < z < MAX_DEPTH:
            x = (p1[0] * w / IMG_W - cx) * z / fx
            y = (p1[1] * h / IMG_H - cy) * z / fy
            pts_3d.append([x, y, z])
            pts_2d.append([p2[0] * w / IMG_W, p2[1] * h / IMG_H])

    if len(pts_3d) < 6:
        return np.eye(4)

    pts_3d = np.array(pts_3d, dtype=np.float64)
    pts_2d = np.array(pts_2d, dtype=np.float64)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts_3d, pts_2d, camera_matrix, None,
        iterationsCount=100, reprojectionError=3.0,
        flags=cv2.SOLVEPNP_ITERATIVE)

    if not success or inliers is None or len(inliers) < 6:
        return np.eye(4)

    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform to (N, 3) points."""
    ones = np.ones((points.shape[0], 1))
    pts_h = np.hstack([points, ones])  # (N, 4)
    transformed = (T @ pts_h.T).T  # (N, 4)
    return transformed[:, :3]


def voxel_downsample(points: np.ndarray, colors: np.ndarray,
                      voxel_size: float = VOXEL_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Simple voxel grid downsampling."""
    if len(points) == 0:
        return points, colors

    # Quantize to voxel grid
    keys = np.floor(points / voxel_size).astype(np.int32)
    # Use unique voxels
    _, unique_idx = np.unique(keys, axis=0, return_index=True)

    if len(unique_idx) > MAX_MAP_POINTS:
        unique_idx = np.random.choice(unique_idx, MAX_MAP_POINTS, replace=False)

    return points[unique_idx], colors[unique_idx]


def on_key(event):
    global current_linear, current_angular, running, auto_mode
    key = event.key
    if key == 'q':
        running = False
        send_control(0, 0)
        plt.close('all')
    elif key == 'm':
        auto_mode = not auto_mode
        current_linear = 0.0
        current_angular = 0.0
        send_control(0, 0)
        print(f"Mode: {'AUTO (MPPI)' if auto_mode else 'MANUAL'}")
        return
    elif key == 'r':
        return 'reset'

    if not auto_mode:
        if key == 'w':
            current_linear = 0.30
            current_angular = 0.0
        elif key == 's':
            current_linear = -0.20
            current_angular = 0.0
        elif key == 'a':
            current_linear = 0.0
            current_angular = 0.35
        elif key == 'd':
            current_linear = 0.0
            current_angular = -0.35
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
    print("Loading DA2...")
    from depth_estimator import DepthEstimator
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    estimator = DepthEstimator(model_size='small', max_depth=10.0, device=device)
    print(f"DA2 ready on {device}")

    # Load MPPI
    from T_mppi_planner import MPPIPlanner, MPPIConfig
    mppi = MPPIPlanner(MPPIConfig(
        num_samples=512, horizon=12, dt=0.15,
        max_linear=0.30, max_angular=0.45,
    ))
    print("MPPI planner ready")

    # Setup figure: 2x2 layout
    fig = plt.figure(figsize=(18, 10))
    fig.canvas.manager.set_window_title('MPPI 3D Mapper — M=auto/manual, WASD=drive, R=reset, Q=quit')

    ax_cam = fig.add_subplot(221)
    ax_depth = fig.add_subplot(222)
    ax_traj = fig.add_subplot(223)
    ax_3d = fig.add_subplot(224, projection='3d')

    ax_cam.set_title("Camera")
    ax_cam.axis('off')
    im_cam = ax_cam.imshow(np.zeros((100, 100, 3), dtype=np.uint8))

    ax_depth.set_title("DA2 Depth")
    ax_depth.axis('off')
    im_depth = ax_depth.imshow(np.zeros((100, 100)), cmap='plasma')

    ax_traj.set_title("MPPI Trajectories")
    ax_traj.set_xlim(-0.5, 2.0)
    ax_traj.set_ylim(-1.2, 1.2)
    ax_traj.set_aspect('equal')
    ax_traj.grid(True, alpha=0.3)

    ax_3d.set_title("3D Map")
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Z (forward)')
    ax_3d.set_zlabel('Y')

    # Reset handler
    reset_flag = [False]
    def key_handler(event):
        result = on_key(event)
        if result == 'reset':
            reset_flag[0] = True

    fig.canvas.mpl_connect('key_press_event', key_handler)
    plt.tight_layout()
    plt.ion()
    plt.show()

    print("\nAUTO MODE (MPPI) — robot explores and maps autonomously")
    print("M=toggle mode, WASD=manual, R=reset map, Q=quit")
    print("Click on the figure window first!\n")

    # Map state
    map_points = np.zeros((0, 3))
    map_colors = np.zeros((0, 3))
    camera_pose = np.eye(4)  # world-from-camera
    prev_gray = None
    prev_depth = None
    frame_count = 0
    pose_trail = [[0, 0, 0]]

    try:
        while running:
            t0 = time.time()

            if reset_flag[0]:
                map_points = np.zeros((0, 3))
                map_colors = np.zeros((0, 3))
                camera_pose = np.eye(4)
                prev_gray = None
                prev_depth = None
                pose_trail = [[0, 0, 0]]
                reset_flag[0] = False
                print("Map reset!")

            frame = get_frame()
            if frame is None:
                plt.pause(0.1)
                continue

            # Resize for processing
            frame_small = cv2.resize(frame, (IMG_W, IMG_H))
            gray = cv2.cvtColor(frame_small, cv2.COLOR_RGB2GRAY)

            # DA2 depth
            depth = estimator.estimate(frame)

            # MPPI planning
            mppi_linear, mppi_angular, mppi_debug = mppi.plan(depth, goal_direction_rad=0.0)
            if auto_mode:
                current_linear = mppi_linear
                current_angular = mppi_angular

            # Estimate motion
            if prev_gray is not None and prev_depth is not None:
                T_new_from_old = estimate_motion(prev_gray, gray, prev_depth)
                # Invert: we want world-from-new = world-from-old * old-from-new
                try:
                    T_old_from_new = np.linalg.inv(T_new_from_old)
                    camera_pose = camera_pose @ T_old_from_new
                except np.linalg.LinAlgError:
                    pass

            prev_gray = gray
            prev_depth = depth

            # Convert current frame to point cloud
            pts_cam, cols_cam = depth_to_pointcloud(depth, frame)

            if len(pts_cam) > 0:
                # Transform to world frame
                pts_world = transform_points(pts_cam, camera_pose)

                # Add to map
                map_points = np.vstack([map_points, pts_world])
                map_colors = np.vstack([map_colors, cols_cam])

                # Downsample if too big
                if len(map_points) > MAX_MAP_POINTS * 1.5:
                    map_points, map_colors = voxel_downsample(map_points, map_colors)

            # Track camera position
            cam_pos = camera_pose[:3, 3]
            pose_trail.append(cam_pos.tolist())

            # --- Update plots ---
            im_cam.set_data(frame)
            im_depth.set_data(depth)
            im_depth.set_clim(depth.min(), min(depth.max(), MAX_DEPTH))

            # MPPI trajectory plot
            ax_traj.clear()
            ax_traj.set_xlim(-0.5, 2.0)
            ax_traj.set_ylim(-1.2, 1.2)
            ax_traj.set_aspect('equal')
            ax_traj.grid(True, alpha=0.3)

            if "top_x" in mppi_debug:
                for i in range(len(mppi_debug["top_x"])):
                    ax_traj.plot(mppi_debug["top_x"][i], mppi_debug["top_y"][i],
                                 'gray', alpha=0.3, linewidth=0.8)

            best_x = mppi_debug["best_x"]
            best_y = mppi_debug["best_y"]
            cost = mppi_debug["best_cost"]
            color = 'green' if cost < 5 else 'orange' if cost < 15 else 'red'
            ax_traj.plot(best_x, best_y, color=color, linewidth=3)
            ax_traj.plot(best_x[-1], best_y[-1], 'o', color=color, markersize=8)
            ax_traj.plot(0, 0, 'ks', markersize=10)
            ax_traj.arrow(0, 0, 0.15, 0, head_width=0.05, head_length=0.03, fc='k', ec='k')
            mode_str = "MPPI" if auto_mode else "MANUAL"
            ax_traj.set_title(f'MPPI [{mode_str}] cost={cost:.1f} cmd=({current_linear:.2f},{current_angular:.2f})')

            # Update 3D plot (every 5 frames to keep it responsive)
            if frame_count % 5 == 0 and len(map_points) > 0:
                ax_3d.clear()

                # Subsample for rendering
                n_render = min(len(map_points), 30000)
                idx = np.random.choice(len(map_points), n_render, replace=False)

                # Plot: X=right, Z=forward, Y=up (flip Y for display)
                ax_3d.scatter(
                    map_points[idx, 0],
                    map_points[idx, 2],   # Z forward
                    -map_points[idx, 1],  # -Y = up
                    c=map_colors[idx],
                    s=0.5,
                    alpha=0.6,
                )

                # Camera trail
                trail = np.array(pose_trail)
                ax_3d.plot(trail[:, 0], trail[:, 2], -trail[:, 1],
                           'r-', linewidth=2, label='Path')
                ax_3d.plot([cam_pos[0]], [cam_pos[2]], [-cam_pos[1]],
                           'ro', markersize=8)

                ax_3d.set_title(f"3D Map ({len(map_points):,} pts)")
                ax_3d.set_xlabel('X (m)')
                ax_3d.set_ylabel('Z forward (m)')
                ax_3d.set_zlabel('Y up (m)')

                # Auto-scale view
                if len(map_points) > 100:
                    center = map_points[idx].mean(axis=0)
                    span = max(map_points[idx].std(axis=0).max() * 3, 2.0)
                    ax_3d.set_xlim(center[0] - span, center[0] + span)
                    ax_3d.set_ylim(center[2] - span, center[2] + span)
                    ax_3d.set_zlim(-center[1] - span, -center[1] + span)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            dt = (time.time() - t0) * 1000
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"[{frame_count}] map={len(map_points):,} pts "
                      f"pos=({cam_pos[0]:.2f},{cam_pos[1]:.2f},{cam_pos[2]:.2f}) "
                      f"{dt:.0f}ms")

            send_control(current_linear, current_angular)

            elapsed = time.time() - t0
            if elapsed < 0.15:
                plt.pause(0.15 - elapsed)

    except KeyboardInterrupt:
        pass
    finally:
        running = False
        send_control(0, 0)
        print(f"\nFinal map: {len(map_points):,} points")
        if len(map_points) > 0:
            np.savez('/tmp/pointcloud_map.npz',
                     points=map_points, colors=map_colors,
                     trail=np.array(pose_trail))
            print("Saved to /tmp/pointcloud_map.npz")
        print("Done.")


if __name__ == "__main__":
    main()
