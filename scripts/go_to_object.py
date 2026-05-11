#!/usr/bin/env python3
"""Voice/text-guided object navigation with YOLO + DA2 + MPPI.

Tell the robot "go to the chair" and it will:
1. YOLO detects all objects in the frame
2. Matches your target to a detection
3. DA2 gives depth → knows how far + what angle the object is
4. MPPI plans a trajectory that avoids obstacles while heading to the object
5. Robot drives there autonomously

Usage:
    python3 T_go_to_object.py

    # Then type target objects:
    > person
    > chair
    > backpack
    > bottle

Controls (click on figure window):
    m       - toggle AUTO/MANUAL
    w/s/a/d - manual drive
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
import cv2
import requests
from PIL import Image

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'third_party', 'Depth-Anything-V2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'third_party', 'Depth-Anything-V2', 'metric_depth'))

SDK_URL = "http://localhost:8000"
FLIP_LINEAR = False

# Camera params
FOV_H_DEG = 90.0

# State
running = True
auto_mode = True
current_linear = 0.0
current_angular = 0.0
target_object = ""  # what the user wants to go to
target_lock = threading.Lock()


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


def pixel_to_angle(px, img_width, fov_deg=FOV_H_DEG):
    """Convert pixel x-coordinate to angle from center. Positive = left."""
    center = img_width / 2.0
    fov_rad = np.radians(fov_deg)
    angle = np.arctan((px - center) / (img_width / (2 * np.tan(fov_rad / 2))))
    return float(angle)


def find_target_detection(results, target_name, img_shape):
    """Find the YOLO detection that best matches the target name.

    Returns (bbox, class_name, confidence, center_x, center_y) or None.
    """
    if not target_name:
        return None

    target_lower = target_name.lower().strip()
    best = None
    best_conf = 0.0

    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            cls_name = r.names[cls_id].lower()
            conf = float(boxes.conf[i])

            # Fuzzy match: target is substring of class name or vice versa
            if target_lower in cls_name or cls_name in target_lower:
                if conf > best_conf:
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    best = {
                        "bbox": (x1, y1, x2, y2),
                        "class": cls_name,
                        "confidence": conf,
                        "center_x": cx,
                        "center_y": cy,
                    }
                    best_conf = conf

    return best


def input_thread():
    """Background thread to read target object from stdin."""
    global target_object, running
    while running:
        try:
            line = input("\n🎯 Target object (or 'list' to see detections): ").strip()
            if line.lower() == 'q':
                running = False
                break
            with target_lock:
                target_object = line
            if line:
                print(f"   → Targeting: '{line}'")
        except (EOFError, KeyboardInterrupt):
            running = False
            break


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
        print(f"Mode: {'AUTO' if auto_mode else 'MANUAL'}")
    elif not auto_mode:
        if key == 'w':
            current_linear, current_angular = 0.30, 0.0
        elif key == 's':
            current_linear, current_angular = -0.20, 0.0
        elif key == 'a':
            current_linear, current_angular = 0.0, 0.35
        elif key == 'd':
            current_linear, current_angular = 0.0, -0.35
        elif key in ('x', ' '):
            current_linear, current_angular = 0.0, 0.0
        send_control(current_linear, current_angular)
    elif key in ('x', ' '):
        current_linear, current_angular = 0.0, 0.0
        send_control(0, 0)


def main():
    global running, current_linear, current_angular, target_object

    # Load models
    print("Loading YOLO...")
    from ultralytics import YOLO
    yolo = YOLO("yolo11n.pt")  # nano model, fast
    print("YOLO ready")

    print("Loading DA2...")
    from depth_estimator import DepthEstimator
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    estimator = DepthEstimator(model_size='small', max_depth=10.0, device=device)
    print(f"DA2 ready on {device}")

    print("Loading MPPI...")
    from T_mppi_planner import MPPIPlanner, MPPIConfig
    mppi = MPPIPlanner(MPPIConfig(
        num_samples=512, horizon=12, dt=0.15,
        max_linear=0.30, max_angular=0.45,
    ))
    print("MPPI ready")

    # Start input thread
    t = threading.Thread(target=input_thread, daemon=True)
    t.start()

    # Setup figure: 2x2
    fig = plt.figure(figsize=(18, 10))
    fig.canvas.manager.set_window_title('Go-To-Object: YOLO + DA2 + MPPI')
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

    ax_yolo = fig.add_subplot(gs[0, 0])       # YOLO detections
    ax_depth = fig.add_subplot(gs[0, 1])       # DA2 depth + target overlay
    ax_traj = fig.add_subplot(gs[1, 0])        # MPPI trajectories
    ax_info = fig.add_subplot(gs[1, 1])        # Status info

    ax_yolo.set_title("YOLO Detections")
    ax_yolo.axis('off')
    im_yolo = ax_yolo.imshow(np.zeros((100, 100, 3), dtype=np.uint8))

    ax_depth.set_title("DA2 Depth")
    ax_depth.axis('off')
    im_depth = ax_depth.imshow(np.zeros((100, 100)), cmap='plasma')

    ax_traj.set_xlim(-0.5, 2.0)
    ax_traj.set_ylim(-1.2, 1.2)
    ax_traj.set_aspect('equal')
    ax_traj.grid(True, alpha=0.3)

    ax_info.axis('off')

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout()
    plt.ion()
    plt.show()

    print("\n" + "=" * 50)
    print("GO-TO-OBJECT NAVIGATOR")
    print("=" * 50)
    print("Type an object name to navigate to it.")
    print("Examples: person, chair, bottle, backpack, tv")
    print("Press M on figure for manual mode, Q to quit")
    print("=" * 50)

    frame_count = 0
    detected_classes = set()

    try:
        while running:
            t0 = time.time()

            frame = get_frame()
            if frame is None:
                plt.pause(0.1)
                continue

            h, w = frame.shape[:2]

            # --- YOLO detection ---
            yolo_results = yolo(frame, verbose=False, conf=0.3)

            # Collect all detected class names
            detected_classes.clear()
            for r in yolo_results:
                if r.boxes is not None:
                    for i in range(len(r.boxes)):
                        cls_id = int(r.boxes.cls[i])
                        detected_classes.add(r.names[cls_id])

            # --- DA2 depth ---
            depth = estimator.estimate(frame)

            # --- Find target object ---
            with target_lock:
                current_target = target_object

            target_det = find_target_detection(yolo_results, current_target, frame.shape)

            # --- Compute goal direction for MPPI ---
            goal_direction = 0.0  # default: forward
            target_distance = None
            target_angle_deg = None

            if target_det is not None:
                # Get angle to target
                target_angle = pixel_to_angle(target_det["center_x"], w)
                goal_direction = target_angle  # MPPI will steer toward this
                target_angle_deg = np.degrees(target_angle)

                # Get depth at target center
                dh, dw = depth.shape
                dy = int(target_det["center_y"] * dh / h)
                dx = int(target_det["center_x"] * dw / w)
                dy = min(max(dy, 0), dh - 1)
                dx = min(max(dx, 0), dw - 1)
                # Average depth in a small region around target center
                r = 5
                region = depth[max(0, dy-r):min(dh, dy+r), max(0, dx-r):min(dw, dx+r)]
                if region.size > 0:
                    target_distance = float(np.median(region))

            # --- MPPI planning ---
            mppi_lin, mppi_ang, mppi_debug = mppi.plan(depth, goal_direction_rad=goal_direction)

            # If target is very close, slow down / stop
            if target_distance is not None and target_distance < 1.2:
                mppi_lin = 0.0
                mppi_ang = 0.0
                status = "ARRIVED"
            elif target_det is not None:
                status = f"NAVIGATING → {target_det['class']}"
            elif current_target:
                status = f"SEARCHING for '{current_target}'..."
                # No target found — spin slowly to search
                if auto_mode:
                    mppi_lin = 0.0
                    mppi_ang = 0.25
            else:
                status = "No target — type object name below"

            if auto_mode:
                current_linear = mppi_lin
                current_angular = mppi_ang

            # --- Draw YOLO detections ---
            frame_draw = frame.copy()
            for r in yolo_results:
                if r.boxes is None:
                    continue
                for i in range(len(r.boxes)):
                    x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                    cls_id = int(r.boxes.cls[i])
                    cls_name = r.names[cls_id]
                    conf = float(r.boxes.conf[i])

                    # Green for target, blue for others
                    is_target = (target_det is not None and
                                 abs(x1 - target_det["bbox"][0]) < 1)
                    color = (0, 255, 0) if is_target else (100, 100, 255)
                    thickness = 3 if is_target else 1

                    cv2.rectangle(frame_draw, (int(x1), int(y1)), (int(x2), int(y2)),
                                  color, thickness)
                    label = f"{cls_name} {conf:.0%}"
                    if is_target and target_distance:
                        label += f" {target_distance:.1f}m"
                    cv2.putText(frame_draw, label, (int(x1), int(y1) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # --- Update plots ---
            im_yolo.set_data(frame_draw)
            ax_yolo.set_title(f"YOLO ({len(detected_classes)} objects)")

            im_depth.set_data(depth)
            im_depth.set_clim(depth.min(), min(depth.max(), 10))
            if target_det and target_distance:
                ax_depth.set_title(f"DA2 Depth — Target: {target_distance:.1f}m @ {target_angle_deg:+.0f}°")
            else:
                ax_depth.set_title("DA2 Depth")

            # MPPI trajectories
            ax_traj.clear()
            ax_traj.set_xlim(-0.5, 2.5)
            ax_traj.set_ylim(-1.5, 1.5)
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

            # Draw goal direction arrow
            if target_det is not None:
                gx = 1.5 * np.cos(goal_direction)
                gy = 1.5 * np.sin(goal_direction)
                ax_traj.annotate("", xy=(gx, gy), xytext=(0, 0),
                                  arrowprops=dict(arrowstyle="->", color="magenta", lw=2))
                ax_traj.plot(gx, gy, '*', color='magenta', markersize=15)

            mode_str = "AUTO" if auto_mode else "MANUAL"
            ax_traj.set_title(f'MPPI [{mode_str}] cmd=({current_linear:.2f},{current_angular:.2f})')
            ax_traj.set_xlabel('Forward (m)')
            ax_traj.set_ylabel('Left (m)')

            # Info panel
            ax_info.clear()
            ax_info.axis('off')
            info_lines = [
                f"STATUS: {status}",
                f"",
                f"Target: '{current_target}'" if current_target else "Target: (none)",
                f"Distance: {target_distance:.1f}m" if target_distance else "Distance: --",
                f"Angle: {target_angle_deg:+.1f}°" if target_angle_deg else "Angle: --",
                f"",
                f"Detected objects:",
            ] + [f"  • {c}" for c in sorted(detected_classes)]

            ax_info.text(0.05, 0.95, '\n'.join(info_lines),
                         transform=ax_info.transAxes, fontsize=12,
                         verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            dt = (time.time() - t0) * 1000
            frame_count += 1
            if frame_count % 15 == 0:
                det_str = ', '.join(sorted(detected_classes)[:5])
                print(f"[{frame_count}] {status} | {det_str} | {dt:.0f}ms")

                # Print detected objects if user asked
                if current_target == 'list':
                    print(f"   Visible: {', '.join(sorted(detected_classes))}")
                    with target_lock:
                        target_object = ""

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
