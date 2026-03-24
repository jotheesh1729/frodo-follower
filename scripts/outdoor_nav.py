#!/usr/bin/env python3
"""Outdoor GPS waypoint navigation runtime for ERC missions.

This script navigates the robot through mission checkpoints using:
- GPS position + compass heading from the SDK
- Simple proportional heading controller
- Optional monocular depth obstacle avoidance
- SDK checkpoint registration

Usage:
    # Dry-run (no motor commands):
    python T_outdoor_runtime.py

    # Live with motor commands:
    python T_outdoor_runtime.py --send-control

    # With depth obstacle avoidance:
    python T_outdoor_runtime.py --send-control --depth-safety

    # Start mission automatically:
    python T_outdoor_runtime.py --send-control --auto-mission
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import requests

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from earthrover_interface import EarthRoverInterface
from T_gps_navigator import GPSWaypointManager, Checkpoint, gps_valid
from T_outdoor_controller import OutdoorController, OutdoorControllerConfig, ControlCommand


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ERC outdoor GPS waypoint navigation.")
    p.add_argument("--sdk-url", default="http://localhost:8000", help="SDK base URL.")
    p.add_argument("--sdk-timeout", type=float, default=5.0)
    p.add_argument("--send-control", action="store_true",
                   help="Actually send motor commands (default: dry-run).")
    p.add_argument("--auto-mission", action="store_true",
                   help="Automatically call /start-mission on startup.")
    p.add_argument("--tick-hz", type=float, default=3.0,
                   help="Control loop frequency.")
    p.add_argument("--max-ticks", type=int, default=None,
                   help="Maximum loop iterations (for testing).")
    p.add_argument("--arrival-radius", type=float, default=8.0,
                   help="Distance (m) to consider checkpoint reached.")
    p.add_argument("--slowdown-radius", type=float, default=20.0,
                   help="Distance (m) to start slowing down.")
    p.add_argument("--max-linear", type=float, default=0.35,
                   help="Max forward speed (0-1).")
    p.add_argument("--max-angular", type=float, default=0.45,
                   help="Max turn speed (0-1).")
    p.add_argument("--heading-gain", type=float, default=0.008,
                   help="P-gain for heading error → angular command.")
    # Depth safety
    p.add_argument("--depth-safety", action="store_true",
                   help="Enable monocular depth obstacle avoidance.")
    p.add_argument("--depth-stop-m", type=float, default=0.5,
                   help="Stop when forward clearance below this (meters).")
    p.add_argument("--depth-slow-m", type=float, default=1.2,
                   help="Slow down when forward clearance below this (meters).")
    # Manual waypoints (if not using mission API)
    p.add_argument("--waypoints", type=str, default=None,
                   help='JSON array of [lat,lon] pairs, e.g. \'[[9.97,-84.37],[9.98,-84.38]]\'')
    # Timeout
    p.add_argument("--mission-timeout-s", type=float, default=2700.0,
                   help="Total mission timeout in seconds (default 45 min).")
    p.add_argument("--checkpoint-timeout-s", type=float, default=600.0,
                   help="Per-checkpoint timeout in seconds (default 10 min).")
    # Output
    p.add_argument("--print-json", action="store_true",
                   help="Print each tick as JSON.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# SDK helpers
# ---------------------------------------------------------------------------

def sdk_start_mission(base_url: str, timeout: float = 10.0) -> dict:
    """POST /start-mission → returns response with checkpoints."""
    resp = requests.post(f"{base_url}/start-mission", timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def sdk_get_checkpoints(base_url: str, timeout: float = 10.0) -> list[dict]:
    """GET /checkpoints-list → returns list of checkpoints."""
    resp = requests.get(f"{base_url}/checkpoints-list", timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data.get("checkpoints_list", [])


def sdk_checkpoint_reached(base_url: str, timeout: float = 10.0) -> dict:
    """POST /checkpoint-reached → tells the mission server we reached one."""
    resp = requests.post(f"{base_url}/checkpoint-reached",
                         json={}, timeout=timeout)
    data = resp.json()
    if resp.status_code != 200:
        detail = data.get("detail", {})
        if isinstance(detail, dict):
            dist = detail.get("proximate_distance_to_checkpoint", "?")
            err = detail.get("error", "unknown")
            return {"success": False, "error": err, "distance": dist}
        return {"success": False, "error": str(detail)}
    return {"success": True, **data}


def sdk_end_mission(base_url: str, timeout: float = 10.0) -> dict:
    """POST /end-mission → terminates the mission (preserves points earned)."""
    try:
        resp = requests.post(f"{base_url}/end-mission", timeout=timeout)
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Depth safety loader
# ---------------------------------------------------------------------------

import numpy as np

# Depth obstacle avoidance thresholds (tuned for FrodoBot wide-angle camera)
ABS_STOP_M = 0.5    # whole scene closer than this → full stop
ABS_SLOW_M = 1.0    # whole scene closer than this → slow down
REL_STOP = 0.15     # forward < 15% of max → stop
REL_SLOW = 0.25     # forward < 25% of max → slow


def load_depth_safety(stop_m: float, slow_m: float):
    """Try to load the depth safety layer; return None on failure."""
    try:
        from depth_estimator import DepthEstimator
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        estimator = DepthEstimator(model_size='small', max_depth=10.0, device=device)
        print(f"[depth] Depth safety loaded on {device}.")
        return estimator
    except Exception as e:
        print(f"[depth] Could not load depth safety: {e}")
        return None


def run_depth_check(estimator, frame, num_bins: int = 16, fov: float = 90.0):
    """Run depth estimation with middle-band clearance.

    Returns (action, fwd_clearance, safe_direction_rad) where action is:
        'CLEAR'  — no obstacle, drive normally
        'STEER'  — obstacle ahead, steer away
        'SLOW'   — close obstacle, slow + steer
        'STOP'   — very close, stop + turn
        'BACKUP' — blocked everywhere, reverse
    """
    try:
        depth_map = estimator.estimate(frame)
        h, w = depth_map.shape

        # Use MIDDLE band (20%-65%) — ignores ceiling and ground
        row_start = int(h * 0.20)
        row_end = int(h * 0.65)
        depth_mid = depth_map[row_start:row_end, :]

        # Compute clearance per angular bin using median
        fov_rad = np.radians(fov)
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

        fwd_clearance = float(np.min(clearance[num_bins // 2 - 2:num_bins // 2 + 2]))
        max_cl = float(clearance.max())

        # Left vs right — decide turn direction
        left_cl = float(np.mean(clearance[:num_bins // 2]))
        right_cl = float(np.mean(clearance[num_bins // 2:]))
        safe_dir = 0.5 if left_cl > right_cl else -0.5  # positive = left

        # Absolute check first
        if max_cl < ABS_STOP_M:
            return 'STOP', fwd_clearance, safe_dir
        if max_cl < ABS_SLOW_M:
            return 'SLOW', fwd_clearance, safe_dir

        # Relative check
        fwd_rel = fwd_clearance / max_cl if max_cl > 0 else 0
        if fwd_rel < REL_STOP:
            return 'STOP', fwd_clearance, safe_dir
        elif fwd_rel < REL_SLOW:
            return 'SLOW', fwd_clearance, safe_dir
        else:
            return 'CLEAR', fwd_clearance, safe_dir

    except Exception:
        return 'CLEAR', None, None


# ---------------------------------------------------------------------------
# Main runtime
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    dry_run = not args.send_control

    rover = EarthRoverInterface(base_url=args.sdk_url, timeout=args.sdk_timeout)
    if not rover.connect():
        print("Failed to connect to SDK. Is the server running?")
        return 1

    # --- Load checkpoints ---
    waypoint_mgr = GPSWaypointManager(
        arrival_radius_m=args.arrival_radius,
        slowdown_radius_m=args.slowdown_radius,
    )

    if args.waypoints:
        # Manual waypoints from CLI
        raw = json.loads(args.waypoints)
        waypoint_mgr.checkpoints = [
            Checkpoint(sequence=i + 1, latitude=pt[0], longitude=pt[1])
            for i, pt in enumerate(raw)
        ]
        print(f"Loaded {len(raw)} manual waypoints.")
    else:
        # Mission API
        if args.auto_mission:
            print("Starting mission...")
            try:
                result = sdk_start_mission(args.sdk_url)
                print(f"  Mission started: {result.get('message', 'OK')}")
            except Exception as e:
                print(f"  Failed to start mission: {e}")
                return 1

        print("Fetching checkpoints...")
        try:
            cp_list = sdk_get_checkpoints(args.sdk_url)
        except Exception as e:
            print(f"  Failed to get checkpoints: {e}")
            print("  Use --waypoints to provide manual waypoints, or --auto-mission.")
            return 1

        if not cp_list:
            print("  No checkpoints returned. Is the mission configured?")
            return 1

        waypoint_mgr.load_from_sdk(cp_list)
        print(f"  Loaded {waypoint_mgr.total_checkpoints} checkpoints:")
        for cp in waypoint_mgr.checkpoints:
            print(f"    #{cp.sequence}: ({cp.latitude:.6f}, {cp.longitude:.6f})")

    # --- Controller ---
    controller = OutdoorController(OutdoorControllerConfig(
        max_linear=args.max_linear,
        max_angular=args.max_angular,
        heading_p_gain=args.heading_gain,
        arrival_radius_m=args.arrival_radius,
        slowdown_radius_m=args.slowdown_radius,
        depth_stop_m=args.depth_stop_m,
        depth_slow_m=args.depth_slow_m,
    ))

    # --- Optional depth safety ---
    depth_estimator = None
    if args.depth_safety:
        depth_estimator = load_depth_safety(args.depth_stop_m, args.depth_slow_m)

    # --- Print header ---
    print()
    print("=" * 60)
    print("OUTDOOR GPS NAVIGATION")
    print("=" * 60)
    print(f"Mode:            {'DRY RUN' if dry_run else 'SEND CONTROL'}")
    print(f"Checkpoints:     {waypoint_mgr.total_checkpoints}")
    print(f"Arrival radius:  {args.arrival_radius:.1f} m")
    print(f"Slowdown radius: {args.slowdown_radius:.1f} m")
    print(f"Max speed:       {args.max_linear:.2f} linear, {args.max_angular:.2f} angular")
    print(f"Heading gain:    {args.heading_gain}")
    print(f"Depth safety:    {'ON' if depth_estimator else 'OFF'}")
    print(f"Tick rate:       {args.tick_hz:.1f} Hz")
    print("=" * 60)
    print()

    period = 1.0 / args.tick_hz
    tick = 0
    mission_start = time.time()
    checkpoint_start = time.time()

    # Consecutive bad-GPS ticks
    bad_gps_count = 0
    BAD_GPS_MAX = 10

    # Arrival confirmation: must be within radius for N consecutive ticks
    arrival_confirm_count = 0
    ARRIVAL_CONFIRM_TICKS = 3

    # Checkpoint-reached API retry cooldown
    last_checkpoint_attempt = 0.0
    CHECKPOINT_RETRY_COOLDOWN = 3.0

    try:
        while True:
            if args.max_ticks is not None and tick >= args.max_ticks:
                print(f"\nReached max ticks ({args.max_ticks}).")
                break

            # --- Timeouts ---
            elapsed_total = time.time() - mission_start
            elapsed_checkpoint = time.time() - checkpoint_start

            if elapsed_total > args.mission_timeout_s:
                print(f"\nMission timeout ({args.mission_timeout_s:.0f}s). Terminating.")
                break

            if elapsed_checkpoint > args.checkpoint_timeout_s:
                print(f"\nCheckpoint timeout ({args.checkpoint_timeout_s:.0f}s). "
                      f"Skipping to save time.")
                waypoint_mgr.advance()
                checkpoint_start = time.time()
                arrival_confirm_count = 0
                if waypoint_mgr.is_mission_complete:
                    print("All checkpoints attempted. Mission complete.")
                    break
                cp = waypoint_mgr.active_checkpoint
                print(f"  Now targeting checkpoint #{cp.sequence}")
                continue

            loop_start = time.time()

            # --- Get sensor data ---
            data = rover.get_data()
            if data is None:
                if not dry_run:
                    rover.stop()
                print(f"[{tick:04d}] No sensor data. Stopping.")
                time.sleep(period)
                tick += 1
                continue

            lat = data.get("latitude")
            lon = data.get("longitude")
            orientation = data.get("orientation")

            if not gps_valid(lat, lon):
                bad_gps_count += 1
                if not dry_run:
                    rover.stop()
                if bad_gps_count <= 3 or bad_gps_count % 10 == 0:
                    print(f"[{tick:04d}] Bad GPS (lat={lat}, lon={lon}). "
                          f"Waiting... ({bad_gps_count}/{BAD_GPS_MAX})")
                if bad_gps_count >= BAD_GPS_MAX:
                    print(f"[{tick:04d}] GPS unavailable for {BAD_GPS_MAX} ticks. "
                          "Continuing to wait...")
                time.sleep(period)
                tick += 1
                continue

            bad_gps_count = 0
            lat, lon = float(lat), float(lon)
            heading_deg = float(orientation) if orientation is not None else 0.0

            # --- Mission complete check ---
            if waypoint_mgr.is_mission_complete:
                if not dry_run:
                    rover.stop()
                print(f"\n[{tick:04d}] ALL CHECKPOINTS REACHED. Mission complete!")
                break

            # --- Compute nav state ---
            nav_state = waypoint_mgr.compute_nav_state(lat, lon, heading_deg)

            # --- Depth obstacle avoidance ---
            depth_action = 'CLEAR'
            fwd_clearance = None
            safe_dir = None
            if depth_estimator is not None:
                frame = rover.get_camera_frame()
                if frame is not None:
                    depth_action, fwd_clearance, safe_dir = run_depth_check(depth_estimator, frame)

            # --- RPM for stall detection ---
            rpm_mean = None
            rpms = data.get("rpms") or []
            if rpms:
                latest_rpm = rpms[-1]
                if isinstance(latest_rpm, (list, tuple)) and len(latest_rpm) >= 4:
                    try:
                        rpm_vals = [abs(float(v)) for v in latest_rpm[:4]]
                        rpm_mean = sum(rpm_vals) / len(rpm_vals)
                    except (TypeError, ValueError):
                        pass

            # --- Compute command (GPS nav + depth override) ---
            command = controller.compute_command(
                nav_state=nav_state,
                depth_clearance_m=fwd_clearance,
                safe_direction_rad=safe_dir,
                rpm_mean=rpm_mean,
            )

            # Depth override: if depth says STOP/SLOW, override GPS command
            if depth_action == 'STOP':
                turn_dir = 1.0 if (safe_dir or 0.5) > 0 else -1.0
                command = ControlCommand(0.0, turn_dir * 0.40, "depth_STOP",
                                         debug={"fwd_cl": fwd_clearance})
            elif depth_action == 'SLOW':
                turn_dir = 1.0 if (safe_dir or 0.5) > 0 else -1.0
                command.linear = min(command.linear, 0.12)
                command.angular = turn_dir * 0.30
                command.reason = "depth_SLOW"
            elif depth_action == 'BACKUP':
                command = ControlCommand(-0.12, 0.0, "depth_BACKUP")


            # --- Arrival / checkpoint-reached logic ---
            if nav_state.get("within_arrival"):
                arrival_confirm_count += 1
            else:
                arrival_confirm_count = 0

            if arrival_confirm_count >= ARRIVAL_CONFIRM_TICKS:
                now = time.time()
                if now - last_checkpoint_attempt >= CHECKPOINT_RETRY_COOLDOWN:
                    last_checkpoint_attempt = now
                    cp = waypoint_mgr.active_checkpoint
                    print(f"\n[{tick:04d}] Within arrival radius of checkpoint "
                          f"#{cp.sequence} ({nav_state['distance_m']:.1f}m). "
                          f"Calling /checkpoint-reached...")

                    if not dry_run:
                        result = sdk_checkpoint_reached(args.sdk_url)
                        if result.get("success"):
                            print(f"  CHECKPOINT #{cp.sequence} REGISTERED! "
                                  f"Next: {result.get('next_checkpoint_sequence', '?')}")
                            waypoint_mgr.advance()
                            checkpoint_start = time.time()
                            arrival_confirm_count = 0
                            controller.reset()
                            if waypoint_mgr.is_mission_complete:
                                rover.stop()
                                print("\n  ALL CHECKPOINTS COMPLETE!")
                                break
                            next_cp = waypoint_mgr.active_checkpoint
                            print(f"  Now targeting checkpoint #{next_cp.sequence} "
                                  f"({next_cp.latitude:.6f}, {next_cp.longitude:.6f})")
                        else:
                            dist_info = result.get("distance", "?")
                            print(f"  Not close enough: {result.get('error', '?')} "
                                  f"(dist={dist_info}m). Continuing...")
                    else:
                        print(f"  [DRY RUN] Would call /checkpoint-reached")
                        # In dry-run, advance anyway for testing
                        waypoint_mgr.advance()
                        checkpoint_start = time.time()
                        arrival_confirm_count = 0
                        if waypoint_mgr.is_mission_complete:
                            print("\n  [DRY RUN] All checkpoints reached!")
                            break

            # --- Send command ---
            if dry_run:
                sent = False
            else:
                sent = rover.send_control(command.linear, command.angular)

            # --- Logging ---
            payload = {
                "tick": tick,
                "lat": lat,
                "lon": lon,
                "heading": heading_deg,
                "cp_seq": nav_state.get("checkpoint_sequence"),
                "cp_idx": f"{nav_state.get('checkpoint_index', '?')}/{nav_state.get('total_checkpoints', '?')}",
                "dist_m": nav_state.get("distance_m"),
                "bearing": nav_state.get("bearing_deg"),
                "h_err": nav_state.get("heading_error_deg"),
                "linear": command.linear,
                "angular": command.angular,
                "reason": command.reason,
                "fwd_cl": fwd_clearance,
                "sent": sent,
            }

            if args.print_json:
                print(json.dumps(payload, default=str))
            else:
                dist = nav_state.get('distance_m', 0)
                h_err = nav_state.get('heading_error_deg', 0)
                cp_seq = nav_state.get('checkpoint_sequence', '?')
                cp_prog = f"{nav_state.get('checkpoint_index', 0)}/{nav_state.get('total_checkpoints', 0)}"
                depth_str = f" depth={fwd_clearance:.1f}m[{depth_action}]" if fwd_clearance is not None else ""
                print(
                    f"[{tick:04d}] cp=#{cp_seq}({cp_prog}) "
                    f"dist={dist:.1f}m h_err={h_err:+.1f}° "
                    f"cmd=({command.linear:.3f},{command.angular:.3f}) "
                    f"{command.reason} sent={sent}"
                    f"{depth_str}"
                )

            elapsed = time.time() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)
            tick += 1

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    finally:
        if not dry_run:
            rover.stop()
            print("Robot stopped.")

    # --- Summary ---
    elapsed_total = time.time() - mission_start
    reached = waypoint_mgr.checkpoints_reached
    total = waypoint_mgr.total_checkpoints
    print()
    print("=" * 60)
    print("MISSION SUMMARY")
    print("=" * 60)
    print(f"Checkpoints: {reached}/{total}")
    print(f"Time: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    if total > 0:
        print(f"Completion: {reached/total*100:.0f}%")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
