# Frodo-AI

Autonomous navigation system for the [FrodoBots Earth Rover](https://frodobots.com/) platform. Combines vision-language object targeting, monocular depth estimation, and sampling-based trajectory planning for real-time obstacle avoidance.

## Architecture

```
                    "go to the chair"
                          |
                    [NLP Parser]
                          |
                   [YOLO Detection]  -----> Object Localization
                          |                    (class, bbox, angle)
                   [DA2 Depth]  ----------> Distance Estimation
                          |                    (metric depth map)
                   [MPPI Planner]  -------> Trajectory Planning
                          |                    (512 samples, 12-step horizon)
                   [Visual Servo]  -------> Motor Control
                          |                    (center target in frame)
                   [FrodoBot SDK]  -------> Robot Actuation
```

## Features

- **Natural Language Navigation** - Tell the robot where to go in plain English
- **YOLO Object Detection** - Real-time detection of 80+ object classes (GPU-accelerated)
- **Depth Anything V2** - Monocular depth estimation for obstacle awareness
- **MPPI Trajectory Planning** - Model Predictive Path Integral control with 512 trajectory samples
- **Visual Servoing** - Centers target object in camera frame while approaching
- **3D Point Cloud Mapping** - Builds a live 3D map using depth + visual odometry
- **Web Interface** - Browser-based control panel with live video, depth, and trajectory visualization
- **ROS2 Integration** - Publishes PointCloud2, Image, Odometry, Path topics
- **GPS Waypoint Navigation** - Outdoor checkpoint-based mission execution with obstacle avoidance

## Quick Start

### 1. Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp config/.env.example config/.env
# Edit config/.env with your SDK_API_TOKEN and BOT_SLUG
```

### 3. Start the SDK Server

```bash
cd earth-rovers-sdk && hypercorn main:app --reload
```

### 4. Run the Web Navigator

```bash
python scripts/web_navigator.py
# Open http://localhost:5000
```

Type commands like:
- `go to the person`
- `find a chair`
- `navigate to the bottle`

## Project Structure

```
frodo-ai/
  frodo_ai/
    perception/
      depth_estimator.py      # Depth Anything V2 wrapper
      object_detector.py      # YOLO detection + target matching
    planning/
      mppi_planner.py         # MPPI trajectory optimization
      gps_navigator.py        # GPS waypoint navigation
    control/
      visual_servo.py         # Visual servoing controller
      outdoor_controller.py   # GPS + depth outdoor controller
    interface/
      rover_interface.py      # FrodoBot SDK communication
      web_server.py           # Web UI backend
  scripts/
    web_navigator.py          # Web-based object navigator
    depth_viewer.py           # Live DA2 depth viewer
    mapper_3d.py              # 3D point cloud mapper
    outdoor_nav.py            # GPS outdoor navigation
    ros2_node.py              # ROS2 publisher node
  config/
    .env.example              # SDK configuration template
    default.yaml              # Default parameters
  earth-rovers-sdk/           # FrodoBot SDK (submodule)
  third_party/
    Depth-Anything-V2/        # DA2 model
  requirements.txt
```

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/web_navigator.py` | Web UI: type objects, robot navigates to them |
| `scripts/depth_viewer.py` | Live depth + obstacle avoidance viewer |
| `scripts/mapper_3d.py` | MPPI driving + 3D point cloud mapping |
| `scripts/outdoor_nav.py` | GPS waypoint navigation with depth safety |
| `scripts/ros2_node.py` | ROS2 node publishing all sensor topics |

## Hardware

- **Robot**: FrodoBots Earth Rover (Mini/Zero)
- **Camera**: Wide-angle front camera (90 FOV)
- **Sensors**: GPS, IMU (accel/gyro/mag), wheel RPM encoders
- **Compute**: Runs on laptop with NVIDIA GPU (tested on RTX 4080)

## Performance

| Component | Latency | Device |
|-----------|---------|--------|
| YOLO 11m | ~15ms | GPU |
| DA2 Small | ~68ms | GPU |
| MPPI (512 samples) | ~6ms | CPU |
| Total pipeline | ~90ms | Mixed |

## License

MIT
