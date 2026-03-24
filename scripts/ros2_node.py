#!/usr/bin/env python3
"""ROS2 node: DA2 depth + MPPI obstacle avoidance + 3D point cloud mapping.

Publishes:
    /camera/image_raw       (sensor_msgs/Image)          - robot camera
    /depth/image            (sensor_msgs/Image)          - DA2 depth map
    /depth/pointcloud       (sensor_msgs/PointCloud2)    - live 3D point cloud
    /map/pointcloud         (sensor_msgs/PointCloud2)    - accumulated 3D map
    /mppi/cmd_vel           (geometry_msgs/Twist)         - MPPI control output
    /mppi/best_path         (nav_msgs/Path)              - MPPI best trajectory
    /mppi/candidate_paths   (visualization_msgs/MarkerArray) - MPPI candidates
    /odom                   (nav_msgs/Odometry)          - visual odometry

Subscribes:
    (nothing — gets camera from SDK directly)

Parameters:
    auto_mode (bool, default True)  - MPPI autonomous driving
    max_linear (float, default 0.30)
    max_angular (float, default 0.45)

Usage:
    ros2 run -- python3 T_ros2_mapper_node.py
    # or simply:
    python3 T_ros2_mapper_node.py

    # Visualize in rviz2:
    rviz2
    # Add displays: PointCloud2 (/map/pointcloud), Image (/camera/image_raw),
    # Image (/depth/image), Path (/mppi/best_path)
"""

import sys
import os
import struct
import base64
import io
import time

import numpy as np
import cv2
import requests
from PIL import Image

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from std_msgs.msg import Header
from sensor_msgs.msg import Image as RosImage, PointCloud2, PointField
from geometry_msgs.msg import Twist, PoseStamped, Point, Quaternion
from nav_msgs.msg import Path, Odometry
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'third_party', 'Depth-Anything-V2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'third_party', 'Depth-Anything-V2', 'metric_depth'))

SDK_URL = "http://localhost:8000"
FLIP_LINEAR = False

# Camera params
FOV_H_DEG = 90.0
IMG_W = 640
IMG_H = 480
FX = IMG_W / (2.0 * np.tan(np.radians(FOV_H_DEG / 2.0)))
FY = FX
CX = IMG_W / 2.0
CY = IMG_H / 2.0
MAX_DEPTH = 8.0
DOWNSAMPLE = 6
MAX_MAP_POINTS = 300_000
VOXEL_SIZE = 0.05


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


def send_control(linear, angular):
    try:
        if FLIP_LINEAR:
            linear = -linear
        requests.post(f"{SDK_URL}/control-legacy",
                      json={"command": {"linear": linear, "angular": angular, "lamp": 0}},
                      timeout=1.0)
    except Exception:
        pass


def depth_to_pointcloud(depth, rgb, downsample=DOWNSAMPLE):
    h, w = depth.shape
    rgb_resized = cv2.resize(rgb, (w, h))
    rows = np.arange(0, h, downsample)
    cols = np.arange(0, w, downsample)
    rr, cc = np.meshgrid(rows, cols, indexing='ij')
    rr, cc = rr.flatten(), cc.flatten()
    z = depth[rr, cc]
    valid = (z > 0.1) & (z < MAX_DEPTH)
    rr, cc, z = rr[valid], cc[valid], z[valid]
    fx = FX * w / IMG_W
    fy = FY * h / IMG_H
    cx = CX * w / IMG_W
    cy = CY * h / IMG_H
    x = (cc - cx) * z / fx
    y = (rr - cy) * z / fy
    points = np.stack([x, y, z], axis=1)
    colors = rgb_resized[rr, cc].astype(np.float32) / 255.0
    return points, colors


def estimate_motion(prev_gray, curr_gray, prev_depth):
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return np.eye(4)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < 10:
        return np.eye(4)
    matches = sorted(matches, key=lambda m: m.distance)[:200]
    h, w = prev_depth.shape
    fx = FX * w / IMG_W
    fy = FY * h / IMG_H
    cx = CX * w / IMG_W
    cy = CY * h / IMG_H
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    pts_3d, pts_2d = [], []
    for m in matches:
        p1 = kp1[m.queryIdx].pt
        p2 = kp2[m.trainIdx].pt
        col = min(max(int(p1[0] * w / IMG_W), 0), w - 1)
        row = min(max(int(p1[1] * h / IMG_H), 0), h - 1)
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
        iterationsCount=100, reprojectionError=3.0, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success or inliers is None or len(inliers) < 6:
        return np.eye(4)
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T


def transform_points(points, T):
    ones = np.ones((points.shape[0], 1))
    pts_h = np.hstack([points, ones])
    return (T @ pts_h.T).T[:, :3]


def voxel_downsample(points, colors, voxel_size=VOXEL_SIZE):
    if len(points) == 0:
        return points, colors
    keys = np.floor(points / voxel_size).astype(np.int32)
    _, unique_idx = np.unique(keys, axis=0, return_index=True)
    if len(unique_idx) > MAX_MAP_POINTS:
        unique_idx = np.random.choice(unique_idx, MAX_MAP_POINTS, replace=False)
    return points[unique_idx], colors[unique_idx]


# ---------------------------------------------------------------------------
# ROS2 message helpers
# ---------------------------------------------------------------------------

def numpy_to_ros_image(frame: np.ndarray, encoding: str = "rgb8") -> RosImage:
    msg = RosImage()
    msg.header.stamp = rclpy.clock.Clock().now().to_msg()
    msg.header.frame_id = "camera_link"
    msg.height, msg.width = frame.shape[:2]
    msg.encoding = encoding
    msg.is_bigendian = False
    if len(frame.shape) == 3:
        msg.step = frame.shape[1] * frame.shape[2]
    else:
        msg.step = frame.shape[1]
    msg.data = frame.tobytes()
    return msg


def depth_to_ros_image(depth: np.ndarray) -> RosImage:
    msg = RosImage()
    msg.header.stamp = rclpy.clock.Clock().now().to_msg()
    msg.header.frame_id = "camera_link"
    msg.height, msg.width = depth.shape
    msg.encoding = "32FC1"
    msg.is_bigendian = False
    msg.step = depth.shape[1] * 4
    msg.data = depth.astype(np.float32).tobytes()
    return msg


def create_pointcloud2(points: np.ndarray, colors: np.ndarray,
                        frame_id: str = "map") -> PointCloud2:
    """Create PointCloud2 from (N,3) points and (N,3) RGB colors (0-1)."""
    msg = PointCloud2()
    msg.header.stamp = rclpy.clock.Clock().now().to_msg()
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = len(points)
    msg.is_dense = True
    msg.is_bigendian = False

    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.point_step = 16
    msg.row_step = msg.point_step * len(points)

    buf = bytearray()
    for i in range(len(points)):
        buf += struct.pack('fff', points[i, 0], points[i, 2], -points[i, 1])
        r = int(colors[i, 0] * 255)
        g = int(colors[i, 1] * 255)
        b = int(colors[i, 2] * 255)
        rgb_packed = struct.unpack('f', struct.pack('I', (r << 16) | (g << 8) | b))[0]
        buf += struct.pack('f', rgb_packed)

    msg.data = bytes(buf)
    return msg


def trajectory_to_path(traj_x: np.ndarray, traj_y: np.ndarray) -> Path:
    msg = Path()
    msg.header.stamp = rclpy.clock.Clock().now().to_msg()
    msg.header.frame_id = "base_link"
    for i in range(len(traj_x)):
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose.position.x = float(traj_x[i])
        pose.pose.position.y = float(traj_y[i])
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0
        msg.poses.append(pose)
    return msg


def pose_to_odometry(camera_pose: np.ndarray) -> Odometry:
    msg = Odometry()
    msg.header.stamp = rclpy.clock.Clock().now().to_msg()
    msg.header.frame_id = "map"
    msg.child_frame_id = "base_link"
    pos = camera_pose[:3, 3]
    msg.pose.pose.position.x = float(pos[0])
    msg.pose.pose.position.y = float(pos[2])
    msg.pose.pose.position.z = float(-pos[1])
    # Simple rotation to quaternion (just yaw for now)
    R = camera_pose[:3, :3]
    yaw = np.arctan2(R[0, 2], R[2, 2])
    msg.pose.pose.orientation.z = float(np.sin(yaw / 2))
    msg.pose.pose.orientation.w = float(np.cos(yaw / 2))
    return msg


# ---------------------------------------------------------------------------
# ROS2 Node
# ---------------------------------------------------------------------------

class MppiMapperNode(Node):
    def __init__(self):
        super().__init__('mppi_3d_mapper')

        # Parameters
        self.declare_parameter('auto_mode', True)
        self.declare_parameter('max_linear', 0.30)
        self.declare_parameter('max_angular', 0.45)
        self.declare_parameter('tick_hz', 5.0)

        self.auto_mode = self.get_parameter('auto_mode').value
        max_lin = self.get_parameter('max_linear').value
        max_ang = self.get_parameter('max_angular').value
        self.tick_hz = self.get_parameter('tick_hz').value

        # Publishers
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                          durability=DurabilityPolicy.VOLATILE)

        self.pub_image = self.create_publisher(RosImage, '/camera/image_raw', qos)
        self.pub_depth = self.create_publisher(RosImage, '/depth/image', qos)
        self.pub_cloud = self.create_publisher(PointCloud2, '/depth/pointcloud', qos)
        self.pub_map = self.create_publisher(PointCloud2, '/map/pointcloud', qos)
        self.pub_cmd = self.create_publisher(Twist, '/mppi/cmd_vel', 10)
        self.pub_path = self.create_publisher(Path, '/mppi/best_path', 10)
        self.pub_odom = self.create_publisher(Odometry, '/odom', 10)

        # DA2
        self.get_logger().info("Loading DA2...")
        from depth_estimator import DepthEstimator
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.estimator = DepthEstimator(model_size='small', max_depth=10.0, device=device)
        self.get_logger().info(f"DA2 ready on {device}")

        # MPPI
        from T_mppi_planner import MPPIPlanner, MPPIConfig
        self.mppi = MPPIPlanner(MPPIConfig(
            num_samples=512, horizon=12, dt=0.15,
            max_linear=max_lin, max_angular=max_ang,
        ))
        self.get_logger().info("MPPI ready")

        # State
        self.map_points = np.zeros((0, 3))
        self.map_colors = np.zeros((0, 3))
        self.camera_pose = np.eye(4)
        self.prev_gray = None
        self.prev_depth = None
        self.frame_count = 0

        # Timer
        period = 1.0 / self.tick_hz
        self.timer = self.create_timer(period, self.tick)
        self.get_logger().info(f"Running at {self.tick_hz} Hz. auto_mode={self.auto_mode}")

    def tick(self):
        t0 = time.time()

        frame = get_frame()
        if frame is None:
            return

        frame_small = cv2.resize(frame, (IMG_W, IMG_H))
        gray = cv2.cvtColor(frame_small, cv2.COLOR_RGB2GRAY)

        # DA2
        depth = self.estimator.estimate(frame)

        # MPPI
        mppi_lin, mppi_ang, mppi_debug = self.mppi.plan(depth, goal_direction_rad=0.0)

        if self.auto_mode:
            send_control(mppi_lin, mppi_ang)

        # Visual odometry
        if self.prev_gray is not None and self.prev_depth is not None:
            T = estimate_motion(self.prev_gray, gray, self.prev_depth)
            try:
                self.camera_pose = self.camera_pose @ np.linalg.inv(T)
            except np.linalg.LinAlgError:
                pass

        self.prev_gray = gray
        self.prev_depth = depth

        # Point cloud
        pts_cam, cols_cam = depth_to_pointcloud(depth, frame)
        if len(pts_cam) > 0:
            pts_world = transform_points(pts_cam, self.camera_pose)
            self.map_points = np.vstack([self.map_points, pts_world])
            self.map_colors = np.vstack([self.map_colors, cols_cam])
            if len(self.map_points) > MAX_MAP_POINTS * 1.5:
                self.map_points, self.map_colors = voxel_downsample(
                    self.map_points, self.map_colors)

        # --- Publish ---
        # Camera image
        self.pub_image.publish(numpy_to_ros_image(frame_small))

        # Depth image
        self.pub_depth.publish(depth_to_ros_image(depth))

        # Current frame point cloud
        if len(pts_cam) > 0:
            n_pub = min(len(pts_cam), 10000)
            idx = np.random.choice(len(pts_cam), n_pub, replace=False) if len(pts_cam) > n_pub else np.arange(len(pts_cam))
            self.pub_cloud.publish(create_pointcloud2(
                pts_cam[idx], cols_cam[idx], frame_id="camera_link"))

        # Map point cloud (every 5 frames)
        if self.frame_count % 5 == 0 and len(self.map_points) > 0:
            n_pub = min(len(self.map_points), 50000)
            idx = np.random.choice(len(self.map_points), n_pub, replace=False) if len(self.map_points) > n_pub else np.arange(len(self.map_points))
            self.pub_map.publish(create_pointcloud2(
                self.map_points[idx], self.map_colors[idx], frame_id="map"))

        # MPPI command
        twist = Twist()
        twist.linear.x = float(mppi_lin)
        twist.angular.z = float(mppi_ang)
        self.pub_cmd.publish(twist)

        # MPPI best path
        self.pub_path.publish(trajectory_to_path(
            mppi_debug["best_x"], mppi_debug["best_y"]))

        # Odometry
        self.pub_odom.publish(pose_to_odometry(self.camera_pose))

        dt = (time.time() - t0) * 1000
        self.frame_count += 1
        if self.frame_count % 10 == 0:
            pos = self.camera_pose[:3, 3]
            self.get_logger().info(
                f"[{self.frame_count}] map={len(self.map_points):,}pts "
                f"pos=({pos[0]:.2f},{pos[2]:.2f}) "
                f"cmd=({mppi_lin:.2f},{mppi_ang:.2f}) "
                f"cost={mppi_debug['best_cost']:.1f} {dt:.0f}ms")


def main():
    rclpy.init()
    node = MppiMapperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        send_control(0, 0)
        # Save map
        if len(node.map_points) > 0:
            np.savez('/tmp/ros2_pointcloud_map.npz',
                     points=node.map_points, colors=node.map_colors)
            node.get_logger().info(f"Saved {len(node.map_points):,} points to /tmp/ros2_pointcloud_map.npz")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
