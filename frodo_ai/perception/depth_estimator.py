"""
Depth Estimator — Depth Anything V3

Wrapper around Depth Anything V3 for monocular metric depth estimation.
Returns depth maps in metres alongside confidence maps.

Falls back to Depth Anything V2 if DA3 is not installed.

Usage:
    from depth_estimator import DepthEstimator

    estimator = DepthEstimator(model_size='base')
    depth, confidence = estimator.estimate(rgb_frame)
    # depth: (H, W) float32 in metres
    # confidence: (H, W) float32 in [0, 1], or None if DA2 fallback

Author: Jotheesh Reddy Kummathi
"""

import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class DepthEstimator:
    """Monocular depth estimation using Depth Anything V3 (or V2 fallback).

    Estimates metric depth (in metres) from a single RGB image.
    Also provides confidence maps when using DA3.
    """

    def __init__(self, model_size='base', device=None, max_depth=10.0, version=3):
        """
        Initialize depth estimator.

        Args:
            model_size: 'small', 'base', or 'large'
            device: 'cuda' or 'cpu'. Auto-detects if None.
            max_depth: Maximum depth in metres (default 10m for indoor)
            version: 2 or 3 to force DA2 or DA3
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.max_depth = max_depth
        self.model_size = model_size
        self._use_da3 = False
        self._da3_model = None
        self._da2_model = None

        if version == 3:
            # Try DA3 first
            try:
                self._init_da3(model_size)
            except Exception as e:
                print(f"DA3 not available ({e}), falling back to DA2...")
                self._init_da2(model_size)
        else:
            # Force DA2
            print("Forcing Depth Anything V2 fallback as requested...")
            self._init_da2(model_size)

    def _init_da3(self, model_size: str):
        """Initialize Depth Anything V3."""
        from depth_anything_3.api import DepthAnything3

        model_name = f"depth-anything/DA3-{model_size.upper()}"
        print(f"Loading Depth Anything V3 ({model_name}) on {self.device}...")

        self._da3_model = DepthAnything3.from_pretrained(model_name)
        self._da3_model = self._da3_model.to(device=self.device)
        self._use_da3 = True
        print(f"DA3 ready on {self.device}")

    def _init_da2(self, model_size: str):
        """Initialize Depth Anything V2 (fallback)."""
        # Add DA2 to path
        base_dir = os.path.join(os.path.dirname(__file__), '..', '..')
        sys.path.insert(0, os.path.join(base_dir, 'third_party', 'Depth-Anything-V2'))
        sys.path.insert(0, os.path.join(base_dir, 'third_party', 'Depth-Anything-V2', 'metric_depth'))

        print(f"Loading Depth Anything V2 ({model_size}) on {self.device}...")

        model_configs = {
            'small': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'base': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'large': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }

        if model_size not in model_configs:
            raise ValueError(f"model_size must be 'small', 'base', or 'large', got '{model_size}'")

        config = model_configs[model_size]

        try:
            from depth_anything_v2.dpt import DepthAnythingV2
        except ImportError:
            da_path = os.path.join(base_dir, 'third_party', 'Depth-Anything-V2')
            sys.path.insert(0, da_path)
            from depth_anything_v2.dpt import DepthAnythingV2

        self._da2_model = DepthAnythingV2(
            encoder=config['encoder'],
            features=config['features'],
            out_channels=config['out_channels'],
            max_depth=self.max_depth,
        )

        # Load checkpoint
        checkpoint_path = self._find_da2_checkpoint(model_size)
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self._da2_model.load_state_dict(state_dict)
            print(f"Loaded DA2 checkpoint: {checkpoint_path}")
        else:
            print(f"WARNING: No DA2 checkpoint found for {model_size} model.")

        self._da2_model.eval()
        self._da2_model.to(self.device)
        self._use_da3 = False
        print(f"DA2 ready on {self.device}")

    def estimate(self, image, target_size=None):
        """
        Estimate depth from RGB image.

        Args:
            image: RGB image as numpy array (H, W, 3) with values 0-255
            target_size: Optional (H, W) to resize output. If None, matches input.

        Returns:
            depth_map: numpy array (H, W) with depth in metres
            confidence: numpy array (H, W) with confidence [0, 1], or None (DA2)
        """
        if self._use_da3:
            return self._estimate_da3(image, target_size)
        else:
            return self._estimate_da2(image, target_size)

    def _estimate_da3(self, image, target_size=None):
        """Estimate depth using DA3."""
        h, w = image.shape[:2]
        if target_size is None:
            target_size = (h, w)

        with torch.no_grad():
            pred = self._da3_model.inference([image])

        depth = pred.depth.squeeze().astype(np.float32)

        # Invert relative depth to pseudo-metric (same as DA2 fallback)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        depth = self.max_depth * (1.0 - depth)
        confidence = pred.conf.squeeze().astype(np.float32) if hasattr(pred, 'conf') and pred.conf is not None else None

        # Resize if needed
        if depth.shape != target_size:
            depth = np.array(Image.fromarray(depth).resize(
                (target_size[1], target_size[0]), Image.BILINEAR))
            if confidence is not None:
                confidence = np.array(Image.fromarray(confidence).resize(
                    (target_size[1], target_size[0]), Image.BILINEAR))

        # Clamp to max depth
        depth = np.clip(depth, 0.0, self.max_depth)

        return depth, confidence

    def _estimate_da2(self, image, target_size=None):
        """Estimate depth using DA2 (fallback)."""
        h, w = image.shape[:2]
        if target_size is None:
            target_size = (h, w)

        img_tensor = self._preprocess_da2(image)

        with torch.no_grad():
            if self.device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    depth = self._da2_model(img_tensor)
            else:
                depth = self._da2_model(img_tensor)

        depth = depth.squeeze().cpu().numpy().astype(np.float32)

        if depth.shape != target_size:
            depth = np.array(Image.fromarray(depth).resize(
                (target_size[1], target_size[0]), Image.BILINEAR))

        return depth, None  # No confidence from DA2

    def _preprocess_da2(self, image):
        """Preprocess image for DA2 model input."""
        img = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        img_tensor = img_tensor.to(self.device)

        h, w = img_tensor.shape[2:]
        new_h = (h // 14) * 14
        new_w = (w // 14) * 14
        if new_h != h or new_w != w:
            img_tensor = F.interpolate(img_tensor, size=(new_h, new_w),
                                       mode='bilinear', align_corners=False)
        return img_tensor

    def _find_da2_checkpoint(self, model_size):
        """Search for DA2 checkpoint file in common locations."""
        base_dir = os.path.join(os.path.dirname(__file__), '..', '..')
        ckpt_dir = os.path.join(base_dir, 'third_party', 'Depth-Anything-V2', 'checkpoints')
        encoder_map = {'small': 'vits', 'base': 'vitb', 'large': 'vitl'}
        enc = encoder_map.get(model_size, model_size)
        search_paths = [
            os.path.join(ckpt_dir, f'depth_anything_v2_metric_hypersim_{enc}.pth'),
            os.path.join(ckpt_dir, f'depth_anything_v2_metric_vkitti_{enc}.pth'),
            os.path.join(ckpt_dir, f'depth_anything_v2_metric_{model_size}.pth'),
            os.path.join(ckpt_dir, f'depth_anything_v2_{model_size}.pth'),
            os.path.join(ckpt_dir, f'depth_anything_v2_{enc}.pth'),
            os.path.join(base_dir, 'models', f'depth_anything_v2_{model_size}.pth'),
        ]
        for path in search_paths:
            if os.path.exists(path):
                return path
        return None

    def get_polar_clearance(self, depth_map, num_bins=32, crop_bottom=0.6,
                            fov_horizontal=90.0, fx=None, cx=None):
        """
        Convert depth map to polar clearance vector.

        For each angular direction,
        what is the minimum distance to an obstacle?

        Args:
            depth_map: (H, W) depth in meters
            num_bins: Number of angular bins (default 32)
            crop_bottom: Fraction of image to keep (bottom portion)
            fov_horizontal: Horizontal field of view in degrees
            fx: Focal length in pixels (auto-computed if None)
            cx: Principal point x (auto-computed if None)

        Returns:
            clearance: numpy array (num_bins,) - min depth per direction
            bin_centers: numpy array (num_bins,) - center angle of each bin
        """
        h, w = depth_map.shape

        # Crop to bottom portion (ground-level obstacles)
        crop_start = int(h * (1.0 - crop_bottom))
        depth_cropped = depth_map[crop_start:, :]

        h_crop, w_crop = depth_cropped.shape

        # Camera intrinsics
        if fx is None:
            fx = w / (2.0 * np.tan(np.radians(fov_horizontal / 2.0)))
        if cx is None:
            cx = w / 2.0

        # Compute yaw angle for each pixel column
        u = np.arange(w_crop)
        yaw_per_col = np.arctan((u - cx) / fx)  # radians

        # Define bin edges
        fov_rad = np.radians(fov_horizontal)
        bin_edges = np.linspace(-fov_rad / 2, fov_rad / 2, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        # Compute clearance per bin (hard min for simplicity at inference)
        clearance = np.full(num_bins, np.inf)

        for b in range(num_bins):
            # Find columns that fall in this bin
            mask = (yaw_per_col >= bin_edges[b]) & (yaw_per_col < bin_edges[b + 1])
            cols_in_bin = np.where(mask)[0]

            if len(cols_in_bin) > 0:
                # Minimum depth across all pixels in these columns
                bin_depths = depth_cropped[:, cols_in_bin]
                valid_depths = bin_depths[bin_depths > 0]  # Ignore zero/invalid
                if len(valid_depths) > 0:
                    clearance[b] = np.min(valid_depths)

        # Replace inf with max_depth
        clearance[clearance == np.inf] = self.max_depth

        return clearance, bin_centers

    def is_waypoint_safe(self, waypoint, clearance, bin_centers, margin=0.5):
        """
        Check if a waypoint direction is safe.

        Args:
            waypoint: (x, y) predicted waypoint in robot frame
            clearance: polar clearance vector from get_polar_clearance()
            bin_centers: bin center angles
            margin: safety margin in meters (default 0.5m)

        Returns:
            safe: True if clearance at waypoint direction >= margin
            clearance_at_wp: clearance value in that direction
        """
        # Get waypoint yaw angle
        wp_yaw = np.arctan2(waypoint[1], waypoint[0])

        # Find nearest bin
        bin_idx = np.argmin(np.abs(bin_centers - wp_yaw))
        clearance_at_wp = clearance[bin_idx]

        return clearance_at_wp >= margin, clearance_at_wp

    def get_safe_direction(self, clearance, bin_centers, margin=0.5):
        """
        Find the safest direction to travel.

        Args:
            clearance: polar clearance vector
            bin_centers: bin center angles
            margin: safety margin

        Returns:
            best_angle: safest direction in radians
            best_clearance: clearance in that direction
        """
        # Prefer center bins (going straight) if safe
        center_idx = len(bin_centers) // 2
        if clearance[center_idx] >= margin:
            return bin_centers[center_idx], clearance[center_idx]

        # Otherwise find bin with maximum clearance
        best_idx = np.argmax(clearance)
        return bin_centers[best_idx], clearance[best_idx]
