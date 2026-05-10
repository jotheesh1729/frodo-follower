"""YOLO-based object detection with natural language target matching."""

from __future__ import annotations

from difflib import get_close_matches
from typing import Optional

import numpy as np


YOLO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

ALIASES = {
    'human': 'person', 'people': 'person', 'man': 'person', 'woman': 'person',
    'guy': 'person', 'someone': 'person', 'somebody': 'person',
    'sofa': 'couch', 'monitor': 'tv', 'screen': 'tv', 'television': 'tv',
    'phone': 'cell phone', 'mobile': 'cell phone', 'cellphone': 'cell phone',
    'pc': 'laptop', 'computer': 'laptop', 'notebook': 'laptop',
    'bag': 'backpack', 'rucksack': 'backpack',
    'mug': 'cup', 'glass': 'wine glass',
    'desk': 'dining table', 'table': 'dining table',
    'plant': 'potted plant', 'flower': 'potted plant',
    'fridge': 'refrigerator', 'bike': 'bicycle', 'motorbike': 'motorcycle',
    'auto': 'car', 'vehicle': 'car', 'van': 'truck',
    'puppy': 'dog', 'kitten': 'cat',
    'ball': 'sports ball', 'toy': 'teddy bear',
}


def parse_target(command: str) -> tuple[str, str]:
    """Parse natural language command into a YOLO class name."""
    cmd = command.lower().strip()

    if not cmd or cmd in ('stop', 'halt', 'freeze', 'wait'):
        return '', 'Stopping robot.'

    for prefix in ['go to the ', 'go to ', 'navigate to the ', 'navigate to ',
                   'find the ', 'find a ', 'find ', 'drive to the ', 'drive to ',
                   'move to the ', 'move to ', 'head to the ', 'head to ',
                   'approach the ', 'approach ', 'get to the ', 'get to ',
                   'look for the ', 'look for a ', 'look for ',
                   'go towards the ', 'go towards ']:
        if cmd.startswith(prefix):
            cmd = cmd[len(prefix):]
            break

    for suffix in [' please', ' now', ' quickly', ' slowly', ' over there',
                   ' near me', ' on the left', ' on the right', ' ahead']:
        if cmd.endswith(suffix):
            cmd = cmd[:-len(suffix)]

    target = cmd.strip()

    if target in YOLO_CLASSES:
        return target, f'Found exact match: {target}'
    if target in ALIASES:
        return ALIASES[target], f'"{target}" -> {ALIASES[target]}'

    matches = get_close_matches(target, YOLO_CLASSES, n=1, cutoff=0.65)
    if matches:
        return matches[0], f'Best match for "{target}": {matches[0]}'

    matches = get_close_matches(target, list(ALIASES.keys()), n=1, cutoff=0.65)
    if matches:
        return ALIASES[matches[0]], f'"{target}" ~ "{matches[0]}" -> {ALIASES[matches[0]]}'

    for cls in YOLO_CLASSES:
        if target in cls or cls in target:
            return cls, f'Partial match: "{target}" -> {cls}'

    return '', f'Cannot find "{target}". Try: {", ".join(YOLO_CLASSES[:10])}...'


class ObjectDetector:
    """YOLO object detector with GPU support."""

    def __init__(self, model_name: str = "yolo11m.pt", device: str = "auto"):
        import torch
        from ultralytics import YOLO

        if device == "auto":
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = YOLO(model_name)
        self.model.to(device)
        self.device = device

    def detect(self, frame: np.ndarray, conf: float = 0.2) -> list[dict]:
        """Run detection on a frame. Returns list of detections."""
        results = self.model(frame, verbose=False, conf=conf, device=self.device)
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for i in range(len(r.boxes)):
                cls_id = int(r.boxes.cls[i])
                x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                detections.append({
                    'class': r.names[cls_id],
                    'confidence': float(r.boxes.conf[i]),
                    'bbox': (x1, y1, x2, y2),
                    'center_x': (x1 + x2) / 2,
                    'center_y': (y1 + y2) / 2,
                })
        return detections

    def find_target(self, detections: list[dict], target_class: str) -> Optional[dict]:
        """Find the best detection matching target_class."""
        best = None
        best_conf = 0
        for det in detections:
            if det['class'] == target_class and det['confidence'] > best_conf:
                best = det
                best_conf = det['confidence']
        return best
