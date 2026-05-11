# frodo-follower

Two autonomous navigation modes for the [FrodoBots Earth Rover](https://frodobots.com/): navigate to a named object in plain English, or lock onto a person and follow them. No maps, no GPS.

## Demo

<!-- demo video placeholder -->

## Modes

### Smart Navigator

Type a query like "go to the chair" or "find the person with the red shirt" into the web UI. The robot finds the target and drives to it.

Run:
```bash
python3 scripts/smart_navigator.py
# Open http://localhost:5002
```

### Person Follower

Click a person in the camera feed to lock on. The robot follows them and stops at a set distance.

Run:
```bash
python3 scripts/person_follower.py
# Open http://localhost:5001
```

## How it works

**Detection** — YOLO-World (`yolov8s-worldv2.pt`) handles open-vocabulary detection. For descriptive queries like "person with brown shirt", the noun is extracted for YOLO and the full description is passed to the VLM for verification.

**Depth** — Depth Anything V2 (metric, indoor) runs every frame producing per-pixel depth in metres. Distance to the target is sampled from the lower portion of the bounding box (feet/legs region) which gives ground-plane distance rather than line-of-sight to the torso.

**Tracking** — An EKF with state `[angle, angular_velocity, distance, approach_velocity]` maintains a smooth estimate across frames. Appearance-based re-ID (HSV histogram) handles occlusions and re-acquisition.

**VLM** — Qwen2-VL-2B runs in a background thread and serves three purposes: verifying YOLO detections against descriptive queries, guiding the search rotation direction when the target is lost, and advising LEFT/RIGHT when the robot is stuck behind an obstacle.

**Control** — PD controller on bearing error with derivative clamping. Obstacle avoidance uses a 5-band depth scan across the forward view; the widest gap determines the bypass arc direction. Emergency backup triggers below 0.6 m and immediately queues a bypass arc on recovery.

## Architecture

```
scripts/
    smart_navigator.py      navigation to named objects, Flask UI on :5002
    person_follower.py      click-to-follow person, Flask UI on :5001

frodo_ai/perception/
    target_tracker.py       EKF tracker + appearance re-ID
    depth_estimator.py      Depth Anything V2 wrapper (metric depth, metres)

web/
    smart_nav.html          UI for smart navigator
    follower.html           UI for person follower
```

## Setup

**1. Clone**

```bash
git clone https://github.com/jotheesh1729/frodo-follower.git
cd frodo-follower
```

**2. Install dependencies**

```bash
pip install torch torchvision
pip install ultralytics transformers qwen-vl-utils
pip install flask opencv-python pillow requests numpy
```

**3. Download model weights**

YOLO-World — put in repo root:
```bash
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-worldv2.pt
```

Depth Anything V2 (metric indoor, small):
```bash
mkdir -p third_party/Depth-Anything-V2/checkpoints
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='depth-anything/Depth-Anything-V2-Metric-Hypersim-Small',
    filename='depth_anything_v2_metric_hypersim_vits.pth',
    local_dir='third_party/Depth-Anything-V2/checkpoints'
)
"
```

Qwen2-VL-2B — downloads automatically on first run via Hugging Face.

**4. Start the SDK**

```bash
cd earth-rovers-sdk && hypercorn main:app --reload
```

**5. Run**

```bash
python3 scripts/smart_navigator.py    # object navigation
# or
python3 scripts/person_follower.py    # person following
```

## Hardware

Tested on FrodoBots Earth Rover (Mini) with an RTX 5070 Ti (12 GB VRAM). A GPU is required for Qwen2-VL-2B; YOLO-World and DA2 will fall back to CPU but will be slow.

## Credits

Based on [frodo-ai](https://github.com/tarunkumarnyu/frodo-ai) by [Tarun Kumar](https://github.com/tarunkumarnyu).

## License

MIT
