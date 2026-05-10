# frodo-follower

Tell the [FrodoBots Earth Rover](https://frodobots.com/) where to go in plain English. Type "go to the chair" and it finds the chair, navigates to it, and stops when it gets there. No maps, no GPS, no LLM — just a camera and vision models.

Built for the [FrodoBots Earth Rover Challenge](https://www.frodobots.com/erc).

## Demo

> Videos coming soon

## How it works

The main loop runs in `scripts/web_navigator.py`:

1. **YOLO 26m** detects objects in the camera frame every iteration (~5 ms with TensorRT FP16)
2. You type a command like "find the bottle" — fuzzy matching maps it to a YOLO class, no LLM needed
3. **Depth Anything V2 Base** runs in a background thread, producing metric depth maps in meters
4. **MPPI planner** samples 512 trajectories, scores them against the depth map, and picks a forward speed that avoids obstacles
5. **Visual servo** keeps the target centred in frame using proportional steering with EMA smoothing
6. The robot stops when the target is within ~1.2m or fills more than 55% of the frame height
7. If the target leaves view, it coasts to the last known position, then does a timed 360° search spin before giving up

The web UI at `localhost:5000` shows the detection feed, depth map, MPPI planner view, and live velocity.

## Setup

### 1. Clone and install

```bash
git clone https://github.com/jotheesh1729/frodo-follower.git
cd frodo-follower
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download model checkpoints

**YOLO 26m** — downloads automatically on first run via ultralytics.

**Depth Anything V2 Base** (metric indoor):

```bash
mkdir -p third_party/Depth-Anything-V2/checkpoints
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='depth-anything/Depth-Anything-V2-Metric-Hypersim-Base',
    filename='depth_anything_v2_metric_hypersim_vitb.pth',
    local_dir='third_party/Depth-Anything-V2/checkpoints'
)
"
```

### 3. (Optional) Export YOLO to TensorRT for faster inference

Requires CUDA + TensorRT. Run once, then the engine is used automatically.

```bash
python scripts/export_trt.py --yolo
```

### 4. Configure

```bash
cp config/.env.example config/.env
# Fill in SDK_API_TOKEN and BOT_SLUG from your FrodoBots account
```

### 5. Start the SDK server

```bash
cd earth-rovers-sdk && hypercorn main:app --reload
```

### 6. Run

```bash
python scripts/web_navigator.py
# Open http://localhost:5000
```

## Commands

Type anything natural into the web UI:

```
go to the chair
find a person
navigate to the bottle
stop
```

Aliases work too — "sofa" maps to couch, "fridge" to refrigerator, "phone" to cell phone, etc. If the input doesn't match anything, you get a clear "cannot find" message instead of a wrong guess.

## Project structure

```
frodo_follower/
├── perception/
│   ├── object_detector.py    # YOLO detection + NLP command parsing
│   └── depth_estimator.py    # Depth Anything V2 wrapper (metric depth in metres)
└── planning/
    └── mppi_planner.py       # MPPI trajectory optimiser (512 samples, 12-step horizon)

scripts/
├── web_navigator.py          # main script — web UI + navigation loop
└── export_trt.py             # export YOLO/DA2 to TensorRT FP16 engines

config/
├── default.yaml              # all tunable parameters
└── .env.example              # SDK credentials template
```

## Configuration

Key parameters in `config/default.yaml`:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `perception.yolo_model` | `yolo26m.pt` | auto-uses `yolo26m.engine` if exported |
| `perception.depth_model` | `base` | small / base / large |
| `control.arrival_distance` | `1.2` | metres — stops this far from target |
| `control.arrival_bbox_frac` | `0.55` | stops if target fills >55% of frame height |
| `control.steer_gain` | `0.25` | proportional gain for visual servo |
| `planning.mppi_samples` | `512` | more samples = better paths, higher CPU |

## Hardware

Tested on:
- FrodoBots Earth Rover (Mini)
- RTX 5070Ti (12 GB VRAM) for YOLO + depth inference

Should work on any CUDA GPU. Falls back to CPU if CUDA isn't available (much slower).

## Credits

Based on [frodo-ai](https://github.com/tarunkumarnyu/frodo-ai) by [Tarun Kumar](https://github.com/tarunkumarnyu).

## License

MIT
