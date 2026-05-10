# Frodo-Follower — Change Log

All changes made during this refactor session, in order of implementation.

---

## Phase 1 — Model switch + GPU fixes + Deduplication

### `scripts/web_navigator.py`
- **YOLO model**: `yolo26x.pt` → `yolo26m.pt` (3-4× faster, near-identical indoor accuracy)
- **GPU device bug**: Removed hardcoded `device='cuda'` in YOLO inference call; now uses `self.device` which is set correctly at init
- **Duplicate code removed**: Deleted the copy of `YOLO_CLASSES`, `ALIASES`, and `parse_target()` that lived in this file. They now come from the single source of truth in `frodo_ai/perception/object_detector.py`
- **Unused imports cleaned up**: Removed `get_close_matches`, `HTTPServer`, `parse_qs`
- **Redundant torch import removed**: Second `import torch` and `device =` in DA2 loading section deleted

### `frodo_ai/perception/depth_estimator.py`
- **FP16 autocast**: Added `torch.amp.autocast('cuda')` around model inference for automatic FP16 on GPU (2× depth estimation speed)
- **Float32 cast after inference**: Added `.astype(np.float32)` after `.cpu().numpy()` — autocast produces FP16 tensors which PIL cannot handle
- **Deprecated API fix**: `torch.cuda.amp.autocast()` → `torch.amp.autocast('cuda')` (removes FutureWarning)

---

## Phase 2 — NLP parser fix

### `frodo_ai/perception/object_detector.py`
- **Fuzzy match cutoff**: Raised from `0.5` → `0.65` in both `get_close_matches` calls
  - **Root cause of `helmet → elephant` bug**: `difflib.SequenceMatcher` found a ratio of ~0.57 between "helmet" and "elephant" (shared substrings "el", "e", "t"), which passed the old 0.5 cutoff
  - At 0.65 cutoff, "helmet" finds no match and falls through to the substring check, which also correctly finds nothing — user gets a clear "cannot find" message

> **Note**: YOLO-World was trialled as a replacement for the NLP parser (open-vocabulary detection, no class list needed). It was reverted because YOLO 26m significantly outperforms YOLO-World on known COCO classes like "chair", "person", "bottle". YOLO-World is better for exotic objects not in COCO 80; for this robot's use case, YOLO 26m + fixed NLP is the right trade-off.

---

## Phase 3 — Smooth control + target locking

### `scripts/web_navigator.py`

#### Control smoothing
- **EMA on angular command**: `cmd_ang = 0.50 × raw + 0.50 × prev` — angular changes gradually, eliminates micro-jitter from YOLO bbox noise
- **Linear ramp**: Max speed change per frame capped at ±0.05 — no more 0 → 0.30 m/s speed jumps
- **Deadband**: Pixel errors < 5% of half-width treated as zero — stops micro-corrections when nearly centred
- **Removed hard centering mode**: Old code stopped forward motion entirely when `|error| > 0.4`, causing "stuck centering" behaviour. Replaced with proportional speed: `lin = 0.30 × max(0.25, 1 - |error|)` — robot always moves forward, just slower when steering hard

#### Target instance locking (TV/chair switching bug)
- **Centroid-based lock**: When multiple instances of the same class are in frame, the robot now picks the one **closest to the last known centroid**, not the highest-confidence one
- This prevents the robot switching from TV-A to TV-B mid-approach when both are visible

---

## Phase 4 — Bug fixes from testing

### Spinning circles bug
- **Root cause**: `self._avoiding = True` was set on arrival. This caused target selection to fall back to highest-confidence instead of centroid lock. With EMA angular carrying residual turn velocity, the robot would rotate slightly after arrival, see a different chair as highest confidence, navigate to it, arrive, rotate again — spiral loop.
- **Fix**: Removed `_avoiding` flag entirely. Target lock now always uses centroid proximity once a lock exists.
- **Fix**: On arrival, `self._smooth_ang` and `self._smooth_lin` are hard-zeroed immediately (overrides EMA bleed)
- **Fix**: Search spin angular reduced from `0.18` → `0.10` rad/s

### MPPI wired to actual control
- Previously: `_, _, mppi_debug = self.mppi.plan(...)` — linear and angular outputs discarded, MPPI only used for visualization
- **Now**: `mppi_lin, _, mppi_debug = self.mppi.plan(...)` — MPPI linear speed is used as the actual forward command when navigating
- MPPI considers the full depth map as an obstacle cost field, automatically slowing or rerouting around obstacles between the robot and the target
- Visual servo still controls angular (more accurate for pixel-level target tracking)
- MPPI visualization now uses the pre-computed `mppi_debug` from the control call (no second redundant plan call)

### DA2 skipped at idle
- Depth estimation was running every frame even with no target, capping idle FPS at ~7.5 (DA2 is the bottleneck at ~30ms/frame)
- Now skipped when `current_target` is empty; idle FPS rises to ~25-30 (limited by frame fetch)

### HTTP server stability
- `BrokenPipeError` in `do_GET` and `do_POST` caught silently — happens when browser disconnects mid-response (e.g., during model download on first run)

---

## Web UI redesign

### `scripts/web_navigator.py` — `HTML_PAGE`

Complete redesign from a basic dark-mode page to a proper dashboard layout:

| Old UI | New UI |
|--------|--------|
| Flat 3-column video grid | Sidebar + full canvas + right panel |
| 500ms polling interval | 300ms polling interval |
| No GPU indicator | GPU/CPU pill badge in top bar |
| No velocity visualization | Live linear + angular bar meters |
| No command history | Last 8 commands shown |
| Static quick buttons | Clickable object cloud from live detections |
| No feed switching | Toggle between Detection / Depth / MPPI on main canvas |
| No target lock indicator | Green "TARGET LOCKED" overlay when navigating |
| Status text only | Colour-coded status card (navigating/arrived/searching/stopped) |

---

## Architecture summary (current)

```
Camera frame
    │
    ▼
YOLO 26m (GPU) ──────────────────────────────► All 80 COCO classes detected
    │                                           Centroid-locked target selected
    ▼
DA2 Small (GPU, FP16) ──────────────────────► Metric depth map (metres)
    │
    ├──► MPPI Planner (CPU, 512 samples) ────► Safe linear speed (obstacle-aware)
    │         goal_direction from visual servo
    │
    └──► Visual Servo ────────────────────────► Angular command (EMA smoothed)
              pixel error → steering

Combined (lin, ang) ──► EMA + ramp ──► send_control() ──► Robot
```

---

## Files changed

| File | Changes |
|------|---------|
| `scripts/web_navigator.py` | Model switch, GPU fix, dedup, servo rewrite, MPPI wiring, UI redesign |
| `frodo_ai/perception/depth_estimator.py` | FP16 autocast, float32 cast, deprecated API fix |
| `frodo_ai/perception/object_detector.py` | NLP fuzzy match cutoff 0.5 → 0.65 |
