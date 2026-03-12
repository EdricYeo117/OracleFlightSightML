# OracleFlightSightML

A practical, webcam-based gaze-to-screen pipeline built on top of **L2CS-Net**.

This repository combines:
- **Gaze estimation** (yaw/pitch) using a pretrained L2CS model,
- **Face mesh + head pose features**, and
- A lightweight **screen-point mapper** trained from a short per-user calibration session.

The result is a local real-time prototype that predicts where on your screen a user is looking.

---

## What this project does

The project has two layers:

1. **Base gaze estimation (L2CS)**
   - Uses `l2cs.Pipeline` with pretrained weights (`models/L2CSNet_gaze360.pkl`) to infer gaze angles.
2. **Personal calibration mapper**
   - Collects calibration samples while the user looks at known points.
   - Builds a feature vector from gaze + head pose + face geometry.
   - Trains two ridge-regression models (`model_x.pkl`, `model_y.pkl`) to map features -> screen coordinates.

---

## Repository structure

- `collect_calibration.py` – interactive fullscreen calibration collection.
- `train_mapper.py` – trains `model_x.pkl` and `model_y.pkl` from calibration CSV.
- `predict_screen_point.py` – realtime prediction loop with webcam preview + predicted dot.
- `calibration_utils.py` – feature definitions and helper utilities.
- `demo.py` – standard L2CS webcam gaze demo (without screen mapping).
- `l2cs/` – package implementation (pipeline, face mesh wrappers, model code, rendering helpers).
- `models/` – model assets (gaze model + face landmark task).
- `requirements.txt` / `pyproject.toml` – dependency and packaging metadata.

---

## Requirements

### Hardware
- Webcam.
- GPU optional (CPU works, lower throughput).

### Software
- Python 3.8+ recommended.
- OpenCV GUI support (for fullscreen calibration/prediction windows).

---

## Setup

### 1) Clone and enter project

```bash
git clone <your-repo-url>
cd OracleFlightSightML
```

### 2) Create virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows PowerShell
```

### 3) Install dependencies

```bash
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

> `pip install -e .` makes the local `l2cs` package importable in editable mode.

---

## Quick start (end-to-end)

### Step A: Collect calibration data

```bash
python collect_calibration.py
```

What happens:
- Opens a fullscreen window and runs multiple target phases:
  - 9-point grid pass 1
  - 9-point grid pass 2
  - center refinement
  - random refinement
- For each target, it waits briefly, aggregates samples, and writes rows to `calibration_samples.csv`.

Controls:
- `SPACE` to start.
- `Q` or `ESC` to quit.

### Step B: Train screen mapper

```bash
python train_mapper.py
```

Outputs:
- `model_x.pkl`
- `model_y.pkl`

Notes:
- Requires `calibration_samples.csv` to exist.
- Expects at least 15 rows (27+ recommended for stability).

### Step C: Run realtime screen-point prediction

```bash
python predict_screen_point.py
```

What you will see:
- Fullscreen gray canvas.
- Webcam preview inset with face landmarks.
- Predicted gaze point rendered as a red dot with white ring.

Exit with `Q` or `ESC`.

---

## Feature engineering used for mapping

The mapper is trained on these per-frame features:

- `gaze_yaw`
- `gaze_pitch`
- `head_yaw`
- `head_pitch`
- `head_roll`
- `face_cx`
- `face_cy`
- `face_w`
- `face_h`

Targets:
- `target_x` (screen x coordinate in pixels)
- `target_y` (screen y coordinate in pixels)

Modeling approach:
- Two independent pipelines (x and y):
  - `StandardScaler`
  - `Ridge(alpha=5.0)`

---

## Running the base L2CS webcam demo

If you only want gaze vectors (without screen coordinate mapping):

```bash
python demo.py --device cpu --cam 0 --arch ResNet50
```

---

## Troubleshooting

### Webcam cannot be opened
- Ensure no other app is using the camera.
- Try another camera index (`0`, `1`, ...).
- Check OS camera permissions.

### Fullscreen size fallback
If screen dimensions cannot be queried, scripts fallback to `1920x1080`.

### Missing model assets
Ensure these files exist:
- `models/L2CSNet_gaze360.pkl`
- `models/face_landmarker.task`

### Poor prediction quality
- Re-run calibration with stable lighting and seated posture.
- Keep head naturally mobile during calibration to capture realistic variation.
- Increase number/diversity of calibration points.

---

## Development tips

- Validate imports quickly:

```bash
python -c "import l2cs; print('ok')"
```

- Keep generated artifacts out of commits when experimenting:
  - `calibration_samples.csv`
  - `model_x.pkl`
  - `model_y.pkl`

---

## License

This repository includes a `LICENSE` file. Check it for usage terms.
