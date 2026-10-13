# YOLOv8 Smart Parking Detection System

Detects **free parking slots (GREEN)**, **correctly parked cars (BLUE)**, and **incorrectly parked cars (RED)** using YOLOv8 detections plus configurable vehicle–slot **overlap-ratio** analysis.

## Dataset findings (do not skip)

Source: Roboflow **PKLot v2 640** at  
`D:\RMS 4\Parking_dataset (1)\Parking_dataset`

| Item | Value |
|---|---|
| Images | 12,416 JPG @ 640×640 |
| Splits | train 8691 / valid 2483 / test 1242 |
| Format | YOLOv8 `.txt` normalized boxes |
| Class `0` | `space-empty` — free valid slot |
| Class `1` | `space-occupied` — occupied slot (**not** “correct parking”) |
| Boxes | ~57 per image; empty≈376k, occupied≈336k |
| Vehicles / wrong-parking labels | **None** |
| Polygons | **None** (axis-aligned boxes approximate original rotated PKLot slots) |

**What this dataset can train directly:** free vs occupied parking slots.

**What it cannot train alone:** a true vehicle detector or a wrong-parking class. Vehicles are handled by (1) remapping occupied slots → `car` for approximate training, and/or (2) COCO-pretrained vehicle classes. Wrong parking is decided only by geometry.

## Architecture

1. **Slot detector (Approach A)** — YOLOv8 on `space-empty` / `space-occupied`.
2. **Vehicle detector (Approach B)** — tiled COCO-pretrained vehicle classes by default; the custom occupied-slot model is available as an explicit alternative.
3. **Overlap classifier (Approach C)** — IoU, \(R_{vehicle}\), \(R_{slot}\), \(R_{outside}\), multi-slot intrusion → FREE / CORRECT / WRONG / UNCERTAIN.

## Setup (Windows PowerShell)

```powershell
cd "D:\RMS 4\parking-detection"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Step-by-step commands

### 1) Inspect dataset

```powershell
python src\data_inspection.py --dataset "D:\RMS 4\Parking_dataset (1)\Parking_dataset" --out reports
```

### 2) Prepare training subsets (original files untouched)

```powershell
# CPU-friendly subsample (800 train images). Use --max-train 0 for full data.
python src\prepare_data.py --src "D:\RMS 4\Parking_dataset (1)\Parking_dataset" --max-train 800 --seed 42
```

### 3) Train slot detector

```powershell
python src\train.py --data dataset\space\data.yaml --model yolov8n.pt --name slot_det --epochs 20 --batch 4 --device cpu
```

### 4) Train approximate car detector

```powershell
python src\train.py --data dataset\car\data.yaml --model yolov8n.pt --name car_det --epochs 20 --batch 4 --device cpu
```

> This machine reports **CUDA unavailable**. **Train on Kaggle GPU for 50 epochs** — see `KAGGLE_TRAINING.md` and `notebooks/ParkingDetection_Kaggle.ipynb`.

### 5) Validate detection (mAP)

```powershell
python src\evaluation.py --weights weights\slot_det_best.pt --data dataset\space\data.yaml --split test --device cpu
```

### 6) Tune overlap thresholds (validation only)

```powershell
python src\tune_thresholds.py --max-images 40
```

### 7) Inference on images

```powershell
# With trained weights:
python src\inference.py --source dataset\space\test\images --out outputs\inference --slot-weights weights\slot_det_best.pt
python src\inference.py --source path\to\parking.jpg --out outputs\vehicles_only --vehicles-only

# Optional approximate car weights (trained from occupied-slot boxes, not car-instance labels):
python src\inference.py --source dataset\space\test\images --out outputs\inference --slot-weights weights\slot_det_best.pt --vehicle-weights weights\car_det_best.pt

# Geometry demo using ground-truth labels (no weights required):
python src\inference.py --source "D:\RMS 4\Parking_dataset (1)\Parking_dataset\test\images" --labels-dir "D:\RMS 4\Parking_dataset (1)\Parking_dataset\test\labels" --use-gt-slots --use-gt-vehicles --out outputs\inference\gt_demo
```

### 8) Video / webcam

```powershell
python src\video_processor.py --source path\to\video.mp4 --out outputs\inference\video_out.mp4 --slot-weights weights\slot_det_best.pt --vehicle-weights weights\car_det_best.pt
python src\video_processor.py --source 0 --display --max-frames 100
```

### 9) Unit tests

```powershell
pytest tests -q
```

## Color legend

| Status | Color | Label |
|---|---|---|
| Free parking slot | GREEN | `FREE SLOT \| Slot-01` |
| Correctly parked car | BLUE | `CORRECT PARKING \| Slot-01 \| Inside: 96%` |
| Incorrectly parked car | RED | `WRONG PARKING \| Outside: 18%` |

## Thresholds

Editable in `configs/thresholds.yaml`:

- `vehicle_inside_min: 0.85` and `outside_max: 0.10` are required for a confident BLUE result.
- `likely_correct_min_inside_ratio: 0.50` permits a higher-recall BLUE result when at least half the car box is inside one assigned slot and no strong adjacent-slot straddle is found; this can introduce false blues when slot boxes are misaligned.
- `wrong_parking_min_outside_ratio: 0.70` requires at least 70% estimated outside-slot area before a single-slot case is marked RED; borderline geometry is UNCERTAIN. This reduces false red labels at the cost of missing some real violations.
- `slot_assign_min: 0.30`
- `minimum_second_slot_intrusion_ratio: 0.20` flags a vehicle that clearly crosses into another slot beyond the primary-slot tolerance band, when its box is at least `minimum_vehicle_slot_size_ratio_for_straddle: 1.50` times the assigned slot size in one dimension. This avoids calling a car wrong just because adjacent predicted slot boxes split a car that otherwise fits one space. If learned slot boxes overlap, `minimum_second_slot_overlap_ratio: 0.50` provides a stricter fallback only for strongly elongated vehicle boxes (`vehicle_straddle_min_aspect_ratio: 1.80`). These geometry rules can still be affected by inaccurate slot boxes.
- `boundary_crossing_tolerance: 0.05` ignores a narrow boundary band equal to 5% of the vehicle box's shorter dimension.
- `vehicle_detection_conf_min: 0.10` keeps lower-confidence boxes visible as uncertain; `vehicle_conf_min` still controls whether they can be classified.
- `vehicle_imgsz: 1280` runs vehicle detection at higher resolution on the full frame; slot detection remains at 640. Set `tiled_inference: true` to enable overlapping crops for small targets, with tile sizes controlled by `vehicle_tile_size` and `slot_tile_size`.
- Crop-edge detections are discarded to avoid counting partial objects, and overlapping full-frame/tile predictions are merged.

When significant vehicle area penetrates a second slot beyond tolerance, or the calibrated fallback confirms an overlap with a second slot, the vehicle is labeled `WRONG PARKING | Straddles Slot-A / Slot-B`, even if it remains inside the combined outer boundary. These values require validation against manually verified examples from the target camera; the available PKLot annotations do not label actual vehicle footprints or parking violations.

The slot detector's `space-occupied` result is used as an occupancy signal. If it says a slot is occupied but no vehicle can be confidently matched, the slot is shown as **occupied, unverified** rather than incorrectly counted as free or as correctly parked. A vehicle with no reliable slot assignment is **UNCERTAIN**, not automatically labeled wrong; geometry must support a violation. Wrong-parking accuracy still requires camera-specific vehicle detections and labeled wrong-parking examples; the PKLot data does not provide those labels.

### Local CPU demo weights (5 epochs, 400-image subset)

| Model | Weights | Test/Val mAP@0.50 |
|---|---|---|
| Slot (`space-empty` / `space-occupied`) | `weights/slot_det_best.pt` (50-epoch Kaggle YOLOv8s) | **0.9795 mAP@0.50**, 0.9770 mAP@0.50:0.95 (test) |
| Car (occupied remapped) | `weights/car_det_best.pt` | **0.969** (val; approximate boxes, not vehicle-instance labels) |

The slot model's test metrics are from `new_model/slot_metrics.json`. They measure parking-slot detection, not vehicle detection or parking-violation classification.

## Honesty / limitations

- Learned detections ≠ geometric decisions ≠ evaluation metrics.
- Occupied slot ≠ correctly parked vehicle.
- Bounding-box overlap can disagree with the true car footprint under perspective.
- Wrong-parking precision/recall needs a manually labeled evaluation set (not present in PKLot Roboflow export).
- Rectangular slots approximate angled parking; polygons + calibration would improve reliability.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: ultralytics` | `pip install -r requirements.txt` |
| Invalid dataset path | Confirm `Parking_dataset (1)\Parking_dataset\data.yaml` exists |
| Invalid YOLO labels | Re-run `data_inspection.py`; fix empty/orphan labels |
| CUDA unavailable | Train on Kaggle GPU (50 epochs) — see `KAGGLE_TRAINING.md` |
| Missing `best.pt` | Run `train.py` or use `--use-gt-slots --use-gt-vehicles` for geometry demos |
| Symlink errors on Windows | Re-run `prepare_data.py --copy` |

## Citation

Almeida et al., PKLot, Expert Systems with Applications, 2015. Dataset via Roboflow CC BY 4.0.
