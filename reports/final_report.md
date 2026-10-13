# Final Report — YOLOv8 Smart Parking Detection

## 1. Dataset (inspected, not assumed)

| Item | Finding |
|---|---|
| Source | Roboflow PKLot v2 640 (`Parking_dataset (1)/Parking_dataset`) |
| Images | 12,416 JPG @ 640×640 |
| Splits | train 8691 / valid 2483 / test 1242 |
| Format | YOLOv8 normalized `.txt` boxes |
| Class `0` | `space-empty` — free valid parking slot |
| Class `1` | `space-occupied` — occupied slot (**occupancy, not correctness**) |
| Instance counts | empty 376,121 (52.8%) / occupied 335,735 (47.2%) |
| Avg boxes/image | ~57.3 |
| Empty label files | 274 (no boxes; still valid images) |
| Corrupt images / bad coords / orphans | 0 |
| Vehicle boxes | **None** |
| Wrong-parking labels | **None** |
| Slot polygons | **None** (axis-aligned boxes approximate original rotated PKLot slots) |

**Class mapping is preserved.** Occupied ≠ correctly parked.

## 2. What the dataset can and cannot train

| Task | Supported? |
|---|---|
| Free vs occupied slot detection | Yes — train directly on classes 0/1 |
| Vehicle-instance detection | Not directly — remap occupied→`car` (approximation) or use COCO vehicles |
| Wrong-parking class | No — must use geometric overlap analysis |
| Research-grade wrong-parking metrics | Needs a separately labeled evaluation set |

## 3. Architecture chosen

1. **Approach A — Slot detector:** YOLOv8n on `space-empty` / `space-occupied`.
2. **Approach B — Vehicle detector:** YOLOv8n trained on remapped occupied boxes (`car`), optional COCO fallback.
3. **Approach C — Overlap classifier:** IoU, \(R_{vehicle}\), \(R_{slot}\), \(R_{outside}\), multi-slot intrusion → FREE (GREEN) / CORRECT (BLUE) / WRONG (RED) / UNCERTAIN.

Wrong parking is **never** a learned YOLO class in this project.

## 4. Configurable thresholds (`configs/thresholds.yaml`)

| Parameter | Value | Role |
|---|---|---|
| `vehicle_inside_min` | 0.90 | Min \(R_{vehicle}\) for CORRECT |
| `outside_max` | 0.10 | Max \(R_{outside}\) for CORRECT |
| `slot_assign_min` | 0.30 | Min \(R_{vehicle}\) to assign a car to a slot |
| `adjacent_intrusion_min` | 0.20 | Both \(R_{vehicle}\) and \(R_{slot}\) on a 2nd slot ⇒ WRONG |
| `slot_coverage_occupied_min` | 0.30 | Physical occupancy of a slot |
| `vehicle_conf_min` / `uncertain_conf_max` | 0.25 / 0.40 | Confidence floors |

Adjacent intrusion requires **both** ratios so packed AABB slot envelopes do not false-trigger WRONG from edge overlap alone.

## 5. Training environment and actual results

- PyTorch 2.7.1+cu118 reported, **CUDA unavailable** → CPU training.
- Prepared CPU subset: 400 train / 100 valid / 100 test (original export untouched).
- Base weights: `weights/yolov8n.pt`.
- Epochs: 5 each (short CPU demo run). Prefer GPU + more epochs for production.

### Slot detector (`weights/slot_det_best.pt`) — test split

| Metric | Value |
|---|---|
| Precision | 0.949 |
| Recall | 0.956 |
| mAP@0.50 | **0.984** |
| mAP@0.50:0.95 | 0.629 |
| space-empty mAP50-95 | 0.660 |
| space-occupied mAP50-95 | 0.599 |

### Approximate car detector (`weights/car_det_best.pt`) — val during training

| Metric | Value |
|---|---|
| Precision | 0.911 |
| Recall | 0.974 |
| mAP@0.50 | **0.969** |
| mAP@0.50:0.95 | 0.594 |

### Geometry proxy (validation labels, 50 images)

| Metric | Value |
|---|---|
| Slot occupancy agreement | 0.934 |
| Proxy CORRECT counts | 1776 / 1937 vehicles |
| Proxy false-WRONG when \(R_{vehicle}\) high | 161 |
| Wrong-parking P/R | **UNAVAILABLE** (no WRONG GT labels) |

### Threshold tuning (validation only)

Selected: `vehicle_inside_min=0.85`, `outside_max=0.10`, `adjacent_intrusion_min=0.15`, `slot_assign_min=0.30`  
(see `outputs/tuning/threshold_search.json`). Unit tests: **14 passed**.

## 6. Evaluation honesty

| Metric source | What it measures |
|---|---|
| YOLO mAP / precision / recall | Slot or approximate-car **detection** quality only |
| Geometry proxy on GT labels | Free-slot agreement and inside-ratio behavior — **not** real wrong-parking accuracy |
| Manually labeled CORRECT/WRONG set | Required for credible wrong-parking precision/recall (not in this export) |

## 7. Strengths

- Clear separation of learned detections vs geometric decisions vs metrics.
- Configurable overlap rules with unit tests (14 scenarios).
- GREEN / BLUE / RED visualization with dashboard overlay.
- Image, folder, video, and webcam inference entry points.
- Original dataset files never modified.

## 8. Limitations and future work

- Axis-aligned boxes are a poor fit for angled PKLot bays; prefer original rotated rectangles / polygons.
- Remapped occupied boxes are not true vehicle footprints; bbox shrink (~0.82) is used only for geometry demos.
- Perspective / camera calibration would improve \(R_{vehicle}\) fidelity.
- Instance segmentation of cars would reduce bbox overhang false alarms.
- CPU-only short training underfits relative to a full GPU run on all 12k images.

## 9. How to reproduce

See project `README.md` for PowerShell commands: inspect → prepare → train → evaluate → tune → infer → pytest.
