# Training & evaluation results

## Slot detector

- Config: `yolov8n`, 5 epochs, batch 4, imgsz 640, CPU, seed 42
- Data: `dataset/space` (400/100/100 subsample)
- Weights: `weights/slot_det_best.pt`
- Runs: `runs/slot_det/`

**Test metrics** (`outputs/evaluation/slot_metrics.json`):

- Precision 0.949 | Recall 0.956 | mAP50 0.984 | mAP50-95 0.629

## Car detector (approximation)

- Occupied slots remapped to class `car`
- Weights: `weights/car_det_best.pt`
- **Test** (CPU demo run): Precision 0.934 | Recall 0.956 | mAP50 **0.977** | mAP50-95 0.586
- Val during training: mAP50 ≈ 0.969 | mAP50-95 ≈ 0.594

## Geometry

- Slot occupancy agreement (proxy): 0.934 on 50 valid images
- Wrong-parking class metrics: not measurable without labeled WRONG set

## Artifacts

- Annotated samples: `outputs/inference/gt_demo/`, `outputs/inference/trained_slot_demo/`, `outputs/inference/full_pipeline/`
- Synthetic RED/BLUE/GREEN demo: `outputs/inference/synthetic_wrong_parking_demo.jpg`
- GT visualization samples: `reports/gt_samples/`
- Threshold search: `outputs/tuning/threshold_search.json`
