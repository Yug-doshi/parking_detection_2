"""Reusable inference pipeline: detect slots + vehicles, classify with overlap ratios, annotate."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geometry import ParkingSlot, VehicleDetection
from parking_classifier import ParkingFrameResult, classify_frame, load_thresholds
from slot_detector import SlotDetector, slots_from_yolo_labels
from vehicle_detector import VehicleDetector, vehicles_from_occupied_labels
from visualization import draw_parking_annotations, draw_vehicle_only_annotations

DEFAULT_THRESH = ROOT / "configs" / "thresholds.yaml"


def load_models(
    slot_weights: Optional[Union[str, Path]] = None,
    vehicle_weights: Optional[Union[str, Path]] = None,
    device: str = "cpu",
    vehicle_conf: float = 0.25,
    slot_conf: float = 0.25,
    nms_iou: float = 0.45,
    vehicle_from_occupied_model: Optional[bool] = None,
    vehicle_tile_size: int = 256,
    slot_tile_size: int = 384,
    tile_overlap: float = 0.2,
    tile_edge_margin: float = 0.04,
    vehicle_imgsz: int = 1280,
    slot_imgsz: int = 640,
    tiled_inference: bool = False,
) -> Tuple[Optional[SlotDetector], Optional[VehicleDetector]]:
    slot_det = (
        SlotDetector(
            slot_weights,
            conf=slot_conf,
            iou=nms_iou,
            device=device,
            tile_size=slot_tile_size,
            tile_overlap=tile_overlap,
            tile_edge_margin=tile_edge_margin,
            imgsz=slot_imgsz,
            tiled_inference=tiled_inference,
        )
        if slot_weights
        else None
    )
    veh_det = None
    if vehicle_weights:
        veh_det = VehicleDetector(
            vehicle_weights,
            conf=vehicle_conf,
            iou=nms_iou,
            device=device,
            treat_all_as_car=vehicle_from_occupied_model,
            allowed_classes=(
                {2, 3, 5, 7} if vehicle_from_occupied_model is False else None
            ),
            tile_size=vehicle_tile_size,
            tile_overlap=tile_overlap,
            tile_edge_margin=tile_edge_margin,
            imgsz=vehicle_imgsz,
            tiled_inference=tiled_inference,
        )
    return slot_det, veh_det


def detect_parking_slots(
    image: np.ndarray,
    slot_detector: Optional[SlotDetector] = None,
    gt_label_path: Optional[Path] = None,
) -> List[ParkingSlot]:
    if gt_label_path is not None:
        h, w = image.shape[:2]
        return slots_from_yolo_labels(gt_label_path, w, h)
    if slot_detector is None:
        raise ValueError("Need slot_detector or gt_label_path")
    return slot_detector.detect(image)


def detect_vehicles(
    image: np.ndarray,
    vehicle_detector: Optional[VehicleDetector] = None,
    gt_label_path: Optional[Path] = None,
    occupied_as_vehicles: bool = False,
) -> List[VehicleDetection]:
    if occupied_as_vehicles and gt_label_path is not None:
        h, w = image.shape[:2]
        return vehicles_from_occupied_labels(gt_label_path, w, h)
    if vehicle_detector is None:
        raise ValueError("Need vehicle_detector or occupied_as_vehicles with labels")
    return vehicle_detector.detect(image)


def process_image(
    image: np.ndarray,
    slot_detector: Optional[SlotDetector],
    vehicle_detector: Optional[VehicleDetector],
    thresholds: Dict[str, float],
    gt_label_path: Optional[Path] = None,
    use_gt_slots: bool = False,
    use_gt_vehicles: bool = False,
) -> Tuple[ParkingFrameResult, np.ndarray, float]:
    t0 = time.perf_counter()
    slots = detect_parking_slots(
        image,
        slot_detector=None if use_gt_slots else slot_detector,
        gt_label_path=gt_label_path if use_gt_slots else None,
    )
    if not use_gt_slots and slot_detector is not None:
        slots = slot_detector.detect(image)

    vehicles = detect_vehicles(
        image,
        vehicle_detector=None if use_gt_vehicles else vehicle_detector,
        gt_label_path=gt_label_path if use_gt_vehicles else None,
        occupied_as_vehicles=use_gt_vehicles,
    )
    if not use_gt_vehicles and vehicle_detector is not None:
        vehicles = vehicle_detector.detect(image)

    # Prefer detected slots; if empty and GT available, fall back
    if not slots and gt_label_path is not None:
        h, w = image.shape[:2]
        slots = slots_from_yolo_labels(gt_label_path, w, h)

    result = classify_frame(vehicles, slots, thresholds)
    annotated = draw_parking_annotations(image, result)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return result, annotated, latency_ms


def process_path(
    input_path: Path,
    out_dir: Path,
    slot_weights: Optional[Path],
    vehicle_weights: Optional[Path],
    thresholds_path: Path,
    device: str = "cpu",
    use_gt_slots: bool = False,
    use_gt_vehicles: bool = False,
    labels_dir: Optional[Path] = None,
    vehicles_only: bool = False,
) -> Dict[str, Any]:
    thr = load_thresholds(thresholds_path)
    slot_det, veh_det = load_models(
        slot_weights=slot_weights,
        vehicle_weights=vehicle_weights,
        device=device,
        vehicle_conf=thr["vehicle_detection_conf_min"],
        slot_conf=thr["slot_conf_min"],
        nms_iou=thr["nms_iou"],
        vehicle_tile_size=int(thr["vehicle_tile_size"]),
        slot_tile_size=int(thr["slot_tile_size"]),
        tile_overlap=thr["inference_tile_overlap"],
        tile_edge_margin=thr["tile_edge_margin_ratio"],
        vehicle_imgsz=int(thr["vehicle_imgsz"]),
        slot_imgsz=int(thr["slot_imgsz"]),
        tiled_inference=bool(thr["tiled_inference"]),
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file() and input_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(
            [p for p in input_path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        )
    else:
        raise FileNotFoundError(input_path)

    summaries = []
    latencies = []
    for fp in files:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        gt = None
        if labels_dir is not None:
            gt = labels_dir / f"{fp.stem}.txt"
        elif use_gt_slots or use_gt_vehicles:
            # sibling labels/
            cand = fp.parent.parent / "labels" / f"{fp.stem}.txt"
            gt = cand if cand.is_file() else None

        result, annotated, ms = process_image(
            img,
            slot_det,
            veh_det,
            thr,
            gt_label_path=gt,
            use_gt_slots=use_gt_slots,
            use_gt_vehicles=use_gt_vehicles,
        )
        latencies.append(ms)
        out_img = out_dir / f"annotated_{fp.name}"
        if vehicles_only:
            annotated = draw_vehicle_only_annotations(img, result)
        cv2.imwrite(str(out_img), annotated)
        summaries.append({"file": fp.name, "summary": result.summary, "latency_ms": round(ms, 2)})

    report = {
        "n_images": len(summaries),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "results": summaries,
    }
    (out_dir / "inference_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("n_images", "avg_latency_ms")}, indent=2))
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Parking detection inference")
    p.add_argument("--source", type=str, required=True, help="Image file or directory")
    p.add_argument("--out", type=str, default=str(ROOT / "outputs" / "inference"))
    p.add_argument("--slot-weights", type=str, default="")
    p.add_argument("--vehicle-weights", type=str, default="")
    p.add_argument("--thresholds", type=str, default=str(DEFAULT_THRESH))
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--use-gt-slots", action="store_true", help="Use YOLO label files as slot geometry")
    p.add_argument("--use-gt-vehicles", action="store_true", help="Use occupied labels as vehicle boxes")
    p.add_argument("--labels-dir", type=str, default="")
    p.add_argument(
        "--vehicles-only",
        action="store_true",
        help="Draw only blue, red, and uncertain vehicle boxes",
    )
    args = p.parse_args()

    slot_w = Path(args.slot_weights) if args.slot_weights else None
    veh_w = Path(args.vehicle_weights) if args.vehicle_weights else None
    # Auto-pick weights if present
    if slot_w is None:
        cand = ROOT / "weights" / "slot_det_best.pt"
        if cand.is_file():
            slot_w = cand
    if veh_w is None:
        cand = ROOT / "weights" / "yolov8n.pt"
        if not cand.is_file():
            cand = ROOT / "weights" / "car_det_best.pt"
        if cand.is_file():
            veh_w = cand

    if not args.use_gt_slots and slot_w is None and not args.use_gt_vehicles:
        # Allow pure GT demo mode
        if not args.use_gt_slots:
            print("WARNING: no slot weights; enabling --use-gt-slots if labels available")
            args.use_gt_slots = True
    if not args.use_gt_vehicles and veh_w is None:
        print("WARNING: no vehicle weights; enabling --use-gt-vehicles if labels available")
        args.use_gt_vehicles = True

    process_path(
        input_path=Path(args.source),
        out_dir=Path(args.out),
        slot_weights=slot_w,
        vehicle_weights=veh_w,
        thresholds_path=Path(args.thresholds),
        device=args.device,
        use_gt_slots=args.use_gt_slots,
        use_gt_vehicles=args.use_gt_vehicles,
        labels_dir=Path(args.labels_dir) if args.labels_dir else None,
        vehicles_only=args.vehicles_only,
    )


if __name__ == "__main__":
    main()
