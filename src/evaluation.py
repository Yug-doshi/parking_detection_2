"""Detection metrics (mAP) and geometric parking-state evaluation.

YOLO mAP measures slot/vehicle detection quality only.
Overlap-based free/correct/wrong metrics require verified parking-state labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geometry import ParkingSlot, VehicleDetection, calculate_vehicle_inside_ratio
from parking_classifier import classify_frame, load_thresholds
from slot_detector import slots_from_yolo_labels
from vehicle_detector import vehicles_from_occupied_labels


def evaluate_detection(weights: Path, data_yaml: Path, split: str = "test", device: str = "cpu") -> Dict[str, Any]:
    model = YOLO(str(weights))
    # Ultralytics val uses the split defined in YAML; override via split arg when supported
    metrics = model.val(data=str(data_yaml), split=split, device=device, plots=True, project=str(ROOT / "runs"), name=f"val_{weights.stem}")
    out = {
        "weights": str(weights),
        "data": str(data_yaml),
        "split": split,
        "map50": float(metrics.box.map50) if hasattr(metrics, "box") else None,
        "map50_95": float(metrics.box.map) if hasattr(metrics, "box") else None,
        "precision": float(metrics.box.mp) if hasattr(metrics, "box") else None,
        "recall": float(metrics.box.mr) if hasattr(metrics, "box") else None,
    }
    if hasattr(metrics, "box") and hasattr(metrics.box, "maps"):
        out["per_class_map50_95"] = [float(x) for x in metrics.box.maps]
    return out


def evaluate_geometry_on_labels(
    images_dir: Path,
    labels_dir: Path,
    thresholds_path: Path,
    max_images: int = 100,
) -> Dict[str, Any]:
    """Proxy evaluation using GT occupied boxes as vehicles and all slots as geometry.

    Correctness labels are synthetic:
      - A vehicle fully inside its source occupied slot is treated as CORRECT proxy.
      - This does NOT prove real-world wrong-parking accuracy.
    """
    thr = load_thresholds(thresholds_path)
    files = sorted(images_dir.glob("*.jpg"))[:max_images]
    n_free_gt = n_free_pred = 0
    n_correct_proxy = n_wrong_pred_on_proxy_correct = 0
    n_vehicles = 0
    slot_occ_ok = 0
    slot_occ_total = 0

    for ip in files:
        lp = labels_dir / f"{ip.stem}.txt"
        import cv2

        img = cv2.imread(str(ip))
        if img is None or not lp.is_file():
            continue
        h, w = img.shape[:2]
        slots = slots_from_yolo_labels(lp, w, h)
        vehicles = vehicles_from_occupied_labels(lp, w, h)
        result = classify_frame(vehicles, slots, thr)

        empty_gt = sum(1 for s in slots if s.source_class == "space-empty")
        free_pred = sum(1 for s in result.slots if s.status == "FREE")
        n_free_gt += empty_gt
        n_free_pred += free_pred

        for vr in result.vehicles:
            n_vehicles += 1
            # Proxy: if R_vehicle to assigned slot >= inside_min => expect CORRECT
            if vr.status == "CORRECT_PARKING":
                n_correct_proxy += 1
            if vr.r_vehicle >= thr["vehicle_inside_min"] and vr.status == "WRONG_PARKING":
                n_wrong_pred_on_proxy_correct += 1

        for sr in result.slots:
            slot_occ_total += 1
            gt_empty = sr.slot.source_class == "space-empty"
            pred_free = sr.status == "FREE"
            if gt_empty == pred_free:
                slot_occ_ok += 1

    report = {
        "note": "Geometry proxy metrics use occupied-slot boxes as vehicles. Not a substitute for manually verified wrong-parking labels.",
        "images": len(files),
        "unique_vehicles_seen": n_vehicles,
        "free_slots_gt_total": n_free_gt,
        "free_slots_pred_total": n_free_pred,
        "slot_occupancy_agreement": round(slot_occ_ok / slot_occ_total, 4) if slot_occ_total else None,
        "proxy_correct_parking_count": n_correct_proxy,
        "proxy_false_wrong_when_high_inside": n_wrong_pred_on_proxy_correct,
        "wrong_parking_metrics": "UNAVAILABLE — dataset has no wrong-parking ground truth",
    }
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default="")
    p.add_argument("--data", type=str, default=str(ROOT / "dataset" / "space" / "data.yaml"))
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--geometry-images", type=str, default="")
    p.add_argument("--geometry-labels", type=str, default="")
    p.add_argument("--thresholds", type=str, default=str(ROOT / "configs" / "thresholds.yaml"))
    p.add_argument("--out", type=str, default=str(ROOT / "outputs" / "evaluation" / "metrics.json"))
    p.add_argument("--max-images", type=int, default=80)
    args = p.parse_args()

    report: Dict[str, Any] = {}
    if args.weights and Path(args.weights).is_file():
        report["detection"] = evaluate_detection(Path(args.weights), Path(args.data), args.split, args.device)

    geo_img = Path(args.geometry_images) if args.geometry_images else ROOT / "dataset" / "space" / "valid" / "images"
    geo_lbl = Path(args.geometry_labels) if args.geometry_labels else ROOT / "dataset" / "space" / "valid" / "labels"
    if geo_img.is_dir() and geo_lbl.is_dir():
        report["geometry_proxy"] = evaluate_geometry_on_labels(
            geo_img, geo_lbl, Path(args.thresholds), max_images=args.max_images
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
