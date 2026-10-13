"""Grid-search overlap thresholds on a validation set (keep test set untouched)."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parking_classifier import classify_frame, load_thresholds
from slot_detector import slots_from_yolo_labels
from vehicle_detector import vehicles_from_occupied_labels


def score_thresholds(
    images_dir: Path,
    labels_dir: Path,
    base: Dict[str, float],
    overrides: Dict[str, float],
    max_images: int,
) -> Dict[str, Any]:
    thr = dict(base)
    thr.update(overrides)
    files = sorted(images_dir.glob("*.jpg"))[:max_images]
    slot_agree = 0
    slot_total = 0
    high_inside_wrong = 0
    high_inside_total = 0
    free_fp = 0
    free_gt = 0

    for ip in files:
        lp = labels_dir / f"{ip.stem}.txt"
        img = cv2.imread(str(ip))
        if img is None or not lp.is_file():
            continue
        h, w = img.shape[:2]
        slots = slots_from_yolo_labels(lp, w, h)
        vehicles = vehicles_from_occupied_labels(lp, w, h)
        result = classify_frame(vehicles, slots, thr)
        for sr in result.slots:
            slot_total += 1
            gt_empty = sr.slot.source_class == "space-empty"
            pred_free = sr.status == "FREE"
            if gt_empty == pred_free:
                slot_agree += 1
            if gt_empty:
                free_gt += 1
                if not pred_free:
                    pass
            else:
                if pred_free:
                    free_fp += 1
        for vr in result.vehicles:
            if vr.r_vehicle >= 0.95:
                high_inside_total += 1
                if vr.status == "WRONG_PARKING":
                    high_inside_wrong += 1

    agree = slot_agree / slot_total if slot_total else 0.0
    false_wrong_rate = high_inside_wrong / high_inside_total if high_inside_total else 0.0
    # Prefer high occupancy agreement and low false-wrong on near-perfect inside ratio
    score = agree - 0.5 * false_wrong_rate
    return {
        "overrides": overrides,
        "slot_occupancy_agreement": round(agree, 4),
        "free_false_positives": free_fp,
        "proxy_false_wrong_rate_high_inside": round(false_wrong_rate, 4),
        "score": round(score, 4),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=str, default=str(ROOT / "dataset" / "space" / "valid" / "images"))
    p.add_argument("--labels", type=str, default=str(ROOT / "dataset" / "space" / "valid" / "labels"))
    p.add_argument("--thresholds", type=str, default=str(ROOT / "configs" / "thresholds.yaml"))
    p.add_argument("--out", type=str, default=str(ROOT / "outputs" / "tuning" / "threshold_search.json"))
    p.add_argument("--max-images", type=int, default=40)
    args = p.parse_args()

    base = load_thresholds(args.thresholds)
    # Small grid for CPU runs; expand on GPU if needed
    grid = {
        "vehicle_inside_min": [0.85, 0.90],
        "outside_max": [0.10, 0.15],
        "slot_assign_min": [0.30],
    }
    keys = list(grid.keys())
    rows: List[Dict[str, Any]] = []
    for values in itertools.product(*[grid[k] for k in keys]):
        overrides = dict(zip(keys, values))
        rows.append(
            score_thresholds(Path(args.images), Path(args.labels), base, overrides, args.max_images)
        )

    rows.sort(key=lambda r: r["score"], reverse=True)
    best = rows[0]
    report = {
        "selection_reason": (
            "Highest validation score = slot occupancy agreement − 0.5 × proxy false-wrong rate "
            "on vehicles with R_vehicle≥0.95. Test set was not used."
        ),
        "selected": best,
        "top5": rows[:5],
        "all": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Write selected thresholds alongside defaults
    selected_path = ROOT / "configs" / "thresholds_selected.yaml"
    selected = dict(base)
    selected.update(best["overrides"])
    import yaml

    selected_path.write_text(yaml.safe_dump(selected, sort_keys=False), encoding="utf-8")
    print(json.dumps({"selected": best, "wrote": str(selected_path)}, indent=2))


if __name__ == "__main__":
    main()
