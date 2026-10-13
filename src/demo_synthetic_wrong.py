"""Create a synthetic wrong-parking visualization to demonstrate RED multi-slot overlap."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geometry import ParkingSlot, VehicleDetection
from parking_classifier import classify_frame, load_thresholds
from visualization import draw_parking_annotations


def main() -> None:
    img_path = next((ROOT / "outputs" / "inference" / "sample_in").glob("*.jpg"), None)
    if img_path is None:
        # blank canvas
        img = np.full((640, 640, 3), 40, dtype=np.uint8)
    else:
        img = cv2.imread(str(img_path))

    slots = [
        ParkingSlot("Slot-01", bbox=[80, 200, 180, 320]),
        ParkingSlot("Slot-02", bbox=[180, 200, 280, 320]),
        ParkingSlot("Slot-03", bbox=[280, 200, 380, 320]),
        ParkingSlot("Slot-04", bbox=[400, 200, 500, 320]),
    ]
    vehicles = [
        VehicleDetection(0, [90, 210, 170, 310], confidence=0.96),   # correct in Slot-01
        VehicleDetection(1, [150, 210, 270, 310], confidence=0.94),  # straddles 01-02
        VehicleDetection(2, [410, 230, 560, 340], confidence=0.91),  # mostly outside Slot-04
    ]
    thr = load_thresholds(ROOT / "configs" / "thresholds.yaml")
    result = classify_frame(vehicles, slots, thr)
    out = draw_parking_annotations(img, result)
    out_dir = ROOT / "outputs" / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "synthetic_wrong_parking_demo.jpg"
    cv2.imwrite(str(out_path), out)
    print("wrote", out_path)
    print("summary", result.summary)
    for v in result.vehicles:
        print(v.status, v.slot_id, v.reason, v.overlapped_slots)


if __name__ == "__main__":
    main()
