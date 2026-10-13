"""Video and webcam inference with optional temporal smoothing of parking status."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inference import load_models, process_image
from parking_classifier import load_thresholds
from visualization import draw_parking_annotations


class TemporalSmoother:
    """Majority-vote smoothing of vehicle statuses keyed by approximate center bins."""

    def __init__(self, window: int = 5, majority: float = 0.6):
        self.window = window
        self.majority = majority
        self.hist: Dict[Tuple[int, int], Deque[str]] = defaultdict(lambda: deque(maxlen=window))

    @staticmethod
    def _key(bbox: List[float], grid: int = 40) -> Tuple[int, int]:
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        return (int(cx // grid), int(cy // grid))

    def update(self, frame_result) -> None:
        for vr in frame_result.vehicles:
            if vr.status == "UNCERTAIN":
                continue
            k = self._key(vr.vehicle.bbox)
            self.hist[k].append(vr.status)
            counts = Counter(self.hist[k])
            status, n = counts.most_common(1)[0]
            if n / max(len(self.hist[k]), 1) >= self.majority:
                vr.status = status


def process_video(
    source: str,
    out_path: Path,
    slot_weights: Optional[Path],
    vehicle_weights: Optional[Path],
    thresholds_path: Path,
    device: str = "cpu",
    max_frames: Optional[int] = None,
    display: bool = False,
) -> Dict:
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
    if slot_det is None and veh_det is None:
        raise RuntimeError("Need at least one model weight for video inference")

    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_in,
        (w, h),
    )

    smoother = TemporalSmoother(
        window=int(thr.get("temporal_window", 5)),
        majority=float(thr.get("temporal_majority", 0.6)),
    )
    latencies: List[float] = []
    n = 0
    t_start = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result, annotated, ms = process_image(frame, slot_det, veh_det, thr)
        smoother.update(result)
        annotated = draw_parking_annotations(frame, result)
        writer.write(annotated)
        latencies.append(ms)
        n += 1
        if display:
            cv2.imshow("parking", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if max_frames is not None and n >= max_frames:
            break

    elapsed = time.perf_counter() - t_start
    cap.release()
    writer.release()
    if display:
        cv2.destroyAllWindows()

    report = {
        "frames": n,
        "elapsed_s": round(elapsed, 2),
        "measured_fps": round(n / elapsed, 2) if elapsed > 0 else None,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "output": str(out_path),
    }
    print(json.dumps(report, indent=2))
    (out_path.with_suffix(".json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Video / webcam parking inference")
    p.add_argument("--source", type=str, required=True, help="Video path or webcam index (0)")
    p.add_argument("--out", type=str, default=str(ROOT / "outputs" / "inference" / "video_out.mp4"))
    p.add_argument("--slot-weights", type=str, default=str(ROOT / "weights" / "slot_det_best.pt"))
    p.add_argument("--vehicle-weights", type=str, default=str(ROOT / "weights" / "yolov8n.pt"))
    p.add_argument("--thresholds", type=str, default=str(ROOT / "configs" / "thresholds.yaml"))
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--display", action="store_true")
    args = p.parse_args()

    slot_w = Path(args.slot_weights) if args.slot_weights and Path(args.slot_weights).is_file() else None
    veh_w = Path(args.vehicle_weights) if args.vehicle_weights and Path(args.vehicle_weights).is_file() else None
    process_video(
        source=args.source,
        out_path=Path(args.out),
        slot_weights=slot_w,
        vehicle_weights=veh_w,
        thresholds_path=Path(args.thresholds),
        device=args.device,
        max_frames=args.max_frames or None,
        display=args.display,
    )


if __name__ == "__main__":
    main()
