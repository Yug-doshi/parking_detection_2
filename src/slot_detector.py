"""Parking-slot detection via trained YOLOv8 space-empty / space-occupied model."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
from ultralytics import YOLO

from detection_utils import iter_image_tiles, suppress_duplicate_detections, touches_tile_edge
from geometry import ParkingSlot, yolo_norm_to_xyxy

# Preserve original dataset class mapping
SLOT_CLASS_NAMES = {0: "space-empty", 1: "space-occupied"}


class SlotDetector:
    def __init__(
        self,
        weights: Union[str, Path],
        conf: float = 0.25,
        iou: float = 0.45,
        device: str = "cpu",
        tile_size: int = 384,
        tile_overlap: float = 0.2,
        tile_edge_margin: float = 0.04,
        imgsz: int = 640,
        tiled_inference: bool = False,
    ):
        self.model = YOLO(str(weights))
        self.conf = conf
        self.iou = iou
        self.device = device
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.tile_edge_margin = tile_edge_margin
        self.imgsz = imgsz
        self.tiled_inference = tiled_inference

    def detect(self, image: np.ndarray) -> List[ParkingSlot]:
        h, w = image.shape[:2]
        candidates = []
        sources = [(image, 0, 0, False)]
        if self.tiled_inference:
            sources.extend(
                (tile, x, y, True)
                for x, y, tile in iter_image_tiles(image, self.tile_size, self.tile_overlap)
                if tile.shape[:2] != image.shape[:2]
            )

        for source, offset_x, offset_y, is_tile in sources:
            results = self.model.predict(
                source=source,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
            if not results or results[0].boxes is None:
                continue
            for box in results[0].boxes:
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                if is_tile and touches_tile_edge(
                    [x1, y1, x2, y2],
                    source.shape[1],
                    source.shape[0],
                    self.tile_edge_margin,
                ):
                    continue
                candidates.append(
                    (
                        float(box.conf[0]),
                        [
                            max(0.0, x1 + offset_x),
                            max(0.0, y1 + offset_y),
                            min(float(w), x2 + offset_x),
                            min(float(h), y2 + offset_y),
                        ],
                        int(box.cls[0]),
                    )
                )

        slots: List[ParkingSlot] = []
        for i, (confidence, bbox, cls_id) in enumerate(
            suppress_duplicate_detections(candidates, self.iou)
        ):
            name = SLOT_CLASS_NAMES.get(cls_id, str(cls_id))
            slots.append(
                ParkingSlot(
                    slot_id=f"Slot-{i + 1:02d}",
                    bbox=bbox,
                    confidence=confidence,
                    source_class=name,
                )
            )
        return slots


def slots_from_yolo_labels(
    label_path: Union[str, Path],
    width: int,
    height: int,
    class_filter: Optional[Sequence[int]] = None,
) -> List[ParkingSlot]:
    """Build stable slot geometry from ground-truth YOLO labels (preferred when available)."""
    path = Path(label_path)
    slots: List[ParkingSlot] = []
    if not path.is_file():
        return slots
    lines = path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
    idx = 0
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(float(parts[0]))
        if class_filter is not None and cid not in class_filter:
            continue
        cx, cy, bw, bh = map(float, parts[1:5])
        bbox = yolo_norm_to_xyxy(cx, cy, bw, bh, width, height)
        idx += 1
        slots.append(
            ParkingSlot(
                slot_id=f"Slot-{idx:02d}",
                bbox=bbox,
                confidence=1.0,
                source_class=SLOT_CLASS_NAMES.get(cid, str(cid)),
            )
        )
    return slots
