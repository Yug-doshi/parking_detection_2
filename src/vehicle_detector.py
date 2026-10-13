"""Vehicle detection.

This PKLot Roboflow export does NOT contain vehicle-instance annotations.
Vehicle boxes are obtained by either:
  1) A detector trained on occupied-slot boxes remapped to class `car` (approximation), or
  2) A COCO-pretrained YOLOv8 model filtered to car/bus/truck classes.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Set, Union

import numpy as np
from ultralytics import YOLO

from detection_utils import iter_image_tiles, suppress_duplicate_detections, touches_tile_edge
from geometry import VehicleDetection, yolo_norm_to_xyxy

# COCO class ids commonly treated as vehicles
COCO_VEHICLE_IDS: Set[int] = {2, 3, 5, 7}  # car, motorcycle, bus, truck
COCO_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class VehicleDetector:
    def __init__(
        self,
        weights: Union[str, Path],
        conf: float = 0.25,
        iou: float = 0.45,
        device: str = "cpu",
        allowed_classes: Optional[Sequence[int]] = None,
        treat_all_as_car: Optional[bool] = None,
        tile_size: int = 256,
        tile_overlap: float = 0.2,
        tile_edge_margin: float = 0.04,
        imgsz: int = 1280,
        tiled_inference: bool = False,
    ):
        self.model = YOLO(str(weights))
        self.conf = conf
        self.iou = iou
        self.device = device
        self.allowed_classes = set(allowed_classes) if allowed_classes is not None else None
        names = self.model.names or {}
        model_classes = [str(name).lower() for name in names.values()]
        self.treat_all_as_car = (
            len(model_classes) == 1 and model_classes[0] == "car"
            if treat_all_as_car is None
            else treat_all_as_car
        )
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.tile_edge_margin = tile_edge_margin
        self.imgsz = imgsz
        self.tiled_inference = tiled_inference

    def detect(self, image: np.ndarray) -> List[VehicleDetection]:
        height, width = image.shape[:2]
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
            result = results[0]
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if self.allowed_classes is not None and cls_id not in self.allowed_classes:
                    continue
                if self.allowed_classes is None and not self.treat_all_as_car:
                    names = result.names or {}
                    if (2 in names or "car" in str(names.values()).lower()) and cls_id not in COCO_VEHICLE_IDS:
                        continue
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                if is_tile and touches_tile_edge(
                    [x1, y1, x2, y2],
                    source.shape[1],
                    source.shape[0],
                    self.tile_edge_margin,
                ):
                    continue
                bbox = [
                    max(0.0, x1 + offset_x),
                    max(0.0, y1 + offset_y),
                    min(float(width), x2 + offset_x),
                    min(float(height), y2 + offset_y),
                ]
                candidates.append((float(box.conf[0]), bbox, cls_id))

        vehicles = []
        for vid, (confidence, bbox, cls_id) in enumerate(
            suppress_duplicate_detections(candidates, self.iou)
        ):
            name = "car" if self.treat_all_as_car else COCO_NAMES.get(cls_id, str(cls_id))
            vehicles.append(
                VehicleDetection(
                    vehicle_id=vid,
                    bbox=bbox,
                    confidence=confidence,
                    class_name=name,
                )
            )
        return vehicles


def vehicles_from_occupied_labels(
    label_path: Union[str, Path],
    width: int,
    height: int,
    occupied_class_id: int = 1,
    shrink: float = 0.82,
) -> List[VehicleDetection]:
    """Approximate vehicle boxes from occupied-slot ground truth (class 1).

    ``shrink`` scales each box about its center. PKLot Roboflow labels are
    axis-aligned envelopes of angled slots that heavily overlap neighbors; using
    the raw slot box as a car causes false multi-slot WRONG flags. A modest
    shrink approximates the vehicle footprint inside the bay for geometry demos.
    """
    path = Path(label_path)
    vehicles: List[VehicleDetection] = []
    if not path.is_file():
        return vehicles
    shrink = float(max(0.05, min(1.0, shrink)))
    for i, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()):
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(float(parts[0]))
        if cid != occupied_class_id:
            continue
        cx, cy, bw, bh = map(float, parts[1:5])
        bw, bh = bw * shrink, bh * shrink
        bbox = yolo_norm_to_xyxy(cx, cy, bw, bh, width, height)
        vehicles.append(VehicleDetection(vehicle_id=i, bbox=bbox, confidence=1.0, class_name="car"))
    return vehicles
