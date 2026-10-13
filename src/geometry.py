"""Vehicle–slot overlap geometry: IoU, inside ratio, coverage, multi-slot intrusion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

from shapely.geometry import Polygon, box as shapely_box
from shapely.validation import make_valid

BBox = List[float]  # [x1, y1, x2, y2] in image pixels
Point = Tuple[float, float]


def _safe_polygon(poly: Polygon) -> Polygon:
    if poly is None or poly.is_empty:
        return Polygon()
    if not poly.is_valid:
        poly = make_valid(poly)
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)
        elif poly.geom_type != "Polygon":
            return Polygon()
    return poly


def bbox_to_polygon(bbox: BBox) -> Polygon:
    x1, y1, x2, y2 = map(float, bbox)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return shapely_box(x1, y1, x2, y2)


def polygon_from_points(points: Sequence[Point]) -> Polygon:
    if len(points) < 3:
        return Polygon()
    return _safe_polygon(Polygon([(float(x), float(y)) for x, y in points]))


def clip_bbox_to_image(bbox: BBox, width: int, height: int) -> BBox:
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(width), float(x1)))
    x2 = max(0.0, min(float(width), float(x2)))
    y1 = max(0.0, min(float(height), float(y1)))
    y2 = max(0.0, min(float(height), float(y2)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def intersection_area(a: Polygon, b: Polygon) -> float:
    a, b = _safe_polygon(a), _safe_polygon(b)
    if a.is_empty or b.is_empty or a.area <= 0 or b.area <= 0:
        return 0.0
    return float(a.intersection(b).area)


def vehicle_intrusion_ratio_beyond_tolerance(
    vehicle: VehicleDetection,
    slot: ParkingSlot,
    primary_slot: ParkingSlot,
    tolerance: float,
) -> float:
    """Return vehicle fraction inside a second slot beyond a primary-slot boundary band."""
    footprint = bbox_to_polygon(vehicle.bbox)
    if footprint.area <= 1e-12:
        return 0.0
    primary_region = primary_slot.as_polygon()
    secondary_region = slot.as_polygon().difference(primary_region.buffer(max(0.0, tolerance)))
    return intersection_area(footprint, secondary_region) / float(footprint.area)


def calculate_iou(region_a: Union[BBox, Polygon], region_b: Union[BBox, Polygon]) -> float:
    """IoU(B,S) = |B ∩ S| / |B ∪ S|."""
    a = region_a if isinstance(region_a, Polygon) else bbox_to_polygon(region_a)
    b = region_b if isinstance(region_b, Polygon) else bbox_to_polygon(region_b)
    a, b = _safe_polygon(a), _safe_polygon(b)
    if a.is_empty or b.is_empty:
        return 0.0
    inter = intersection_area(a, b)
    union = float(a.area + b.area - inter)
    if union <= 1e-12:
        return 0.0
    return inter / union


def calculate_vehicle_inside_ratio(
    vehicle: Union[BBox, Polygon],
    slot: Union[BBox, Polygon, Sequence[Point]],
) -> float:
    """R_vehicle = |B ∩ S| / |B|."""
    v = vehicle if isinstance(vehicle, Polygon) else bbox_to_polygon(vehicle)
    if isinstance(slot, Polygon):
        s = slot
    elif isinstance(slot, (list, tuple)) and slot and isinstance(slot[0], (list, tuple)):
        s = polygon_from_points(slot)  # type: ignore[arg-type]
    else:
        s = bbox_to_polygon(list(slot))  # type: ignore[arg-type]
    v, s = _safe_polygon(v), _safe_polygon(s)
    if v.area <= 1e-12:
        return 0.0
    return intersection_area(v, s) / float(v.area)


def calculate_slot_coverage_ratio(
    vehicle: Union[BBox, Polygon],
    slot: Union[BBox, Polygon, Sequence[Point]],
) -> float:
    """R_slot = |B ∩ S| / |S|."""
    v = vehicle if isinstance(vehicle, Polygon) else bbox_to_polygon(vehicle)
    if isinstance(slot, Polygon):
        s = slot
    elif isinstance(slot, (list, tuple)) and slot and isinstance(slot[0], (list, tuple)):
        s = polygon_from_points(slot)  # type: ignore[arg-type]
    else:
        s = bbox_to_polygon(list(slot))  # type: ignore[arg-type]
    v, s = _safe_polygon(v), _safe_polygon(s)
    if s.area <= 1e-12:
        return 0.0
    return intersection_area(v, s) / float(s.area)


def calculate_outside_ratio(
    vehicle: Union[BBox, Polygon],
    slot: Union[BBox, Polygon, Sequence[Point]],
) -> float:
    """R_outside = 1 - R_vehicle."""
    return 1.0 - calculate_vehicle_inside_ratio(vehicle, slot)


@dataclass
class ParkingSlot:
    """Stable parking-slot geometry with an identifier such as Slot-01."""

    slot_id: str
    polygon: Optional[List[Point]] = None
    bbox: Optional[BBox] = None  # [x1,y1,x2,y2]
    confidence: float = 1.0
    source_class: Optional[str] = None

    def __post_init__(self) -> None:
        if self.polygon is None and self.bbox is None:
            raise ValueError(f"{self.slot_id}: need polygon or bbox")
        if self.polygon is not None and len(self.polygon) < 3 and self.bbox is None:
            raise ValueError(f"{self.slot_id}: polygon needs >= 3 points")

    def as_polygon(self) -> Polygon:
        if self.polygon is not None and len(self.polygon) >= 3:
            return polygon_from_points(self.polygon)
        assert self.bbox is not None
        return bbox_to_polygon(self.bbox)

    def as_bbox(self) -> BBox:
        if self.bbox is not None:
            return list(self.bbox)
        poly = self.as_polygon()
        minx, miny, maxx, maxy = poly.bounds
        return [float(minx), float(miny), float(maxx), float(maxy)]


@dataclass
class VehicleDetection:
    vehicle_id: int
    bbox: BBox
    confidence: float = 1.0
    class_name: str = "car"


@dataclass
class OverlapMetrics:
    slot_id: str
    iou: float
    r_vehicle: float
    r_slot: float
    r_outside: float


@dataclass
class MultiSlotOverlap:
    primary: Optional[OverlapMetrics] = None
    all_overlaps: List[OverlapMetrics] = field(default_factory=list)
    adjacent_intrusions: List[OverlapMetrics] = field(default_factory=list)


def compute_overlap_metrics(vehicle: VehicleDetection, slot: ParkingSlot) -> OverlapMetrics:
    v = bbox_to_polygon(vehicle.bbox)
    s = slot.as_polygon()
    return OverlapMetrics(
        slot_id=slot.slot_id,
        iou=calculate_iou(v, s),
        r_vehicle=calculate_vehicle_inside_ratio(v, s),
        r_slot=calculate_slot_coverage_ratio(v, s),
        r_outside=calculate_outside_ratio(v, s),
    )


def assign_vehicle_to_slot(
    vehicle: VehicleDetection,
    slots: Sequence[ParkingSlot],
    assign_min: float = 0.30,
) -> MultiSlotOverlap:
    """Assign vehicle to the slot with highest R_vehicle among those meeting assign_min."""
    metrics = [compute_overlap_metrics(vehicle, slot) for slot in slots]
    metrics.sort(key=lambda m: m.r_vehicle, reverse=True)
    primary = metrics[0] if metrics and metrics[0].r_vehicle >= assign_min else None
    adjacent = []
    if primary is not None:
        for m in metrics[1:]:
            # Ignore infinitesimal edge-touch noise from adjacent rectangles.
            if m.r_vehicle > 1e-4 or m.r_slot > 1e-4:
                adjacent.append(m)
    return MultiSlotOverlap(primary=primary, all_overlaps=metrics, adjacent_intrusions=adjacent)


def yolo_norm_to_xyxy(cx: float, cy: float, bw: float, bh: float, width: int, height: int) -> BBox:
    x1 = (cx - bw / 2.0) * width
    y1 = (cy - bh / 2.0) * height
    x2 = (cx + bw / 2.0) * width
    y2 = (cy + bh / 2.0) * height
    return [x1, y1, x2, y2]


def xyxy_to_yolo_norm(bbox: BBox, width: int, height: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2.0) / max(width, 1)
    cy = ((y1 + y2) / 2.0) / max(height, 1)
    bw = (x2 - x1) / max(width, 1)
    bh = (y2 - y1) / max(height, 1)
    return cx, cy, bw, bh
