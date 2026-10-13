"""Draw GREEN free / BLUE correct / RED wrong annotations and dashboard overlay."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from parking_classifier import ParkingFrameResult, SlotOccupancyResult, VehicleParkingResult

# Exact project color rules (BGR for OpenCV)
COLOR_FREE = (0, 255, 0)       # GREEN
COLOR_CORRECT = (255, 0, 0)    # BLUE
COLOR_WRONG = (0, 0, 255)      # RED
COLOR_UNCERTAIN = (180, 180, 180)
COLOR_SLOT_OUTLINE = (200, 200, 50)
COLOR_TEXT = (255, 255, 255)
COLOR_PANEL = (30, 30, 30)


def _put_label(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    color: Tuple[int, int, int],
    enabled: bool = False,
) -> None:
    if not enabled:
        return
    x, y = org
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.45, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 4, y + 4), color, -1)
    cv2.putText(img, text, (x + 2, y - 2), font, scale, COLOR_TEXT, thickness, cv2.LINE_AA)


def _draw_slot(img: np.ndarray, slot_res: SlotOccupancyResult, thin_outline: bool = False, show_labels: bool = False) -> None:
    slot = slot_res.slot
    if slot_res.status == "FREE" and slot_res.usable_available:
        color = COLOR_FREE
        label = f"FREE SLOT | {slot.slot_id}"
        thickness = 2
    elif slot_res.status in ("OCCUPIED_WRONG", "PARTIALLY_OBSTRUCTED"):
        color = COLOR_WRONG
        label = f"{slot.slot_id} | obstructed"
        thickness = 1 if thin_outline else 2
    elif slot_res.status == "OCCUPIED_UNVERIFIED":
        color = COLOR_UNCERTAIN
        label = f"OCCUPIED | {slot.slot_id} | unverified"
        thickness = 1
    else:
        # Occupied correctly — thin outline only; vehicle box is primary BLUE signal
        color = COLOR_SLOT_OUTLINE
        label = f"{slot.slot_id}"
        thickness = 1

    pts = None
    if slot.polygon is not None and len(slot.polygon) >= 3:
        pts = np.array(slot.polygon, dtype=np.int32)
        cv2.polylines(img, [pts], True, color, thickness)
        org = (int(pts[:, 0].min()), int(pts[:, 1].min()) - 2)
    else:
        x1, y1, x2, y2 = map(int, slot.as_bbox())
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        org = (x1, y1 - 2)

    if slot_res.status in ("FREE", "OCCUPIED_UNVERIFIED") and show_labels:
        _put_label(img, label, (max(0, org[0]), max(14, org[1])), color, enabled=True)


def _draw_vehicle(img: np.ndarray, vr: VehicleParkingResult, show_labels: bool = False) -> None:
    x1, y1, x2, y2 = map(int, vr.vehicle.bbox)
    conf = vr.vehicle.confidence
    if vr.status == "CORRECT_PARKING":
        color = COLOR_CORRECT
        text = f"CORRECT PARKING | {vr.slot_id} | Inside: {vr.r_vehicle:.0%} | {conf:.2f}"
    elif vr.status == "WRONG_PARKING":
        color = COLOR_WRONG
        if vr.reason.startswith("straddles "):
            text = f"WRONG PARKING | Straddles {' / '.join(vr.overlapped_slots)}"
        elif len(vr.overlapped_slots) > 1:
            text = f"WRONG PARKING | Overlaps {', '.join(vr.overlapped_slots)} | {conf:.2f}"
        else:
            text = f"WRONG PARKING | Outside: {vr.r_outside:.0%} | {conf:.2f}"
    else:
        color = COLOR_UNCERTAIN
        text = f"UNCERTAIN | {conf:.2f}"

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if show_labels:
        _put_label(img, text, (x1, max(14, y1 - 2)), color, enabled=True)


def draw_dashboard(img: np.ndarray, summary: Dict[str, Any]) -> np.ndarray:
    lines = [
        f"Total slots: {summary.get('total_slots', 0)}",
        f"Free: {summary.get('free_slots', 0)}",
        f"Correct occupied: {summary.get('correctly_occupied_slots', 0)}",
        f"Occupied unverified: {summary.get('unverified_occupied_slots', 0)}",
        f"Wrong vehicles: {summary.get('wrong_parking_vehicles', 0)}",
        f"Uncertain: {summary.get('uncertain_detections', 0)}",
        f"Occupancy: {summary.get('occupancy_percent', 0)}%",
        f"Available: {summary.get('available_spaces', 0)}",
    ]
    legend = [
        "GREEN = FREE SLOT",
        "BLUE = CORRECT PARKING",
        "RED = WRONG PARKING",
    ]
    panel_h = 20 + 18 * (len(lines) + len(legend) + 1)
    panel_w = 280
    overlay = img.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), COLOR_PANEL, -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    y = 28
    for line in lines:
        cv2.putText(img, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)
        y += 18
    y += 4
    for line, col in zip(legend, [COLOR_FREE, COLOR_CORRECT, COLOR_WRONG]):
        cv2.putText(img, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
        y += 16
    return img


def draw_parking_annotations(
    image: np.ndarray,
    frame_result: ParkingFrameResult,
    draw_dashboard_overlay: bool = False,
    show_labels: bool = False,
) -> np.ndarray:
    out = image.copy()
    for sr in frame_result.slots:
        _draw_slot(out, sr, show_labels=show_labels)
    for vr in frame_result.vehicles:
        _draw_vehicle(out, vr, show_labels=show_labels)
    if draw_dashboard_overlay and show_labels:
        draw_dashboard(out, frame_result.summary)
    return out


def draw_vehicle_only_annotations(
    image: np.ndarray,
    frame_result: ParkingFrameResult,
    show_labels: bool = False,
) -> np.ndarray:
    """Draw vehicle classifications without slot boxes or free-space labels."""
    out = image.copy()
    for vehicle_result in frame_result.vehicles:
        _draw_vehicle(out, vehicle_result, show_labels=show_labels)
    return out


def draw_ground_truth_yolo(
    image: np.ndarray,
    labels: List[Tuple[int, float, float, float, float]],
    class_names: Dict[int, str],
    colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
) -> np.ndarray:
    """Visualize original YOLO ground-truth boxes (dataset inspection)."""
    out = image.copy()
    h, w = out.shape[:2]
    colors = colors or {0: COLOR_FREE, 1: (0, 165, 255)}  # empty=green, occupied=orange
    for cid, cx, cy, bw, bh in labels:
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        color = colors.get(cid, (255, 255, 255))
        name = class_names.get(cid, str(cid))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
        _put_label(out, name, (x1, max(14, y1 - 2)), color)
    return out
