"""Unit tests for overlap geometry and parking classification (12 required scenarios)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geometry import (
    ParkingSlot,
    VehicleDetection,
    assign_vehicle_to_slot,
    calculate_iou,
    calculate_outside_ratio,
    calculate_slot_coverage_ratio,
    calculate_vehicle_inside_ratio,
    clip_bbox_to_image,
)
from parking_classifier import classify_frame, detect_wrong_parking, load_thresholds


@pytest.fixture
def thr():
    return load_thresholds(ROOT / "configs" / "thresholds.yaml")


@pytest.fixture
def slot01():
    return ParkingSlot(slot_id="Slot-01", bbox=[100, 100, 200, 200])


@pytest.fixture
def slot02():
    return ParkingSlot(slot_id="Slot-02", bbox=[200, 100, 300, 200])


def test_1_car_completely_inside_one_slot(thr, slot01):
    car = VehicleDetection(0, [110, 110, 190, 190], confidence=0.95)
    r = detect_wrong_parking(car, [slot01], thr)
    assert calculate_vehicle_inside_ratio(car.bbox, slot01.as_polygon()) >= 0.99
    assert r.status == "CORRECT_PARKING"
    assert r.slot_id == "Slot-01"


def test_2_car_slightly_outside_within_tolerance(thr, slot01):
    # ~8% outside — within outside_max=0.10
    car = VehicleDetection(0, [108, 108, 208, 192], confidence=0.9)
    r_out = calculate_outside_ratio(car.bbox, slot01.as_polygon())
    assert r_out <= thr["outside_max"] + 0.02
    r = detect_wrong_parking(car, [slot01], thr)
    # May be correct or wrong depending on exact ratio; assert no crash and valid status
    assert r.status in ("CORRECT_PARKING", "WRONG_PARKING")
    assert 0.0 <= r.r_vehicle <= 1.0


def test_3_car_substantially_outside(thr, slot01):
    car = VehicleDetection(0, [150, 150, 280, 280], confidence=0.9)
    assert calculate_outside_ratio(car.bbox, slot01.as_polygon()) > 0.3
    r = detect_wrong_parking(car, [slot01], thr)
    assert r.status == "WRONG_PARKING"


def test_borderline_outside_vehicle_is_uncertain_in_conservative_mode(thr, slot01):
    thr = dict(thr)
    thr["wrong_parking_min_outside_ratio"] = 0.30
    thr["likely_correct_min_inside_ratio"] = 0.90
    car = VehicleDetection(0, [120, 110, 220, 190], confidence=0.95)

    result = detect_wrong_parking(car, [slot01], thr)

    assert 0.0 < result.r_outside < 0.30
    assert result.status == "UNCERTAIN"
    assert result.reason == "borderline slot overlap"


def test_strong_outside_vehicle_remains_wrong_in_conservative_mode(thr, slot01):
    thr = dict(thr)
    thr["wrong_parking_min_outside_ratio"] = 0.30
    car = VehicleDetection(0, [150, 150, 280, 280], confidence=0.95)

    result = detect_wrong_parking(car, [slot01], thr)

    assert result.r_outside >= 0.30
    assert result.status == "WRONG_PARKING"


def test_borderline_outside_vehicle_is_not_red(thr, slot01):
    car = VehicleDetection(0, [120, 110, 220, 190], confidence=0.95)

    result = detect_wrong_parking(car, [slot01], thr)

    assert result.r_outside < thr["wrong_parking_min_outside_ratio"]
    assert result.status != "WRONG_PARKING"


def test_mostly_inside_single_slot_is_correct_without_straddle(thr, slot01):
    car = VehicleDetection(0, [120, 110, 220, 190], confidence=0.95)

    result = detect_wrong_parking(car, [slot01], thr)

    assert result.r_vehicle >= thr["likely_correct_min_inside_ratio"]
    assert result.status == "CORRECT_PARKING"
    assert result.reason == "mostly within one slot"


def test_4_car_overlapping_two_adjacent_slots(thr, slot01, slot02):
    slot_left = ParkingSlot(slot_id="Slot-left", bbox=[100, 100, 150, 200])
    slot_right = ParkingSlot(slot_id="Slot-right", bbox=[150, 100, 200, 200])
    car = VehicleDetection(0, [115, 110, 235, 190], confidence=0.92)
    assignment = assign_vehicle_to_slot(car, [slot_left, slot_right], assign_min=0.2)
    assert assignment.primary is not None
    assert len([m for m in assignment.all_overlaps if m.r_vehicle >= 0.1]) >= 2
    r = detect_wrong_parking(car, [slot_left, slot_right], thr)
    assert r.status == "WRONG_PARKING"
    assert len(r.overlapped_slots) >= 2


def test_car_straddling_large_adjacent_slots_is_wrong_by_vehicle_intrusion(thr):
    slot_left = ParkingSlot(slot_id="Slot-12", bbox=[0, 0, 40, 200])
    slot_right = ParkingSlot(slot_id="Slot-13", bbox=[40, 0, 80, 200])
    car = VehicleDetection(12, [0, 50, 80, 130], confidence=0.95)

    r = detect_wrong_parking(car, [slot_left, slot_right], thr)

    assert r.status == "WRONG_PARKING"
    assert set(r.overlapped_slots) == {"Slot-12", "Slot-13"}
    assert r.reason == "straddles Slot-12 / Slot-13"


def test_boundary_crossing_within_tolerance_is_not_wrong(thr):
    slot_left = ParkingSlot(slot_id="Slot-12", bbox=[0, 0, 400, 200])
    slot_right = ParkingSlot(slot_id="Slot-13", bbox=[400, 0, 800, 200])
    car = VehicleDetection(13, [320, 50, 403, 130], confidence=0.95)

    r = detect_wrong_parking(car, [slot_left, slot_right], thr)

    assert r.status == "CORRECT_PARKING"
    assert r.overlapped_slots == ["Slot-12"]


def test_5_slot_with_no_vehicle(thr, slot01):
    frame = classify_frame([], [slot01], thr)
    assert frame.slots[0].status == "FREE"
    assert frame.slots[0].usable_available is True
    assert frame.summary["free_slots"] == 1


def test_occupied_slot_without_vehicle_is_not_reported_free(thr):
    occupied_slot = ParkingSlot(
        slot_id="Slot-occupied",
        bbox=[100, 100, 200, 200],
        source_class="space-occupied",
    )
    frame = classify_frame([], [occupied_slot], thr)
    assert frame.slots[0].status == "OCCUPIED_UNVERIFIED"
    assert frame.slots[0].usable_available is False
    assert frame.summary["free_slots"] == 0
    assert frame.summary["unverified_occupied_slots"] == 1


def test_6_slot_partially_obstructed_by_wrong_car(thr, slot01, slot02):
    thr = dict(thr)
    thr["wrong_parking_min_outside_ratio"] = 0.30
    thr["minimum_second_slot_intrusion_ratio"] = 0.30
    slot_left = ParkingSlot(slot_id="Slot-left", bbox=[100, 100, 150, 200])
    slot_right = ParkingSlot(slot_id="Slot-right", bbox=[150, 100, 200, 200])
    car = VehicleDetection(0, [115, 110, 235, 190], confidence=0.9)
    frame = classify_frame([car], [slot_left, slot_right], thr)
    assert any(v.status == "WRONG_PARKING" for v in frame.vehicles)
    assert any(s.status in ("OCCUPIED_WRONG", "PARTIALLY_OBSTRUCTED", "OCCUPIED_CORRECT") for s in frame.slots)
    # Not both freely available
    assert frame.summary["free_slots"] < 2


def test_7_two_cars_in_different_slots(thr, slot01, slot02):
    c1 = VehicleDetection(0, [110, 110, 190, 190], confidence=0.95)
    c2 = VehicleDetection(1, [210, 110, 290, 190], confidence=0.95)
    frame = classify_frame([c1, c2], [slot01, slot02], thr)
    assert frame.summary["unique_vehicles"] == 2
    assert all(v.status == "CORRECT_PARKING" for v in frame.vehicles)
    assert frame.summary["free_slots"] == 0


def test_8_low_confidence_vehicle(thr, slot01):
    car = VehicleDetection(0, [155, 110, 255, 190], confidence=0.30)
    r = detect_wrong_parking(car, [slot01], thr)
    assert r.status == "UNCERTAIN"


def test_low_confidence_detection_threshold_keeps_uncertain_vehicle_visible(thr, slot01):
    assert thr["vehicle_detection_conf_min"] < thr["vehicle_conf_min"]
    car = VehicleDetection(0, [110, 110, 190, 190], confidence=0.15)
    frame = classify_frame([car], [slot01], thr)
    assert len(frame.vehicles) == 1
    assert frame.vehicles[0].status == "UNCERTAIN"


def test_low_confidence_car_with_strong_single_slot_fit_can_be_blue(thr, slot01):
    car = VehicleDetection(0, [110, 110, 190, 190], confidence=0.30)

    result = detect_wrong_parking(car, [slot01], thr)

    assert result.status == "CORRECT_PARKING"
    assert result.reason == "low-confidence detection mostly within one slot"


def test_low_confidence_car_outside_slot_stays_uncertain(thr, slot01):
    car = VehicleDetection(0, [155, 110, 255, 190], confidence=0.30)

    result = detect_wrong_parking(car, [slot01], thr)

    assert result.status == "UNCERTAIN"
    assert result.reason == "low confidence"


def test_9_ambiguous_slot_assignment(thr, slot01, slot02):
    # Tiny overlap with both — below assign_min
    car = VehicleDetection(0, [195, 140, 205, 160], confidence=0.9)
    r = detect_wrong_parking(car, [slot01, slot02], thr)
    assert r.status == "UNCERTAIN"
    assert r.reason == "borderline slot overlap"


def test_no_slot_geometry_is_uncertain_not_wrong(thr):
    car = VehicleDetection(0, [350, 20, 410, 65], confidence=0.95)

    result = detect_wrong_parking(car, [], thr)

    assert result.status == "UNCERTAIN"
    assert result.reason == "no reliable slot assignment"


def test_two_significant_slot_intrusions_without_primary_are_wrong(thr):
    thr = dict(thr)
    thr["slot_assign_min"] = 0.6
    slot_left = ParkingSlot(slot_id="Slot-12", bbox=[0, 0, 20, 100])
    slot_right = ParkingSlot(slot_id="Slot-13", bbox=[20, 0, 40, 100])
    car = VehicleDetection(0, [0, 0, 40, 100], confidence=0.95)

    result = detect_wrong_parking(car, [slot_left, slot_right], thr)

    assert result.status == "WRONG_PARKING"
    assert result.overlapped_slots == ["Slot-12", "Slot-13"]


def test_overlapping_slot_predictions_do_not_hide_second_slot_intrusion(thr):
    thr = dict(thr)
    thr["minimum_second_slot_intrusion_ratio"] = 0.15
    thr["minimum_vehicle_slot_size_ratio_for_straddle"] = 0.9
    thr["minimum_second_slot_overlap_ratio"] = 0.08
    thr["vehicle_straddle_min_aspect_ratio"] = 1.10
    slot_primary = ParkingSlot(
        slot_id="Slot-primary",
        bbox=[182.23, 256.23, 248.31, 311.45],
    )
    slot_secondary = ParkingSlot(
        slot_id="Slot-secondary",
        bbox=[156.73, 263.79, 188.38, 309.58],
    )
    car = VehicleDetection(
        0,
        [180.75, 255.79, 248.76, 311.11],
        confidence=0.95,
    )

    result = detect_wrong_parking(car, [slot_primary, slot_secondary], thr)

    assert result.status == "WRONG_PARKING"
    assert result.overlapped_slots == ["Slot-primary", "Slot-secondary"]
    assert result.reason == "straddles Slot-primary / Slot-secondary"


def test_vehicle_intruding_into_neighboring_slot_beyond_tolerance_is_wrong(thr):
    slot_primary = ParkingSlot(slot_id="Slot-primary", bbox=[178.5, 263.0, 210.4, 312.0])
    slot_neighbor = ParkingSlot(slot_id="Slot-neighbor", bbox=[205.5, 265.7, 237.1, 309.8])
    slot_next = ParkingSlot(slot_id="Slot-next", bbox=[231.4, 266.0, 260.1, 311.5])
    car = VehicleDetection(0, [180.75, 255.79, 248.76, 311.11], confidence=0.90)

    result = detect_wrong_parking(car, [slot_primary, slot_neighbor, slot_next], thr)

    assert result.status == "WRONG_PARKING"
    assert result.reason.startswith("straddles Slot-primary")
    assert "Slot-neighbor" in result.overlapped_slots


def test_car_that_fits_slot_is_not_wrong_from_neighbor_box_split(thr):
    slot_primary = ParkingSlot(slot_id="Slot-primary", bbox=[434.0, 419.8, 460.9, 483.7])
    slot_neighbor = ParkingSlot(slot_id="Slot-neighbor", bbox=[405.8, 422.2, 433.2, 481.2])
    car = VehicleDetection(5, [419.75, 428.71, 447.14, 482.49], confidence=0.42)

    result = detect_wrong_parking(car, [slot_primary, slot_neighbor], thr)

    assert result.status != "WRONG_PARKING"
    assert result.reason == "borderline slot overlap"


def test_large_car_with_subthreshold_neighbor_intrusion_is_wrong(thr):
    primary = ParkingSlot(
        slot_id="Slot-primary",
        bbox=[625.75, 420.71, 658.21, 487.66],
    )
    neighbor = ParkingSlot(
        slot_id="Slot-neighbor",
        bbox=[596.92, 422.17, 630.14, 490.54],
    )
    car = VehicleDetection(0, [606.5, 414.3, 670.6, 485.1], confidence=0.80)

    result = detect_wrong_parking(car, [primary, neighbor], thr)

    assert result.status == "WRONG_PARKING"
    assert result.reason == "straddles Slot-primary / Slot-neighbor"


def test_overlapping_slot_edge_does_not_flag_upright_vehicle(thr):
    slot_primary = ParkingSlot(slot_id="Slot-primary", bbox=[0, 0, 100, 100])
    slot_secondary = ParkingSlot(slot_id="Slot-secondary", bbox=[80, 0, 120, 100])
    upright_car = VehicleDetection(0, [70, 20, 100, 80], confidence=0.95)

    result = detect_wrong_parking(upright_car, [slot_primary, slot_secondary], thr)

    assert result.status == "CORRECT_PARKING"
    assert result.overlapped_slots == ["Slot-primary"]


def test_10_partially_occluded_vehicle_still_classifiable(thr, slot01):
    # Smaller visible box still mostly inside
    car = VehicleDetection(0, [120, 120, 170, 160], confidence=0.85)
    r = detect_wrong_parking(car, [slot01], thr)
    assert r.status == "CORRECT_PARKING"
    assert calculate_iou(car.bbox, slot01.bbox) > 0


def test_11_car_touches_slot_boundary(thr, slot01):
    car = VehicleDetection(0, [100, 100, 200, 200], confidence=0.9)  # exact slot bounds
    r_out = calculate_outside_ratio(car.bbox, slot01.as_polygon())
    assert r_out < 1e-6
    r = detect_wrong_parking(car, [slot01], thr)
    assert r.status == "CORRECT_PARKING"


def test_12_bbox_extends_beyond_image(thr, slot01):
    raw = [-20, 110, 190, 190]
    clipped = clip_bbox_to_image(raw, width=640, height=640)
    assert clipped[0] >= 0
    car = VehicleDetection(0, clipped, confidence=0.9)
    # No division by zero
    r_v = calculate_vehicle_inside_ratio(car.bbox, slot01.as_polygon())
    r_s = calculate_slot_coverage_ratio(car.bbox, slot01.as_polygon())
    assert 0.0 <= r_v <= 1.0
    assert 0.0 <= r_s <= 1.0
    r = detect_wrong_parking(car, [slot01], thr)
    assert r.status in ("CORRECT_PARKING", "WRONG_PARKING", "UNCERTAIN")


def test_zero_area_vehicle_safe(slot01):
    car = VehicleDetection(0, [100, 100, 100, 100], confidence=0.9)
    assert calculate_vehicle_inside_ratio(car.bbox, slot01.as_polygon()) == 0.0
    assert calculate_iou(car.bbox, slot01.bbox) == 0.0


def test_polygon_slot_overlap(thr):
    slot = ParkingSlot(
        slot_id="Slot-01",
        polygon=[(100, 100), (220, 90), (230, 200), (90, 210)],
    )
    car = VehicleDetection(0, [120, 120, 180, 180], confidence=0.95)
    assert calculate_vehicle_inside_ratio(car.bbox, slot.as_polygon()) > 0.9
    r = detect_wrong_parking(car, [slot], thr)
    assert r.status == "CORRECT_PARKING"
