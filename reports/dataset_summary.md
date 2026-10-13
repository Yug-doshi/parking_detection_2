# Dataset Summary Report

**Root:** `D:\RMS 4\Parking_dataset (1)\Parking_dataset`

## Class mapping (preserved)

- Class `0`: space-empty — valid parking slot with no vehicle (free)
- Class `1`: space-occupied — valid parking slot containing a vehicle (occupancy, NOT correctness)

## Splits

- **train**: {'images': 8691, 'labels': 8691, 'class_counts': {'space-occupied': 231948, 'space-empty': 265908}, 'issues': {'empty_label': 189}}
- **valid**: {'images': 2483, 'labels': 2483, 'class_counts': {'space-occupied': 69687, 'space-empty': 73629}, 'issues': {'empty_label': 59}}
- **test**: {'images': 1242, 'labels': 1242, 'class_counts': {'space-occupied': 34100, 'space-empty': 36584}, 'issues': {'empty_label': 26}}

## Class distribution

- space-occupied: 335735 (47.16%)
- space-empty: 376121 (52.84%)

## Issues

```json
{
  "empty_label": 274
}
```

## Architecture recommendation

- **Approach_A_slot_detection**: SUPPORTED — train YOLOv8 on space-empty / space-occupied
- **Approach_B_vehicle_detection**: NOT directly supported — no vehicle boxes. Options: (1) remap occupied slots to a car class for approximate vehicle training; (2) use COCO-pretrained vehicle classes.
- **Approach_C_wrong_parking_geometry**: REQUIRED — no wrong-parking labels; use vehicle–slot overlap ratios.
- **what_dataset_trains_directly**: ['free parking slots', 'occupied parking slots']
- **what_needs_extra_info**: ['vehicle-level boxes or segmentation for accurate car footprint', 'manually verified wrong-parking labels for classification metrics', 'original PKLot rotated rectangles / polygons for angled slots']

## Limitations

- Labels are parking-slot occupancy states, not vehicle instances.
- An occupied slot is NOT the same as a correctly parked car.
- No wrong-parking class exists; wrong parking must use geometric overlap analysis.
- Axis-aligned boxes approximate angled PKLot slots — geometry has rectangular approximation error.
- Wrong-parking accuracy cannot be measured without a separately labeled evaluation set.
