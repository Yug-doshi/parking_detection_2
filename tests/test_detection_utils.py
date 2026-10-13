from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from detection_utils import iter_image_tiles, suppress_duplicate_detections, touches_tile_edge


def test_tiles_cover_short_and_far_image_edges():
    image = np.zeros((145, 542, 3), dtype=np.uint8)

    tiles = list(iter_image_tiles(image, tile_size=256, overlap=0.2))

    assert tiles[0][0:2] == (0, 0)
    assert tiles[-1][0] + tiles[-1][2].shape[1] == image.shape[1]
    assert all(tile.shape[0] == image.shape[0] for _, _, tile in tiles)


def test_partial_detections_near_tile_edges_are_rejected():
    assert touches_tile_edge([0, 20, 30, 50], 256, 256, 0.04)
    assert not touches_tile_edge([40, 30, 120, 100], 256, 256, 0.04)


def test_duplicate_boxes_from_overlapping_tiles_are_suppressed():
    kept = suppress_duplicate_detections(
        [
            (0.7, [10, 10, 50, 50], 2),
            (0.6, [11, 11, 51, 51], 2),
            (0.8, [80, 10, 120, 50], 2),
        ],
        iou_threshold=0.5,
    )

    assert len(kept) == 2
    assert kept[0][0] == 0.8


def test_duplicate_box_at_exact_threshold_is_suppressed_and_classes_stay_separate():
    kept = suppress_duplicate_detections(
        [
            (0.9, [0, 0, 100, 100], 2),
            (0.8, [50, 0, 150, 100], 2),
            (0.7, [0, 0, 100, 100], 3),
        ],
        iou_threshold=1 / 3,
    )

    assert len(kept) == 2
    assert {item[2] for item in kept} == {2, 3}
