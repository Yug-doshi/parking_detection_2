"""Shared helpers for tiled object detection."""

from __future__ import annotations

from typing import Iterator, List, Sequence, Tuple

import numpy as np

from geometry import BBox

DetectionCandidate = Tuple[float, BBox, int]


def _tile_starts(length: int, tile_length: int, overlap: float) -> List[int]:
    if length <= tile_length:
        return [0]
    step = max(1, int(tile_length * (1.0 - overlap)))
    starts = list(range(0, length - tile_length + 1, step))
    if starts[-1] != length - tile_length:
        starts.append(length - tile_length)
    return starts


def iter_image_tiles(
    image: np.ndarray, tile_size: int, overlap: float
) -> Iterator[Tuple[int, int, np.ndarray]]:
    """Yield overlapping crops that cover the full image, including its far edges."""
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be between 0 and 1")

    height, width = image.shape[:2]
    tile_height = min(tile_size, height)
    tile_width = min(tile_size, width)
    for y in _tile_starts(height, tile_height, overlap):
        for x in _tile_starts(width, tile_width, overlap):
            yield x, y, image[y : y + tile_height, x : x + tile_width]


def touches_tile_edge(
    bbox: Sequence[float],
    tile_width: int,
    tile_height: int,
    margin_ratio: float,
) -> bool:
    """Identify likely partial detections clipped by a crop boundary."""
    margin = min(tile_width, tile_height) * margin_ratio
    x1, y1, x2, y2 = bbox
    return x1 <= margin or y1 <= margin or x2 >= tile_width - margin or y2 >= tile_height - margin


def suppress_duplicate_detections(
    candidates: Sequence[DetectionCandidate], iou_threshold: float
) -> List[DetectionCandidate]:
    """Greedily merge duplicate boxes from the full-frame and tiled passes."""
    kept: List[DetectionCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item[0], reverse=True):
        _, bbox, _ = candidate
        x1, y1, x2, y2 = bbox
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        duplicate = False
        for _, other, _ in kept:
            ox1, oy1, ox2, oy2 = other
            intersection = max(0.0, min(x2, ox2) - max(x1, ox1)) * max(
                0.0, min(y2, oy2) - max(y1, oy1)
            )
            other_area = max(0.0, ox2 - ox1) * max(0.0, oy2 - oy1)
            union = area + other_area - intersection
            if union > 0.0 and intersection / union > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept
