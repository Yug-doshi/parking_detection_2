"""Inspect the actual PKLot / Roboflow dataset and write a summary report + GT visualizations."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visualization import draw_ground_truth_yolo

DEFAULT_DATASET = Path(r"D:\RMS 4\Parking_dataset (1)\Parking_dataset")
CLASS_NAMES = {0: "space-empty", 1: "space-occupied"}


def find_data_yaml(dataset_root: Path) -> Path:
    for cand in [dataset_root / "data.yaml", dataset_root.parent / "data.yaml"]:
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"data.yaml not found under {dataset_root}")


def parse_label_file(path: Path) -> List[Tuple[int, float, float, float, float]]:
    rows = []
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return rows
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(float(parts[0]))
        cx, cy, bw, bh = map(float, parts[1:5])
        rows.append((cid, cx, cy, bw, bh))
    return rows


def inspect_dataset(dataset_root: Path, out_dir: Path, sample_n: int = 12, seed: int = 42) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = find_data_yaml(dataset_root)
    with open(yaml_path, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    names = data_cfg.get("names", CLASS_NAMES)
    if isinstance(names, list):
        class_names = {i: n for i, n in enumerate(names)}
    else:
        class_names = {int(k): v for k, v in names.items()}

    report: Dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "data_yaml": str(yaml_path),
        "yaml_contents": data_cfg,
        "documented_class_mapping": {
            "0": "space-empty — valid parking slot with no vehicle (free)",
            "1": "space-occupied — valid parking slot containing a vehicle (occupancy, NOT correctness)",
        },
        "annotation_format": "YOLOv8 normalized bounding boxes (.txt); rectangular approximations of original PKLot rotated rectangles",
        "polygons_available": False,
        "vehicle_instance_annotations": False,
        "wrong_parking_class_present": False,
        "limitations": [
            "Labels are parking-slot occupancy states, not vehicle instances.",
            "An occupied slot is NOT the same as a correctly parked car.",
            "No wrong-parking class exists; wrong parking must use geometric overlap analysis.",
            "Axis-aligned boxes approximate angled PKLot slots — geometry has rectangular approximation error.",
            "Wrong-parking accuracy cannot be measured without a separately labeled evaluation set.",
        ],
        "splits": {},
        "class_counts": {},
        "issues": Counter(),
        "image_dimensions": Counter(),
        "image_formats": Counter(),
    }

    total_class = Counter()
    boxes_per_image: List[int] = []
    all_stems_by_split: Dict[str, set] = {}

    for split in ("train", "valid", "test"):
        img_dir = dataset_root / split / "images"
        lbl_dir = dataset_root / split / "labels"
        if not img_dir.is_dir():
            report["splits"][split] = {"missing": True}
            continue
        images = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}}
        labels = {p.stem: p for p in lbl_dir.iterdir() if p.suffix.lower() == ".txt"} if lbl_dir.is_dir() else {}
        all_stems_by_split[split] = set(images)
        split_classes = Counter()
        issues = Counter()

        for stem, ip in images.items():
            report["image_formats"][ip.suffix.lower()] += 1
            try:
                with Image.open(ip) as im:
                    im.verify()
                with Image.open(ip) as im:
                    w, h = im.size
                    report["image_dimensions"][f"{w}x{h}"] += 1
            except Exception:
                issues["corrupt_image"] += 1
                report["issues"]["corrupt_image"] += 1
                continue

            lp = labels.get(stem)
            if lp is None:
                issues["missing_label"] += 1
                report["issues"]["missing_label"] += 1
                continue
            rows = parse_label_file(lp)
            if not rows:
                issues["empty_label"] += 1
                report["issues"]["empty_label"] += 1
            boxes_per_image.append(len(rows))
            for cid, cx, cy, bw, bh in rows:
                if cid not in class_names:
                    issues["unknown_class"] += 1
                    report["issues"]["unknown_class"] += 1
                split_classes[cid] += 1
                total_class[cid] += 1
                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 <= bw <= 1.0 and 0.0 <= bh <= 1.0):
                    issues["invalid_normalized_coords"] += 1
                    report["issues"]["invalid_normalized_coords"] += 1
                x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
                if x1 < -0.02 or y1 < -0.02 or x2 > 1.02 or y2 > 1.02:
                    issues["bbox_out_of_bounds"] += 1
                    report["issues"]["bbox_out_of_bounds"] += 1

        for stem in labels:
            if stem not in images:
                issues["orphan_label"] += 1
                report["issues"]["orphan_label"] += 1

        report["splits"][split] = {
            "images": len(images),
            "labels": len(labels),
            "class_counts": {class_names.get(k, str(k)): v for k, v in split_classes.items()},
            "issues": dict(issues),
        }

    # Near-duplicate check across splits by stem basename prefix (date_time) — leak heuristic
    leak_pairs = []
    splits = list(all_stems_by_split)
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            inter = all_stems_by_split[splits[i]] & all_stems_by_split[splits[j]]
            if inter:
                leak_pairs.append({"splits": [splits[i], splits[j]], "identical_stems": len(inter)})
    report["cross_split_identical_stems"] = leak_pairs

    report["class_counts"] = {class_names.get(k, str(k)): v for k, v in total_class.items()}
    total_boxes = sum(total_class.values()) or 1
    report["class_balance_percent"] = {
        class_names.get(k, str(k)): round(100.0 * v / total_boxes, 2) for k, v in total_class.items()
    }
    report["boxes_per_image"] = {
        "min": min(boxes_per_image) if boxes_per_image else 0,
        "max": max(boxes_per_image) if boxes_per_image else 0,
        "avg": round(sum(boxes_per_image) / len(boxes_per_image), 2) if boxes_per_image else 0,
    }
    report["image_dimensions"] = dict(report["image_dimensions"])
    report["image_formats"] = dict(report["image_formats"])
    report["issues"] = dict(report["issues"])
    report["architecture_recommendation"] = {
        "Approach_A_slot_detection": "SUPPORTED — train YOLOv8 on space-empty / space-occupied",
        "Approach_B_vehicle_detection": "NOT directly supported — no vehicle boxes. Options: (1) remap occupied slots to a car class for approximate vehicle training; (2) use COCO-pretrained vehicle classes.",
        "Approach_C_wrong_parking_geometry": "REQUIRED — no wrong-parking labels; use vehicle–slot overlap ratios.",
        "what_dataset_trains_directly": ["free parking slots", "occupied parking slots"],
        "what_needs_extra_info": [
            "vehicle-level boxes or segmentation for accurate car footprint",
            "manually verified wrong-parking labels for classification metrics",
            "original PKLot rotated rectangles / polygons for angled slots",
        ],
    }

    # Visualizations
    vis_dir = out_dir / "gt_samples"
    vis_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    for split in ("train", "valid", "test"):
        img_dir = dataset_root / split / "images"
        lbl_dir = dataset_root / split / "labels"
        if not img_dir.is_dir():
            continue
        files = list(img_dir.glob("*.jpg"))
        sample = rng.sample(files, min(sample_n // 3 + 1, len(files)))
        for ip in sample:
            lp = lbl_dir / f"{ip.stem}.txt"
            img = cv2.imread(str(ip))
            if img is None:
                continue
            rows = parse_label_file(lp) if lp.is_file() else []
            vis = draw_ground_truth_yolo(img, rows, class_names)
            cv2.imwrite(str(vis_dir / f"{split}_{ip.name}"), vis)

    json_path = out_dir / "dataset_summary.json"
    md_path = out_dir / "dataset_summary.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"GT samples -> {vis_dir}")
    return report


def _to_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Dataset Summary Report",
        "",
        f"**Root:** `{report['dataset_root']}`",
        "",
        "## Class mapping (preserved)",
        "",
    ]
    for k, v in report["documented_class_mapping"].items():
        lines.append(f"- Class `{k}`: {v}")
    lines += ["", "## Splits", ""]
    for split, info in report["splits"].items():
        lines.append(f"- **{split}**: {info}")
    lines += ["", "## Class distribution", ""]
    for k, v in report["class_counts"].items():
        pct = report["class_balance_percent"].get(k, 0)
        lines.append(f"- {k}: {v} ({pct}%)")
    lines += ["", "## Issues", "", f"```json\n{json.dumps(report['issues'], indent=2)}\n```", ""]
    lines += ["## Architecture recommendation", ""]
    for k, v in report["architecture_recommendation"].items():
        lines.append(f"- **{k}**: {v}")
    lines += ["", "## Limitations", ""]
    for lim in report["limitations"]:
        lines.append(f"- {lim}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect parking dataset")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--out", type=str, default=str(ROOT / "reports"))
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    inspect_dataset(Path(args.dataset), Path(args.out), sample_n=args.samples, seed=args.seed)


if __name__ == "__main__":
    main()
