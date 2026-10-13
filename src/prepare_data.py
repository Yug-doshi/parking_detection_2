"""Prepare YOLO datasets without modifying the original Roboflow export.

Creates:
  dataset/space/  — original classes 0=space-empty, 1=space-occupied (symlinks/copies)
  dataset/car/    — occupied slots remapped to class 0=car (approximation for vehicle training)

Also writes data.yaml files with absolute paths suitable for Ultralytics.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Set

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = Path(r"D:\RMS 4\Parking_dataset (1)\Parking_dataset")


def _link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        try:
            dst.symlink_to(src)
        except OSError:
            shutil.copy2(src, dst)


def _subsample(stems: List[str], max_n: Optional[int], seed: int) -> List[str]:
    if max_n is None or max_n >= len(stems):
        return stems
    rng = random.Random(seed)
    return sorted(rng.sample(stems, max_n))


def prepare_space_dataset(
    src: Path,
    dst: Path,
    max_per_split: Optional[int] = None,
    seed: int = 42,
    copy_files: bool = False,
) -> None:
    """Keep original class IDs: 0=space-empty, 1=space-occupied."""
    for split in ("train", "valid", "test"):
        img_src = src / split / "images"
        lbl_src = src / split / "labels"
        stems = sorted({p.stem for p in img_src.glob("*.jpg")})
        stems = _subsample(stems, max_per_split if split == "train" else None, seed)
        # Always keep all valid/test unless max set for all
        if max_per_split is not None and split != "train":
            # Cap val/test proportionally for CPU experiments
            cap = max(50, max_per_split // 4)
            stems = _subsample(sorted({p.stem for p in img_src.glob("*.jpg")}), cap, seed + hash(split) % 1000)

        for stem in stems:
            ip = img_src / f"{stem}.jpg"
            lp = lbl_src / f"{stem}.txt"
            if not ip.is_file():
                continue
            _link_or_copy(ip, dst / split / "images" / ip.name, copy_files)
            if lp.is_file():
                _link_or_copy(lp, dst / split / "labels" / lp.name, copy_files)
            else:
                (dst / split / "labels" / f"{stem}.txt").write_text("", encoding="utf-8")

    yaml_path = dst / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {dst.resolve().as_posix()}",
                "train: train/images",
                "val: valid/images",
                "test: test/images",
                "nc: 2",
                "names: ['space-empty', 'space-occupied']",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Space dataset -> {dst}")
    print(f"Config -> {yaml_path}")


def prepare_car_dataset(
    src: Path,
    dst: Path,
    occupied_class: int = 1,
    max_per_split: Optional[int] = None,
    seed: int = 42,
    copy_files: bool = False,
) -> None:
    """Remap occupied slots -> class 0 (car). Empty slots are dropped (not vehicles)."""
    for split in ("train", "valid", "test"):
        img_src = src / split / "images"
        lbl_src = src / split / "labels"
        stems = sorted({p.stem for p in img_src.glob("*.jpg")})
        if max_per_split is not None:
            cap = max_per_split if split == "train" else max(50, max_per_split // 4)
            stems = _subsample(stems, cap, seed + hash(split) % 1000)

        for stem in stems:
            ip = img_src / f"{stem}.jpg"
            lp = lbl_src / f"{stem}.txt"
            if not ip.is_file():
                continue
            _link_or_copy(ip, dst / split / "images" / ip.name, copy_files)
            out_lines: List[str] = []
            if lp.is_file():
                for line in lp.read_text(encoding="utf-8", errors="ignore").splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cid = int(float(parts[0]))
                    if cid != occupied_class:
                        continue
                    out_lines.append("0 " + " ".join(parts[1:5]))
            (dst / split / "labels").mkdir(parents=True, exist_ok=True)
            (dst / split / "labels" / f"{stem}.txt").write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")

    yaml_path = dst / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {dst.resolve().as_posix()}",
                "train: train/images",
                "val: valid/images",
                "test: test/images",
                "nc: 1",
                "names: ['car']",
                "",
                "# NOTE: car boxes are remapped from space-occupied slot boxes (approximation).",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Car dataset -> {dst}")
    print(f"Config -> {yaml_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=str, default=str(DEFAULT_SRC))
    p.add_argument("--out", type=str, default=str(ROOT / "dataset"))
    p.add_argument("--max-train", type=int, default=800, help="Subsample train images (CPU-friendly). Use 0 for all.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--copy", action="store_true", help="Copy files instead of symlink")
    args = p.parse_args()
    src = Path(args.src)
    out = Path(args.out)
    max_train = None if args.max_train == 0 else args.max_train
    prepare_space_dataset(src, out / "space", max_per_split=max_train, seed=args.seed, copy_files=args.copy)
    prepare_car_dataset(src, out / "car", max_per_split=max_train, seed=args.seed, copy_files=args.copy)
    # Also write project-level data.yaml pointing to space (primary slot task)
    (ROOT / "data.yaml").write_text((out / "space" / "data.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    print("Done.")


if __name__ == "__main__":
    main()
