"""Train YOLOv8 on prepared parking datasets (space or car)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def detect_device(requested: str) -> str:
    if requested and requested != "auto":
        return requested
    return "0" if torch.cuda.is_available() else "cpu"


def train(
    data_yaml: Path,
    model: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    project: Path,
    name: str,
    seed: int,
    lr0: float,
    optimizer: str,
    patience: int,
    workers: int,
) -> Path:
    device = detect_device(device)
    print(f"Device: {device} | CUDA available: {torch.cuda.is_available()}")
    print(f"Data: {data_yaml}")
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Missing dataset YAML: {data_yaml}")

    yolo = YOLO(model)
    results = yolo.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(project),
        name=name,
        seed=seed,
        lr0=lr0,
        optimizer=optimizer,
        patience=patience,
        workers=workers,
        exist_ok=True,
    )
    best = Path(project) / name / "weights" / "best.pt"
    last = Path(project) / name / "weights" / "last.pt"
    meta = {
        "data": str(data_yaml),
        "model": model,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "best_weights": str(best) if best.is_file() else None,
        "last_weights": str(last) if last.is_file() else None,
        "results_dir": str(Path(project) / name),
    }
    out_meta = Path(project) / name / "train_meta.json"
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    if not best.is_file():
        raise RuntimeError("Training finished but best.pt was not found.")
    # Copy best into project weights/
    dest = ROOT / "weights" / f"{name}_best.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(best.read_bytes())
    print(f"Copied best weights -> {dest}")
    return best


def main() -> None:
    default_cfg = ROOT / "configs" / "train.yaml"
    cfg = {}
    if default_cfg.is_file():
        with open(default_cfg, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    p = argparse.ArgumentParser(description="Train YOLOv8 parking models")
    p.add_argument("--data", type=str, default=str(ROOT / "dataset" / "space" / "data.yaml"))
    p.add_argument("--model", type=str, default=cfg.get("model", "yolov8n.pt"))
    p.add_argument("--epochs", type=int, default=int(cfg.get("epochs", 30)))
    p.add_argument("--imgsz", type=int, default=int(cfg.get("imgsz", 640)))
    p.add_argument("--batch", type=int, default=int(cfg.get("batch", 8)))
    p.add_argument("--device", type=str, default=str(cfg.get("device", "auto")))
    p.add_argument("--project", type=str, default=str(ROOT / "runs"))
    p.add_argument("--name", type=str, default=str(cfg.get("name", "slot_det")))
    p.add_argument("--seed", type=int, default=int(cfg.get("seed", 42)))
    p.add_argument("--lr0", type=float, default=float(cfg.get("lr0", 0.01)))
    p.add_argument("--optimizer", type=str, default=str(cfg.get("optimizer", "auto")))
    p.add_argument("--patience", type=int, default=int(cfg.get("patience", 10)))
    p.add_argument("--workers", type=int, default=int(cfg.get("workers", 2)))
    args = p.parse_args()

    train(
        data_yaml=Path(args.data),
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=Path(args.project),
        name=args.name,
        seed=args.seed,
        lr0=args.lr0,
        optimizer=args.optimizer,
        patience=args.patience,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
