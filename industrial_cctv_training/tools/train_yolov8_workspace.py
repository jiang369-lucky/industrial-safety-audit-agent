# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
from pathlib import Path


WORKSPACE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKSPACE_DIR.parent
DATASET_DIR = WORKSPACE_DIR / "dataset"
RUNS_DIR = WORKSPACE_DIR / "runs"
MODEL_DIR = REPO_ROOT / "model"

LOCAL_YOLO_CONFIG = WORKSPACE_DIR / ".ultralytics"
LOCAL_YOLO_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(LOCAL_YOLO_CONFIG))
os.environ.setdefault("YOLO_VERBOSE", "True")

from ultralytics import YOLO  # noqa: E402


ROUTER_EXPERIMENTS = {
    "router_raw_no_aug": {
        "data": DATASET_DIR / "dataset_gate",
        "name": "router_exp1_raw_no_aug",
        "augment": False,
    },
    "router_raw_yolo_aug": {
        "data": DATASET_DIR / "dataset_gate",
        "name": "router_exp2_raw_yolo_aug",
        "augment": True,
    },
    "router_dataset_aug_no_aug": {
        "data": DATASET_DIR / "dataset_gate_augmented",
        "name": "router_exp3_dataset_aug_no_aug",
        "augment": False,
    },
    "router_dataset_aug_yolo_aug": {
        "data": DATASET_DIR / "dataset_gate_augmented",
        "name": "router_exp4_dataset_aug_yolo_aug",
        "augment": True,
    },
}


def remove_yolo_caches(root: Path) -> int:
    count = 0
    for cache_path in root.rglob("*.cache"):
        cache_path.unlink()
        count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 models inside industrial_cctv_training workspace.")
    parser.add_argument(
        "--target",
        choices=[
            "all",
            "router_all",
            "router_raw_no_aug",
            "router_raw_yolo_aug",
            "router_dataset_aug_no_aug",
            "router_dataset_aug_yolo_aug",
            "detect_all",
            "weld",
            "cut",
        ],
        default="all",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--router-epochs", type=int, default=150)
    parser.add_argument("--detect-epochs", type=int, default=100)
    parser.add_argument("--router-batch", type=int, default=32)
    parser.add_argument("--detect-batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--router-cache", action="store_true", help="Cache classification images. Faster, but may hang on some Windows setups.")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def ensure_detection_yaml() -> tuple[Path, Path]:
    weld_yaml = WORKSPACE_DIR / "weld_detect.yaml"
    cut_yaml = WORKSPACE_DIR / "cut_detect.yaml"
    weld_yaml.write_text(
        "\n".join(
            [
                f"path: {str((DATASET_DIR / 'dataset_detect' / 'welding').resolve()).replace(os.sep, '/')}",
                "train: images/train",
                "val: images/val",
                "nc: 3",
                "names:",
                "  0: collector",
                "  1: worker",
                "  2: mask",
                "",
            ]
        ),
        encoding="utf-8",
    )
    cut_yaml.write_text(
        "\n".join(
            [
                f"path: {str((DATASET_DIR / 'dataset_detect' / 'cutting').resolve()).replace(os.sep, '/')}",
                "train: images/train",
                "val: images/val",
                "nc: 2",
                "names:",
                "  0: worker",
                "  1: extinguisher",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return weld_yaml, cut_yaml


def train_router_one(key: str, args: argparse.Namespace) -> None:
    spec = ROUTER_EXPERIMENTS[key]
    data_dir = Path(spec["data"])
    if not data_dir.exists():
        raise FileNotFoundError(f"找不到场景分类数据集: {data_dir}")

    removed = remove_yolo_caches(data_dir)
    if removed:
        print(f"[缓存清理] 已删除 {data_dir} 下的 {removed} 个 YOLO cache 文件")

    model = YOLO(str(MODEL_DIR / "yolov8n-cls.pt"))
    common = dict(
        data=str(data_dir),
        epochs=args.router_epochs,
        imgsz=224,
        batch=args.router_batch,
        project=str(RUNS_DIR / "classify" / "router"),
        name=str(spec["name"]),
        exist_ok=args.exist_ok,
        device=args.device,
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        warmup_epochs=5,
        augment=bool(spec["augment"]),
        dropout=0.2,
        weight_decay=0.0005,
        patience=30,
        workers=args.workers,
        cache=args.router_cache,
        plots=True,
        verbose=True,
    )
    if spec["augment"]:
        common.update(
            hsv_h=0.01,
            hsv_s=0.4,
            hsv_v=0.3,
            degrees=5.0,
            translate=0.05,
            scale=0.2,
            fliplr=0.2,
            mosaic=0.2,
            mixup=0.1,
        )
    print(f"\n[场景分类] {spec['name']} | data={data_dir} | YOLO增强={spec['augment']}")
    model.train(**common)


def train_detector(yaml_path: Path, name: str, args: argparse.Namespace) -> None:
    if not yaml_path.exists():
        raise FileNotFoundError(f"找不到检测 YAML: {yaml_path}")
    # YAML path points to a dataset root through its path field; clear all workspace
    # caches to avoid stale labels/images after in-place degradation.
    removed = remove_yolo_caches(DATASET_DIR / "dataset_detect")
    if removed:
        print(f"[缓存清理] 已删除检测数据集下的 {removed} 个 YOLO cache 文件")

    model = YOLO(str(MODEL_DIR / "yolov8n.pt"))
    print(f"\n[检测器] {name} | data={yaml_path}")
    model.train(
        data=str(yaml_path),
        epochs=args.detect_epochs,
        imgsz=640,
        batch=args.detect_batch,
        project=str(RUNS_DIR / "detect"),
        name=name,
        exist_ok=args.exist_ok,
        device=args.device,
        patience=100,
        workers=0,
        amp=False,
        plots=True,
    )


def main() -> None:
    args = parse_args()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    router_targets = []
    if args.target in ("all", "router_all"):
        router_targets = list(ROUTER_EXPERIMENTS.keys())
    elif args.target in ROUTER_EXPERIMENTS:
        router_targets = [args.target]

    for key in router_targets:
        train_router_one(key, args)

    if args.target in ("all", "detect_all", "weld", "cut"):
        weld_yaml, cut_yaml = ensure_detection_yaml()
        if args.target in ("all", "detect_all", "weld"):
            train_detector(weld_yaml, "yolo_weld_expert", args)
        if args.target in ("all", "detect_all", "cut"):
            train_detector(cut_yaml, "yolo_cut_expert", args)

    print("\n训练流程结束。结果位于:")
    print(RUNS_DIR)


if __name__ == "__main__":
    main()
