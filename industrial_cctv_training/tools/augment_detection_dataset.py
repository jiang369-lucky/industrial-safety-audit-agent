# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TASKS = ("welding", "cutting")
VARIANTS = ("cctv_lowres", "dust_haze", "smoke_compression")


def read_image(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_jpg(path: Path, image: np.ndarray, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    buffer.tofile(str(path))


def clip_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).astype(np.uint8)


def adjust_brightness_contrast(image: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    return clip_uint8(image.astype(np.float32) * alpha + beta)


def add_sensor_noise(image: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
    noise = rng.normal(0.0, sigma, image.shape).astype(np.float32)
    return clip_uint8(image.astype(np.float32) + noise)


def jpeg_roundtrip(image: np.ndarray, quality: int) -> np.ndarray:
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return image
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return image if decoded is None else decoded


def smooth_noise_mask(shape: tuple[int, int], rng: np.random.Generator, grid: int = 48) -> np.ndarray:
    h, w = shape
    small_h = max(4, h // grid)
    small_w = max(4, w // grid)
    base = rng.random((small_h, small_w)).astype(np.float32)
    mask = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(w, h) / 28.0)
    mask -= mask.min()
    denom = float(mask.max() - mask.min())
    if denom > 1e-6:
        mask /= denom
    return mask[..., None]


def cctv_lowres(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = image.shape[:2]
    scale = float(rng.uniform(0.48, 0.70))
    small = cv2.resize(image, (max(8, int(w * scale)), max(8, int(h * scale))), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    restored = cv2.GaussianBlur(restored, (3, 3), sigmaX=float(rng.uniform(0.25, 0.75)))
    restored = add_sensor_noise(restored, rng, sigma=float(rng.uniform(2.5, 5.5)))
    restored = adjust_brightness_contrast(restored, alpha=float(rng.uniform(0.88, 1.05)), beta=float(rng.uniform(-8, 6)))
    return jpeg_roundtrip(restored, quality=int(rng.integers(48, 68)))


def dust_haze(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = image.shape[:2]
    mask = smooth_noise_mask((h, w), rng, grid=36)
    alpha = 0.07 + 0.16 * mask
    dust_color = np.array([rng.uniform(122, 150), rng.uniform(132, 158), rng.uniform(145, 176)], dtype=np.float32)
    blended = image.astype(np.float32) * (1.0 - alpha) + dust_color.reshape(1, 1, 3) * alpha
    blended = add_sensor_noise(clip_uint8(blended), rng, sigma=float(rng.uniform(1.5, 4.0)))
    blended = cv2.GaussianBlur(blended, (3, 3), sigmaX=float(rng.uniform(0.18, 0.45)))
    return jpeg_roundtrip(blended, quality=int(rng.integers(55, 76)))


def smoke_compression(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = image.shape[:2]
    mask_a = smooth_noise_mask((h, w), rng, grid=52)
    mask_b = smooth_noise_mask((h, w), rng, grid=28)
    smoke = np.clip(0.62 * mask_a + 0.38 * mask_b, 0, 1)
    alpha = 0.08 + 0.18 * smoke
    smoke_color = np.array([rng.uniform(150, 175), rng.uniform(150, 175), rng.uniform(150, 175)], dtype=np.float32)
    blended = image.astype(np.float32) * (1.0 - alpha) + smoke_color.reshape(1, 1, 3) * alpha

    hsv = cv2.cvtColor(clip_uint8(blended), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= float(rng.uniform(0.72, 0.90))
    hsv[..., 2] *= float(rng.uniform(0.88, 1.02))
    blended = cv2.cvtColor(clip_uint8(hsv), cv2.COLOR_HSV2BGR)
    blended = cv2.GaussianBlur(blended, (3, 3), sigmaX=float(rng.uniform(0.25, 0.60)))
    blended = add_sensor_noise(blended, rng, sigma=float(rng.uniform(2.0, 5.0)))
    return jpeg_roundtrip(blended, quality=int(rng.integers(44, 66)))


def augment_image(image: np.ndarray, variant: str, rng: np.random.Generator) -> np.ndarray:
    if variant == "cctv_lowres":
        return cctv_lowres(image, rng)
    if variant == "dust_haze":
        return dust_haze(image, rng)
    if variant == "smoke_compression":
        return smoke_compression(image, rng)
    raise ValueError(f"Unknown augmentation variant: {variant}")


def copy_tree_without_cache(source: Path, target: Path) -> None:
    for src_path in source.rglob("*"):
        rel = src_path.relative_to(source)
        dst_path = target / rel
        if src_path.is_dir():
            dst_path.mkdir(parents=True, exist_ok=True)
            continue
        if src_path.suffix.lower() == ".cache":
            continue
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)


def image_files(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def matching_label_path(root: Path, task: str, split: str, image_path: Path) -> Path:
    return root / task / "labels" / split / f"{image_path.stem}.txt"


def augment_task(
    source: Path,
    output: Path,
    task: str,
    variants: tuple[str, ...],
    seed: int,
    dry_run: bool,
) -> list[dict[str, object]]:
    train_images = image_files(source / task / "images" / "train")
    records: list[dict[str, object]] = []

    for index, image_path in enumerate(train_images):
        label_path = matching_label_path(source, task, "train", image_path)
        has_label = label_path.exists()
        image = None if dry_run else read_image(image_path)
        if image is None and not dry_run:
            records.append(
                {
                    "task": task,
                    "source_image": str(image_path),
                    "variant": "read_failed",
                    "output_image": "",
                    "output_label": "",
                    "label_status": "missing" if not has_label else "ok",
                }
            )
            continue

        for variant_index, variant in enumerate(variants):
            out_stem = f"{image_path.stem}__aug_{variant}"
            out_image = output / task / "images" / "train" / f"{out_stem}.jpg"
            out_label = output / task / "labels" / "train" / f"{out_stem}.txt"
            records.append(
                {
                    "task": task,
                    "source_image": str(image_path),
                    "variant": variant,
                    "output_image": str(out_image),
                    "output_label": str(out_label),
                    "label_status": "copied" if has_label else "empty_created",
                }
            )
            if dry_run:
                continue

            rng = np.random.default_rng(seed + index * 97 + variant_index * 1009 + (0 if task == "welding" else 50000))
            augmented = augment_image(image, variant, rng)
            write_jpg(out_image, augmented, quality=88)
            out_label.parent.mkdir(parents=True, exist_ok=True)
            if has_label:
                shutil.copy2(label_path, out_label)
            else:
                out_label.write_text("", encoding="utf-8")

    return records


def write_report(output: Path, records: list[dict[str, object]], dry_run: bool) -> None:
    if dry_run:
        return
    report_path = output / "augmentation_report.csv"
    with report_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["task", "source_image", "variant", "output_image", "output_label", "label_status"],
        )
        writer.writeheader()
        writer.writerows(records)

    readme = output / "README_augmentation.md"
    readme.write_text(
        "\n".join(
            [
                "# dataset_detect_augmented",
                "",
                "This dataset is generated from `dataset_detect` for YOLO detection training.",
                "",
                "Augmentation is applied only to the training split. Validation images and labels are copied without augmentation.",
                "",
                "Offline variants:",
                "- `cctv_lowres`: low-resolution CCTV degradation, blur, sensor noise and JPEG compression.",
                "- `dust_haze`: low-frequency dust veil, mild blur and compression.",
                "- `smoke_compression`: soft smoke-like haze, contrast reduction, blur and compression.",
                "",
                "All variants preserve object geometry, so YOLO labels are copied unchanged.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def count_images(root: Path, task: str, split: str) -> int:
    path = root / task / "images" / split
    return len(image_files(path)) if path.exists() else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an augmented YOLO detection dataset for welding and cutting.")
    parser.add_argument(
        "--source",
        default=str(Path("industrial_cctv_training") / "dataset" / "dataset_detect"),
        help="Source dataset_detect directory.",
    )
    parser.add_argument("--output", default="", help="Output directory. Default: <source>_augmented.")
    parser.add_argument("--seed", type=int, default=20260711, help="Random seed for deterministic augmentation.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned counts without writing files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve() if args.output else source.with_name(f"{source.name}_augmented")

    if not source.exists():
        raise FileNotFoundError(f"Source dataset not found: {source}")
    for task in TASKS:
        for split in ("train", "val"):
            if not (source / task / "images" / split).exists():
                raise FileNotFoundError(f"Missing image directory: {source / task / 'images' / split}")
            if not (source / task / "labels" / split).exists():
                raise FileNotFoundError(f"Missing label directory: {source / task / 'labels' / split}")

    if output.exists() and not args.dry_run:
        raise FileExistsError(
            f"Output already exists: {output}\n"
            "Please remove it manually or pass a different --output path to avoid overwriting data."
        )

    print(f"Source: {source}")
    print(f"Output: {output}")
    print(f"Variants per training image: {len(VARIANTS)} ({', '.join(VARIANTS)})")

    if not args.dry_run:
        copy_tree_without_cache(source, output)

    all_records: list[dict[str, object]] = []
    for task in TASKS:
        train_count = count_images(source, task, "train")
        val_count = count_images(source, task, "val")
        expected_train = train_count * (1 + len(VARIANTS))
        print(f"{task}: train {train_count} -> {expected_train}, val {val_count} unchanged")
        all_records.extend(augment_task(source, output, task, VARIANTS, args.seed, args.dry_run))

    write_report(output, all_records, args.dry_run)

    if not args.dry_run:
        for task in TASKS:
            print(
                f"{task} output: train={count_images(output, task, 'train')}, "
                f"val={count_images(output, task, 'val')}"
            )
        print(f"Augmentation report: {output / 'augmentation_report.csv'}")


if __name__ == "__main__":
    main()
