# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


WORKSPACE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = WORKSPACE_DIR / "dataset"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply industrial CCTV degradation to YOLO datasets in this workspace."
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["dataset_gate", "dataset_gate_augmented", "dataset_detect"],
        help="Dataset folders under industrial_cctv_training/dataset to overwrite.",
    )
    parser.add_argument(
        "--profile",
        choices=["standard", "paper"],
        default="paper",
        help="paper is close to the current low-clarity monitoring videos; standard is milder.",
    )
    parser.add_argument("--delete-cache", action="store_true", help="Delete YOLO *.cache files after rewriting images.")
    parser.add_argument("--dry-run", action="store_true", help="Only list how many images would be processed.")
    return parser.parse_args()


def stable_seed(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
    return int(digest[:8], 16)


def imread_unicode(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    ok, encoded = cv2.imencode(path.suffix.lower(), image)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def image_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def make_vignette(height: int, width: int, strength: float) -> np.ndarray:
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    radius = np.sqrt(xx * xx + yy * yy)
    return (1.0 - np.clip(radius * strength, 0.0, 0.28))[:, :, None]


def smooth_noise(height: int, width: int, rng: np.random.Generator, scale: int = 12) -> np.ndarray:
    small_h = max(8, height // scale)
    small_w = max(8, width // scale)
    field = rng.random((small_h, small_w), dtype=np.float32)
    field = cv2.resize(field, (width, height), interpolation=cv2.INTER_CUBIC)
    field = cv2.GaussianBlur(field, (0, 0), sigmaX=max(height, width) * 0.035)
    field -= float(field.min())
    denom = float(field.max()) + 1e-6
    return field / denom


def lens_dust(height: int, width: int, rng: np.random.Generator, count: int, alpha: float) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.float32)
    for _ in range(count):
        cx = int(rng.integers(0, width))
        cy = int(rng.integers(0, height))
        radius = int(rng.integers(1, 5))
        value = float(rng.uniform(0.25, 1.0))
        cv2.circle(mask, (cx, cy), radius, value, -1, lineType=cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return np.clip(mask * alpha, 0.0, 1.0)[:, :, None]


def smoke_mask(height: int, width: int, rng: np.random.Generator, strength: float) -> np.ndarray:
    field_a = smooth_noise(height, width, rng, scale=10)
    field_b = smooth_noise(height, width, rng, scale=18)
    field = 0.62 * field_a + 0.38 * field_b
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    band = np.exp(-((yy - 0.42) ** 2) / 0.34)
    center = 0.55 + 0.45 * np.exp(-((xx - 0.5) ** 2) / 0.20)
    field = np.clip((field - 0.33) * 1.50, 0.0, 1.0)
    return (field * band * center * strength)[:, :, None].astype(np.float32)


def motion_blur(frame: np.ndarray, rng: np.random.Generator, probability: float) -> np.ndarray:
    if rng.random() > probability:
        return frame
    kernel_size = 3 if rng.random() < 0.58 else 5
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    direction = rng.integers(0, 3)
    if direction == 0:
        kernel[kernel_size // 2, :] = 1.0
    elif direction == 1:
        np.fill_diagonal(kernel, 1.0)
    else:
        np.fill_diagonal(np.fliplr(kernel), 1.0)
    kernel /= kernel.sum()
    blurred = cv2.filter2D(frame, -1, kernel)
    mix = float(rng.uniform(0.25, 0.55))
    return cv2.addWeighted(frame, 1.0 - mix, blurred, mix, 0)


def degrade_image(image: np.ndarray, seed_key: str, profile: str) -> np.ndarray:
    rng = np.random.default_rng(stable_seed(seed_key))
    height, width = image.shape[:2]

    if profile == "standard":
        clarity_scale = 0.50
        contrast = 0.84
        dust_alpha = 0.090
        dust_count = 56
        smoke_strength = 0.070
        sensor_std = 2.2
        defocus_sigma = 0.35
        motion_probability = 0.15
        block_alpha = 0.04
    else:
        clarity_scale = 0.38
        contrast = 0.80
        dust_alpha = 0.105
        dust_count = 66
        smoke_strength = 0.082
        sensor_std = 3.0
        defocus_sigma = 0.75
        motion_probability = 0.28
        block_alpha = 0.11

    small_w = max(32, int(width * clarity_scale))
    small_h = max(32, int(height * clarity_scale))
    low = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
    out = cv2.resize(low, (width, height), interpolation=cv2.INTER_LINEAR)
    out = cv2.GaussianBlur(out, (3, 3), 0)
    out = cv2.GaussianBlur(out, (0, 0), sigmaX=defocus_sigma, sigmaY=defocus_sigma)
    out = motion_blur(out, rng, motion_probability)

    ycrcb = cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb)
    y_channel, cr_channel, cb_channel = cv2.split(ycrcb)
    cr_channel = cv2.GaussianBlur(cr_channel, (5, 5), 0)
    cb_channel = cv2.GaussianBlur(cb_channel, (5, 5), 0)
    out = cv2.cvtColor(cv2.merge([y_channel, cr_channel, cb_channel]), cv2.COLOR_YCrCb2BGR)

    if block_alpha > 0:
        block = 8
        bw = max(16, width // block)
        bh = max(16, height // block)
        low_block = cv2.resize(out, (bw, bh), interpolation=cv2.INTER_AREA)
        restored = cv2.resize(low_block, (width, height), interpolation=cv2.INTER_NEAREST)
        out = cv2.addWeighted(out, 1.0 - block_alpha, restored, block_alpha, 0)

    out_f = out.astype(np.float32)
    mean = np.array([128.0, 128.0, 128.0], dtype=np.float32)
    exposure = float(rng.uniform(-5.5, 5.5))
    wb = rng.uniform(-3.0, 3.0, size=(3,)).astype(np.float32)
    out_f = (out_f - mean) * contrast + mean + exposure + wb

    dust_color = np.array([150.0, 151.0, 145.0], dtype=np.float32)
    out_f = out_f * (1.0 - dust_alpha) + dust_color * dust_alpha
    dust_mask = lens_dust(height, width, rng, dust_count, 0.16)
    out_f = out_f * (1.0 - dust_mask) + dust_color * dust_mask

    smoke = smoke_mask(height, width, rng, smoke_strength)
    smoke_color = np.array([185.0, 188.0, 184.0], dtype=np.float32)
    out_f = out_f * (1.0 - smoke) + smoke_color * smoke

    out_f += rng.normal(0.0, sensor_std, out_f.shape).astype(np.float32)
    out_f *= make_vignette(height, width, 0.10 if profile == "standard" else 0.12)
    return np.clip(out_f, 0, 255).astype(np.uint8)


def delete_caches(root: Path) -> int:
    count = 0
    for cache in root.rglob("*.cache"):
        cache.unlink()
        count += 1
    return count


def main() -> None:
    args = parse_args()
    roots = [DATASET_DIR / target for target in args.targets]
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(f"找不到数据集目录: {root}")

    all_images: list[Path] = []
    for root in roots:
        all_images.extend(image_files(root))

    print("工业监控风格图像覆盖脚本")
    print(f"工作区: {WORKSPACE_DIR}")
    print(f"处理目标: {', '.join(args.targets)}")
    print(f"噪声配置: {args.profile}")
    print(f"图像数量: {len(all_images)}")

    if args.dry_run:
        print("dry-run: 未修改任何文件。")
        return

    unreadable: list[Path] = []
    for path in tqdm(all_images, desc="覆盖图像"):
        image = imread_unicode(path)
        if image is None:
            unreadable.append(path)
            continue
        degraded = degrade_image(image, str(path.relative_to(WORKSPACE_DIR)), args.profile)
        if not imwrite_unicode(path, degraded):
            raise RuntimeError(f"写入失败: {path}")

    if args.delete_cache:
        removed = delete_caches(DATASET_DIR)
        print(f"已删除 YOLO cache 文件: {removed}")

    if unreadable:
        print("\n以下图像无法读取，未被覆盖：")
        for path in unreadable[:50]:
            print(f"  {path}")
        if len(unreadable) > 50:
            print(f"  ... 还有 {len(unreadable) - 50} 个")
        raise RuntimeError(f"存在 {len(unreadable)} 个无法读取的图像，请先检查文件完整性。")

    print("完成：所有可读图像已原地覆盖为工业监控风格。")


if __name__ == "__main__":
    main()
