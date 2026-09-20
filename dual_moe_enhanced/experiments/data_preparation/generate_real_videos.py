# -*- coding: utf-8 -*-
"""
Generate realistic welding videos for a more practical "Clean" condition.

The generated videos keep the original FPS, frame count and frame size, so the
existing frame-level annotations can still be reused. Only three kinds of
realistic factory-monitor degradation are added:

1. Lower surveillance-camera clarity.
2. Industrial dust film / lens dust.
3. Mild drifting welding-site smoke.

No artificial strong occlusion, sparks, color blocks, or stress-test artifacts
are added here. Those should stay in separate stress-test datasets.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from common.experiment_paths import TEST_VIDEO_DIR  # noqa: E402


DEFAULT_INPUT_DIR = TEST_VIDEO_DIR / "welding"
DEFAULT_OUTPUT_DIR = TEST_VIDEO_DIR / "clean" / "welding"


@dataclass(frozen=True)
class RealDegradationParams:
    clarity_scale: float = 0.42
    blur_kernel: int = 3
    contrast: float = 0.80
    brightness_shift: float = 2.5
    dust_film_alpha: float = 0.105
    lens_dust_count: int = 70
    lens_dust_alpha: float = 0.160
    lens_dust_radius_min: int = 1
    lens_dust_radius_max: int = 5
    smoke_alpha: float = 0.090
    smoke_field_scale: float = 0.32
    smoke_low_threshold: float = 0.34
    smoke_contrast: float = 1.45
    smoke_center_y_ratio: float = 0.42
    smoke_band_height_ratio: float = 0.58
    sensor_noise_std: float = 3.0
    compression_quality: int = 100
    exposure_wave_amp: float = 7.0
    contrast_wave_amp: float = 0.065
    white_balance_wave_amp: float = 3.5
    defocus_sigma_base: float = 0.20
    defocus_sigma_amp: float = 0.75
    motion_blur_max_kernel: int = 5
    motion_blur_threshold: float = 0.82
    motion_blur_mix_max: float = 0.55
    chroma_blur_kernel: int = 5
    fov_scale: float = 1.0
    fov_shift_ratio: float = 0.0
    block_artifact_size: int = 1
    block_artifact_alpha: float = 0.0
    vignette_strength: float = 0.10


@dataclass(frozen=True)
class ViewTransform:
    scale: float = 1.0
    shift_x: float = 0.0
    shift_y: float = 0.0


PROFILE_PARAMS = {
    "standard": RealDegradationParams(
        clarity_scale=0.50,
        contrast=0.84,
        brightness_shift=4.0,
        dust_film_alpha=0.090,
        lens_dust_count=58,
        lens_dust_alpha=0.145,
        smoke_alpha=0.082,
        sensor_noise_std=2.2,
        exposure_wave_amp=4.0,
        contrast_wave_amp=0.035,
        white_balance_wave_amp=2.0,
        defocus_sigma_base=0.10,
        defocus_sigma_amp=0.35,
        motion_blur_max_kernel=3,
        chroma_blur_kernel=3,
        vignette_strength=0.09,
    ),
    "strong": RealDegradationParams(),
    "hard_clean": RealDegradationParams(
        clarity_scale=0.36,
        blur_kernel=3,
        contrast=0.80,
        brightness_shift=2.0,
        dust_film_alpha=0.100,
        lens_dust_count=66,
        lens_dust_alpha=0.155,
        smoke_alpha=0.076,
        sensor_noise_std=3.2,
        exposure_wave_amp=7.5,
        contrast_wave_amp=0.070,
        white_balance_wave_amp=3.8,
        defocus_sigma_base=0.28,
        defocus_sigma_amp=0.90,
        motion_blur_max_kernel=5,
        motion_blur_threshold=0.72,
        motion_blur_mix_max=0.62,
        chroma_blur_kernel=5,
        fov_scale=0.94,
        fov_shift_ratio=0.018,
        block_artifact_size=8,
        block_artifact_alpha=0.14,
        vignette_strength=0.10,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate realistic CCTV/dust/smoke welding videos."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_PARAMS.keys()),
        default="strong",
        help="Natural degradation strength. Use standard for a milder clean/welding set.",
    )
    parser.add_argument(
        "--clarity-scale",
        type=float,
        default=None,
        help="Internal downsample ratio used to simulate surveillance clarity.",
    )
    return parser.parse_args()


def stable_seed(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def make_vignette(height: int, width: int, strength: float) -> np.ndarray:
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    radius = np.sqrt(xx * xx + yy * yy)
    return (1.0 - np.clip(radius * strength, 0.0, 0.25))[:, :, None]


def make_view_transform(width: int, height: int, rng: np.random.Generator,
                        params: RealDegradationParams) -> ViewTransform:
    if params.fov_scale >= 0.999:
        return ViewTransform()
    max_shift_x = width * params.fov_shift_ratio
    max_shift_y = height * params.fov_shift_ratio
    return ViewTransform(
        scale=float(params.fov_scale),
        shift_x=float(rng.uniform(-max_shift_x, max_shift_x)),
        shift_y=float(rng.uniform(-max_shift_y, max_shift_y)),
    )


def apply_view_transform(frame: np.ndarray, transform: ViewTransform) -> np.ndarray:
    if transform.scale >= 0.999:
        return frame

    height, width = frame.shape[:2]
    scaled_w = max(16, int(width * transform.scale))
    scaled_h = max(16, int(height * transform.scale))
    small = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

    background = cv2.GaussianBlur(frame, (0, 0), sigmaX=max(height, width) * 0.030)
    background = cv2.addWeighted(background, 0.82, frame, 0.18, 0)

    x0 = int(round((width - scaled_w) / 2 + transform.shift_x))
    y0 = int(round((height - scaled_h) / 2 + transform.shift_y))
    x0 = int(np.clip(x0, 0, width - scaled_w))
    y0 = int(np.clip(y0, 0, height - scaled_h))
    output = background.copy()
    output[y0:y0 + scaled_h, x0:x0 + scaled_w] = small
    return output


def make_lens_dust(height: int, width: int, rng: np.random.Generator,
                   params: RealDegradationParams) -> np.ndarray:
    dust = np.zeros((height, width), dtype=np.float32)
    for _ in range(params.lens_dust_count):
        cx = int(rng.integers(0, width))
        cy = int(rng.integers(0, height))
        radius = int(rng.integers(params.lens_dust_radius_min,
                                  params.lens_dust_radius_max + 1))
        value = float(rng.uniform(0.35, 1.0))
        cv2.circle(dust, (cx, cy), radius, value, -1, lineType=cv2.LINE_AA)
    dust = cv2.GaussianBlur(dust, (5, 5), 0)
    return np.clip(dust, 0.0, 1.0)


def normalize01(field: np.ndarray) -> np.ndarray:
    field = field.astype(np.float32)
    min_value = float(field.min())
    max_value = float(field.max())
    if max_value - min_value < 1e-6:
        return np.zeros_like(field, dtype=np.float32)
    return (field - min_value) / (max_value - min_value)


def make_smooth_field(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    coarse_h = max(8, height // 12)
    coarse_w = max(8, width // 12)
    coarse = rng.random((coarse_h, coarse_w), dtype=np.float32)
    field = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    sigma = max(height, width) * 0.035
    field = cv2.GaussianBlur(field, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return normalize01(field)


def make_smoke_components(height: int, width: int, rng: np.random.Generator,
                          params: RealDegradationParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low_h = max(64, int(height * params.smoke_field_scale))
    low_w = max(64, int(width * params.smoke_field_scale))
    field_a = make_smooth_field(low_h, low_w, rng)
    field_b = make_smooth_field(low_h, low_w, rng)

    y = np.linspace(0.0, 1.0, low_h, dtype=np.float32)
    x = np.linspace(0.0, 1.0, low_w, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    center_y = params.smoke_center_y_ratio
    band = np.exp(-((yy - center_y) ** 2) / max(params.smoke_band_height_ratio ** 2, 1e-6))
    center_bias = 0.55 + 0.45 * np.exp(-((xx - 0.5) ** 2) / 0.18)
    envelope = normalize01(band * center_bias)
    return field_a, field_b, envelope


def translate_field(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    height, width = field.shape[:2]
    matrix = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
    return cv2.warpAffine(
        field,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def make_smoke_mask(height: int, width: int, frame_idx: int,
                    smoke_components: tuple[np.ndarray, np.ndarray, np.ndarray],
                    params: RealDegradationParams) -> np.ndarray:
    field_a, field_b, envelope = smoke_components
    low_h, low_w = field_a.shape[:2]

    dx_a = frame_idx * 0.055 + 2.6 * np.sin(frame_idx * 0.010)
    dy_a = -frame_idx * 0.015 + 1.8 * np.cos(frame_idx * 0.008)
    dx_b = -frame_idx * 0.032 + 2.0 * np.sin(frame_idx * 0.007 + 1.7)
    dy_b = -frame_idx * 0.010 + 1.5 * np.cos(frame_idx * 0.011 + 0.8)

    moving_a = translate_field(field_a, dx_a, dy_a)
    moving_b = translate_field(field_b, dx_b, dy_b)
    smoke = 0.66 * moving_a + 0.34 * moving_b
    smoke = cv2.GaussianBlur(smoke, (0, 0), sigmaX=max(low_h, low_w) * 0.010)
    smoke = np.clip((smoke - params.smoke_low_threshold) * params.smoke_contrast, 0.0, 1.0)
    smoke *= envelope
    smoke *= 0.92 + 0.08 * np.sin(frame_idx * 0.013)
    return cv2.resize(smoke, (width, height), interpolation=cv2.INTER_LINEAR)


def apply_chroma_blur(frame: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size < 3:
        return frame
    if kernel_size % 2 == 0:
        kernel_size += 1
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y_channel, cr_channel, cb_channel = cv2.split(ycrcb)
    cr_channel = cv2.GaussianBlur(cr_channel, (kernel_size, kernel_size), 0)
    cb_channel = cv2.GaussianBlur(cb_channel, (kernel_size, kernel_size), 0)
    return cv2.cvtColor(cv2.merge([y_channel, cr_channel, cb_channel]), cv2.COLOR_YCrCb2BGR)


def apply_temporal_defocus(frame: np.ndarray, frame_idx: int,
                           params: RealDegradationParams) -> np.ndarray:
    focus_wave = 0.5 + 0.5 * np.sin(frame_idx * 0.017 + 0.8)
    sigma = params.defocus_sigma_base + params.defocus_sigma_amp * (focus_wave ** 2)
    if sigma < 0.25:
        return frame
    return cv2.GaussianBlur(frame, (0, 0), sigmaX=sigma, sigmaY=sigma)


def apply_motion_blur(frame: np.ndarray, frame_idx: int,
                      params: RealDegradationParams) -> np.ndarray:
    max_kernel = max(1, int(params.motion_blur_max_kernel))
    if max_kernel < 3:
        return frame
    if max_kernel % 2 == 0:
        max_kernel += 1

    motion_wave = 0.5 + 0.5 * np.sin(frame_idx * 0.049 + 1.2)
    if motion_wave < params.motion_blur_threshold:
        return frame

    kernel_size = 3 if max_kernel <= 3 else 5
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    direction_wave = np.sin(frame_idx * 0.021)
    if direction_wave > 0.35:
        np.fill_diagonal(kernel, 1.0)
    elif direction_wave < -0.35:
        np.fill_diagonal(np.fliplr(kernel), 1.0)
    else:
        kernel[kernel_size // 2, :] = 1.0
    kernel /= kernel.sum()

    blurred = cv2.filter2D(frame, -1, kernel)
    denom = max(1.0 - params.motion_blur_threshold, 1e-6)
    mix = min(params.motion_blur_mix_max,
              (motion_wave - params.motion_blur_threshold) / denom * params.motion_blur_mix_max)
    return cv2.addWeighted(frame, 1.0 - mix, blurred, mix, 0)


def apply_block_artifacts(frame: np.ndarray, params: RealDegradationParams) -> np.ndarray:
    if params.block_artifact_size <= 1 or params.block_artifact_alpha <= 1e-6:
        return frame
    height, width = frame.shape[:2]
    block = int(params.block_artifact_size)
    small_w = max(16, width // block)
    small_h = max(16, height // block)
    low = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(low, (width, height), interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(frame, 1.0 - params.block_artifact_alpha,
                           restored, params.block_artifact_alpha, 0)


def degrade_frame(frame: np.ndarray, frame_idx: int,
                  rng: np.random.Generator,
                  lens_dust: np.ndarray,
                  smoke_components: tuple[np.ndarray, np.ndarray, np.ndarray],
                  view_transform: ViewTransform,
                  vignette: np.ndarray,
                  params: RealDegradationParams) -> np.ndarray:
    height, width = frame.shape[:2]

    frame = apply_view_transform(frame, view_transform)

    small_w = max(32, int(width * params.clarity_scale))
    small_h = max(32, int(height * params.clarity_scale))
    low_res = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
    degraded = cv2.resize(low_res, (width, height), interpolation=cv2.INTER_LINEAR)

    if params.blur_kernel >= 3 and params.blur_kernel % 2 == 1:
        degraded = cv2.GaussianBlur(degraded, (params.blur_kernel, params.blur_kernel), 0)
    degraded = apply_temporal_defocus(degraded, frame_idx, params)
    degraded = apply_motion_blur(degraded, frame_idx, params)
    degraded = apply_chroma_blur(degraded, params.chroma_blur_kernel)
    degraded = apply_block_artifacts(degraded, params)

    if params.compression_quality < 100:
        ok, encoded = cv2.imencode(
            ".jpg",
            degraded,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(params.compression_quality)],
        )
        if ok:
            degraded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    out = degraded.astype(np.float32)

    mean = np.array([128.0, 128.0, 128.0], dtype=np.float32)
    exposure_shift = params.exposure_wave_amp * (
        0.65 * np.sin(frame_idx * 0.011 + 0.6)
        + 0.35 * np.sin(frame_idx * 0.027 + 2.1)
    )
    contrast_wave = 1.0 + params.contrast_wave_amp * np.sin(frame_idx * 0.015 + 1.4)
    wb_shift = np.array([
        np.sin(frame_idx * 0.009 + 0.3),
        np.sin(frame_idx * 0.007 + 1.6),
        np.sin(frame_idx * 0.010 + 2.4),
    ], dtype=np.float32) * params.white_balance_wave_amp

    out = (out - mean) * (params.contrast * contrast_wave) + mean
    out = out + params.brightness_shift + exposure_shift + wb_shift

    dust_color = np.array([150.0, 151.0, 145.0], dtype=np.float32)
    out = out * (1.0 - params.dust_film_alpha) + dust_color * params.dust_film_alpha

    dust_alpha = (lens_dust * params.lens_dust_alpha)[:, :, None]
    out = out * (1.0 - dust_alpha) + dust_color * dust_alpha

    smoke_mask = make_smoke_mask(height, width, frame_idx, smoke_components, params)
    smoke_color = np.array([185.0, 188.0, 184.0], dtype=np.float32)
    smoke_alpha = (smoke_mask * params.smoke_alpha)[:, :, None]
    out = out * (1.0 - smoke_alpha) + smoke_color * smoke_alpha

    if params.sensor_noise_std > 0:
        sensor_noise = rng.normal(0.0, params.sensor_noise_std, out.shape).astype(np.float32)
        out += sensor_noise

    out *= vignette
    return np.clip(out, 0, 255).astype(np.uint8)


def process_video(input_path: Path, output_path: Path,
                  params: RealDegradationParams,
                  overwrite: bool) -> tuple[int, float]:
    if output_path.exists() and not overwrite:
        print(f"[skip] {output_path.name} already exists")
        return 0, 0.0

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-6:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create video: {output_path}")

    rng = np.random.default_rng(stable_seed(input_path.name))
    lens_dust = make_lens_dust(height, width, rng, params)
    smoke_components = make_smoke_components(height, width, rng, params)
    view_transform = make_view_transform(width, height, rng, params)
    vignette = make_vignette(height, width, params.vignette_strength)

    start = time.time()
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(degrade_frame(frame, count, rng, lens_dust, smoke_components,
                                   view_transform, vignette, params))
        count += 1
        if total > 0 and count % 500 == 0:
            print(f"    {input_path.name}: {count}/{total} frames")

    cap.release()
    writer.release()
    elapsed = time.time() - start
    return count, elapsed


def main() -> None:
    args = parse_args()
    base_params = PROFILE_PARAMS[args.profile]
    if args.clarity_scale is None:
        params = base_params
    else:
        params = RealDegradationParams(**{**base_params.__dict__, "clarity_scale": args.clarity_scale})

    try:
        cv2.setNumThreads(0)
    except Exception:
        pass

    videos = sorted(args.input_dir.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No mp4 videos found in: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Generate clean/welding videos with CCTV clarity loss, dust, and mild smoke")
    print(f"Input : {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Videos: {len(videos)}")
    print(f"Profile: {args.profile}")
    print(f"Clarity scale: {params.clarity_scale}")

    total_start = time.time()
    total_frames = 0
    for idx, input_path in enumerate(videos, 1):
        output_path = args.output_dir / input_path.name
        print(f"\n[{idx}/{len(videos)}] {input_path.name}")
        frames, elapsed = process_video(input_path, output_path, params, args.overwrite)
        total_frames += frames
        if frames > 0:
            speed = frames / elapsed if elapsed > 0 else 0.0
            print(f"    done: {frames} frames, {elapsed:.1f}s, {speed:.1f} FPS")

    total_elapsed = time.time() - total_start
    print("\nAll done.")
    print(f"Generated frames: {total_frames}")
    print(f"Elapsed: {total_elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
