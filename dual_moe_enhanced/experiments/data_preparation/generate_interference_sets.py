# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from common.experiment_paths import TEST_VIDEO_DIR  # noqa: E402
from generate_real_videos import RealDegradationParams, process_video


INTERFERENCE_PROFILES = {
    "light": RealDegradationParams(
        clarity_scale=0.78,
        blur_kernel=3,
        contrast=0.96,
        brightness_shift=0.5,
        dust_film_alpha=0.025,
        lens_dust_count=24,
        lens_dust_alpha=0.060,
        smoke_alpha=0.022,
        smoke_field_scale=0.40,
        smoke_low_threshold=0.38,
        smoke_contrast=1.15,
        sensor_noise_std=1.2,
        exposure_wave_amp=2.0,
        contrast_wave_amp=0.020,
        white_balance_wave_amp=1.2,
        defocus_sigma_base=0.05,
        defocus_sigma_amp=0.18,
        motion_blur_max_kernel=3,
        motion_blur_threshold=0.88,
        motion_blur_mix_max=0.20,
        chroma_blur_kernel=3,
        block_artifact_size=4,
        block_artifact_alpha=0.025,
        vignette_strength=0.035,
    ),
    "medium": RealDegradationParams(
        clarity_scale=0.64,
        blur_kernel=3,
        contrast=0.91,
        brightness_shift=0.0,
        dust_film_alpha=0.048,
        lens_dust_count=42,
        lens_dust_alpha=0.090,
        smoke_alpha=0.045,
        smoke_field_scale=0.36,
        smoke_low_threshold=0.36,
        smoke_contrast=1.25,
        sensor_noise_std=2.2,
        exposure_wave_amp=4.0,
        contrast_wave_amp=0.040,
        white_balance_wave_amp=2.0,
        defocus_sigma_base=0.12,
        defocus_sigma_amp=0.38,
        motion_blur_max_kernel=5,
        motion_blur_threshold=0.80,
        motion_blur_mix_max=0.34,
        chroma_blur_kernel=5,
        block_artifact_size=6,
        block_artifact_alpha=0.060,
        vignette_strength=0.060,
    ),
    "heavy": RealDegradationParams(
        clarity_scale=0.52,
        blur_kernel=3,
        contrast=0.86,
        brightness_shift=-1.0,
        dust_film_alpha=0.070,
        lens_dust_count=62,
        lens_dust_alpha=0.120,
        smoke_alpha=0.068,
        smoke_field_scale=0.32,
        smoke_low_threshold=0.34,
        smoke_contrast=1.35,
        sensor_noise_std=3.4,
        exposure_wave_amp=6.0,
        contrast_wave_amp=0.060,
        white_balance_wave_amp=3.0,
        defocus_sigma_base=0.20,
        defocus_sigma_amp=0.58,
        motion_blur_max_kernel=5,
        motion_blur_threshold=0.74,
        motion_blur_mix_max=0.48,
        chroma_blur_kernel=5,
        block_artifact_size=8,
        block_artifact_alpha=0.095,
        vignette_strength=0.085,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="由 clean/cutting 与 clean/welding 生成 light/medium/heavy 干扰数据集。"
    )
    parser.add_argument("--clean-root", type=Path, default=TEST_VIDEO_DIR / "clean")
    parser.add_argument("--output-root", type=Path, default=TEST_VIDEO_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--levels",
        nargs="+",
        choices=sorted(INTERFERENCE_PROFILES.keys()),
        default=["light", "medium", "heavy"],
        help="需要生成的干扰等级。",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        choices=["welding", "cutting"],
        default=["welding", "cutting"],
        help="需要处理的场景子目录。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass

    total_frames = 0
    total_start = time.time()

    for level in args.levels:
        params = INTERFERENCE_PROFILES[level]
        for scene in args.scenes:
            input_dir = args.clean_root / scene
            output_dir = args.output_root / level / scene
            videos = sorted(input_dir.glob("*.mp4"))
            if not videos:
                print(f"[skip] 未找到视频: {input_dir}")
                continue

            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n[{level.upper()} | {scene}]")
            print(f"输入: {input_dir}")
            print(f"输出: {output_dir}")
            print(f"视频数: {len(videos)}")

            for idx, input_path in enumerate(videos, 1):
                output_path = output_dir / input_path.name
                print(f"  ({idx}/{len(videos)}) {input_path.name}")
                frames, elapsed = process_video(input_path, output_path, params, args.overwrite)
                total_frames += frames
                if frames > 0:
                    fps = frames / elapsed if elapsed > 0 else 0.0
                    print(f"      完成: {frames} 帧, {elapsed:.1f}s, {fps:.1f} FPS")

    total_elapsed = time.time() - total_start
    print("\n全部完成")
    print(f"生成帧数: {total_frames}")
    print(f"总耗时: {total_elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
