# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from industrial_audit_agent.api import _sample_video_frames
from industrial_audit_agent.config import get_settings
from industrial_audit_agent.schemas import AuditRequest, SceneType
from industrial_audit_agent.workflow import SafetyAuditWorkflow


def _resolve_path(path: str) -> str:
    settings = get_settings()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = settings.repo_root / candidate
    return str(candidate)


async def run_image(args: argparse.Namespace) -> None:
    workflow = SafetyAuditWorkflow(get_settings())
    request = AuditRequest(image_path=_resolve_path(args.image), scene_hint=SceneType(args.scene), stream_id=args.stream_id)
    report = await workflow.run(request)
    print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
    print("\n" + report.markdown)


async def run_video(args: argparse.Namespace) -> None:
    settings = get_settings()
    workflow = SafetyAuditWorkflow(settings)
    video_path = Path(_resolve_path(args.video))
    frame_paths = _sample_video_frames(video_path, args.stride, args.max_frames)
    reports = []
    for idx, frame_path in enumerate(frame_paths):
        request = AuditRequest(
            image_path=str(frame_path),
            scene_hint=SceneType(args.scene),
            stream_id=args.stream_id,
            frame_id=idx,
            metadata={"source_video": str(video_path)},
        )
        reports.append(await workflow.run(request))

    payload = {
        "video_path": str(video_path),
        "sampled_frames": len(reports),
        "max_risk_score": max((report.risk_score for report in reports), default=0.0),
        "non_compliant_frames": sum(1 for report in reports if report.verdict.value == "non_compliant"),
        "reports": [report.model_dump(mode="json") for report in reports],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Industrial safety audit agent demo.")
    sub = parser.add_subparsers(dest="command", required=True)

    image = sub.add_parser("image", help="Audit one image frame.")
    image.add_argument("--image", required=True)
    image.add_argument("--scene", default="auto", choices=["auto", "welding", "cutting", "idle"])
    image.add_argument("--stream-id", default="cli-image")

    video = sub.add_parser("video", help="Audit sampled frames from a video.")
    video.add_argument("--video", required=True)
    video.add_argument("--scene", default="auto", choices=["auto", "welding", "cutting", "idle"])
    video.add_argument("--stream-id", default="cli-video")
    video.add_argument("--stride", type=int, default=80)
    video.add_argument("--max-frames", type=int, default=6)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "image":
        asyncio.run(run_image(args))
    elif args.command == "video":
        asyncio.run(run_video(args))


if __name__ == "__main__":
    main()
