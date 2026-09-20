# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException

from industrial_audit_agent.config import get_settings
from industrial_audit_agent.schemas import AuditRequest, VideoAuditRequest, VideoAuditSummary
from industrial_audit_agent.tools.mcp_tools import tool_manifest
from industrial_audit_agent.workflow import SafetyAuditWorkflow


settings = get_settings()
workflow = SafetyAuditWorkflow(settings)

app = FastAPI(
    title="工业安全智能审核 Agent",
    description="YOLO + SFT-VLM + RAG + LangGraph dynamic routing service.",
    version="0.1.0",
)


@app.get("/health")
async def health() -> dict[str, object]:
    return {
        "status": "ok",
        "detector_backend": settings.detector_backend,
        "vlm_backend": settings.vlm_backend,
        "router_model_exists": settings.router_model_path.exists(),
        "weld_detector_exists": settings.weld_detector_path.exists(),
        "cut_detector_exists": settings.cut_detector_path.exists(),
    }


@app.get("/tools")
async def tools() -> list[dict[str, object]]:
    return tool_manifest()


@app.post("/audit/image")
async def audit_image(request: AuditRequest):
    path = Path(request.image_path)
    if not path.is_absolute():
        path = settings.repo_root / path
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {path}")
    request.image_path = str(path)
    report = await workflow.run(request)
    return report.model_dump()


@app.post("/audit/video")
async def audit_video(request: VideoAuditRequest) -> VideoAuditSummary:
    video_path = Path(request.video_path)
    if not video_path.is_absolute():
        video_path = settings.repo_root / video_path
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")

    frame_paths = await asyncio.to_thread(_sample_video_frames, video_path, request.stride, request.max_frames)
    reports = []
    for idx, frame_path in enumerate(frame_paths):
        item = AuditRequest(
            image_path=str(frame_path),
            scene_hint=request.scene_hint,
            stream_id=request.stream_id,
            frame_id=idx,
            metadata={"source_video": str(video_path)},
        )
        reports.append(await workflow.run(item))

    max_risk = max((report.risk_score for report in reports), default=0.0)
    bad_frames = sum(1 for report in reports if report.verdict.value == "non_compliant")
    return VideoAuditSummary(
        video_path=str(video_path),
        sampled_frames=len(reports),
        max_risk_score=max_risk,
        non_compliant_frames=bad_frames,
        reports=reports,
    )


def _sample_video_frames(video_path: Path, stride: int, max_frames: int) -> list[Path]:
    import cv2

    frame_dir = settings.runtime_dir / "frames" / video_path.stem
    frame_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_paths: list[Path] = []
    idx = 0
    saved = 0
    while saved < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            out = frame_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(out), frame)
            frame_paths.append(out)
            saved += 1
        idx += 1
    cap.release()
    return frame_paths
