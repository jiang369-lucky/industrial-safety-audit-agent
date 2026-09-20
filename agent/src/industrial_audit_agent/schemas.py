# -*- coding: utf-8 -*-
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class SceneType(str, Enum):
    AUTO = "auto"
    WELDING = "welding"
    CUTTING = "cutting"
    IDLE = "idle"


class ComplianceVerdict(str, Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    UNCERTAIN = "uncertain"
    NEED_HUMAN = "need_human"


class RouteType(str, Enum):
    DIRECT_REPORT = "direct_report"
    VLM_REVIEW = "vlm_review"
    TEMPORAL_CHECK = "temporal_check"
    HUMAN_REVIEW = "human_review"


class AuditRequest(BaseModel):
    image_path: str = Field(..., description="Path to an image frame.")
    scene_hint: SceneType = Field(default=SceneType.AUTO, description="Optional scene hint.")
    stream_id: str = Field(default="default", description="Video stream id for temporal memory.")
    frame_id: int | None = Field(default=None, description="Optional frame index.")
    metadata: dict[str, Any] = Field(default_factory=dict)


class VideoAuditRequest(BaseModel):
    video_path: str
    scene_hint: SceneType = SceneType.AUTO
    stream_id: str = "video-demo"
    stride: int = Field(default=80, ge=1)
    max_frames: int = Field(default=6, ge=1, le=100)


class DetectionBox(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    xyxy: list[float] = Field(default_factory=list)


class FrameDetection(BaseModel):
    scene: SceneType
    boxes: list[DetectionBox] = Field(default_factory=list)
    risk_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    missing_items: list[str] = Field(default_factory=list)
    detector_backend: str = "mock"
    notes: list[str] = Field(default_factory=list)


class RouteDecision(BaseModel):
    route: RouteType
    reason: str
    confidence_band: Literal["high", "medium", "low"]


class VLMReview(BaseModel):
    verdict: ComplianceVerdict
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    model_backend: str = "mock"
    raw_text: str | None = None


class PolicyEvidence(BaseModel):
    policy_id: str
    title: str
    content: str
    score: float = Field(ge=0.0)


class TemporalEvidence(BaseModel):
    smoothed_risk: float = Field(ge=0.0, le=1.0)
    smoothed_confidence: float = Field(ge=0.0, le=1.0)
    oscillation: float = Field(ge=0.0, le=1.0)
    frame_count: int


class AuditReport(BaseModel):
    report_id: str
    scene: SceneType
    verdict: ComplianceVerdict
    risk_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    route: RouteDecision
    missing_items: list[str] = Field(default_factory=list)
    policy_evidence: list[PolicyEvidence] = Field(default_factory=list)
    vlm_review: VLMReview | None = None
    temporal_evidence: TemporalEvidence | None = None
    actions: list[str] = Field(default_factory=list)
    markdown: str = ""


class VideoAuditSummary(BaseModel):
    video_path: str
    sampled_frames: int
    max_risk_score: float
    non_compliant_frames: int
    reports: list[AuditReport]
