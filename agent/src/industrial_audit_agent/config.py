# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


@dataclass(frozen=True)
class Settings:
    repo_root: Path = field(default_factory=lambda: _env_path("AGENT_REPO_ROOT", _repo_root()))
    detector_backend: str = field(default_factory=lambda: os.getenv("AGENT_DETECTOR_BACKEND", "auto"))
    vlm_backend: str = field(default_factory=lambda: os.getenv("AGENT_VLM_BACKEND", "mock"))

    high_confidence: float = field(default_factory=lambda: _env_float("AGENT_HIGH_CONFIDENCE", 0.78))
    low_confidence: float = field(default_factory=lambda: _env_float("AGENT_LOW_CONFIDENCE", 0.42))
    alert_risk: float = field(default_factory=lambda: _env_float("AGENT_ALERT_RISK", 0.62))
    safe_risk: float = field(default_factory=lambda: _env_float("AGENT_SAFE_RISK", 0.28))

    @property
    def agent_dir(self) -> Path:
        return self.repo_root / "agent"

    @property
    def runtime_dir(self) -> Path:
        return self.agent_dir / "runtime"

    @property
    def router_model_path(self) -> Path:
        default = first_existing(
            self.repo_root / "runs" / "classify" / "runs" / "router" / "exp2_raw_yolo_aug" / "weights" / "best.pt",
            self.repo_root / "runs" / "classify" / "runs" / "router" / "exp4_augmented_yolo_aug" / "weights" / "best.pt",
        )
        return _env_path("AGENT_ROUTER_MODEL_PATH", default)

    @property
    def weld_detector_path(self) -> Path:
        default = self.repo_root / "runs" / "detect" / "runs" / "detect" / "yolo_weld_expert" / "weights" / "best.pt"
        return _env_path("AGENT_WELD_DETECTOR_PATH", default)

    @property
    def cut_detector_path(self) -> Path:
        default = self.repo_root / "runs" / "detect" / "runs" / "detect" / "yolo_cut_expert" / "weights" / "best.pt"
        return _env_path("AGENT_CUT_DETECTOR_PATH", default)

    @property
    def vlm_base_model_path(self) -> Path:
        default = first_existing(
            self.repo_root / "model" / "Qwen2.5-VL-3B-Instruct",
            self.repo_root / "model" / "Qwen3-VL-4B-Instruct",
            self.repo_root / "model" / "Qwen",
        )
        return _env_path("AGENT_VLM_BASE_MODEL_PATH", default)

    @property
    def vlm_lora_path(self) -> Path:
        default = first_existing(
            self.repo_root / "LlamaFactory-main" / "saves" / "qwen2_5-vl-weld-lora",
            self.repo_root / "LlamaFactory-main" / "saves" / "qwen3-vl-weld-3epoch",
        )
        return _env_path("AGENT_VLM_LORA_PATH", default)

    @property
    def policy_path(self) -> Path:
        default = self.agent_dir / "src" / "industrial_audit_agent" / "resources" / "safety_policy.md"
        return _env_path("AGENT_POLICY_PATH", default)


def get_settings() -> Settings:
    return Settings()
