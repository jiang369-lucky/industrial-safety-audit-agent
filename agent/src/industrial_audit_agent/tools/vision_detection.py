# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from industrial_audit_agent.config import Settings
from industrial_audit_agent.schemas import AuditRequest, DetectionBox, FrameDetection, SceneType


def _norm_label(label: str) -> str:
    text = label.lower()
    if any(key in text for key in ["person", "worker", "human", "工人"]):
        return "worker"
    if any(key in text for key in ["mask", "helmet", "face", "shield", "面罩"]):
        return "mask"
    if any(key in text for key in ["smoke", "collector", "fume", "烟雾", "收集"]):
        return "smoke_collector"
    if any(key in text for key in ["extinguisher", "fire", "灭火器"]):
        return "extinguisher"
    return text.replace(" ", "_")


def _infer_scene(path: str, hint: SceneType) -> SceneType:
    if hint != SceneType.AUTO:
        return hint
    text = path.lower()
    if "weld" in text or "dht" in text or "焊" in text:
        return SceneType.WELDING
    if "cut" in text or "qgt" in text or "割" in text:
        return SceneType.CUTTING
    return SceneType.IDLE


class VisionDetectionTool:
    """YOLO detector adapter with a deterministic fallback for demos."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._models: dict[str, Any] = {}
        self._yolo_cls: Any | None = None

    async def detect_image(self, request: AuditRequest) -> FrameDetection:
        return await asyncio.to_thread(self._detect_image_sync, request)

    def _detect_image_sync(self, request: AuditRequest) -> FrameDetection:
        path = Path(request.image_path)
        scene = _infer_scene(str(path), request.scene_hint)

        if self.settings.detector_backend != "mock":
            model = self._get_detector(scene)
            if model is not None and path.exists():
                try:
                    return self._predict_with_yolo(model, path, scene)
                except Exception as exc:  # pragma: no cover - depends on local GPU/model
                    fallback = self._mock_detection(path, scene)
                    fallback.notes.append(f"YOLO推理失败，已回退到模拟检测: {exc}")
                    return fallback

        return self._mock_detection(path, scene)

    def _get_detector(self, scene: SceneType) -> Any | None:
        if self.settings.detector_backend == "mock":
            return None
        if self._yolo_cls is None:
            try:
                from ultralytics import YOLO

                self._yolo_cls = YOLO
            except Exception:
                return None

        key = scene.value
        if key in self._models:
            return self._models[key]

        path = {
            SceneType.WELDING: self.settings.weld_detector_path,
            SceneType.CUTTING: self.settings.cut_detector_path,
        }.get(scene)
        if path is None or not path.exists():
            return None

        self._models[key] = self._yolo_cls(str(path))
        return self._models[key]

    def _predict_with_yolo(self, model: Any, path: Path, scene: SceneType) -> FrameDetection:
        results = model.predict(str(path), verbose=False)
        boxes: list[DetectionBox] = []
        names = getattr(model.model, "names", {}) or getattr(model, "names", {}) or {}
        for result in results:
            result_boxes = getattr(result, "boxes", None)
            if result_boxes is None:
                continue
            for box in result_boxes:
                cls_id = int(box.cls[0])
                raw_label = str(names.get(cls_id, cls_id))
                label = _norm_label(raw_label)
                conf = float(box.conf[0])
                xyxy = [float(v) for v in box.xyxy[0].tolist()]
                boxes.append(DetectionBox(label=label, confidence=conf, xyxy=xyxy))

        return self._score_detection(scene, boxes, backend="yolo")

    def _score_detection(self, scene: SceneType, boxes: list[DetectionBox], backend: str) -> FrameDetection:
        labels = {box.label for box in boxes if box.confidence >= 0.25}
        conf_values = [box.confidence for box in boxes] or [0.35]
        avg_conf = float(np.clip(np.mean(conf_values), 0.0, 1.0))
        missing: list[str] = []

        if scene == SceneType.WELDING:
            if "worker" in labels and "mask" not in labels:
                missing.append("mask")
            if "worker" in labels and "smoke_collector" not in labels:
                missing.append("smoke_collector")
        elif scene == SceneType.CUTTING:
            if "worker" in labels and "extinguisher" not in labels:
                missing.append("extinguisher")

        base_risk = 0.18 if not missing else 0.72
        if "worker" not in labels and scene in {SceneType.WELDING, SceneType.CUTTING}:
            base_risk = max(base_risk, 0.48)
        risk = float(np.clip(base_risk + (0.55 - avg_conf) * 0.18, 0.0, 1.0))

        notes = []
        if backend == "mock":
            notes.append("当前为模拟视觉检测结果，用于无模型环境下演示Agent流程。")

        return FrameDetection(
            scene=scene,
            boxes=boxes,
            risk_score=risk,
            confidence=avg_conf,
            missing_items=missing,
            detector_backend=backend,
            notes=notes,
        )

    def _mock_detection(self, path: Path, scene: SceneType) -> FrameDetection:
        digest = hashlib.sha256(str(path).encode("utf-8")).digest()
        marker = digest[0] / 255.0
        text = str(path).lower()

        boxes = [DetectionBox(label="worker", confidence=0.72 + marker * 0.18, xyxy=[80, 60, 260, 420])]
        explicit_violation = any(key in text for key in ["missing", "violation", "danger", "no_", "bad", "违规"])

        if scene == SceneType.WELDING:
            if not explicit_violation and marker > 0.32:
                boxes.append(DetectionBox(label="mask", confidence=0.64 + marker * 0.20, xyxy=[120, 70, 210, 150]))
            if marker > 0.22:
                boxes.append(DetectionBox(label="smoke_collector", confidence=0.56 + marker * 0.24, xyxy=[300, 80, 520, 260]))
        elif scene == SceneType.CUTTING:
            if not explicit_violation and marker > 0.24:
                boxes.append(DetectionBox(label="extinguisher", confidence=0.58 + marker * 0.23, xyxy=[360, 240, 430, 430]))

        return self._score_detection(scene, boxes, backend="mock")
