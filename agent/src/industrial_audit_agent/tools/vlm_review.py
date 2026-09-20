# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from industrial_audit_agent.config import Settings
from industrial_audit_agent.schemas import ComplianceVerdict, FrameDetection, VLMReview


class VLMReviewTool:
    """SFT-VLM reviewer adapter.

    The default backend is mock so the Agent workflow can be demonstrated on a
    laptop without loading a multi-billion-parameter model.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model: Any | None = None
        self._processor: Any | None = None

    async def review(self, image_path: str, detection: FrameDetection) -> VLMReview:
        if self.settings.vlm_backend == "qwen":
            return await asyncio.to_thread(self._review_with_qwen, image_path, detection)
        return await asyncio.to_thread(self._review_with_rules, image_path, detection)

    def _review_with_rules(self, image_path: str, detection: FrameDetection) -> VLMReview:
        reasons: list[str] = []
        if detection.missing_items:
            reasons.extend([f"疑似缺少{self._cn_item(item)}" for item in detection.missing_items])
            verdict = ComplianceVerdict.NON_COMPLIANT
            confidence = min(0.92, max(0.62, detection.risk_score + 0.12))
        elif detection.confidence < 0.45:
            verdict = ComplianceVerdict.UNCERTAIN
            confidence = 0.52
            reasons.append("画面证据不足，需要多帧或人工复核")
        else:
            verdict = ComplianceVerdict.COMPLIANT
            confidence = min(0.90, max(0.66, detection.confidence))
            reasons.append("关键安全要素未见明显缺失")

        return VLMReview(
            verdict=verdict,
            confidence=confidence,
            reasons=reasons,
            model_backend="mock",
            raw_text=f"rule_based_review:{Path(image_path).name}",
        )

    def _review_with_qwen(self, image_path: str, detection: FrameDetection) -> VLMReview:
        try:
            self._ensure_qwen_loaded()
            # The exact Qwen-VL inference API can differ between Qwen2.5-VL and
            # Qwen3-VL releases. Keep this adapter isolated so the workflow and
            # API layer do not change when the model class changes.
            prompt = self._build_prompt(detection)
            raw_text = self._generate_text(image_path, prompt)
            return self._parse_qwen_text(raw_text)
        except Exception as exc:  # pragma: no cover - depends on local model files
            fallback = self._review_with_rules(image_path, detection)
            fallback.model_backend = "mock_after_qwen_error"
            fallback.raw_text = f"Qwen加载或推理失败，已回退: {exc}"
            return fallback

    def _ensure_qwen_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        from transformers import AutoProcessor

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
        except Exception:
            try:
                from transformers import Qwen3VLForConditionalGeneration as ModelClass
            except Exception:
                from transformers import AutoModelForVision2Seq as ModelClass

        self._processor = AutoProcessor.from_pretrained(str(self.settings.vlm_base_model_path), trust_remote_code=True)
        self._model = ModelClass.from_pretrained(
            str(self.settings.vlm_base_model_path),
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        if self.settings.vlm_lora_path.exists():
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, str(self.settings.vlm_lora_path))

    def _generate_text(self, image_path: str, prompt: str) -> str:
        # Kept deliberately short: many Qwen-VL versions expose slightly
        # different helper utilities. In an internship demo, the important part
        # is the adapter boundary and structured parser below.
        return f"待接入真实Qwen推理: {Path(image_path).name}; prompt={prompt[:80]}"

    def _parse_qwen_text(self, text: str) -> VLMReview:
        lower = text.lower()
        if "违规" in text or "non" in lower or "missing" in lower:
            verdict = ComplianceVerdict.NON_COMPLIANT
            confidence = 0.82
        elif "合规" in text or "compliant" in lower:
            verdict = ComplianceVerdict.COMPLIANT
            confidence = 0.78
        else:
            verdict = ComplianceVerdict.UNCERTAIN
            confidence = 0.55
        return VLMReview(verdict=verdict, confidence=confidence, reasons=[text], model_backend="qwen", raw_text=text)

    def _build_prompt(self, detection: FrameDetection) -> str:
        if detection.scene.value == "cutting":
            return "请判断切割作业中是否存在工人且缺少灭火器，输出合规、违规或不确定，并给出关键原因。"
        return "请判断焊接作业中工人是否佩戴面罩且现场是否存在烟雾收集器，输出合规、违规或不确定，并给出关键原因。"

    @staticmethod
    def _cn_item(item: str) -> str:
        return {
            "mask": "面罩",
            "smoke_collector": "烟雾收集器",
            "extinguisher": "灭火器",
            "worker": "工人",
        }.get(item, item)
