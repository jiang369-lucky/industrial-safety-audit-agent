# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
from datetime import datetime

from industrial_audit_agent.schemas import (
    AuditReport,
    ComplianceVerdict,
    FrameDetection,
    PolicyEvidence,
    RouteDecision,
    TemporalEvidence,
    VLMReview,
)


def make_report_id(seed: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8]
    return f"AUD-{stamp}-{digest}"


def build_actions(verdict: ComplianceVerdict, missing_items: list[str]) -> list[str]:
    if verdict == ComplianceVerdict.COMPLIANT:
        return ["维持当前作业状态，保留本次审核记录。"]
    if verdict == ComplianceVerdict.UNCERTAIN:
        return ["进入多帧复核或人工审核，暂不自动放行。"]
    if verdict == ComplianceVerdict.NEED_HUMAN:
        return ["通知安全员进行现场复查。"]

    actions = ["触发安全告警，暂停当前动火作业。"]
    if "mask" in missing_items:
        actions.append("要求作业人员正确佩戴焊接面罩。")
    if "smoke_collector" in missing_items:
        actions.append("检查并开启烟雾收集装置。")
    if "extinguisher" in missing_items:
        actions.append("补齐灭火器后再恢复切割作业。")
    return actions


def decide_final_verdict(
    detection: FrameDetection,
    route: RouteDecision,
    vlm_review: VLMReview | None,
    temporal: TemporalEvidence | None,
) -> tuple[ComplianceVerdict, float, float]:
    risk = detection.risk_score
    confidence = detection.confidence

    if temporal is not None:
        risk = 0.55 * risk + 0.45 * temporal.smoothed_risk
        confidence = max(confidence, temporal.smoothed_confidence)

    if vlm_review is not None:
        if vlm_review.verdict == ComplianceVerdict.NON_COMPLIANT:
            risk = max(risk, 0.72 + 0.18 * vlm_review.confidence)
        elif vlm_review.verdict == ComplianceVerdict.COMPLIANT:
            risk = min(risk, 0.32)
        confidence = max(confidence, vlm_review.confidence)

    if route.route.value == "human_review":
        return ComplianceVerdict.NEED_HUMAN, float(risk), float(confidence)
    if risk >= 0.62:
        return ComplianceVerdict.NON_COMPLIANT, float(risk), float(confidence)
    if risk <= 0.35 and confidence >= 0.55:
        return ComplianceVerdict.COMPLIANT, float(risk), float(confidence)
    return ComplianceVerdict.UNCERTAIN, float(risk), float(confidence)


def build_audit_report(
    request_path: str,
    detection: FrameDetection,
    route: RouteDecision,
    policies: list[PolicyEvidence],
    vlm_review: VLMReview | None = None,
    temporal: TemporalEvidence | None = None,
) -> AuditReport:
    verdict, risk, confidence = decide_final_verdict(detection, route, vlm_review, temporal)
    actions = build_actions(verdict, detection.missing_items)
    report = AuditReport(
        report_id=make_report_id(request_path),
        scene=detection.scene,
        verdict=verdict,
        risk_score=round(risk, 4),
        confidence=round(confidence, 4),
        route=route,
        missing_items=detection.missing_items,
        policy_evidence=policies,
        vlm_review=vlm_review,
        temporal_evidence=temporal,
        actions=actions,
    )
    report.markdown = render_markdown(report)
    return report


def render_markdown(report: AuditReport) -> str:
    scene_cn = {"welding": "焊接", "cutting": "切割", "idle": "空闲", "auto": "自动"}.get(report.scene.value, report.scene.value)
    verdict_cn = {
        "compliant": "合规",
        "non_compliant": "违规",
        "uncertain": "不确定",
        "need_human": "人工复核",
    }[report.verdict.value]
    missing = "、".join(report.missing_items) if report.missing_items else "无"
    actions = "\n".join([f"- {item}" for item in report.actions])
    evidence = "\n".join([f"- {item.title}: {item.content[:90]}" for item in report.policy_evidence]) or "- 未检索到匹配规范"
    return (
        f"## 工业安全审核报告\n\n"
        f"- 报告编号：{report.report_id}\n"
        f"- 作业场景：{scene_cn}\n"
        f"- 审核结论：{verdict_cn}\n"
        f"- 风险分数：{report.risk_score:.3f}\n"
        f"- 置信度：{report.confidence:.3f}\n"
        f"- 路由策略：{report.route.route.value}（{report.route.reason}）\n"
        f"- 缺失要素：{missing}\n\n"
        f"### 规范依据\n{evidence}\n\n"
        f"### 处置建议\n{actions}\n"
    )
