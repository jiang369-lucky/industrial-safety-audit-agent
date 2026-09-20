# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, TypedDict

from industrial_audit_agent.config import Settings, get_settings
from industrial_audit_agent.report import build_audit_report
from industrial_audit_agent.schemas import (
    AuditReport,
    AuditRequest,
    ComplianceVerdict,
    FrameDetection,
    PolicyEvidence,
    RouteDecision,
    RouteType,
    TemporalEvidence,
    VLMReview,
)
from industrial_audit_agent.tools.rag_retriever import PolicyRetriever
from industrial_audit_agent.tools.temporal import TemporalValidator
from industrial_audit_agent.tools.vision_detection import VisionDetectionTool
from industrial_audit_agent.tools.vlm_review import VLMReviewTool


class AuditState(TypedDict, total=False):
    request: AuditRequest
    detection: FrameDetection
    route: RouteDecision
    vlm_review: VLMReview | None
    temporal: TemporalEvidence | None
    policies: list[PolicyEvidence]
    report: AuditReport


class SafetyAuditWorkflow:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.detector = VisionDetectionTool(self.settings)
        self.vlm = VLMReviewTool(self.settings)
        self.retriever = PolicyRetriever(self.settings)
        self.temporal = TemporalValidator(window_size=8)
        self._graph = self._try_build_langgraph()

    async def run(self, request: AuditRequest) -> AuditReport:
        if self._graph is not None:
            state: AuditState = await self._graph.ainvoke({"request": request})
            return state["report"]
        return await self._run_fallback(request)

    async def _run_fallback(self, request: AuditRequest) -> AuditReport:
        detection = await self.detector.detect_image(request)
        route = self._route(detection)
        temporal: TemporalEvidence | None = None
        vlm_review: VLMReview | None = None

        if route.route == RouteType.TEMPORAL_CHECK:
            temporal = self.temporal.update(request.stream_id, detection)
            if temporal.frame_count < 3 or temporal.oscillation > 0.18:
                route = RouteDecision(
                    route=RouteType.HUMAN_REVIEW,
                    reason="低置信或跳变样本不足以自动判定",
                    confidence_band="low",
                )
            elif 0.35 < temporal.smoothed_risk < 0.65:
                vlm_review = await self.vlm.review(request.image_path, detection)
        elif route.route == RouteType.VLM_REVIEW:
            vlm_review = await self.vlm.review(request.image_path, detection)

        policies = self._retrieve_policy(detection, route, vlm_review)
        return build_audit_report(request.image_path, detection, route, policies, vlm_review, temporal)

    def _route(self, detection: FrameDetection) -> RouteDecision:
        conf = detection.confidence
        risk = detection.risk_score

        if conf < self.settings.low_confidence:
            return RouteDecision(route=RouteType.TEMPORAL_CHECK, reason="检测置信度较低，进入多帧校验", confidence_band="low")

        if conf >= self.settings.high_confidence and (risk <= self.settings.safe_risk or risk >= self.settings.alert_risk):
            return RouteDecision(route=RouteType.DIRECT_REPORT, reason="视觉证据充分，直接生成审核结论", confidence_band="high")

        if self.settings.safe_risk < risk < self.settings.alert_risk:
            return RouteDecision(route=RouteType.VLM_REVIEW, reason="风险处于边界区间，调用VLM复核", confidence_band="medium")

        return RouteDecision(route=RouteType.TEMPORAL_CHECK, reason="证据存在不稳定性，进入时序校验", confidence_band="medium")

    def _retrieve_policy(
        self,
        detection: FrameDetection,
        route: RouteDecision,
        vlm_review: VLMReview | None,
    ) -> list[PolicyEvidence]:
        scene = detection.scene.value
        missing = " ".join(detection.missing_items)
        verdict = vlm_review.verdict.value if vlm_review else "visual_only"
        query = f"{scene} {missing} {verdict} {route.reason} 动火 作业 安全 合规"
        return self.retriever.retrieve(query, top_k=3)

    def _try_build_langgraph(self) -> Any | None:
        try:
            from langgraph.graph import END, StateGraph
        except Exception:
            return None

        graph = StateGraph(AuditState)

        async def detect_node(state: AuditState) -> AuditState:
            detection = await self.detector.detect_image(state["request"])
            return {"detection": detection}

        def route_node(state: AuditState) -> AuditState:
            return {"route": self._route(state["detection"])}

        async def vlm_node(state: AuditState) -> AuditState:
            review = await self.vlm.review(state["request"].image_path, state["detection"])
            return {"vlm_review": review}

        def temporal_node(state: AuditState) -> AuditState:
            evidence = self.temporal.update(state["request"].stream_id, state["detection"])
            return {"temporal": evidence}

        def policy_node(state: AuditState) -> AuditState:
            policies = self._retrieve_policy(state["detection"], state["route"], state.get("vlm_review"))
            return {"policies": policies}

        def report_node(state: AuditState) -> AuditState:
            report = build_audit_report(
                state["request"].image_path,
                state["detection"],
                state["route"],
                state.get("policies", []),
                state.get("vlm_review"),
                state.get("temporal"),
            )
            return {"report": report}

        def human_node(state: AuditState) -> AuditState:
            review = VLMReview(
                verdict=ComplianceVerdict.NEED_HUMAN,
                confidence=0.5,
                reasons=["系统证据不足，转人工复核"],
                model_backend="human_fallback",
            )
            return {"vlm_review": review}

        def choose_after_route(state: AuditState) -> str:
            route = state["route"].route
            if route == RouteType.VLM_REVIEW:
                return "vlm"
            if route == RouteType.TEMPORAL_CHECK:
                return "temporal"
            if route == RouteType.HUMAN_REVIEW:
                return "human"
            return "policy"

        def choose_after_temporal(state: AuditState) -> str:
            temporal = state["temporal"]
            if temporal.frame_count < 3 or temporal.oscillation > 0.18:
                return "human"
            if 0.35 < temporal.smoothed_risk < 0.65:
                return "vlm"
            return "policy"

        graph.add_node("detect", detect_node)
        graph.add_node("route", route_node)
        graph.add_node("vlm", vlm_node)
        graph.add_node("temporal", temporal_node)
        graph.add_node("human", human_node)
        graph.add_node("policy", policy_node)
        graph.add_node("report", report_node)

        graph.set_entry_point("detect")
        graph.add_edge("detect", "route")
        graph.add_conditional_edges("route", choose_after_route, {"vlm": "vlm", "temporal": "temporal", "human": "human", "policy": "policy"})
        graph.add_conditional_edges("temporal", choose_after_temporal, {"vlm": "vlm", "human": "human", "policy": "policy"})
        graph.add_edge("vlm", "policy")
        graph.add_edge("human", "policy")
        graph.add_edge("policy", "report")
        graph.add_edge("report", END)
        return graph.compile()
