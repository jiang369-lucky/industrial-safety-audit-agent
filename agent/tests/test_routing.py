from industrial_audit_agent.config import Settings
from industrial_audit_agent.schemas import FrameDetection, SceneType
from industrial_audit_agent.workflow import SafetyAuditWorkflow


def test_high_confidence_direct_route():
    workflow = SafetyAuditWorkflow(Settings(detector_backend="mock", vlm_backend="mock"))
    detection = FrameDetection(scene=SceneType.WELDING, risk_score=0.85, confidence=0.9, missing_items=["mask"])
    route = workflow._route(detection)
    assert route.route.value == "direct_report"


def test_medium_risk_uses_vlm():
    workflow = SafetyAuditWorkflow(Settings(detector_backend="mock", vlm_backend="mock"))
    detection = FrameDetection(scene=SceneType.WELDING, risk_score=0.48, confidence=0.65)
    route = workflow._route(detection)
    assert route.route.value == "vlm_review"
