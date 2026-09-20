# -*- coding: utf-8 -*-
from __future__ import annotations

from industrial_audit_agent.config import get_settings
from industrial_audit_agent.schemas import AuditRequest, SceneType
from industrial_audit_agent.workflow import SafetyAuditWorkflow


try:
    from mcp.server.fastmcp import FastMCP
except Exception as exc:  # pragma: no cover
    raise SystemExit("MCP dependency is missing. Install with: pip install mcp") from exc


mcp = FastMCP("industrial-safety-audit")
workflow = SafetyAuditWorkflow(get_settings())


@mcp.tool()
async def audit_image(image_path: str, scene_hint: str = "auto", stream_id: str = "mcp") -> dict:
    """Audit one industrial hot-work image and return a structured report."""

    request = AuditRequest(image_path=image_path, scene_hint=SceneType(scene_hint), stream_id=stream_id)
    report = await workflow.run(request)
    return report.model_dump(mode="json")


@mcp.tool()
async def retrieve_policy(query: str, top_k: int = 3) -> list[dict]:
    """Retrieve relevant hot-work safety policy snippets."""

    rows = workflow.retriever.retrieve(query, top_k=top_k)
    return [row.model_dump(mode="json") for row in rows]


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
