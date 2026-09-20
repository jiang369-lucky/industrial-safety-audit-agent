# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any


def tool_manifest() -> list[dict[str, Any]]:
    """MCP-style tool descriptors with JSON Schema input contracts."""

    return [
        {
            "name": "detect_frame",
            "description": "Detect industrial hot-work safety objects with YOLO or fallback rules.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string"},
                    "scene_hint": {"type": "string", "enum": ["auto", "welding", "cutting", "idle"]},
                    "stream_id": {"type": "string"},
                },
                "required": ["image_path"],
            },
        },
        {
            "name": "vlm_review",
            "description": "Review ambiguous frames with an SFT-VLM adapter.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string"},
                    "detection": {"type": "object"},
                },
                "required": ["image_path", "detection"],
            },
        },
        {
            "name": "retrieve_policy",
            "description": "Retrieve hot-work safety policy evidence with BM25 and optional Milvus extension.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
            },
        },
        {
            "name": "generate_audit_report",
            "description": "Generate a JSON-Schema-constrained compliance audit report.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string"},
                    "scene_hint": {"type": "string"},
                },
                "required": ["image_path"],
            },
        },
    ]
