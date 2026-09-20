# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from industrial_audit_agent.config import Settings
from industrial_audit_agent.schemas import PolicyEvidence


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    words = re.findall(r"[a-z0-9_]+", text)
    cjk = re.findall(r"[\u4e00-\u9fff]", text)
    return words + cjk


@dataclass
class _PolicyChunk:
    policy_id: str
    title: str
    content: str
    tokens: list[str]


class PolicyRetriever:
    """Small BM25 retriever; can be replaced by Milvus in production."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._chunks = self._load_chunks(settings.policy_path)
        self._avg_len = sum(len(chunk.tokens) for chunk in self._chunks) / max(1, len(self._chunks))
        self._df = self._document_frequency()

    def retrieve(self, query: str, top_k: int = 3) -> list[PolicyEvidence]:
        query_tokens = _tokenize(query)
        scored = []
        for chunk in self._chunks:
            score = self._bm25(query_tokens, chunk)
            scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            PolicyEvidence(
                policy_id=chunk.policy_id,
                title=chunk.title,
                content=chunk.content,
                score=round(float(score), 4),
            )
            for score, chunk in scored[:top_k]
            if score > 0
        ]

    def _load_chunks(self, path: Path) -> list[_PolicyChunk]:
        if not path.exists():
            return []
        text = path.read_text(encoding="utf-8")
        blocks = re.split(r"\n(?=##\s+)", text)
        chunks: list[_PolicyChunk] = []
        for idx, block in enumerate(blocks, start=1):
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if not lines:
                continue
            title = lines[0].lstrip("# ").strip()
            content = "\n".join(lines[1:]) if len(lines) > 1 else title
            chunks.append(_PolicyChunk(f"P{idx:03d}", title, content, _tokenize(block)))
        return chunks

    def _document_frequency(self) -> dict[str, int]:
        df: dict[str, int] = {}
        for chunk in self._chunks:
            for token in set(chunk.tokens):
                df[token] = df.get(token, 0) + 1
        return df

    def _bm25(self, query_tokens: list[str], chunk: _PolicyChunk) -> float:
        if not query_tokens or not chunk.tokens:
            return 0.0
        k1 = 1.5
        b = 0.75
        n_docs = max(1, len(self._chunks))
        length_norm = k1 * (1 - b + b * len(chunk.tokens) / max(1.0, self._avg_len))
        score = 0.0
        counts: dict[str, int] = {}
        for token in chunk.tokens:
            counts[token] = counts.get(token, 0) + 1
        for token in query_tokens:
            tf = counts.get(token, 0)
            if tf == 0:
                continue
            df = self._df.get(token, 0)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            score += idf * (tf * (k1 + 1)) / (tf + length_norm)
        return score
