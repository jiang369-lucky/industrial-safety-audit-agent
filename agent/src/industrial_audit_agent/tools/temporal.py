# -*- coding: utf-8 -*-
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from industrial_audit_agent.schemas import FrameDetection, TemporalEvidence


class TemporalValidator:
    """Multi-frame memory for low-confidence or oscillating samples."""

    def __init__(self, window_size: int = 8):
        self.window_size = window_size
        self._history: dict[str, deque[tuple[float, float]]] = defaultdict(lambda: deque(maxlen=window_size))

    def update(self, stream_id: str, detection: FrameDetection) -> TemporalEvidence:
        history = self._history[stream_id]
        history.append((detection.risk_score, detection.confidence))
        risks = np.array([item[0] for item in history], dtype=np.float32)
        confs = np.array([item[1] for item in history], dtype=np.float32)
        if len(risks) <= 1:
            oscillation = 0.0
        else:
            oscillation = float(np.mean(np.abs(np.diff(risks))))
        return TemporalEvidence(
            smoothed_risk=float(np.clip(np.mean(risks), 0.0, 1.0)),
            smoothed_confidence=float(np.clip(np.mean(confs), 0.0, 1.0)),
            oscillation=float(np.clip(oscillation, 0.0, 1.0)),
            frame_count=len(history),
        )
