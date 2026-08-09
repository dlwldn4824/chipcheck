from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Prediction:
    label: str
    confidence: float
    needs_review: bool
    latency_ms: float


class Predictor:
    """단일 이미지 추론 + 신뢰도 임계값 기반 재검토 플래그."""

    def __init__(self, model, class_names: list[str], cfg: dict[str, Any], device: str):
        self.model = model
        self.class_names = class_names
        self.cfg = cfg
        self.device = device
        self.conf_th = cfg["evaluate"]["confidence_threshold"]
        self.review_th = cfg["evaluate"]["review_threshold"]

    def predict(self, batch) -> list[Prediction]:
        import time

        import torch
        import torch.nn.functional as F

        self.model.eval()
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self.model(batch.to(self.device))
            probs = F.softmax(logits, dim=1)
            confs, idxs = probs.max(dim=1)
        latency = (time.perf_counter() - t0) * 1000 / max(batch.size(0), 1)

        results = []
        for conf, idx in zip(confs.tolist(), idxs.tolist()):
            results.append(
                Prediction(
                    label=self.class_names[idx],
                    confidence=conf,
                    needs_review=conf < self.review_th,
                    latency_ms=latency,
                )
            )
        return results
