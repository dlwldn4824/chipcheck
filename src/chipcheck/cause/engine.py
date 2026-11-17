"""원인 후보를 좁히는 엔진. LLM 없음.

세 가지가 순서대로 후보를 줄인다.

  1. 하드 필터   스텝·장비가 안 맞는 원인을 통째로 뺀다. 가장 많이 줄어든다.
  2. 이미지      분류기가 준 클래스 확률로, 이 결함 모양을 만들 수 없는
                 원인을 끌어내린다.
  3. 텍스트      BM25로 로그 문구와 겹치는 원인을 끌어올린다.

이미지가 후보를 자르고 텍스트가 순서를 매긴다. 둘 중 하나만으로는 안 된다.
이미지만으로는 "스크래치다"까지만 가고 그걸 만든 원인 수십 개를 구분하지
못한다. 로그만으로는 어느 결함을 말하는지 모른다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .schema import Cause, CauseHit, DefectEvent, Diagnosis
from .textmine import BM25


@dataclass
class Weights:
    text: float = 1.0
    image: float = 1.0
    rule: float = 0.4


class CauseRanker:
    def __init__(self, causes: list[Cause], weights: Weights | None = None):
        self.causes = {c.cause_id: c for c in causes}
        self.w = weights or Weights()
        self.index = BM25()
        for c in causes:
            self.index.add(c.cause_id, c.document())
        self.index.build()

    # ---- 1. 하드 필터 ----
    def _survivors(self, ev: DefectEvent) -> list[Cause]:
        out = []
        for c in self.causes.values():
            if c.step and ev.step and c.step.lower() != ev.step.lower():
                continue
            out.append(c)
        return out

    # ---- 2. 이미지 지지도 ----
    @staticmethod
    def _class_support(c: Cause, class_probs: dict[str, float] | None) -> float:
        if not class_probs:
            return 1.0           # 이미지가 없으면 모든 원인을 동등하게 둔다
        if not c.defect_classes:
            return 0.5           # 클래스를 안 적어 둔 원인은 중립
        return float(sum(class_probs.get(k, 0.0) for k in c.defect_classes))

    def rank(
        self,
        ev: DefectEvent,
        class_probs: dict[str, float] | None = None,
        ood_ok: bool = True,
        top_k: int = 5,
        min_class_support: float = 0.02,
    ) -> Diagnosis:
        t0 = time.perf_counter()
        n_before = len(self.causes)

        if not ood_ok:
            return Diagnosis(event=ev, hits=[], abstained=True,
                             abstain_reason="학습 분포 밖의 이미지. 원인을 추정하지 않고 사람에게 넘김.",
                             n_candidates_before=n_before, n_candidates_after=0,
                             elapsed_ms=(time.perf_counter() - t0) * 1000)

        survivors = self._survivors(ev)
        query = ev.query()
        text_scores = self.index.score(query)
        max_text = max(text_scores.values()) if text_scores else 0.0

        hits: list[CauseHit] = []
        for c in survivors:
            support = self._class_support(c, class_probs)
            if class_probs and support < min_class_support:
                continue        # 이 결함 모양을 못 만드는 원인
            t_raw = text_scores.get(c.cause_id, 0.0)
            t_n = t_raw / max_text if max_text else 0.0
            rule = 0.0
            if c.equipment_hint and ev.equipment and c.equipment_hint.lower() in ev.equipment.lower():
                rule += 0.6
            if c.step and ev.step and c.step.lower() == ev.step.lower():
                rule += 0.4
            parts = {"text": self.w.text * t_n,
                     "image": self.w.image * support,
                     "rule": self.w.rule * rule}
            hits.append(CauseHit(cause=c, score=sum(parts.values()), parts=parts,
                                 evidence=self.index.matched_terms(c.cause_id, query),
                                 class_support=support))

        hits.sort(key=lambda h: -h.score)
        return Diagnosis(event=ev, hits=hits[:top_k],
                         n_candidates_before=n_before, n_candidates_after=len(hits),
                         elapsed_ms=(time.perf_counter() - t0) * 1000)
