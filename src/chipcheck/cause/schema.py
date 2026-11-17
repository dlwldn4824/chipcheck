"""엔진과 바깥 세상 사이의 계약.

이 파일이 이 프로젝트의 경계다. 팹마다 로그 포맷이 다르지만, 어댑터가
`DefectEvent` 하나로 맞춰 주면 엔진은 그 뒤로 아무것도 몰라도 된다.
새 장비·새 회사에 넣을 때 손대는 곳은 어댑터뿐이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DefectEvent:
    """결함 한 건. 어떤 로그에서 왔든 엔진은 이 모양으로만 받는다."""

    defect_id: str
    lot_id: str = ""
    wafer_id: str = ""
    image_path: str | None = None

    # 하드 필터에 쓰는 구조화 필드. 모르면 None으로 두면 필터가 그냥 건너뛴다.
    step: str | None = None        # 공정 스텝 (etch, cmp, litho, cvd, clean, implant)
    equipment: str | None = None   # 장비 또는 챔버 ID
    recipe: str | None = None

    # 결함 형상. 검사 장비가 주는 값.
    geometry: dict[str, float] = field(default_factory=dict)

    # 알람 문구·이벤트 라인을 이어붙인 자유 텍스트. 텍스트 마이닝의 입력.
    text: str = ""

    # 원본 보존 — 어댑터가 버린 필드를 나중에 되살릴 수 있게.
    raw: dict[str, Any] = field(default_factory=dict)

    def query(self) -> str:
        """검색 질의로 쓸 문자열. 구조화 필드도 토큰으로 같이 넣는다."""
        parts = [self.text]
        for v in (self.step, self.equipment, self.recipe):
            if v:
                parts.append(str(v))
        return "  ".join(p for p in parts if p)


@dataclass
class Cause:
    """지식베이스 한 줄. 원인 하나."""

    cause_id: str
    name: str
    step: str = ""                      # 비어 있으면 모든 스텝에 해당
    equipment_hint: str = ""
    defect_classes: tuple[str, ...] = ()  # 이 원인이 만들 수 있는 결함 클래스
    morphology: str = ""                # 이미지에서 어떻게 보이는지
    log_keywords: str = ""              # 로그에 같이 뜨는 문구·코드
    action: str = ""                    # 엔지니어가 취할 조치
    reference: str = ""

    def document(self) -> str:
        """BM25가 색인할 문서."""
        return "  ".join([self.name, self.morphology, self.log_keywords,
                          self.step, self.equipment_hint])


@dataclass
class CauseHit:
    cause: Cause
    score: float
    parts: dict[str, float]      # 점수를 무엇이 만들었는지
    evidence: list[str]          # 로그에서 걸린 토큰
    class_support: float         # 이미지가 이 원인을 얼마나 지지하는지


@dataclass
class Diagnosis:
    """엔진의 출력. 콘솔과 KLARF 되쓰기가 이걸 공유한다."""

    event: DefectEvent
    hits: list[CauseHit]
    abstained: bool = False
    abstain_reason: str = ""
    n_candidates_before: int = 0
    n_candidates_after: int = 0
    elapsed_ms: float = 0.0
