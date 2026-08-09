#!/usr/bin/env python3
"""데이터셋 조사 템플릿을 콘솔에 출력하고 체크리스트를 안내한다."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "데이터셋_조사.md"

CRITERIA = [
    "전체 이미지 수",
    "정상/불량 비율",
    "불량 유형 수",
    "해상도",
    "촬영 장비 및 조건",
    "라벨 형식",
    "상업적·연구적 사용 가능 여부",
    "클래스 불균형",
    "라벨 신뢰도",
]


def main() -> None:
    print("=== 칩췍 Phase 1 · 데이터셋 조사 ===\n")
    print(f"조사 시트: {DOC}")
    print("\n확인 기준:")
    for i, c in enumerate(CRITERIA, 1):
        print(f"  {i}. {c}")
    print("\n초기 후보: WM-811K, NEU Surface Defect, MVTec AD, Kaggle 웨이퍼/반도체")
    print("→ docs/데이터셋_조사.md 표를 채운 뒤 최종 선정 섹션을 업데이트하세요.")


if __name__ == "__main__":
    main()
