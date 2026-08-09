# 칩췍 (Chip Check) — AI 티키타카 프로젝트

**AI 기반 반도체 불량 이미지 분석 SW + FPGA 기반 추론 가속기**

GitHub: https://github.com/dlwldn4824/chipcheck

| 항목 | 내용 |
| --- | --- |
| 팀명 | 칩췍 (chip check) |
| 팀원 | 이지우, 김민성, 정회림, 최서영 |
| 멘토 | 이지우 |
| 기간 | 2026.07.01 ~ 2026.11.30 |
| 연구비 | 700,000원 (RISE) |

## 목표

반도체 공정 이미지를 AI로 분석해 불량 여부와 유형을 자동 판별하고, FPGA 가속을 통해 현장에서 즉시 활용할 수 있는 고속 추론 시스템을 구축한다.

## 파이프라인

```
반도체 이미지 입력
        ↓
이미지 전처리
        ↓
관심 영역 검출
        ↓
AI 모델 추론
        ↓
정상·불량 판별
        ↓
불량 유형 분류 및 위치 표시
        ↓
결과 시각화·저장
```

## 폴더 구조

```
AI_티키타카_칩췍/
├── configs/          # 학습·추론 설정
├── data/             # raw / processed / splits
├── docs/             # 회의록, 행정서류, 발표자료
├── notebooks/        # 탐색·실험 노트북
├── results/          # 지표, 그래프, 체크포인트, 오류 사례
├── scripts/          # CLI 진입점
└── src/chipcheck/    # 본 코드
    ├── preprocess/
    ├── models/
    ├── train/
    ├── evaluate/
    ├── inference/
    ├── fpga/
    ├── camera/
    └── viz/
```

## 진행 Phase

1. **이미지 데이터 확보** — 공개 데이터셋 조사·선정
2. **판독 기준 및 파이프라인 설계** — 전처리·판정 규칙
3. **모델 학습 및 성능 평가** — 베이스라인 → 비교
4. **FPGA 기반 추론 가속** — 경량화·양자화·배포
5. **실시간 카메라 연동** (여유 시)

## 빠른 시작

```bash
cd "AI_티키타카_칩췍"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 분할 CSV 생성 → 학습 → 평가
python scripts/preprocess.py
python scripts/train.py --epochs 5          # 빠른 스모크 테스트
python scripts/train.py                     # 기본 15 epoch
python scripts/evaluate.py
```


## 문서

- [프로젝트 로드맵](docs/로드맵.md)
- [역할 분담](docs/역할분담.md)
- [데이터셋 조사](docs/데이터셋_조사.md)
- [회의록 #1](docs/회의록/회의록_01.md)
- [행정 서류](docs/행정/)

## 한 줄 정리

먼저 신뢰할 수 있는 반도체 불량 이미지 데이터와 판독 파이프라인을 구축한 뒤, AI 모델을 학습하고 FPGA 기반 가속을 통해 실시간 불량 분석 시스템으로 확장한다.
