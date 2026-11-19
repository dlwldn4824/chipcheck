"""Mahalanobis 관문을 보드에서 쓰기 위한 파라미터 묶음.

보드 쪽에는 이 저장소가 없다. 그래서 거리 계산에 필요한 값만 .npz 한 장으로
내보내고, 같은 파일을 읽어 점수를 내는 순수 numpy 함수를 여기 같이 둔다.
PS의 ARM 코어에서 돌아가야 하므로 torch에 의존하지 않는다.

파일 안에 들어가는 것
  means       (C, D)  클래스별 특징 평균
  precision   (D, D)  공통 공분산의 역행렬
  thresholds  이름 -> 임계값. score < 임계값이면 사람에게 넘긴다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def gate_score(feats: np.ndarray, means: np.ndarray, precision: np.ndarray) -> np.ndarray:
    """가장 가까운 클래스 중심까지의 마할라노비스 거리(부호 반전).

    값이 클수록 학습 분포에 가깝다. torch 없이 돈다.
    """
    feats = np.asarray(feats, dtype=np.float64)
    if feats.ndim == 1:
        feats = feats[None, :]
    out = np.empty((len(feats), len(means)))
    for c, mu in enumerate(means):
        diff = feats - mu
        out[:, c] = np.einsum("ij,jk,ik->i", diff, precision, diff)
    return -out.min(axis=1)


def save_gate(path: str | Path, means: np.ndarray, precision: np.ndarray,
              thresholds: dict[str, float], meta: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        means=np.asarray(means, dtype=np.float64),
        precision=np.asarray(precision, dtype=np.float64),
        threshold_names=np.array(list(thresholds), dtype=object),
        threshold_values=np.array(list(thresholds.values()), dtype=np.float64),
        meta=np.array(json.dumps(meta or {}, ensure_ascii=False), dtype=object),
    )
    return path


def load_gate(path: str | Path) -> dict:
    z = np.load(path, allow_pickle=True)
    return {
        "means": z["means"],
        "precision": z["precision"],
        "thresholds": dict(zip(z["threshold_names"].tolist(), z["threshold_values"].tolist())),
        "meta": json.loads(str(z["meta"])),
    }
