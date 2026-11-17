"""신뢰도 보정과 선택적 예측(재검토 큐) 평가.

정확도만 보면 모델이 쓸 만해 보이지만, 자동 판정을 걸려면 "모델이 자기가
틀릴 때를 아는가"를 따로 재야 한다. 이 모듈은 그 두 가지를 다룬다.

  1. 보정 — softmax 확률이 실제 정답률과 맞는지 (ECE), temperature scaling
  2. 선택적 예측 — 확신이 낮은 건을 사람에게 넘길 때 남는 구간의 오류율

신뢰도 점수는 세 가지를 비교한다.
  MSP          최대 softmax 확률. 가장 흔하지만 과신하는 것으로 알려져 있다.
  Energy       -logsumexp(logits). 로짓 크기를 그대로 쓴다.
  Mahalanobis  마지막 특징 공간에서 클래스 중심까지의 거리. 공분산은 클래스
               공통(tied)으로 추정한다 — 표본이 2장뿐인 클래스가 있어
               클래스별 공분산은 세울 수 없다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """구간별 |평균 신뢰도 - 실제 정답률|의 표본 가중 평균."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """검증셋 NLL을 최소화하는 온도 T. 로짓을 T로 나눠 쓴다."""
    best_t, best_nll = 1.0, float("inf")
    for t in np.concatenate([np.arange(0.20, 3.01, 0.01), np.arange(3.1, 10.1, 0.1)]):
        p = softmax(logits / t)
        nll = -np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, None)).mean()
        if nll < best_nll:
            best_nll, best_t = float(nll), float(t)
    return best_t


def msp_score(logits: np.ndarray) -> np.ndarray:
    return softmax(logits).max(axis=1)


def energy_score(logits: np.ndarray, t: float = 1.0) -> np.ndarray:
    """에너지가 낮을수록 분포 안쪽. 부호를 뒤집어 '클수록 확신'으로 맞춘다."""
    z = logits / t
    m = z.max(axis=1, keepdims=True)
    return float(t) * (m.squeeze(1) + np.log(np.exp(z - m).sum(axis=1)))


@dataclass
class MahalanobisScorer:
    means: np.ndarray      # (C, D)
    precision: np.ndarray  # (D, D)

    @classmethod
    def fit(cls, feats: np.ndarray, labels: np.ndarray, n_classes: int, shrink: float = 1e-3):
        d = feats.shape[1]
        means = np.zeros((n_classes, d), dtype=np.float64)
        centered = []
        for c in range(n_classes):
            m = labels == c
            if m.sum() == 0:
                continue
            means[c] = feats[m].mean(axis=0)
            centered.append(feats[m] - means[c])
        centered = np.concatenate(centered, axis=0)
        cov = centered.T @ centered / max(len(centered) - n_classes, 1)
        cov += shrink * np.trace(cov) / d * np.eye(d)   # 표본이 적은 클래스 때문에 정칙화
        return cls(means=means, precision=np.linalg.inv(cov))

    def score(self, feats: np.ndarray) -> np.ndarray:
        """가장 가까운 클래스 중심까지의 마할라노비스 거리(부호 반전)."""
        out = np.empty((len(feats), len(self.means)))
        for c, mu in enumerate(self.means):
            diff = feats - mu
            out[:, c] = np.einsum("ij,jk,ik->i", diff, self.precision, diff)
        return -out.min(axis=1)


def risk_coverage(score: np.ndarray, correct: np.ndarray) -> dict[str, np.ndarray]:
    """점수가 높은 순으로 자동 통과시킬 때의 커버리지별 오류율."""
    order = np.argsort(-score)
    ok = correct[order].astype(float)
    n = len(ok)
    cum_err = np.cumsum(1.0 - ok)
    k = np.arange(1, n + 1)
    return {"coverage": k / n, "risk": cum_err / k, "n_errors": cum_err}


def zero_error_coverage(score: np.ndarray, correct: np.ndarray) -> float:
    """오류가 한 건도 통과하지 않는 최대 자동 통과 비율."""
    rc = risk_coverage(score, correct)
    clean = rc["n_errors"] == 0
    return float(rc["coverage"][clean].max()) if clean.any() else 0.0


def coverage_at_risk(score: np.ndarray, correct: np.ndarray, max_risk: float) -> float:
    """허용 오류율 이하를 지키는 최대 자동 통과 비율."""
    rc = risk_coverage(score, correct)
    ok = rc["risk"] <= max_risk
    return float(rc["coverage"][ok].max()) if ok.any() else 0.0


def aurc(score: np.ndarray, correct: np.ndarray) -> float:
    """risk-coverage 곡선 아래 넓이. 낮을수록 '틀릴 때를 잘 안다'."""
    rc = risk_coverage(score, correct)
    return float(np.trapezoid(rc["risk"], rc["coverage"]))


def clopper_pearson_upper(k: int, n: int, delta: float = 0.05) -> float:
    """오류 k건/표본 n건일 때 참 오류율의 (1-delta) 상한."""
    from scipy.stats import beta

    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(beta.ppf(1.0 - delta, k + 1, n - k))


def risk_controlled_threshold(
    score: np.ndarray, correct: np.ndarray, alpha: float = 0.01, delta: float = 0.05
) -> dict[str, float]:
    """자동 통과 구간의 오류율이 alpha 이하임을 (1-delta) 신뢰도로 보증하는 임계값.

    검증셋 오류율의 점추정이 아니라 Clopper-Pearson 상한을 쓴다. 표본이 적을수록
    상한이 느슨해져 임계값이 저절로 보수적으로 잡힌다 — 희귀 클래스가 1~2장뿐인
    이 데이터에서 중요한 성질이다.
    """
    order = np.argsort(-score)
    s_sorted = score[order]
    ok = correct[order].astype(int)
    n_err = np.cumsum(1 - ok)
    best = {"threshold": float("inf"), "val_coverage": 0.0, "val_risk_ucb": 0.0}
    for i in range(len(ok)):
        ucb = clopper_pearson_upper(int(n_err[i]), i + 1, delta)
        if ucb <= alpha:
            best = {
                "threshold": float(s_sorted[i]),
                "val_coverage": (i + 1) / len(ok),
                "val_risk_ucb": ucb,
            }
    return best
