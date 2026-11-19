#!/usr/bin/env python3
"""보드에서 쓸 Mahalanobis 관문 파라미터를 내보낸다.

    python scripts/export_gate.py

왜 필요한가. DPU가 로짓만 내보내면 관문을 보드에서 쓸 수 없다. 거리 계산은
avgpool 출력(512-d)을 입력으로 받는데, 그 계산에 필요한 클래스 평균과 공분산
역행렬이 인계 패키지에 없었다. 이 스크립트가 그 두 행렬과 임계값을 .npz로
내보낸다. 보드 쪽은 chipcheck.fpga.gate.gate_score()만 복사해 가면 된다.

INT8 주의. 양자화하면 특징값이 float과 달라진다. 양자화된 모델로 특징을 다시
뽑아 이 스크립트를 --feats 옵션으로 한 번 더 돌려야 보드에서 임계값이 맞는다.

출력
  results/handoff/model/mahalanobis_gate.npz
  results/handoff/metadata/gate.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chipcheck.evaluate.calibration import (  # noqa: E402
    MahalanobisScorer, clopper_pearson_upper,
)
from chipcheck.fpga.gate import gate_score, save_gate  # noqa: E402
from chipcheck.utils import load_config  # noqa: E402


def threshold_at_fpr(scores_in: np.ndarray, fpr: float) -> float:
    """분포 안쪽 표본을 fpr 비율만큼 희생하는 임계값."""
    return float(np.quantile(scores_in, fpr))


def zero_error_threshold(score: np.ndarray, correct: np.ndarray) -> float:
    """오류가 처음 나타나기 직전까지 통과시키는 임계값."""
    order = np.argsort(-score)
    wrong = np.where(~correct[order])[0]
    if len(wrong) == 0:
        return float(score.min())
    k = wrong[0]
    return float(score[order][max(k - 1, 0)])


def risk_controlled(score: np.ndarray, correct: np.ndarray,
                    alpha: float = 0.01, delta: float = 0.05) -> float:
    """통과분의 오류율 상한이 alpha를 넘지 않는 가장 낮은 임계값."""
    order = np.argsort(-score)
    s, c = score[order], correct[order]
    best = float(s[0])
    errs = 0
    for i in range(len(s)):
        errs += int(not c[i])
        if clopper_pearson_upper(errs, i + 1, delta) <= alpha:
            best = float(s[i])
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="Mahalanobis 관문 파라미터 내보내기")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--cache", default="results/metrics/_logit_cache.npz")
    ap.add_argument("--feats", default=None,
                    help="INT8 모델로 다시 뽑은 특징 npz (train/val/test_feats, *_y, *_logits)")
    ap.add_argument("--out", default="results/handoff/model/mahalanobis_gate.npz")
    ap.add_argument("--shrink", type=float, default=1e-3)
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    n_cls = cfg["data"]["num_classes"]
    names = cfg["data"]["class_names"]

    src = ROOT / (args.feats or args.cache)
    if not src.exists():
        sys.exit(f"특징 캐시가 없습니다: {src}\n먼저 python scripts/calibrate.py 를 돌리세요.")
    z = np.load(src)
    d = {s: (z[f"{s}_logits"].astype(np.float64),
             z[f"{s}_feats"].astype(np.float64),
             z[f"{s}_y"]) for s in ("train", "val", "test")}

    scorer = MahalanobisScorer.fit(d["train"][1], d["train"][2], n_cls, shrink=args.shrink)

    sc, th = {}, {}
    for s in ("train", "val", "test"):
        logits, feats, y = d[s]
        sc[s] = {"score": gate_score(feats, scorer.means, scorer.precision),
                 "correct": logits.argmax(1) == y}

    th["zero_error_val"] = zero_error_threshold(sc["val"]["score"], sc["val"]["correct"])
    th["risk_1pct_val"] = risk_controlled(sc["val"]["score"], sc["val"]["correct"])
    th["novelty_fpr5_train"] = threshold_at_fpr(sc["train"]["score"], 0.05)
    th["novelty_fpr1_train"] = threshold_at_fpr(sc["train"]["score"], 0.01)

    report = {}
    for name, t in th.items():
        passed = sc["test"]["score"] >= t
        n = int(passed.sum())
        errs = int((~sc["test"]["correct"][passed]).sum())
        report[name] = {
            "threshold": t,
            "test_coverage": round(n / len(passed), 4),
            "test_errors_passed": errs,
            "test_risk_on_passed": round(errs / n, 5) if n else None,
        }
        print(f"{name:22s} thr={t:12.2f}  통과 {n/len(passed):6.1%}  "
              f"오답 통과 {errs}건")

    meta = {
        "feature_layer": "resnet18 avgpool, 512-d, flatten",
        "feature_source": str(src.relative_to(ROOT)),
        "quantized_features": bool(args.feats),
        "n_classes": n_cls,
        "class_names": names,
        "shrink": args.shrink,
        "covariance": "클래스 공통(tied). class_5의 학습 표본이 2장이라 "
                      "클래스별 공분산은 세울 수 없다.",
        "usage": "score = gate_score(feats, means, precision); "
                 "score < threshold 이면 사람에게 넘긴다.",
        "thresholds": report,
        "warning": "이 값은 float32 특징으로 계산했다. INT8 양자화 후에는 "
                   "--feats 로 양자화 특징을 넣어 다시 내보내야 한다.",
    }
    out = save_gate(ROOT / args.out, scorer.means, scorer.precision,
                    {k: float(v) for k, v in th.items()}, meta)

    jp = ROOT / "results/handoff/metadata/gate.json"
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nmeans {scorer.means.shape}  precision {scorer.precision.shape}")
    print(out.relative_to(ROOT))
    print(jp.relative_to(ROOT))


if __name__ == "__main__":
    main()
