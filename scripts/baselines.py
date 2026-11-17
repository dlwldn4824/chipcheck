#!/usr/bin/env python3
"""학습 모델이 실제로 얼마나 기여했는지 재기 위한 기준선.

    python scripts/baselines.py

같은 분할, 같은 테스트셋으로 비교한다.
  최빈 클래스   항상 가장 흔한 결함으로 찍기
  규칙 기반     픽셀 통계 8개 + 얕은 결정트리 (사람이 손으로 짤 법한 수준)
  ResNet18      학습 모델
결과는 results/metrics/baselines.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chipcheck.utils import ensure_dir, load_config  # noqa: E402

FEATURE_NAMES = ["밝기 평균", "밝기 표준편차", "하위 5% 밝기", "상위 95% 밝기",
                 "가로 엣지 밀도", "세로 엣지 밀도", "어두운 점 비율", "밝은 점 비율"]


def handcrafted(path, side: int = 64):
    """사람이 눈으로 고를 법한 단순 영상 통계."""
    import numpy as np
    from PIL import Image

    g = np.asarray(Image.open(path).convert("L").resize((side, side), Image.BILINEAR), dtype=np.float32)
    mu, sd = g.mean(), g.std()
    return [mu, sd, np.percentile(g, 5), np.percentile(g, 95),
            np.abs(np.diff(g, axis=1)).mean(), np.abs(np.diff(g, axis=0)).mean(),
            float((g < mu - 2 * sd).mean()), float((g > mu + 2 * sd).mean())]


def main() -> None:
    ap = argparse.ArgumentParser(description="규칙 기반 기준선 비교")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--depth", type=int, default=4, help="결정트리 깊이")
    args = ap.parse_args()

    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.tree import DecisionTreeClassifier, export_text

    cfg = load_config(ROOT / args.config)
    names = cfg["data"]["class_names"]
    rows = list(csv.DictReader(open(ROOT / cfg["data"]["splits_dir"] / "split.csv", encoding="utf-8")))

    X, Y = {"train": [], "test": []}, {"train": [], "test": []}
    t0 = time.perf_counter()
    for r in rows:
        s = r["split"]
        if s == "val":
            continue
        X[s].append(handcrafted(ROOT / r["path"]))
        Y[s].append(int(r["label"]))
    Xtr, Ytr = np.array(X["train"]), np.array(Y["train"])
    Xte, Yte = np.array(X["test"]), np.array(Y["test"])
    print(f"특징 추출 {len(Ytr) + len(Yte)}장 · {time.perf_counter() - t0:.1f}s")

    def score(y_pred):
        return {"accuracy": float(accuracy_score(Yte, y_pred)),
                "f1_macro": float(f1_score(Yte, y_pred, average="macro", zero_division=0))}

    out = {"n_train": int(len(Ytr)), "n_test": int(len(Yte)), "baselines": {}}

    out["baselines"]["최빈 클래스"] = score(np.full_like(Yte, np.bincount(Ytr).argmax()))
    out["baselines"]["최빈 클래스"]["설명"] = f"항상 {names[int(np.bincount(Ytr).argmax())]}로 찍기"

    tree = DecisionTreeClassifier(max_depth=args.depth, random_state=cfg["project"]["seed"]).fit(Xtr, Ytr)
    out["baselines"]["규칙 기반"] = score(tree.predict(Xte))
    out["baselines"]["규칙 기반"]["설명"] = f"픽셀 통계 {Xtr.shape[1]}개 + 깊이 {args.depth} 결정트리"
    out["baselines"]["규칙 기반"]["사용한 특징"] = FEATURE_NAMES
    out["baselines"]["규칙 기반"]["트리"] = export_text(
        tree, feature_names=FEATURE_NAMES, max_depth=2).strip().splitlines()[:12]

    model_metrics = json.loads((ROOT / cfg["evaluate"]["output_dir"] / "test_metrics.json").read_text(encoding="utf-8"))
    out["baselines"]["ResNet18 (우리 모델)"] = {
        "accuracy": model_metrics["accuracy"],
        "f1_macro": model_metrics["f1_macro"],
        "설명": "ImageNet 사전학습 + 가중 샘플링 5 epoch",
    }

    dst = ensure_dir(ROOT / cfg["evaluate"]["output_dir"]) / "baselines.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'기준선':<22}{'accuracy':>10}{'macro-F1':>11}")
    for k, v in out["baselines"].items():
        print(f"{k:<22}{v['accuracy'] * 100:>9.2f}%{v['f1_macro']:>11.3f}")
    print(f"\n→ {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
