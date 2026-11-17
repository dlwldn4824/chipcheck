#!/usr/bin/env python3
"""희귀 결함을 분류가 아니라 이상 탐지로 잡을 수 있는가.

    python scripts/novelty.py

사진이 4장뿐인 클래스는 어떤 학습 기법으로도 분류기가 일반화할 표본이 아니다.
그러면 "이건 평소와 다르다"로 잡아서 사람에게 넘기면 되는가 ─ 이 스크립트가
그걸 수치로 답한다.

클래스를 하나씩 빼고(leave-one-class-out) 나머지 train 특징으로 Mahalanobis를
세운 뒤, 빠진 클래스의 모든 이미지를 얼마나 잘 골라내는지 AUROC로 잰다.

주의: 특징 추출기(ResNet18)는 6종을 모두 보고 학습했다. 따라서 완전한 미지
클래스 상황보다 낙관적인 수치다. 보고서에 이 단서를 함께 적어야 한다.
결과는 results/metrics/novelty.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chipcheck.evaluate.calibration import MahalanobisScorer  # noqa: E402
from chipcheck.utils import ensure_dir, load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="희귀 클래스 이상 탐지 평가")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--cache", default="results/metrics/_logit_cache.npz")
    ap.add_argument("--fpr", type=float, default=0.05, help="허용 오경보율")
    args = ap.parse_args()

    import numpy as np
    from sklearn.metrics import roc_auc_score

    cfg = load_config(ROOT / args.config)
    names = cfg["data"]["class_names"]
    z = np.load(ROOT / args.cache)
    feats = {s: z[f"{s}_feats"] for s in ("train", "val", "test")}
    ys = {s: z[f"{s}_y"] for s in ("train", "val", "test")}
    all_f = np.concatenate([feats[s] for s in ("train", "val", "test")])
    all_y = np.concatenate([ys[s] for s in ("train", "val", "test")])
    n_cls = len(names)
    counts = {names[c]: int((all_y == c).sum()) for c in range(n_cls)}

    out = {"counts": counts, "fpr_budget": args.fpr, "per_class": {},
           "note": "특징 추출기는 6종을 모두 보고 학습했으므로 완전한 미지 클래스보다 낙관적."}

    print(f"{'빠진 클래스':<10}{'전체 장수':>9}{'AUROC':>9}{'오경보 5%에서 검출률':>22}")
    for c in range(n_cls):
        tr_mask = ys["train"] != c
        if tr_mask.sum() < 50 or (all_y == c).sum() == 0:
            continue
        # 해당 클래스를 빼고 적합
        scorer = MahalanobisScorer.fit(feats["train"][tr_mask], ys["train"][tr_mask], n_cls)
        novel = -scorer.score(all_f[all_y == c])          # 클수록 이상
        normal = -scorer.score(feats["test"][ys["test"] != c])
        s = np.concatenate([normal, novel])
        lab = np.concatenate([np.zeros(len(normal)), np.ones(len(novel))])
        auroc = float(roc_auc_score(lab, s))
        thr = float(np.quantile(normal, 1 - args.fpr))    # 정상에서 5%만 걸리는 선
        tpr = float((novel > thr).mean())
        out["per_class"][names[c]] = {
            "n_images": int(len(novel)), "auroc": round(auroc, 4),
            "detect_rate_at_fpr": round(tpr, 4), "threshold": thr,
        }
        print(f"{names[c]:<10}{len(novel):>9}{auroc:>9.3f}{tpr*100:>21.1f}%")

    # 희귀 3종을 한꺼번에: 흔한 3종만으로 세운 탐지기가 얼마나 잡는가
    RARE = [0, 1, 4]                                       # class_1, class_2, class_5
    common_mask = ~np.isin(ys["train"], RARE)
    scorer = MahalanobisScorer.fit(feats["train"][common_mask], ys["train"][common_mask], n_cls)
    novel = -scorer.score(all_f[np.isin(all_y, RARE)])
    normal = -scorer.score(feats["test"][~np.isin(ys["test"], RARE)])
    s = np.concatenate([normal, novel]); lab = np.concatenate([np.zeros(len(normal)), np.ones(len(novel))])
    thr = float(np.quantile(normal, 1 - args.fpr))
    out["rare_combined"] = {
        "classes": [names[c] for c in RARE],
        "n_rare_images": int(len(novel)),
        "auroc": round(float(roc_auc_score(lab, s)), 4),
        "detect_rate_at_fpr": round(float((novel > thr).mean()), 4),
        "n_common_test": int(len(normal)),
    }
    r = out["rare_combined"]
    print(f"\n흔한 3종만으로 세운 탐지기 → 희귀 3종 {r['n_rare_images']}장")
    print(f"  AUROC {r['auroc']:.3f} · 오경보 {args.fpr*100:.0f}%에서 검출률 {r['detect_rate_at_fpr']*100:.1f}%")

    # 비교: 분류기는 같은 희귀 3종을 얼마나 맞히나
    te_lg = z["test_logits"]; te_y = z["test_y"]
    rare_te = np.isin(te_y, RARE)
    out["classifier_on_rare_test"] = {
        "n": int(rare_te.sum()),
        "correct": int((te_lg[rare_te].argmax(1) == te_y[rare_te]).sum()),
    }
    print(f"  (분류기는 테스트셋 희귀 {int(rare_te.sum())}장 중 "
          f"{out['classifier_on_rare_test']['correct']}장 정답)")

    dst = ensure_dir(ROOT / cfg["evaluate"]["output_dir"]) / "novelty.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
