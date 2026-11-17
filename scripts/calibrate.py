#!/usr/bin/env python3
"""모델이 '자기가 틀릴 때를 아는가'를 측정한다.

    python scripts/calibrate.py

train 특징으로 Mahalanobis를 세우고, val에서 온도와 임계값을 고른 뒤,
test에서만 보고한다. 결과는 results/metrics/calibration.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chipcheck.evaluate.calibration import (  # noqa: E402
    MahalanobisScorer, aurc, coverage_at_risk, energy_score, risk_controlled_threshold,
    expected_calibration_error, fit_temperature, msp_score, risk_coverage,
    softmax, zero_error_coverage,
)
from chipcheck.models import build_model  # noqa: E402
from chipcheck.utils import ensure_dir, load_config, resolve_device  # noqa: E402


def collect(model, paths, size, device, bs=32):
    """로짓과 마지막 풀링 특징(512-d)을 함께 뽑는다."""
    import numpy as np
    import torch
    from chipcheck.fpga.standalone_model import preprocess_batch

    feats = {}
    h = model.avgpool.register_forward_hook(
        lambda m, i, o: feats.__setitem__("f", o.flatten(1).detach().cpu().numpy())
    )
    L, F = [], []
    with torch.no_grad():
        for i in range(0, len(paths), bs):
            xb = preprocess_batch(paths[i : i + bs], size).to(device)
            L.append(model(xb).cpu().numpy())
            F.append(feats["f"])
    h.remove()
    return np.concatenate(L).astype(np.float64), np.concatenate(F).astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description="신뢰도 보정 · 선택적 예측 평가")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", default="results/checkpoints/best.pt")
    ap.add_argument("--cache", default="results/metrics/_logit_cache.npz")
    args = ap.parse_args()

    import numpy as np
    import torch

    cfg = load_config(ROOT / args.config)
    size = cfg["data"]["image_size"]
    n_cls = cfg["data"]["num_classes"]
    names = cfg["data"]["class_names"]
    device = resolve_device(cfg["train"].get("device", "auto"))

    rows = list(csv.DictReader(open(ROOT / cfg["data"]["splits_dir"] / "split.csv", encoding="utf-8")))
    split = {s: [(ROOT / r["path"], int(r["label"])) for r in rows if r["split"] == s]
             for s in ("train", "val", "test")}

    cache = ROOT / args.cache
    if cache.exists():
        z = np.load(cache)
        data = {s: (z[f"{s}_logits"], z[f"{s}_feats"], z[f"{s}_y"]) for s in split}
        print(f"캐시 사용: {cache.relative_to(ROOT)}")
    else:
        model = build_model(cfg)
        model.load_state_dict(torch.load(ROOT / args.ckpt, map_location="cpu", weights_only=False)["model"])
        model.eval().to(device)
        data, blob = {}, {}
        for s, items in split.items():
            lg, ft = collect(model, [p for p, _ in items], size, device)
            y = np.array([c for _, c in items])
            data[s] = (lg, ft, y)
            blob[f"{s}_logits"], blob[f"{s}_feats"], blob[f"{s}_y"] = lg, ft, y
            print(f"  {s}: {len(y)}장 → 로짓 {lg.shape}, 특징 {ft.shape}")
        ensure_dir(cache.parent)
        np.savez_compressed(cache, **blob)

    (tr_lg, tr_ft, tr_y) = data["train"]
    (va_lg, va_ft, va_y) = data["val"]
    (te_lg, te_ft, te_y) = data["test"]

    # ---- 1. 보정 ----
    T = fit_temperature(va_lg, va_y)
    ece_before = expected_calibration_error(softmax(te_lg), te_y)
    ece_after = expected_calibration_error(softmax(te_lg / T), te_y)
    pred = te_lg.argmax(axis=1)
    correct = pred == te_y
    conf_raw = msp_score(te_lg)
    print(f"\n온도 T = {T:.3f} (val NLL 최소)")
    print(f"ECE  보정 전 {ece_before:.4f} → 보정 후 {ece_after:.4f}")
    print(f"정답 평균 신뢰도 {conf_raw[correct].mean():.3f} / 오답 평균 {conf_raw[~correct].mean():.3f}")

    # ---- 2. 신뢰도 점수 세 가지 ----
    maha = MahalanobisScorer.fit(tr_ft, tr_y, n_cls)
    scores = {
        "MSP": msp_score(te_lg),
        "MSP+T": msp_score(te_lg / T),
        "Energy": energy_score(te_lg, t=1.0),
        "Mahalanobis": maha.score(te_ft),
    }

    out = {
        "n_test": int(len(te_y)),
        "accuracy": float(correct.mean()),
        "n_errors": int((~correct).sum()),
        "temperature": T,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "mean_conf_correct": float(conf_raw[correct].mean()),
        "mean_conf_wrong": float(conf_raw[~correct].mean()),
        "overconfident_errors": int(((~correct) & (conf_raw >= 0.9)).sum()),
        "scores": {},
        "note": "train으로 Mahalanobis 적합, val로 온도 선택, test에서만 보고.",
    }
    print(f"\n{'점수':<13}{'AURC↓':>9}{'오류0 자동통과':>16}{'오류≤0.5% 자동통과':>20}")
    for k, s in scores.items():
        rec = {
            "aurc": aurc(s, correct),
            "zero_error_coverage": zero_error_coverage(s, correct),
            "coverage_at_0.5pct": coverage_at_risk(s, correct, 0.005),
            "coverage_at_1pct": coverage_at_risk(s, correct, 0.01),
        }
        out["scores"][k] = rec
        print(f"{k:<13}{rec['aurc']:>9.5f}{rec['zero_error_coverage']*100:>15.1f}%{rec['coverage_at_0.5pct']*100:>19.1f}%")

    # ---- 3. 설정된 임계값이 실제로 거르는 양 ----
    out["configured_thresholds"] = {}
    for key in ("confidence_threshold", "review_threshold"):
        t = cfg["evaluate"][key]
        keep = conf_raw >= t
        out["configured_thresholds"][key] = {
            "value": t,
            "review_rate": float(1 - keep.mean()),
            "errors_passed": int((~correct & keep).sum()),
        }
    print("\n현재 설정값이 실제로 거르는 양")
    for k, v in out["configured_thresholds"].items():
        print(f"  {k}={v['value']}: 재검토 {v['review_rate']*100:.1f}%, 통과한 오류 {v['errors_passed']}건")

    # ---- 4. 정직한 운영점: 임계값은 val에서 고르고 test에서만 보고 ----
    va_pred = va_lg.argmax(axis=1)
    va_correct = va_pred == va_y
    va_scores = {
        "MSP": msp_score(va_lg),
        "MSP+T": msp_score(va_lg / T),
        "Energy": energy_score(va_lg, t=1.0),
        "Mahalanobis": maha.score(va_ft),
    }
    out["val_selected"] = {}
    print(f"\n임계값을 val에서 고르고 test에 적용한 결과")
    print(f"{'점수':<13}{'test 자동통과':>14}{'재검토':>9}{'통과한 오류':>12}")
    for k in scores:
        vs, ts = va_scores[k], scores[k]
        wrong = vs[~va_correct]
        tau = float(wrong.max()) if len(wrong) else float(vs.min() - 1.0)  # val 오답을 모두 거르는 선
        keep = ts > tau
        rec = {
            "threshold_from_val": tau,
            "test_coverage": float(keep.mean()),
            "test_review_rate": float(1 - keep.mean()),
            "test_errors_passed": int(((~correct) & keep).sum()),
            "test_risk_on_passed": float((~correct)[keep].mean()) if keep.any() else 0.0,
        }
        out["val_selected"][k] = rec
        print(f"{k:<13}{rec['test_coverage']*100:>13.1f}%{rec['test_review_rate']*100:>8.1f}%{rec['test_errors_passed']:>11}건")

    # 재검토로 넘어간 건의 클래스 구성 (희귀 클래스를 잡아내는지)
    import collections
    best_val = min(out["val_selected"], key=lambda k: (out["val_selected"][k]["test_errors_passed"],
                                                       -out["val_selected"][k]["test_coverage"]))
    tau = out["val_selected"][best_val]["threshold_from_val"]
    sent = scores[best_val] <= tau
    out["review_queue_composition"] = {
        names[c]: int(n) for c, n in sorted(collections.Counter(te_y[sent]).items())
    }
    out["best_operating_point"] = best_val
    print(f"\n{best_val} 기준 재검토 큐 {int(sent.sum())}건의 실제 클래스 구성: {out['review_queue_composition']}")

    # ---- 5. 오류율 보증이 붙은 임계값 (Clopper-Pearson 상한) ----
    out["risk_controlled"] = {}
    print(f"\n오류율 1% 이하를 95% 신뢰도로 보증하는 임계값 (val에서 결정)")
    print(f"{'점수':<13}{'val 자동통과':>13}{'test 자동통과':>14}{'test 실제 오류율':>17}{'통과 오류':>10}")
    for k in scores:
        sel = risk_controlled_threshold(va_scores[k], va_correct, alpha=0.01, delta=0.05)
        keep = scores[k] > sel["threshold"]
        rec = dict(sel)
        rec["test_coverage"] = float(keep.mean())
        rec["test_errors_passed"] = int(((~correct) & keep).sum())
        rec["test_risk_on_passed"] = float((~correct)[keep].mean()) if keep.any() else 0.0
        out["risk_controlled"][k] = rec
        print(f"{k:<13}{sel['val_coverage']*100:>12.1f}%{rec['test_coverage']*100:>13.1f}%"
              f"{rec['test_risk_on_passed']*100:>16.2f}%{rec['test_errors_passed']:>9}건")

    # ---- 6. 최고 점수의 risk-coverage 곡선 저장 ----
    best = min(out["scores"], key=lambda k: out["scores"][k]["aurc"])
    rc = risk_coverage(scores[best], correct)
    out["best_score"] = best
    out["curve"] = {"name": best,
                    "coverage": [round(float(v), 4) for v in rc["coverage"][::5]],
                    "risk": [round(float(v), 5) for v in rc["risk"][::5]]}

    dst = ensure_dir(ROOT / cfg["evaluate"]["output_dir"]) / "calibration.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n가장 좋은 점수: {best}  →  {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
