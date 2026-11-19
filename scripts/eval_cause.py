#!/usr/bin/env python3
"""원인 축소 엔진을 조건을 바꿔 가며 측정한다.

    python scripts/eval_cause.py

두 가지를 본다.

  1. 이미지가 순위에 얼마나 보태는가. 같은 로그를 텍스트만으로 랭킹한 경우와
     분류 확률을 함께 쓴 경우를 비교한다.
  2. 원인 수가 늘어도 콘솔에 띄울 만큼 빠른가. 42개를 1,000개까지 불려서 잰다.

가상 로그의 문구는 지식베이스에서 파생되므로, 여기 적중률은 잡음·방해 알람
속에서 올바른 문서를 찾는 검색 성능이다. 실제 팹의 원인 추정 정확도가 아니다.

출력: results/metrics/cause_engine.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if (HERE / "kb.csv").exists():
    # 인계 패키지 안에서 실행 (chipcheck_handoff_*/cause/)
    ROOT = HERE
    sys.path.insert(0, str(HERE))
    KB_DEFAULT, LOGS, CLASSES_CSV = "kb.csv", HERE / "logs", HERE / "classes.csv"
else:
    # 저장소에서 실행
    ROOT = HERE.parent
    sys.path.insert(0, str(ROOT / "src"))
    KB_DEFAULT = "data/cause_kb/kb.csv"
    LOGS = ROOT / "data/cause_kb/logs"
    CLASSES_CSV = ROOT / "data/cause_kb/classes.csv"

from chipcheck.cause.engine import CauseRanker  # noqa: E402
from chipcheck.cause.kb import load_kb  # noqa: E402
from chipcheck.cause.synth import make_case  # noqa: E402

sys.path.insert(0, str(HERE))
from demo_cause import inflate, simulated_probs  # noqa: E402

CLASSES = [f"class_{i}" for i in range(1, 7)]


def run(ranker, cases, use_image: bool, seed: int) -> dict:
    from chipcheck.cause.schema import DefectEvent

    rng = random.Random(seed)
    top1 = top5 = 0
    ms, after = [], []
    for c in cases:
        ev = DefectEvent(defect_id=c.case_id, lot_id=c.lot_id, wafer_id=c.wafer_id,
                         step=c.step or None, equipment=c.equipment, recipe=c.recipe,
                         geometry=c.geometry, text="  ".join(c.alarms))
        probs = simulated_probs(c.truth_class, CLASSES, rng) if use_image else None
        d = ranker.rank(ev, class_probs=probs, top_k=5)
        ids = [h.cause.cause_id for h in d.hits]
        top1 += ids[:1] == [c.truth_cause_id]
        top5 += c.truth_cause_id in ids
        ms.append(d.elapsed_ms)
        after.append(d.n_candidates_after)
    n = len(cases)
    return {"n": n, "top1": round(top1 / n, 4), "top5": round(top5 / n, 4),
            "median_ms": round(statistics.median(ms), 2),
            "max_ms": round(max(ms), 2),
            "median_candidates_after_filter": statistics.median(after)}


def main() -> None:
    ap = argparse.ArgumentParser(description="원인 엔진 측정")
    ap.add_argument("--kb", default=KB_DEFAULT)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/metrics/cause_engine.json"
                    if (ROOT / "results").exists() else "cause_engine.json")
    args = ap.parse_args()

    causes = load_kb(ROOT / args.kb)
    ranker = CauseRanker(causes)

    difficulty = {}
    for keep in (0.3, 0.6, 1.0):
        for dis in (2, 6):
            rng = random.Random(args.seed)
            cases = [make_case(causes, rng, i + 1, keep, dis) for i in range(args.n)]
            key = f"keep{int(keep*100)}_distract{dis}"
            difficulty[key] = {
                "keep_ratio": keep, "n_distractors": dis,
                "text_only": run(ranker, cases, False, args.seed),
                "text_plus_image": run(ranker, cases, True, args.seed),
            }
            t, ti = difficulty[key]["text_only"], difficulty[key]["text_plus_image"]
            print(f"키워드 {keep:.0%} 유지 · 방해 {dis}개  "
                  f"텍스트만 top1 {t['top1']:.1%}  →  이미지 포함 {ti['top1']:.1%}"
                  f"   (top5 {t['top5']:.1%} → {ti['top5']:.1%})")

    print()
    scale = {}
    rng = random.Random(args.seed)
    cases = [make_case(causes, rng, i + 1, 0.6, 2) for i in range(args.n)]
    for size in (42, 200, 1000):
        r = CauseRanker(inflate(causes, size) if size > len(causes) else causes)
        res = run(r, cases, True, args.seed)
        scale[str(size)] = {"median_ms": res["median_ms"], "max_ms": res["max_ms"]}
        print(f"원인 {size:5d}개   건당 {res['median_ms']:5.2f} ms (중앙값), "
              f"최대 {res['max_ms']:5.2f} ms")

    out = {
        "n_causes_real": len(causes),
        "n_cases_per_setting": args.n,
        "note": "가상 로그 문구는 지식베이스에서 파생된다. top1/top5는 잡음과 방해 "
                "알람이 섞인 상태에서 올바른 원인 문서를 찾는 검색 성능이며, "
                "실제 팹의 원인 추정 정확도가 아니다.",
        "latency_note": "건당 시간은 하드 필터 + 이미지 지지도 + BM25 랭킹을 모두 "
                        "포함한 값이다. BM25 점수 계산만 따로 재면 더 짧다.",
        "difficulty": difficulty,
        "scale": scale,
    }
    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
