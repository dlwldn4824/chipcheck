#!/usr/bin/env python3
"""원인 축소 엔진 시연.

    python scripts/demo_cause.py                       # 기본 (KLARF 가상 로그)
    python scripts/demo_cause.py --format flat_csv     # 다른 포맷, 같은 엔진
    python scripts/demo_cause.py --inflate 1000        # 원인 1,000개로 불려서 속도 측정
    python scripts/demo_cause.py --cases 3 --verbose   # 콘솔 출력 자세히

로그 한 줄도 바뀌지 않고 어댑터만 바뀐다는 점, 그리고 후보 N개가 상위 5개로
줄어든다는 점을 보이는 것이 목적이다.

정확도 수치에 대한 주의. 가상 로그의 문구는 지식베이스에서 파생된다. 그래서
여기 나오는 적중률은 '잡음과 방해 알람이 섞였을 때 올바른 문서를 찾아내는
검색 성능'이지, 실제 팹에서의 원인 추정 정확도가 아니다.
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
import sys
import time
from dataclasses import replace
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

from chipcheck.cause.adapters import REGISTRY  # noqa: E402
from chipcheck.cause.engine import CauseRanker  # noqa: E402
from chipcheck.cause.kb import load_kb  # noqa: E402

BAR = "─" * 78


def load_class_names() -> dict[str, str]:
    p = CLASSES_CSV
    if not p.exists():
        return {}
    return {r["class_id"]: r["team_name"] for r in csv.DictReader(open(p, encoding="utf-8"))}


def simulated_probs(truth_class: str, classes: list[str], rng: random.Random,
                    peak: float = 0.88) -> dict[str, float]:
    """분류기 출력을 흉내 낸다. 실제 모델 대신 쓰는 값임을 호출부가 표시한다."""
    rest = [c for c in classes if c != truth_class]
    out = {truth_class: peak}
    w = [rng.random() for _ in rest]
    s = sum(w) or 1.0
    for c, x in zip(rest, w):
        out[c] = (1 - peak) * x / s
    return out


def inflate(causes, n: int):
    """원인 수를 n개까지 불린다. 1,000건 규모에서의 검색 속도를 재기 위한 것."""
    out = list(causes)
    i = 0
    while len(out) < n:
        base = causes[i % len(causes)]
        k = len(out)
        out.append(replace(base, cause_id=f"{base.cause_id}-V{k}",
                           name=f"{base.name} 변형 {k}"))
        i += 1
    return out


def show(d, names, truth_id, rank) -> None:
    ev = d.event
    print(BAR)
    print(f"  {ev.defect_id}   lot {ev.lot_id} / wafer {ev.wafer_id}   "
          f"step {ev.step or '-'}   장비 {ev.equipment or '-'}")
    print(f"  후보 {d.n_candidates_before}개 → {d.n_candidates_after}개 "
          f"→ 상위 {len(d.hits)}개 표시   ({d.elapsed_ms:.1f} ms)")
    if d.abstained:
        print(f"  판정 보류: {d.abstain_reason}")
        return
    for i, h in enumerate(d.hits, 1):
        mark = "◀ 정답" if h.cause.cause_id == truth_id else ""
        print(f"  {i}. [{h.score:5.2f}] {h.cause.cause_id:12s} {h.cause.name} {mark}")
        print(f"      텍스트 {h.parts['text']:.2f} · 이미지 {h.parts['image']:.2f} "
              f"· 규칙 {h.parts['rule']:.2f}")
        if h.evidence:
            print(f"      근거 토큰: {', '.join(h.evidence[:6])}")
        print(f"      조치: {h.cause.action}")
    print(f"  정답 {truth_id} 순위: {rank if rank else '5위 밖'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="원인 축소 엔진 시연")
    ap.add_argument("--kb", default=KB_DEFAULT)
    ap.add_argument("--format", default="klarf_like", choices=["klarf_like", "flat_csv"])
    ap.add_argument("--cases", type=int, default=5, help="콘솔에 자세히 출력할 건수")
    ap.add_argument("--inflate", type=int, default=0, help="원인 수를 이만큼으로 불린다")
    ap.add_argument("--no-image", action="store_true", help="이미지 없이 텍스트만으로 랭킹")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    causes = load_kb(ROOT / args.kb)
    n_real = len(causes)
    if args.inflate:
        causes = inflate(causes, args.inflate)
    ranker = CauseRanker(causes)
    names = load_class_names()
    classes = sorted(names) or [f"class_{i}" for i in range(1, 7)]

    src = LOGS / ("sample.klarf" if args.format == "klarf_like" else "sample_flat.csv")
    if not src.exists():
        sys.exit(f"가상 로그가 없습니다: {src}\n먼저 python scripts/make_synthetic_logs.py 를 돌리세요.")
    events = REGISTRY[args.format].parse(src.read_text(encoding="utf-8"))
    truth = {r["case_id"]: r for r in
             csv.DictReader(open(LOGS / "ground_truth.csv", encoding="utf-8"))}

    print(f"지식베이스 {n_real}개 원인" + (f" → {len(causes)}개로 확장" if args.inflate else ""))
    print(f"로그 {src.relative_to(ROOT)}  ({args.format} 어댑터)  사건 {len(events)}건")
    print(f"이미지 분류 확률: {'사용 안 함' if args.no_image else '시뮬레이션 값 (실제 모델 아님)'}\n")

    rng = random.Random(args.seed)
    top1 = top5 = 0
    ms = []
    for i, ev in enumerate(events):
        t = truth.get(ev.defect_id)
        if not t:
            continue
        probs = None if args.no_image else simulated_probs(t["truth_class"], classes, rng)
        d = ranker.rank(ev, class_probs=probs, top_k=5)
        ms.append(d.elapsed_ms)
        ids = [h.cause.cause_id for h in d.hits]
        rank = ids.index(t["truth_cause_id"]) + 1 if t["truth_cause_id"] in ids else 0
        top1 += rank == 1
        top5 += rank > 0
        if i < args.cases:
            show(d, names, t["truth_cause_id"], rank)

    n = len(ms)
    print(BAR)
    print(f"  사건 {n}건   1순위 적중 {top1}/{n} ({top1/n:.1%})   "
          f"5순위 안 적중 {top5}/{n} ({top5/n:.1%})")
    print(f"  건당 {statistics.median(ms):.2f} ms (중앙값), "
          f"최대 {max(ms):.2f} ms   원인 {len(causes)}개 기준")
    print("  주의: 가상 로그 문구는 지식베이스에서 파생된다. 위 적중률은 잡음·방해")
    print("        알람 속 검색 성능이고, 실제 팹의 원인 추정 정확도가 아니다.")


if __name__ == "__main__":
    main()
