#!/usr/bin/env python3
"""가상 로그 샘플을 만든다.

    python scripts/make_synthetic_logs.py --n 60

Carinthia-S에는 로그가 없다. 실제 팹 로그의 구조만 본떠 세 가지 포맷으로
같은 사건을 적어 둔다. 어댑터가 포맷을 흡수하는지, 엔진이 잡음 속에서 맞는
원인을 끌어올리는지 확인하는 용도다. 여기서 나온 수치는 실제 팹 정확도가
아니다. 로그 문구가 지식베이스에서 파생되기 때문이다.

출력
  data/cause_kb/logs/sample.klarf
  data/cause_kb/logs/sample_flat.csv
  data/cause_kb/logs/sample_secs_gem.log
  data/cause_kb/logs/ground_truth.csv
"""

from __future__ import annotations

import argparse
import random
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

from chipcheck.cause.kb import load_kb  # noqa: E402
from chipcheck.cause.synth import (  # noqa: E402
    make_case, to_flat_csv, to_klarf, to_secs_gem, truth_csv,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="가상 공정 로그 생성")
    ap.add_argument("--kb", default=KB_DEFAULT)
    ap.add_argument("--out", default=str(LOGS))
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-ratio", type=float, default=0.6,
                    help="정답 원인의 키워드 중 로그에 남길 비율")
    ap.add_argument("--distractors", type=int, default=2,
                    help="다른 원인에서 끌어와 섞을 방해 알람 수")
    args = ap.parse_args()

    causes = load_kb(ROOT / args.kb)
    rng = random.Random(args.seed)
    cases = [make_case(causes, rng, i + 1, args.keep_ratio, args.distractors)
             for i in range(args.n)]

    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "sample.klarf": to_klarf(cases),
        "sample_flat.csv": to_flat_csv(cases),
        "sample_secs_gem.log": to_secs_gem(cases),
        "ground_truth.csv": truth_csv(cases),
    }
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8")
        print(f"{(out / name).name:24s} {len(text.splitlines()):5d}줄")

    print(f"\n원인 {len(causes)}개에서 사건 {len(cases)}건 생성 "
          f"(키워드 유지 {args.keep_ratio:.0%}, 방해 알람 {args.distractors}개)")


if __name__ == "__main__":
    main()
