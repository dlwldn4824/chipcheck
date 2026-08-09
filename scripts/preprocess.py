#!/usr/bin/env python3
"""데이터 검증 + train/val/test 분할 CSV 생성 (학습 시 증강은 온더플라이)."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chipcheck.data import make_splits
from chipcheck.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="전처리: 분할 CSV 생성")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    image_root = ROOT / cfg["data"]["raw_dir"]
    counts: Counter[str] = Counter()
    for name in cfg["data"]["class_names"]:
        folder = image_root / name
        n = len(list(folder.glob("*"))) if folder.is_dir() else 0
        counts[name] = n
        print(f"  {name}: {n}")

    total = sum(counts.values())
    if total == 0:
        print(f"[중단] 이미지 없음: {image_root}")
        return

    out = make_splits(cfg, ROOT)
    print(f"총 {total}장 → 분할 저장: {out}")


if __name__ == "__main__":
    main()
