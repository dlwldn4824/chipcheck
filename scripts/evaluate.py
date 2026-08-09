#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chipcheck.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="성능 평가")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    print("평가 지표:", ", ".join(cfg["evaluate"]["metrics"]))
    print("출력:", cfg["evaluate"]["output_dir"], "/", cfg["evaluate"]["figure_dir"])
    print("체크포인트 학습 후 y_true/y_pred로 compute_classification_metrics()를 호출하세요.")
    print("모듈: src/chipcheck/evaluate/metrics.py")


if __name__ == "__main__":
    main()
