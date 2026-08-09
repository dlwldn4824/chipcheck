#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chipcheck.utils import load_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="베이스라인 모델 학습")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    set_seed(cfg["project"]["seed"])

    if not cfg["data"].get("num_classes"):
        print("[중단] configs/default.yaml 의 data.num_classes / class_names 를 먼저 설정하세요.")
        print("데이터셋 선정(Phase 1) 완료 후 train DataLoader를 연결하면 학습이 시작됩니다.")
        print("스켈레톤: src/chipcheck/train/trainer.py , models/builder.py")
        return

    # 데이터로더 연결 후:
    # from chipcheck.models import build_model
    # from chipcheck.train import Trainer
    # model = build_model(cfg)
    # Trainer(cfg).fit(train_loader, val_loader, model)
    print("num_classes가 설정됨. DataLoader 구현 후 Trainer.fit()을 호출하세요.")


if __name__ == "__main__":
    main()
