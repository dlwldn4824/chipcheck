#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chipcheck.data import build_dataloaders
from chipcheck.models import build_model
from chipcheck.train import Trainer
from chipcheck.utils import ensure_dir, load_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="베이스라인 모델 학습")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    set_seed(cfg["project"]["seed"])

    # checkpoint 경로를 프로젝트 루트 기준으로
    cfg["train"]["checkpoint_dir"] = str(ROOT / cfg["train"]["checkpoint_dir"])

    print("device will resolve inside Trainer; building loaders...")
    train_loader, val_loader, test_loader = build_dataloaders(cfg, ROOT)
    model = build_model(cfg)
    trainer = Trainer(cfg)
    print(f"device={trainer.device} model={cfg['model']['name']} epochs={cfg['train']['epochs']}")
    result = trainer.fit(train_loader, val_loader, model)

    out = ensure_dir(ROOT / "results" / "metrics")
    hist_path = out / "train_history.json"
    hist_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"best_val_acc={result['best_val_acc']:.4f}")
    print(f"checkpoint → {cfg['train']['checkpoint_dir']}/best.pt")
    print(f"history → {hist_path}")
    print(f"test batches ready: {len(test_loader)} (evaluate.py 로 평가)")


if __name__ == "__main__":
    main()
