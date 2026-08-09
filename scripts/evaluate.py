#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chipcheck.data import build_dataloaders
from chipcheck.evaluate import compute_classification_metrics, save_confusion_matrix
from chipcheck.evaluate.metrics import dump_metrics
from chipcheck.models import build_model
from chipcheck.utils import ensure_dir, load_config, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="테스트셋 성능 평가")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ckpt", default="results/checkpoints/best.pt")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    device = resolve_device(cfg["train"].get("device", "auto"))
    _, _, test_loader = build_dataloaders(cfg, ROOT)

    import torch

    model = build_model(cfg)
    ckpt_path = ROOT / args.ckpt
    if not ckpt_path.exists():
        print(f"[중단] 체크포인트 없음: {ckpt_path}  (먼저 train.py 실행)")
        return
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb).argmax(dim=1).cpu().tolist()
            y_pred.extend(pred)
            y_true.extend(yb.tolist())

    names = cfg["data"]["class_names"]
    metrics = compute_classification_metrics(y_true, y_pred, class_names=names)
    out = ensure_dir(ROOT / cfg["evaluate"]["output_dir"])
    fig = ensure_dir(ROOT / cfg["evaluate"]["figure_dir"])
    dump_metrics(metrics, out / "test_metrics.json")
    save_confusion_matrix(metrics["confusion_matrix"], names, fig / "confusion_matrix.png")
    print(
        f"acc={metrics['accuracy']:.4f}  "
        f"f1_macro={metrics['f1_macro']:.4f}  "
        f"precision={metrics['precision_macro']:.4f}  "
        f"recall={metrics['recall_macro']:.4f}"
    )
    print(f"→ {out / 'test_metrics.json'}")
    print(f"→ {fig / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
