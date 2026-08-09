#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chipcheck.preprocess import PreprocessPipeline
from chipcheck.utils import ensure_dir, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="기본 전처리 파이프라인")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    raw = ROOT / cfg["data"]["raw_dir"]
    out = ensure_dir(ROOT / cfg["data"]["processed_dir"])
    pipe = PreprocessPipeline(image_size=cfg["data"]["image_size"])

    images = [
        p
        for p in raw.rglob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    ]
    if not images:
        print(f"[안내] raw 이미지가 없습니다: {raw}")
        print("데이터셋 다운로드 후 data/raw/ 아래에 넣고 다시 실행하세요.")
        return

    for src in images:
        rel = src.relative_to(raw)
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        pipe.process_file(src).save(dst.with_suffix(".png"))
    print(f"전처리 완료: {len(images)}장 → {out}")


if __name__ == "__main__":
    main()
