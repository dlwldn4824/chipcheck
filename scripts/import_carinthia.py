#!/usr/bin/env python3
"""Carinthia-S SEM 데이터셋을 CSV 라벨 기준으로 ImageFolder 형식으로 정리한다."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DOWNLOADS = RAW / "downloads"
OUT = RAW / "carinthia_s"

# HTML에 명시적 클래스명이 없어 번호 폴더 사용. 추후 논문/문서에서 매핑 가능.
CLASS_NAMES = {
    "1": "class_1",
    "2": "class_2",
    "3": "class_3",
    "4": "class_4",
    "5": "class_5",
    "6": "class_6",
}


def find_zip() -> Path:
    candidates = [
        DOWNLOADS / "carinthia_s_data.zip",
        DOWNLOADS / "data.zip",
    ]
    for c in candidates:
        if c.exists() and c.stat().st_size > 1_000_000:
            return c
    raise FileNotFoundError(
        f"Carinthia zip이 없습니다: {DOWNLOADS}\n"
        "URL: https://zenodo.org/records/16895427/files/data.zip"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=None)
    args = parser.parse_args()

    zip_path = args.zip or find_zip()
    if OUT.exists():
        shutil.rmtree(OUT)
    images_out = OUT / "images"
    masks_out = OUT / "masks"
    images_out.mkdir(parents=True)
    masks_out.mkdir(parents=True)

    print(f"Reading {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)...")
    counts: Counter[str] = Counter()
    manifest = []

    with zipfile.ZipFile(zip_path) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith("carinthia-s.csv"))
        rows = list(csv.DictReader(zf.open(csv_name).read().decode().splitlines(), delimiter=";"))
        print(f"CSV rows: {len(rows)}")

        for row in rows:
            label = str(row["label"]).strip()
            cls = CLASS_NAMES.get(label, f"class_{label}")
            img_rel = row["image_path"].lstrip("/")
            mask_rel = row["mask_path"].lstrip("/")
            # zip 내부 경로: data/images/...
            img_zip = f"data/{img_rel}"
            mask_zip = f"data/{mask_rel}"

            img_dst_dir = images_out / cls
            mask_dst_dir = masks_out / cls
            img_dst_dir.mkdir(parents=True, exist_ok=True)
            mask_dst_dir.mkdir(parents=True, exist_ok=True)

            img_name = Path(img_rel).name
            mask_name = Path(mask_rel).name
            with zf.open(img_zip) as src, open(img_dst_dir / img_name, "wb") as dst:
                shutil.copyfileobj(src, dst)
            with zf.open(mask_zip) as src, open(mask_dst_dir / mask_name, "wb") as dst:
                shutil.copyfileobj(src, dst)

            counts[cls] += 1
            manifest.append(
                {
                    "image": f"images/{cls}/{img_name}",
                    "mask": f"masks/{cls}/{mask_name}",
                    "label_id": int(label),
                    "class": cls,
                }
            )

        # keep original csv
        (OUT / "carinthia-s.csv").write_bytes(zf.read(csv_name))

    summary = {
        "dataset": "Carinthia-S",
        "source": "https://zenodo.org/records/16895427",
        "n_images": sum(counts.values()),
        "n_masks": sum(counts.values()),
        "num_classes": len(counts),
        "classes": dict(sorted(counts.items())),
        "class_id_map": CLASS_NAMES,
        "layout": "data/raw/carinthia_s/images/<class>/*.{jpg,png}",
        "note": "SEM semiconductor wafer defect images with binary segmentation masks. Highly imbalanced.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in manifest),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Done →", OUT)


if __name__ == "__main__":
    main()
