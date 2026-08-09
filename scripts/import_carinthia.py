#!/usr/bin/env python3
"""Carinthia-S SEM 데이터셋을 data/raw → ImageFolder 형식으로 정리한다."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DOWNLOADS = RAW / "downloads"
OUT = RAW / "carinthia_s"


def find_zip() -> Path:
    candidates = [
        DOWNLOADS / "carinthia_s_data.zip",
        DOWNLOADS / "data.zip",
    ]
    for c in candidates:
        if c.exists() and c.stat().st_size > 1_000_000:
            return c
    raise FileNotFoundError(
        f"Carinthia zip이 없습니다. {DOWNLOADS} 에 data.zip을 두세요.\n"
        "URL: https://zenodo.org/records/16895427/files/data.zip"
    )


def extract_zip(zip_path: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    # find image root
    for p in dest.rglob("*"):
        if p.is_dir() and any(p.glob("*.png")) or any(p.glob("*.jpg")):
            return p if p != dest else dest
    return dest


def infer_class_from_path(path: Path) -> str | None:
    """경로/파일명에서 클래스명을 추론."""
    parts = [p.lower() for p in path.parts]
    name = path.stem.lower()
    # common carinthia class folder names / keywords
    keywords = [
        "particle",
        "scratch",
        "residue",
        "pattern",
        "bridge",
        "missing",
        "hole",
        "bubble",
        "crack",
        "contamination",
        "defect",
        "normal",
        "good",
    ]
    for part in reversed(parts[:-1]):
        for kw in keywords:
            if kw in part:
                return part
        if part not in {"images", "masks", "mask", "img", "data", "train", "test", "val"}:
            # likely class folder
            if part.replace("_", "").replace("-", "").isalnum() and len(part) < 40:
                return part
    for kw in keywords:
        if kw in name:
            return kw
    return None


def organize_imagefolder(extracted: Path, out: Path) -> dict:
    """images/ 와 masks/ 를 분리하고 클래스별 ImageFolder로 복사."""
    images_out = out / "images"
    masks_out = out / "masks"
    if images_out.exists():
        shutil.rmtree(images_out)
    if masks_out.exists():
        shutil.rmtree(masks_out)
    images_out.mkdir(parents=True)
    masks_out.mkdir(parents=True)

    all_files = [
        p
        for p in extracted.rglob("*")
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    ]

    # Prefer paths that look like images vs masks
    image_files = []
    mask_files = []
    for p in all_files:
        low = str(p).lower()
        if "mask" in low or "/gt/" in low or "ground" in low or "label" in low and "images" not in low:
            mask_files.append(p)
        else:
            image_files.append(p)

    # If everything mixed, treat non-mask as images
    if not image_files and all_files:
        image_files = all_files

    counts: Counter[str] = Counter()
    manifest = []

    for src in image_files:
        cls = infer_class_from_path(src) or "unknown"
        # skip if parent is literally 'masks'
        if "mask" in cls:
            continue
        dst_dir = images_out / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        # avoid overwrite collisions
        if dst.exists():
            dst = dst_dir / f"{src.stem}_{abs(hash(str(src))) % 10_000_000}{src.suffix}"
        shutil.copy2(src, dst)
        counts[cls] += 1
        manifest.append({"path": str(dst.relative_to(out)), "class": cls, "source": str(src)})

    for src in mask_files:
        cls = infer_class_from_path(src) or "unknown"
        dst_dir = masks_out / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        if dst.exists():
            dst = dst_dir / f"{src.stem}_{abs(hash(str(src))) % 10_000_000}{src.suffix}"
        shutil.copy2(src, dst)

    summary = {
        "dataset": "Carinthia-S",
        "source": "https://zenodo.org/records/16895427",
        "n_images": sum(counts.values()),
        "n_masks": len(mask_files),
        "classes": dict(sorted(counts.items())),
        "layout": "data/raw/carinthia_s/images/<class>/*.png",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in manifest),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=None)
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    zip_path = args.zip or find_zip()
    extract_dir = DOWNLOADS / "carinthia_s_extracted"
    if not args.skip_extract:
        print(f"Extracting {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)...")
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_zip(zip_path, extract_dir)
    else:
        if not extract_dir.exists():
            raise FileNotFoundError(extract_dir)

    print(f"Organizing into {OUT}...")
    summary = organize_imagefolder(extract_dir, OUT)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Done.")


if __name__ == "__main__":
    main()
