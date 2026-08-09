#!/usr/bin/env python3
"""공개 반도체 불량 데이터셋 다운로드 진입점."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = ROOT / "data" / "raw" / "downloads"

DATASETS = {
    "carinthia_s": {
        "url": "https://zenodo.org/records/16895427/files/data.zip?download=1",
        "filename": "carinthia_s_data.zip",
        "meta_url": "https://zenodo.org/records/16895427/files/carinthia-s_dataset.html?download=1",
        "meta_filename": "carinthia_s_dataset.html",
        "note": "SEM 반도체 불량 이미지 4,591장 + 세그멘테이션 마스크 (권장 1순위)",
    },
    "magnetic_tile": {
        "url": "https://github.com/abin24/Magnetic-tile-defect-datasets./archive/refs/heads/master.zip",
        "filename": "magnetic_tile.zip",
        "note": "표면 결함 보조 데이터 (반도체 직접 아님, 파이프라인 검증용)",
    },
}


def curl_download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"[skip] already exists: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"[download] {url}\n  → {dest}")
    subprocess.check_call(
        ["curl", "-L", "--fail", "--retry", "5", "--retry-delay", "2", "-o", str(dest), url]
    )


def download_hf_waferguard() -> None:
    out = ROOT / "data" / "raw" / "waferguard"
    out.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            repo_id="oliversinn/waferguard-dataset",
            repo_type="dataset",
            local_dir=str(out),
        )
        print(f"[ok] waferguard → {path}")
    except Exception as e:
        print(f"[warn] HuggingFace waferguard 실패: {e}")
        print("  pip install huggingface_hub 후 재시도하세요.")


def main() -> None:
    parser = argparse.ArgumentParser(description="칩췍 데이터셋 다운로드")
    parser.add_argument(
        "--dataset",
        choices=["all", "carinthia_s", "magnetic_tile", "waferguard"],
        default="carinthia_s",
    )
    args = parser.parse_args()

    targets = (
        ["carinthia_s", "magnetic_tile", "waferguard"]
        if args.dataset == "all"
        else [args.dataset]
    )

    for name in targets:
        if name == "waferguard":
            download_hf_waferguard()
            continue
        info = DATASETS[name]
        print(f"\n=== {name}: {info['note']} ===")
        curl_download(info["url"], DOWNLOADS / info["filename"])
        if info.get("meta_url"):
            curl_download(info["meta_url"], DOWNLOADS / info["meta_filename"])

    if "carinthia_s" in targets:
        print("\n다음: python scripts/import_carinthia.py")


if __name__ == "__main__":
    main()
