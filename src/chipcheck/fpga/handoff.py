"""양자화/FPGA 담당자에게 넘길 핸드오프 패키지 생성.

구성물
  model/        float 가중치(.pth) + 독립 모델 정의 코드 + (선택) ONNX
  calibration/  INT8 보정용 이미지 (train 스플릿에서만 추출)
  test/         테스트셋 이미지 + 라벨 + float 모델 예측/로짓
  metadata/     전처리 규격 · 클래스 순서 · float 정확도
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from chipcheck.utils import ensure_dir

PKG_DIRS = ("model", "calibration/images", "test/images", "metadata")


def read_split(root: Path, cfg: dict[str, Any]) -> list[dict[str, str]]:
    split_csv = root / cfg["data"]["splits_dir"] / "split.csv"
    if not split_csv.exists():
        raise FileNotFoundError(f"분할 파일 없음: {split_csv} (scripts/preprocess.py 먼저 실행)")
    with open(split_csv, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pick_calibration(
    rows: list[dict[str, str]], total: int, num_classes: int, seed: int
) -> list[tuple[str, int]]:
    """train 스플릿에서 클래스 라운드로빈으로 보정용 샘플 추출.

    소수 클래스를 먼저 채우고 남는 자리를 다수 클래스로 메워, 희소 클래스의
    activation 범위가 보정 집합에서 빠지지 않게 한다.
    """
    by_class: dict[int, list[str]] = defaultdict(list)
    for r in rows:
        if r["split"] == "train":
            by_class[int(r["label"])].append(r["path"])

    rng = random.Random(seed)
    for paths in by_class.values():
        rng.shuffle(paths)

    picked: list[tuple[str, int]] = []
    cursor = {c: 0 for c in by_class}
    while len(picked) < total:
        progressed = False
        for c in range(num_classes):
            if len(picked) >= total:
                break
            pool = by_class.get(c, [])
            i = cursor.get(c, 0)
            if i < len(pool):
                picked.append((pool[i], c))
                cursor[c] = i + 1
                progressed = True
        if not progressed:  # 전체 train을 다 썼다
            break
    return picked


def save_resized(src: Path, dst: Path, size: int) -> None:
    from PIL import Image

    img = Image.open(src).convert("RGB").resize((size, size), Image.BILINEAR)
    img.save(dst, format="PNG")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(pkg: Path) -> Path:
    files = []
    for p in sorted(pkg.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.json":
            files.append(
                {
                    "path": str(p.relative_to(pkg)),
                    "bytes": p.stat().st_size,
                    "sha256": _sha256(p),
                }
            )
    out = pkg / "MANIFEST.json"
    out.write_text(
        json.dumps(
            {"generated": date.today().isoformat(), "n_files": len(files), "files": files},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return out


def make_package_dirs(out_dir: Path) -> Path:
    pkg = ensure_dir(out_dir)
    for d in PKG_DIRS:
        ensure_dir(pkg / d)
    return pkg


def copy_model_def(pkg: Path) -> Path:
    src = Path(__file__).with_name("standalone_model.py")
    dst = pkg / "model" / "model_def.py"
    shutil.copyfile(src, dst)
    return dst
