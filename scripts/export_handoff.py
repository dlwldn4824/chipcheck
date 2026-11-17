#!/usr/bin/env python3
"""양자화/FPGA 담당자에게 넘길 핸드오프 패키지를 한 번에 생성.

    python scripts/export_handoff.py                  # results/handoff/ 에 생성 + zip
    python scripts/export_handoff.py --n-calib 500    # 보정 이미지 장수 변경
    python scripts/export_handoff.py --no-test-images --no-zip
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chipcheck.fpga.handoff import (  # noqa: E402
    read_split,
    save_resized,
    copy_model_def,
    make_package_dirs,
    pick_calibration,
    write_manifest,
)
from chipcheck.models import build_model  # noqa: E402
from chipcheck.utils import load_config, resolve_device  # noqa: E402


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="FPGA/양자화 인계 패키지 생성")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", default="results/checkpoints/best.pt")
    ap.add_argument("--out", default="results/handoff")
    ap.add_argument("--n-calib", type=int, default=300, help="보정용 이미지 장수")
    ap.add_argument("--onnx", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--test-images", action=argparse.BooleanOptionalAction, default=True,
        help="테스트셋 원본 이미지 동봉 (약 25MB)",
    )
    ap.add_argument("--zip", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    import numpy as np
    import torch

    cfg = load_config(ROOT / args.config)
    ckpt_path = ROOT / args.ckpt
    if not ckpt_path.exists():
        sys.exit(f"[중단] 체크포인트 없음: {ckpt_path} (먼저 scripts/train.py 실행)")

    arch = cfg["model"]["name"]
    num_classes = cfg["data"]["num_classes"]
    class_names = cfg["data"]["class_names"]
    size = cfg["data"]["image_size"]
    mean = cfg["preprocess"]["mean"]
    std = cfg["preprocess"]["std"]
    device = resolve_device(cfg["train"].get("device", "auto"))

    pkg = make_package_dirs(ROOT / args.out)
    print(f"패키지 경로: {pkg}")

    # ── 1. float 모델 가중치 + 모델 정의 코드 ────────────────────────────
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    weights = state["model"] if isinstance(state, dict) and "model" in state else state
    weights_name = f"chipcheck_{arch}_float.pth"
    torch.save(weights, pkg / "model" / weights_name)
    copy_model_def(pkg)

    model = build_model(cfg)
    model.load_state_dict(weights)
    model.eval().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  모델: {arch}  파라미터 {n_params:,}개  → model/{weights_name}")

    onnx_name = None
    if args.onnx:
        try:
            from chipcheck.fpga import export_onnx

            onnx_name = f"chipcheck_{arch}.onnx"
            dummy = torch.zeros(1, 3, size, size)
            try:
                export_onnx(model.cpu(), dummy, pkg / "model" / onnx_name)
            finally:
                model.to(device)  # 실패해도 추론용 디바이스로 되돌린다
            print(f"  ONNX → model/{onnx_name}")
        except Exception as exc:  # onnx/onnxscript 미설치 등
            onnx_name = None
            print(f"  [건너뜀] ONNX 내보내기 실패: {type(exc).__name__}: {exc}")

    # ── 2. 보정용 이미지 (train 스플릿에서만) ────────────────────────────
    rows = read_split(ROOT, cfg)
    calib = pick_calibration(rows, args.n_calib, num_classes, cfg["project"]["seed"])
    calib_rows = []
    for i, (rel, label) in enumerate(calib):
        dst_name = f"calib_{i:04d}_{class_names[label]}.png"
        save_resized(ROOT / rel, pkg / "calibration" / "images" / dst_name, size)
        calib_rows.append({"file": f"images/{dst_name}", "label": label,
                           "class_name": class_names[label], "source": rel})
    with open(pkg / "calibration" / "calib_list.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "label", "class_name", "source"])
        w.writeheader()
        w.writerows(calib_rows)
    calib_dist = Counter(r["class_name"] for r in calib_rows)
    print(f"  보정 이미지 {len(calib_rows)}장 ({size}x{size} PNG) → calibration/  {dict(calib_dist)}")

    # ── 3. 테스트셋 + float 모델 기준 출력 ───────────────────────────────
    test_rows = [r for r in rows if r["split"] == "test"]
    from chipcheck.fpga.standalone_model import preprocess_batch

    paths = [ROOT / r["path"] for r in test_rows]
    labels = np.array([int(r["label"]) for r in test_rows], dtype=np.int64)
    all_logits = []
    bs = 32
    with torch.no_grad():
        for i in range(0, len(paths), bs):
            xb = preprocess_batch(paths[i : i + bs], size).to(device)
            all_logits.append(model(xb).cpu().numpy())
    logits = np.concatenate(all_logits).astype(np.float32)
    preds = logits.argmax(axis=1)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()

    test_manifest = []
    for i, r in enumerate(test_rows):
        name = Path(r["path"]).name
        if args.test_images:
            shutil.copyfile(ROOT / r["path"], pkg / "test" / "images" / name)
        test_manifest.append(
            {
                "file": f"images/{name}" if args.test_images else "",
                "source": r["path"],
                "label": int(labels[i]),
                "class_name": class_names[int(labels[i])],
                "float_pred": int(preds[i]),
                "float_pred_name": class_names[int(preds[i])],
                "float_confidence": round(float(probs[i].max()), 6),
            }
        )
    with open(pkg / "test" / "test_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(test_manifest[0].keys()))
        w.writeheader()
        w.writerows(test_manifest)
    np.savez_compressed(
        pkg / "test" / "float_reference_outputs.npz",
        logits=logits,
        labels=labels,
        preds=preds,
        files=np.array([Path(r["path"]).name for r in test_rows]),
    )
    print(f"  테스트셋 {len(test_rows)}장 + float 로짓 → test/")

    # ── 4. 메타데이터 ────────────────────────────────────────────────────
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    acc = float((preds == labels).mean())
    float_acc = {
        "split": "test",
        "n_samples": int(len(labels)),
        "accuracy": acc,
        "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, preds, labels=list(range(num_classes))).tolist(),
        "per_class": classification_report(
            labels, preds, labels=list(range(num_classes)),
            target_names=class_names, output_dict=True, zero_division=0,
        ),
        "note": "INT8 양자화 후 같은 테스트셋으로 재측정해 이 수치와 비교하세요.",
    }
    _write_json(pkg / "metadata" / "float_accuracy.json", float_acc)

    sample = preprocess_batch(paths[: min(64, len(paths))], size).numpy()
    _write_json(
        pkg / "metadata" / "preprocess.json",
        {
            "input_tensor": {
                "layout": "NCHW", "channels": 3, "color_space": "RGB",
                "height": size, "width": size, "dtype": "float32",
            },
            "source_images": {
                "native_size": [480, 480], "format": "jpg",
                "note": "원본은 480x480 SEM 이미지, 학습 입력은 아래 순서로 변환",
            },
            "steps": [
                {"op": "load", "detail": "PIL Image.open(path).convert('RGB')"},
                {"op": "resize", "size": [size, size], "interpolation": "bilinear",
                 "detail": "torchvision.transforms.Resize((H, W))"},
                {"op": "to_tensor", "detail": "HWC uint8[0,255] → CHW float32[0,1] (÷255)"},
                {"op": "normalize", "mean": mean, "std": std,
                 "detail": "채널별 (x - mean) / std"},
            ],
            "value_range_after_normalize": {
                "min": round(float(sample.min()), 4),
                "max": round(float(sample.max()), 4),
                "measured_on": f"test {sample.shape[0]}장",
                "note": "고정소수점 범위 잡을 때 참고",
            },
            "reference_code": "model/model_def.py 의 preprocess_image / preprocess_batch",
        },
    )

    split_counts = {s: Counter(class_names[int(r["label"])] for r in rows if r["split"] == s)
                    for s in ("train", "val", "test")}
    _write_json(
        pkg / "metadata" / "classes.json",
        {
            "num_classes": num_classes,
            "index_to_name": {str(i): n for i, n in enumerate(class_names)},
            "output_order_note": "모델 출력 logits의 열 순서 = 위 인덱스 순서",
            "dataset": "Carinthia-S (SEM 웨이퍼 결함) https://zenodo.org/records/16895427",
            "counts_per_split": {s: dict(c) for s, c in split_counts.items()},
            "imbalance_note": "class_3이 전체의 약 87%, class_5는 전체 4장뿐",
        },
    )

    _write_json(
        pkg / "metadata" / "model_info.json",
        {
            "arch": arch,
            "num_classes": num_classes,
            "weights_file": weights_name,
            "onnx_file": onnx_name,
            "framework": f"pytorch {torch.__version__}",
            "n_parameters": int(n_params),
            "precision": "float32",
            "task": "image classification (single label)",
            "output": "logits (softmax 미적용)",
            "generated": date.today().isoformat(),
        },
    )
    print("  메타데이터 4종 → metadata/")

    # ── 5. README + 체크섬 ───────────────────────────────────────────────
    (pkg / "README.md").write_text(
        _readme(arch, weights_name, onnx_name, size, mean, std, num_classes,
                class_names, len(calib_rows), len(test_rows), acc, float_acc["f1_macro"]),
        encoding="utf-8",
    )
    write_manifest(pkg)

    if args.zip:
        zip_base = pkg.parent / f"chipcheck_handoff_{date.today().isoformat()}"
        shutil.make_archive(str(zip_base), "zip", root_dir=pkg)
        mb = (zip_base.with_suffix(".zip")).stat().st_size / 1e6
        print(f"\n압축본: {zip_base}.zip  ({mb:.1f} MB)")

    print(f"\n완료 — float test accuracy {acc:.4f}, macro-F1 {float_acc['f1_macro']:.4f}")


def _readme(arch, weights, onnx, size, mean, std, nc, names, n_calib, n_test, acc, f1) -> str:
    onnx_line = f"- `model/{onnx}` — ONNX (opset 17, 동적 batch)\n" if onnx else ""
    return f"""# 칩췍 — 양자화/FPGA 인계 패키지

Carinthia-S SEM 웨이퍼 결함 분류 모델. 아래 자료만으로 INT8 양자화와 보드 추론을
학습 환경 없이 재현할 수 있습니다.

## 1. 모델
- `model/{weights}` — 학습된 float32 state_dict
- `model/model_def.py` — 독립 모델 정의 (torch/torchvision만 있으면 동작)
{onnx_line}
```python
import sys; sys.path.insert(0, "model")
from model_def import load_from_package, preprocess_image
model, info = load_from_package(".")          # eval 모드
logits = model(preprocess_image("test/images/<파일>.jpg"))
```

## 2. 보정(calibration) 이미지
- `calibration/images/` — {n_calib}장, 이미 {size}×{size} PNG로 리사이즈됨
- `calibration/calib_list.csv` — 파일 ↔ 라벨 ↔ 원본 경로
- **train 스플릿에서만** 뽑았습니다 (test 오염 없음). 클래스 라운드로빈으로
  희소 클래스가 빠지지 않게 배분했습니다.

## 3. 전처리 규격
`metadata/preprocess.json` 에 전부 있습니다. 요약:

| 항목 | 값 |
| --- | --- |
| 입력 | {size}×{size}, 3채널 RGB, NCHW, float32 |
| 원본 | 480×480 jpg |
| 순서 | resize(bilinear) → ÷255 → 채널별 정규화 |
| mean | {mean} |
| std | {std} |

보드 쪽 전처리가 이 순서와 다르면 정확도가 떨어집니다. 기준 구현은
`model/model_def.py` 의 `preprocess_image()` 입니다.

## 4. 클래스 순서
`metadata/classes.json`. 출력 logits 열 순서:

{chr(10).join(f"- `{i}` = {n}" for i, n in enumerate(names))}

({nc}클래스. class_3이 전체의 약 87%로 매우 불균형합니다.)

## 5. 테스트셋 & float 기준 성능
- `test/images/` — 테스트셋 {n_test}장 (원본 480×480)
- `test/test_manifest.csv` — 정답 라벨 + float 모델 예측 + confidence
- `test/float_reference_outputs.npz` — float 모델 로짓 `(N, {nc})`, 라벨, 예측
- `metadata/float_accuracy.json` — 아래 수치의 전체 내역

| 지표 | float32 |
| --- | --- |
| accuracy | {acc:.4f} |
| macro-F1 | {f1:.4f} |

양자화 후 같은 테스트셋으로 재측정해서 이 두 수치와 비교해 주세요.
accuracy는 불균형 때문에 쉽게 높게 나오니 **macro-F1을 같이** 봐야 합니다.
로짓을 직접 비교하려면 `float_reference_outputs.npz` 를 쓰면 샘플 단위로
어디서 틀어졌는지 볼 수 있습니다.

## 6. 체크섬
`MANIFEST.json` 에 파일별 크기와 sha256이 있습니다.
"""


if __name__ == "__main__":
    main()
