from __future__ import annotations

from pathlib import Path
from typing import Any


def export_onnx(model, sample_input, out_path: str | Path) -> Path:
    """학습 모델을 ONNX로 보내 FPGA/엣지 변환 파이프라인 진입점."""
    import torch

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.onnx.export(
        model,
        sample_input,
        str(out_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    return out_path


def compare_backends_placeholder() -> dict[str, Any]:
    """Phase 4에서 채울 CPU/GPU/FPGA 비교표."""
    return {
        "cpu": {"latency_ms": None, "power_w": None, "accuracy": None, "cost": None},
        "gpu": {"latency_ms": None, "power_w": None, "accuracy": None, "cost": None},
        "fpga": {"latency_ms": None, "power_w": None, "accuracy": None, "cost": None},
        "notes": [
            "양자화에 따른 정확도 저하",
            "지원 연산 / 메모리 / 입력 크기",
            "실시간 FPS / HW-SW 데이터 전달 속도",
        ],
    }


def export_onnx_dual(model, sample_input, out_path: str | Path,
                     fixed_batch: bool = True) -> Path:
    """(특징 512-d, 로짓 6개) 두 출력을 내보낸다.

    보드에서 Mahalanobis 관문을 쓰려면 로짓만으로는 부족하다. `fixed_batch=True`가
    기본인데, Vitis AI 계열 컴파일러는 배치 축이 동적인 그래프를 그대로 받지 못하는
    경우가 많기 때문이다. GPU 추론용으로 가변 배치가 필요하면 False로 둔다.
    """
    import torch

    from .standalone_model import wrap_dual_output

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wrapped = wrap_dual_output(model)
    dyn = None if fixed_batch else {"input": {0: "batch"},
                                    "features": {0: "batch"}, "logits": {0: "batch"}}
    torch.onnx.export(
        wrapped,
        sample_input,
        str(out_path),
        input_names=["input"],
        output_names=["features", "logits"],
        dynamic_axes=dyn,
        opset_version=17,
    )
    return out_path
