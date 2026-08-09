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
