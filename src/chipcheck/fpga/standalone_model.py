"""칩췍 결함 분류 모델 — 독립 실행용 정의 파일.

이 파일 + `*.pth` 가중치만 있으면 학습 환경(= 이 저장소) 없이 모델을 복원할 수 있다.
양자화 담당자에게 전달되는 핸드오프 패키지에 `model/model_def.py` 로 복사된다.

의존성: torch, torchvision, Pillow, numpy
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# 학습 때 쓴 값 — 바꾸면 정확도가 떨어진다.
IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def build_model(arch: str = "resnet18", num_classes: int = 6) -> nn.Module:
    """학습 때와 동일한 구조를 pretrained 가중치 없이 생성."""
    if arch == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif arch == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"미지원 아키텍처: {arch}")
    return model


def load_model(
    weights_path: str | Path,
    arch: str = "resnet18",
    num_classes: int = 6,
    device: str = "cpu",
) -> nn.Module:
    """state_dict(.pth) 로드 → eval 모드 모델 반환."""
    state = torch.load(weights_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model" in state and "fc.weight" not in state:
        state = state["model"]
    model = build_model(arch, num_classes)
    model.load_state_dict(state)
    model.eval().to(device)
    return model


def load_from_package(pkg_dir: str | Path, device: str = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    """핸드오프 패키지 루트를 주면 메타데이터를 읽어 모델까지 복원."""
    pkg = Path(pkg_dir)
    info = json.loads((pkg / "metadata" / "model_info.json").read_text(encoding="utf-8"))
    model = load_model(
        pkg / "model" / info["weights_file"],
        arch=info["arch"],
        num_classes=info["num_classes"],
        device=device,
    )
    return model, info


def build_eval_transform(
    image_size: int = IMAGE_SIZE,
    mean: Sequence[float] = MEAN,
    std: Sequence[float] = STD,
):
    """학습/평가에서 쓴 것과 동일한 전처리."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ]
    )


def preprocess_image(path: str | Path, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    """이미지 1장 → (1, 3, H, W) float32 텐서."""
    img = Image.open(path).convert("RGB")
    return build_eval_transform(image_size)(img).unsqueeze(0)


def preprocess_batch(paths: Sequence[str | Path], image_size: int = IMAGE_SIZE) -> torch.Tensor:
    """이미지 여러 장 → (N, 3, H, W) float32 텐서."""
    tf = build_eval_transform(image_size)
    return torch.stack([tf(Image.open(p).convert("RGB")) for p in paths])


@torch.no_grad()
def predict(model: nn.Module, paths: Sequence[str | Path], device: str = "cpu"):
    """경로 목록 → (logits, 예측 클래스 인덱스)."""
    x = preprocess_batch(paths).to(device)
    logits = model(x)
    return logits.cpu(), logits.argmax(dim=1).cpu()


class FeatureAndLogits(nn.Module):
    """특징 512-d와 로짓 6개를 함께 내보내는 래퍼.

    보드에서 Mahalanobis 관문을 쓰려면 분류 확률만으로는 부족하다. 거리 계산이
    avgpool 출력(512-d)을 입력으로 받기 때문이다. 학습 시점에는 forward hook으로
    꺼냈지만, DPU 컴파일러에는 hook이 없다. 그래서 양자화 전에 이 래퍼로 감싸서
    출력이 두 개인 그래프로 만든다. DPU 컴파일러는 다중 출력을 지원한다.

    반환 순서는 (features, logits)로 고정한다. 보드 쪽 코드가 이 순서에 의존한다.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.backbone
        x = m.conv1(x)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.maxpool(x)
        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        x = m.layer4(x)
        x = m.avgpool(x)
        feats = torch.flatten(x, 1)
        return feats, m.fc(feats)


def wrap_dual_output(model: nn.Module) -> FeatureAndLogits:
    """학습된 ResNet을 (특징, 로짓) 두 출력 모델로 감싼다."""
    return FeatureAndLogits(model).eval()
