from __future__ import annotations

from typing import Any


CANDIDATES = [
    "resnet18",
    "efficientnet_b0",
    "mobilenet_v3_small",
    "yolo",  # detection path — Phase 2 이후
]


def list_model_candidates() -> list[str]:
    return list(CANDIDATES)


def build_model(cfg: dict[str, Any]):
    """분류 베이스라인 모델 생성."""
    import torch.nn as nn
    from torchvision import models

    name = cfg["model"]["name"]
    num_classes = cfg["data"].get("num_classes")
    if not num_classes:
        raise ValueError("configs.data.num_classes를 데이터셋 선정 후 설정하세요.")

    pretrained = cfg["model"].get("pretrained", True)
    weights = "DEFAULT" if pretrained else None

    if name == "resnet18":
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError(
            f"미지원 모델: {name}. 후보={CANDIDATES} (yolo는 detection 브랜치에서 별도)"
        )
    return model
