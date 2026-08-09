from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass
class PreprocessPipeline:
    """입력 이미지 → 학습/추론용 텐서 변환."""

    image_size: int = 224
    mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    std: tuple[float, ...] = (0.229, 0.224, 0.225)
    augment: bool = False

    def process_file(self, path: str | Path) -> Image.Image:
        img = Image.open(path).convert("RGB")
        return img.resize((self.image_size, self.image_size))


def build_transforms(cfg: dict[str, Any], train: bool = True):
    """torchvision transforms 구성. torch 미설치 시 None."""
    try:
        from torchvision import transforms
    except ImportError:
        return None

    size = cfg["data"]["image_size"]
    mean = cfg["preprocess"]["mean"]
    std = cfg["preprocess"]["std"]
    aug = cfg["preprocess"].get("augment", {})

    ops: list = [transforms.Resize((size, size))]
    if train and aug:
        if aug.get("horizontal_flip"):
            ops.append(transforms.RandomHorizontalFlip())
        rot = aug.get("rotation_degrees", 0)
        if rot:
            ops.append(transforms.RandomRotation(rot))
        if aug.get("brightness_contrast"):
            ops.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return transforms.Compose(ops)
