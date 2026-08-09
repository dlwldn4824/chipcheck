from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from chipcheck.preprocess import build_transforms
from chipcheck.utils import ensure_dir


def _list_samples(image_root: Path, class_names: list[str]) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    for idx, name in enumerate(class_names):
        folder = image_root / name
        if not folder.is_dir():
            continue
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                samples.append((p, idx))
    return samples


def make_splits(cfg: dict[str, Any], root: Path) -> Path:
    """계층적 분할 CSV 저장 → data/splits/split.csv"""
    from sklearn.model_selection import train_test_split

    image_root = root / cfg["data"]["raw_dir"]
    class_names = cfg["data"]["class_names"]
    samples = _list_samples(image_root, class_names)
    if not samples:
        raise FileNotFoundError(f"이미지 없음: {image_root}")

    paths = [str(p.relative_to(root)) for p, _ in samples]
    labels = [y for _, y in samples]
    seed = cfg["project"]["seed"]
    test_ratio = cfg["data"]["test_ratio"]
    val_ratio = cfg["data"]["val_ratio"]

    def _split(x, y, size):
        try:
            return train_test_split(x, y, test_size=size, random_state=seed, stratify=y)
        except ValueError:
            # 소수 클래스 때문에 stratify 실패 시 비계층 분할
            return train_test_split(x, y, test_size=size, random_state=seed)

    p_tmp, p_test, y_tmp, y_test = _split(paths, labels, test_ratio)
    val_frac = val_ratio / (1.0 - test_ratio)
    p_train, p_val, y_train, y_val = _split(p_tmp, y_tmp, val_frac)

    out = ensure_dir(root / cfg["data"]["splits_dir"]) / "split.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "split"])
        for p, y in zip(p_train, y_train):
            w.writerow([p, y, "train"])
        for p, y in zip(p_val, y_val):
            w.writerow([p, y, "val"])
        for p, y in zip(p_test, y_test):
            w.writerow([p, y, "test"])
    return out


def build_dataloaders(cfg: dict[str, Any], root: Path):
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

    split_csv = root / cfg["data"]["splits_dir"] / "split.csv"
    if not split_csv.exists():
        make_splits(cfg, root)

    rows = []
    with open(split_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    class SplitDataset(Dataset):
        def __init__(self, items, transform):
            self.items = items
            self.transform = transform

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            path, label = self.items[i]
            img = Image.open(root / path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label

    def subset(split: str):
        return [(r["path"], int(r["label"])) for r in rows if r["split"] == split]

    train_items = subset("train")
    val_items = subset("val")
    test_items = subset("test")

    train_tf = build_transforms(cfg, train=True)
    eval_tf = build_transforms(cfg, train=False)

    train_ds = SplitDataset(train_items, train_tf)
    val_ds = SplitDataset(val_items, eval_tf)
    test_ds = SplitDataset(test_items, eval_tf)

    # 불균형 대응: 클래스 역빈도 샘플링
    label_counts = torch.zeros(cfg["data"]["num_classes"])
    for _, y in train_items:
        label_counts[y] += 1
    weights = [1.0 / label_counts[y].item() for _, y in train_items]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    bs = cfg["train"]["batch_size"]
    nw = cfg["train"].get("num_workers", 0)
    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw)
    return train_loader, val_loader, test_loader
