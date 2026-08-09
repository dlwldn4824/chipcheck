from __future__ import annotations

from pathlib import Path
from typing import Any

from chipcheck.utils import ensure_dir, resolve_device


class Trainer:
    """베이스라인 학습 루프 스켈레톤."""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.device = resolve_device(cfg["train"].get("device", "auto"))
        self.ckpt_dir = ensure_dir(cfg["train"]["checkpoint_dir"])

    def fit(self, train_loader, val_loader, model) -> dict[str, Any]:
        """실제 학습은 데이터셋 연결 후 구현. 현재는 뼈대만 제공."""
        import torch
        import torch.nn as nn

        model = model.to(self.device)
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg["train"]["lr"],
            weight_decay=self.cfg["train"]["weight_decay"],
        )
        # 가중 샘플링을 쓰므로 CE는 기본값 (이중 보정 방지)
        criterion = nn.CrossEntropyLoss()
        history: dict[str, list] = {"train_loss": [], "val_acc": []}

        epochs = self.cfg["train"]["epochs"]
        patience = self.cfg["train"].get("early_stopping_patience", 7)
        best_acc = -1.0
        bad = 0
        for epoch in range(epochs):
            model.train()
            running = 0.0
            n = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                opt.step()
                running += loss.item() * xb.size(0)
                n += xb.size(0)
            train_loss = running / max(n, 1)
            val_acc = self._eval_acc(model, val_loader)
            history["train_loss"].append(train_loss)
            history["val_acc"].append(val_acc)
            print(f"[{epoch+1}/{epochs}] loss={train_loss:.4f} val_acc={val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                bad = 0
                path = Path(self.ckpt_dir) / "best.pt"
                torch.save({"model": model.state_dict(), "cfg": self.cfg}, path)
            else:
                bad += 1
                if bad >= patience:
                    print(f"early stopping @ epoch {epoch+1}")
                    break

        return {"history": history, "best_val_acc": best_acc}

    def _eval_acc(self, model, loader) -> float:
        import torch

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = model(xb).argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        return correct / max(total, 1)
