"""원인 지식베이스 적재.

CSV 한 장이면 된다. 공정이 바뀌면 이 표만 다시 채운다.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .schema import Cause

COLUMNS = ["cause_id", "name", "step", "equipment_hint", "defect_classes",
           "morphology", "log_keywords", "action", "reference"]


def load_kb(path: str | Path) -> list[Cause]:
    out: list[Cause] = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("cause_id") or row["cause_id"].startswith("#"):
                continue
            classes = tuple(c.strip() for c in (row.get("defect_classes") or "").split(";") if c.strip())
            out.append(Cause(
                cause_id=row["cause_id"].strip(),
                name=(row.get("name") or "").strip(),
                step=(row.get("step") or "").strip(),
                equipment_hint=(row.get("equipment_hint") or "").strip(),
                defect_classes=classes,
                morphology=(row.get("morphology") or "").strip(),
                log_keywords=(row.get("log_keywords") or "").strip(),
                action=(row.get("action") or "").strip(),
                reference=(row.get("reference") or "").strip(),
            ))
    return out


def write_template(path: str | Path, rows: list[dict] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows or []:
            w.writerow(r)
    return path
