"""자체 포맷을 쓰는 팹을 위한 예시 어댑터.

어댑터 하나가 이 정도 분량이면 된다는 것을 보이는 게 목적이다.
"""

from __future__ import annotations

import csv
import io

from ..schema import DefectEvent
from .base import register


class FlatCsvAdapter:
    name = "flat_csv"
    #  팹에서 쓰는 컬럼명 → 우리 필드명. 보통 여기만 고치면 끝난다.
    COLMAP = {"defect_id": "defect_id", "lot": "lot_id", "wafer": "wafer_id",
              "process_step": "step", "tool": "equipment", "recipe": "recipe",
              "image": "image_path", "message": "text"}

    def parse(self, raw: str) -> list[DefectEvent]:
        out = []
        for row in csv.DictReader(io.StringIO(raw)):
            kw = {dst: (row.get(src) or "").strip()
                  for src, dst in self.COLMAP.items() if row.get(src)}
            geom = {}
            for src, dst in (("size_um", "size_um"), ("area_um2", "area_um2")):
                try:
                    geom[dst] = float(row[src])
                except (KeyError, TypeError, ValueError):
                    pass
            out.append(DefectEvent(geometry=geom, raw=dict(row),
                                   **{k: v for k, v in kw.items() if k != "defect_id"},
                                   defect_id=kw.get("defect_id", "?")))
        return out


register(FlatCsvAdapter())
