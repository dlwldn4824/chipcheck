"""KLARF를 본뜬 결함 결과 파일 + 장비 알람 라인을 읽는다.

실제 KLARF는 장비와 버전마다 필드 구성이 다르다. 여기서는 공개된 구조만
따라가고, 실제 팹에 넣을 때는 이 파일만 그 장비 포맷에 맞춰 고치면 된다.
"""

from __future__ import annotations

import re

from ..schema import DefectEvent
from .base import register

_KV = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s+(.*?);?\s*$")


class KlarfLikeAdapter:
    name = "klarf_like"

    def parse(self, raw: str) -> list[DefectEvent]:
        header: dict[str, str] = {}
        columns: list[str] = []
        rows: list[list[str]] = []
        alarms: list[str] = []
        mode = None

        for line in raw.splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            if s.startswith("DefectRecordSpec"):
                columns = s.replace(";", "").split()[2:]
                mode = None
                continue
            if s.startswith("DefectList"):
                mode = "defects"
                continue
            if s.startswith("EventList"):
                mode = "events"
                continue
            if mode == "defects":
                if s.startswith(("SummaryList", "EndOfFile")):
                    mode = None
                    continue
                rows.append(s.replace(";", "").split())
                continue
            if mode == "events":
                alarms.append(s)
                continue
            m = _KV.match(s)
            if m:
                header[m.group(1)] = m.group(2).strip().strip('"')

        text_blob = "  ".join(alarms)
        out: list[DefectEvent] = []
        for r in rows:
            rec = dict(zip(columns, r))
            geom = {}
            for k_src, k_dst in (("XREL", "x_rel"), ("YREL", "y_rel"),
                                 ("DSIZE", "size_um"), ("DEFECTAREA", "area_um2")):
                try:
                    geom[k_dst] = float(rec[k_src])
                except (KeyError, ValueError):
                    pass
            out.append(DefectEvent(
                defect_id=rec.get("DEFECTID", "?"),
                lot_id=header.get("LotID", ""),
                wafer_id=header.get("WaferID", ""),
                image_path=rec.get("IMAGEPATH") or None,
                step=header.get("StepID") or None,
                equipment=header.get("InspectionStationID") or None,
                recipe=header.get("Recipe") or None,
                geometry=geom,
                text=text_blob,
                raw={"header": header, "record": rec},
            ))
        return out


register(KlarfLikeAdapter())
