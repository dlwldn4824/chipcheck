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
        """파일 하나에 웨이퍼 레코드가 여러 개 들어 있어도 각각 따로 읽는다.

        실제 KLARF도 한 로트의 웨이퍼를 이어 붙여 쓴다. 블록이 끝나면
        (EndOfFile 또는 다음 LotID) 지금까지 모은 헤더·결함·알람을 한 덩어리로
        닫는다. 그래야 알람 문구가 다른 웨이퍼로 새지 않는다.
        """
        out: list[DefectEvent] = []
        header: dict[str, str] = {}
        columns: list[str] = []
        rows: list[list[str]] = []
        alarms: list[str] = []
        mode = None

        def flush() -> None:
            nonlocal header, rows, alarms, mode
            if rows:
                out.extend(self._events(header, columns, rows, alarms))
            header, rows, alarms, mode = {}, [], [], None

        for line in raw.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith("EndOfFile"):
                flush()
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
            if s.startswith("SummaryList"):
                mode = None
                continue
            if mode == "defects":
                rows.append(s.replace(";", "").split())
                continue
            if mode == "events":
                alarms.append(s)
                continue
            m = _KV.match(s)
            if m:
                key = m.group(1)
                if key == "LotID" and rows:
                    flush()          # EndOfFile 없이 다음 레코드가 시작된 경우
                header[key] = m.group(2).strip().strip('"')
        flush()
        return out

    @staticmethod
    def _events(header, columns, rows, alarms) -> list[DefectEvent]:
        text_blob = "  ".join(alarms)
        out = []
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
                raw={"header": dict(header), "record": rec},
            ))
        return out


register(KlarfLikeAdapter())
