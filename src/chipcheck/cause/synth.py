"""가상 로그 생성기.

Carinthia-S에는 로그도 원인 라벨도 없다. 그래서 실제 팹 로그의 구조만
본뜬 가상 로그를 만든다. 무엇을 할 수 있고 무엇을 할 수 없는지 먼저 적는다.

할 수 있는 것
  어댑터가 포맷을 제대로 흡수하는지, 엔진이 잡음과 방해 알람 속에서 맞는
  원인을 끌어올리는지 확인한다.

할 수 없는 것
  이 로그는 지식베이스의 키워드에서 파생된다. 그러므로 여기서 나온 정확도는
  '실제 팹에서의 원인 추정 정확도'가 아니다. 그 수치를 얻으려면 실제 로그와
  엔지니어가 확정한 원인 라벨이 있어야 한다.

그래서 생성기는 일부러 문제를 어렵게 만든다.
  - 정답 원인의 키워드 중 일부만 남긴다 (keep_ratio)
  - 다른 원인의 알람을 방해물로 섞는다 (n_distractors)
  - 공정 로그에 흔한 무해한 라인을 채운다
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .schema import Cause

NOISE = [
    "INFO wafer loaded slot {slot} carrier {lot}",
    "INFO recipe {recipe} started",
    "INFO chamber pressure stable 5.0 mTorr",
    "INFO 온도 안정화 완료",
    "WARN queue time 12 min remaining",
    "INFO robot handoff complete",
    "INFO exhaust flow nominal",
    "INFO 슬롯 매핑 확인 완료",
    "INFO MFC setpoint reached",
    "INFO interlock cleared",
]

STEP_TOOL = {
    "litho": ["TRACK01", "SCANNER02"], "etch": ["ETCHER03", "ETCHER07"],
    "cmp": ["POLISHER01", "CLEANER02"], "cvd": ["CVD04", "CVD09"],
    "clean": ["WETBENCH02", "SPINDRYER01"], "implant": ["IMPLANTER01"],
    "thermal": ["RTP02", "FURNACE01"], "metal": ["SPUTTER01", "PLATER03"],
    "": ["INSPECTOR01", "ROBOT02", "LOADPORT01", "SEM01"],
}


@dataclass
class SynthCase:
    """가상 로그 한 건과 그 정답."""

    case_id: str
    truth_cause_id: str
    truth_class: str
    step: str
    equipment: str
    recipe: str
    lot_id: str
    wafer_id: str
    alarms: list[str] = field(default_factory=list)
    geometry: dict[str, float] = field(default_factory=dict)


def _keywords(c: Cause) -> list[str]:
    return [w for w in c.log_keywords.split() if w]


def _alarm_lines(c: Cause, rng: random.Random, keep_ratio: float) -> list[str]:
    kw = _keywords(c)
    n = max(2, int(round(len(kw) * keep_ratio)))
    idx = sorted(rng.sample(range(len(kw)), min(n, len(kw))))
    picked = [kw[i] for i in idx]          # 원문 순서를 지켜야 로그처럼 읽힌다
    sev = rng.choice(["ALARM", "ERROR", "WARN"])
    lines = [f"{sev} {' '.join(picked[:3])}"]
    if len(picked) > 3:
        lines.append(f"DETAIL {' '.join(picked[3:])}")
    return lines


def make_case(causes: list[Cause], rng: random.Random, idx: int,
              keep_ratio: float = 0.6, n_distractors: int = 2,
              n_noise: int = 4) -> SynthCase:
    """원인 하나를 정답으로 고르고, 그 원인이 남길 법한 로그를 만든다."""
    truth = rng.choice([c for c in causes if c.defect_classes])
    step = truth.step
    tools = STEP_TOOL.get(step, STEP_TOOL[""])
    equip = next((t for t in tools if truth.equipment_hint
                  and truth.equipment_hint.lower() in t.lower()), rng.choice(tools))
    lot = f"LOT{rng.randint(1000, 9999)}"
    wafer = f"W{rng.randint(1, 25):02d}"
    recipe = f"{(step or 'gen').upper()}_{rng.choice('ABCD')}{rng.randint(10, 99)}"

    alarms = _alarm_lines(truth, rng, keep_ratio)
    pool = [c for c in causes if c.cause_id != truth.cause_id]
    for d in rng.sample(pool, min(n_distractors, len(pool))):
        kw = _keywords(d)
        if kw:
            j = sorted(rng.sample(range(len(kw)), min(2, len(kw))))
            alarms.append("WARN " + " ".join(kw[i] for i in j))
    for t in rng.sample(NOISE, min(n_noise, len(NOISE))):
        alarms.append(t.format(slot=rng.randint(1, 25), lot=lot, recipe=recipe))
    rng.shuffle(alarms)

    return SynthCase(
        case_id=f"CASE{idx:04d}",
        truth_cause_id=truth.cause_id,
        truth_class=rng.choice(truth.defect_classes),
        step=step, equipment=equip, recipe=recipe, lot_id=lot, wafer_id=wafer,
        alarms=alarms,
        geometry={"x_rel": round(rng.uniform(-150, 150), 2),
                  "y_rel": round(rng.uniform(-150, 150), 2),
                  "size_um": round(rng.uniform(0.2, 12.0), 2)},
    )


def to_klarf(cases: list[SynthCase]) -> str:
    """KLARF 구조를 본뜬 검사 결과 파일. 케이스마다 한 덩어리."""
    out = ["# KLARF를 본뜬 가상 파일. 실제 KLARF는 장비·버전마다 필드가 다르다.",
           "FileVersion 1 2;", "FileTimestamp 11-19-26 09:00:00;"]
    for c in cases:
        out += [
            "",
            f'LotID "{c.lot_id}";',
            f'WaferID "{c.wafer_id}";',
            f'StepID "{c.step or "unknown"}";',
            f'InspectionStationID "{c.equipment}";',
            f'Recipe "{c.recipe}";',
            "DefectRecordSpec 5 DEFECTID XREL YREL DSIZE IMAGEPATH;",
            "DefectList",
            f"  {c.case_id} {c.geometry['x_rel']} {c.geometry['y_rel']} "
            f"{c.geometry['size_um']} images/{c.truth_class}/{c.case_id}.jpg;",
            "EventList",
        ] + [f"  {a}" for a in c.alarms] + ["EndOfFile;"]
    return "\n".join(out) + "\n"


def to_flat_csv(cases: list[SynthCase]) -> str:
    """자체 포맷을 쓰는 팹을 가정한 평면 CSV."""
    rows = ["defect_id,lot,wafer,process_step,tool,recipe,image,size_um,message"]
    for c in cases:
        msg = " | ".join(c.alarms).replace('"', "'")
        rows.append(f'{c.case_id},{c.lot_id},{c.wafer_id},{c.step},{c.equipment},'
                    f'{c.recipe},images/{c.truth_class}/{c.case_id}.jpg,'
                    f'{c.geometry["size_um"]},"{msg}"')
    return "\n".join(rows) + "\n"


def to_secs_gem(cases: list[SynthCase]) -> str:
    """SECS/GEM(SEMI E5·E30) 알람 리포트를 본뜬 텍스트 로그."""
    out = ["# SEMI E5/E30 알람 리포트를 본뜬 가상 로그 (S5F1 / S6F11 형태)"]
    for i, c in enumerate(cases):
        out.append(f"\n[{c.equipment}] S1F1 W  ARE YOU THERE  -> S1F2 ONLINE")
        for j, a in enumerate(c.alarms):
            sev = a.split()[0]
            if sev in ("ALARM", "ERROR"):
                out.append(f"[{c.equipment}] S5F1 ALID={2000 + i * 10 + j} "
                           f"ALCD=SET  ALTX=\"{a}\"")
            else:
                out.append(f"[{c.equipment}] S6F11 CEID={7000 + i * 10 + j} "
                           f"RPT=\"{a}\"")
        out.append(f"[{c.equipment}] S6F11 CEID=9001 RPT=\"PROCESS END "
                   f"LOT={c.lot_id} WAFER={c.wafer_id} PPID={c.recipe}\"")
    return "\n".join(out) + "\n"


def truth_csv(cases: list[SynthCase]) -> str:
    rows = ["case_id,truth_cause_id,truth_class,step,equipment"]
    for c in cases:
        rows.append(f"{c.case_id},{c.truth_cause_id},{c.truth_class},{c.step},{c.equipment}")
    return "\n".join(rows) + "\n"
