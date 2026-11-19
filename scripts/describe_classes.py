"""클래스마다 결함이 실제로 어떻게 생겼는지 마스크로 측정한다.

Carinthia-S는 라벨이 1~6 숫자뿐이고 결함 이름이 없다. 이름을 지어내면
지식베이스의 근거가 사라진다. 그래서 세그멘테이션 마스크에서 형태 통계를
재고, 코드로 적어 둔 규칙(name_from_stats)으로 이름을 붙인다. 붙인 이름은
측정값과 함께 남겨서 누구든 다시 확인할 수 있게 한다.

출력
  results/metrics/class_morphology.json   측정값 + 팀이 붙인 형태 이름
  data/cause_kb/classes.csv               지식베이스가 참조하는 클래스 표
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
MASK_DIR = ROOT / "data/raw/carinthia_s/masks"
OUT_JSON = ROOT / "results/metrics/class_morphology.json"
OUT_CSV = ROOT / "data/cause_kb/classes.csv"


def mask_stats(path: Path) -> dict | None:
    """마스크 한 장의 형태 통계. 결함 영역이 없으면 None."""
    m = np.array(Image.open(path).convert("L")) > 127
    area = int(m.sum())
    if area == 0:
        return None
    lab, n = ndimage.label(m)
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    big = int(np.argmax(sizes)) + 1
    ys, xs = np.nonzero(lab == big)
    h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    return {
        "area_frac": area / m.size,                   # 결함이 화면에서 차지하는 비율
        "n_blobs": float(n),                          # 떨어져 있는 덩어리 개수
        "elongation": max(h, w) / max(1, min(h, w)),  # 길쭉한 정도
        "fill": float(sizes.max()) / float(h * w),    # 외접 사각형을 얼마나 채우는지
    }


def name_from_stats(s: dict) -> tuple[str, str]:
    """측정값만 보고 형태 이름을 정한다. 규칙을 코드로 남겨 근거를 남긴다."""
    if s["empty_frac"] >= 0.90:
        return "정상", "결함 영역이 표시되지 않은 이미지. 마스크가 비어 있다"
    if s["fill_med"] <= 0.15 or s["elongation_p90"] >= 8.0:
        return "선형 결함", "한 방향으로 가늘고 길게 이어진 형태 (스크래치·균열 계열)"
    if s["n_blobs_med"] >= 3 and s["fill_med"] <= 0.40:
        return "확산형 다중 결함", "여러 덩어리가 흩어져 넓게 퍼진 형태 (파티클 산포·얼룩 계열)"
    if s["area_frac_med"] >= 0.05:
        return "대형 단일 결함", "경계가 뚜렷한 큰 덩어리 하나. 화면의 5% 이상을 차지한다"
    if s["area_frac_med"] >= 0.025:
        return "중형 단일 결함", "경계가 뚜렷한 덩어리 하나. 화면의 2.5~5%를 차지한다"
    return "소형 단일 결함", "경계가 뚜렷한 작은 덩어리 하나. 화면의 2.5% 미만"


def main() -> None:
    per_class: dict[str, list[dict]] = defaultdict(list)
    empty_count: dict[str, int] = defaultdict(int)
    total_count: dict[str, int] = defaultdict(int)

    for cls_dir in sorted(p for p in MASK_DIR.iterdir() if p.is_dir()):
        for f in sorted(cls_dir.glob("*.png")):
            total_count[cls_dir.name] += 1
            st = mask_stats(f)
            if st is None:
                empty_count[cls_dir.name] += 1
            else:
                per_class[cls_dir.name].append(st)

    out, rows = {}, []
    for cls in sorted(total_count):
        vals = per_class[cls]
        n_tot = total_count[cls]
        agg = {"empty_frac": empty_count[cls] / n_tot}
        if vals:
            agg |= {
                "area_frac_med": float(np.median([v["area_frac"] for v in vals])),
                "n_blobs_med": float(np.median([v["n_blobs"] for v in vals])),
                "elongation_med": float(np.median([v["elongation"] for v in vals])),
                "elongation_p90": float(np.percentile([v["elongation"] for v in vals], 90)),
                "fill_med": float(np.median([v["fill"] for v in vals])),
            }
        else:
            agg |= {k: 0.0 for k in ("area_frac_med", "n_blobs_med",
                                     "elongation_med", "elongation_p90", "fill_med")}
        name, desc = name_from_stats(agg)
        out[cls] = {"n_images": n_tot, "n_with_defect_mask": len(vals),
                    "stats": {k: round(v, 4) for k, v in agg.items()},
                    "team_name": name, "description": desc}
        rows.append({"class_id": cls, "team_name": name, "morphology": desc,
                     "n_images": n_tot,
                     "empty_mask_frac": round(agg["empty_frac"], 4),
                     "area_frac_median": round(agg["area_frac_med"], 4),
                     "n_blobs_median": round(agg["n_blobs_med"], 1),
                     "elongation_p90": round(agg["elongation_p90"], 2),
                     "fill_median": round(agg["fill_med"], 2)})
        print(f"{cls:9s} n={n_tot:5d} empty={agg['empty_frac']:5.1%} "
              f"area={agg['area_frac_med']:.4f} blobs={agg['n_blobs_med']:.1f} "
              f"elong_p90={agg['elongation_p90']:5.2f} fill={agg['fill_med']:.2f}"
              f"  ->  {name}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "source": "Carinthia-S 세그멘테이션 마스크 4,591장 전수 측정",
        "note": "데이터셋은 결함 이름을 제공하지 않는다. team_name은 측정된 형태 "
                "통계에 name_from_stats() 규칙을 적용해 팀이 붙인 이름이며, "
                "공정상의 공식 결함 분류명이 아니다.",
        "classes": out,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{OUT_JSON.relative_to(ROOT)}\n{OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
