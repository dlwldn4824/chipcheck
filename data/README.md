# data/

| 경로 | 용도 |
| --- | --- |
| `raw/` | 원본 데이터셋 (다운로드 후 보관, git 제외) |
| `processed/` | 전처리된 이미지·캐시 |
| `splits/` | train/val/test 분할 목록 (csv/json) |

## Carinthia-S (메인)

```bash
python scripts/download_datasets.py --dataset carinthia_s
# 이어받기
curl -L -C - -o data/raw/downloads/carinthia_s_data.zip \
  "https://zenodo.org/records/16895427/files/data.zip?download=1"
python scripts/import_carinthia.py
```

레이아웃:

```
raw/carinthia_s/
  images/class_1..class_6/*.jpg
  masks/class_1..class_6/*.png
  carinthia-s.csv
  summary.json
  manifest.jsonl
```

출처: https://zenodo.org/records/16895427

