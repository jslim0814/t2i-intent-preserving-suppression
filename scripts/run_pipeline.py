"""캐릭터 이름 -> 최종 이미지.

  python -m scripts.run_pipeline --name Mario
"""
import argparse

from src import config, pipeline

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="Mario", help="캐릭터 이름 (data/keywords/final/{slug}.json 이 있어야 함)")
ap.add_argument("--out", default="outputs")
ap.add_argument("--num", type=int, default=config.NUM_CANDIDATES, help="후보 이미지 수 (기본 50)")
a = ap.parse_args()

pipeline.run(a.name, out_root=a.out, seeds=range(config.SEED, config.SEED + a.num))
