"""최종 이미지(또는 폴더의 이미지들)의 DETECT / CONS / p-CONS 계산.  (프로젝트 루트에서 실행)

  # 한 환경에서 cons, detect 모두 실행 (InternVL3 와 t2v_metrics 가 둘 다 되는 환경일 경우)
  python -m scripts.run_eval --name Mario --images outputs/mario/final.png

  # 환경 분리하여 실행 (DETECT 는 최신 transformers 환경, CONS 는 t2v_metrics 환경)
  python -m scripts.run_eval --name Mario --images outputs/mario/final.png --only detect
  python -m scripts.run_eval --name Mario --images outputs/mario/final.png --only cons

  # 후보 전체 평균 (Standard 기준선)
  python -m scripts.run_eval --name Mario --images outputs/mario/candidates --out outputs/mario/eval_candidates.csv
"""
import argparse
import glob
import os

from src import metrics
from src.utils import slugify

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="Mario", help="캐릭터 이름")
ap.add_argument("--images", required=True, help="이미지 파일 또는 폴더")
ap.add_argument("--keyword", default=None, help="CONS 평가 문구 (없으면 csv 에서 찾음)")
ap.add_argument("--keyword-csv", default="data/eval/key_entitie_n1.csv")
ap.add_argument("--prompt", default=None, help="p-CONS 텍스트 (기본: 캐릭터 이름)")
ap.add_argument("--out", default=None, help="결과 csv 경로")
ap.add_argument("--only", choices=["detect", "cons"], default=None,
                help="detect: DETECT 만 / cons: CONS, p-CONS 만. 안 주면 전부 계산")
a = ap.parse_args()

paths = sorted(glob.glob(os.path.join(a.images, "*.png"))) if os.path.isdir(a.images) else [a.images]

# --only detect 이면 CONS 를 건너뛰고, --only cons 이면 DETECT 를 건너뛴다
do_detect, do_vqa = a.only != "cons", a.only != "detect"
# CONS 평가 문구는 CONS 를 계산할 때만 필요하다
kw = (a.keyword or metrics.load_keyword(a.name, a.keyword_csv)) if do_vqa else None

df, mean = metrics.evaluate(a.name, paths, kw, a.prompt, do_detect=do_detect, do_vqa=do_vqa)

suffix = f"_{a.only}" if a.only else ""                 # eval.csv / eval_detect.csv / eval_cons.csv
out = a.out or f"outputs/{slugify(a.name)}/eval{suffix}.csv"
os.makedirs(os.path.dirname(out), exist_ok=True)
df.to_csv(out, index=False)

print(f"{a.name} (n={mean['n']}" + (f", CONS keyword='{kw}')" if kw else ")"))
print(" | ".join(f"{k} {mean[k]:.3f}" for k in ("DETECT", "CONS", "pCONS") if k in mean) + f"\nsaved: {out}")