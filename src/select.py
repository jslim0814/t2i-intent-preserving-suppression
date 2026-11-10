"""4단계: 저작권 위험도가 가장 낮은 후보 1장을 고른다.

저작권 위험도 점수 = 다른 후보들과의 평균 CLIP 거리 (1 - 평균 코사인 유사도).
값이 낮을수록 후보들이 서로 비슷하다 = 모델이 한 곳(원작 캐릭터)으로 수렴하지 않고
일관되게 생성한 안전한 이미지라는 뜻이므로, 점수가 가장 낮은 이미지를 최종 이미지로 고른다.

선택 규칙
  1) VLM 필터를 통과한 후보(어떤 캐릭터도 검출 안 됨, 판정 실패 제외)만 남긴다
  2) 남은 후보끼리 점수를 다시 계산해 가장 낮은 1장을 고른다 (후보가 1장이면 그대로)
  3) 후보가 모두 검출되었다면 전체 후보를 대상으로 같은 선택을 한다 (fallback)
"""
import gc
import os

import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from . import config


class OriginalityScorer:
    def __init__(self, model_id: str = config.CLIP_ID, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def embed(self, paths: list) -> torch.Tensor:
        feats = []
        for p in paths:
            px = self.processor(images=Image.open(p).convert("RGB"),
                                return_tensors="pt")["pixel_values"].to(self.device)
            vo = self.model.vision_model(pixel_values=px)
            feats.append(self.model.visual_projection(vo.pooler_output))
        return torch.nn.functional.normalize(torch.cat(feats, 0), dim=-1)

    def scores(self, paths: list) -> pd.DataFrame:
        """각 이미지의 (1 - 나머지 이미지들과의 평균 코사인 유사도). 낮을수록 안전."""
        n = len(paths)
        f = self.embed(paths)
        sim = f @ f.T
        avg_sim = (sim.sum(1) - sim.diag()) / (n - 1)
        return pd.DataFrame({"file": [os.path.basename(p) for p in paths],
                             "score": (1 - avg_sim).cpu().tolist()})

    def unload(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


def select_final(paths: list, detections: dict, scorer: OriginalityScorer) -> dict:
    """paths: 후보 이미지 경로, detections: {파일명: detect 결과 dict} (detect.py)

    반환: dict(selected, selected_path, score, n_pool, n_candidates, fallback, table)
    """
    by_name = {os.path.basename(p): p for p in paths}
    ok = [n for n, r in detections.items()
          if n in by_name and not r["has_error"] and r["score"] == 0]
    fallback = len(ok) == 0
    pool = list(by_name) if fallback else ok

    if len(pool) >= 2:
        table = scorer.scores([by_name[n] for n in pool])
        best = table.loc[table["score"].idxmin()]
        name, score = best["file"], float(best["score"])
    else:                                    # 후보 1장: 비교할 상대가 없으니 그대로 선택
        table = pd.DataFrame({"file": pool, "score": [float("nan")]})
        name, score = pool[0], float("nan")

    table["passed_filter"] = table["file"].isin(ok)
    return dict(selected=name, selected_path=by_name[name], score=score,
                n_pool=len(paths), n_candidates=len(pool), fallback=fallback, table=table)
