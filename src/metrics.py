"""평가 지표.

DETECT : 최종 이미지에서 저작권 캐릭터가 검출되는 비율 (InternVL3-8B, 필터와 다른 모델). 낮을수록 좋음
CONS   : 캐릭터의 핵심 특징(예: Mario -> plumber)이 이미지에 남아 있는 정도 (VQAScore). 높을수록 좋음
p-CONS : 사용자가 입력한 프롬프트(캐릭터 이름)와 이미지의 일치도 (VQAScore). 높을수록 좋음

DETECT 와 CONS 는 필요한 라이브러리 버전이 달라서 가상환경을 나눠 돌릴 수 있다.
  - DETECT         : 최신 transformers 환경 (InternVL3 필요)      -> run_eval.py --only detect
  - CONS / p-CONS  : t2v_metrics 가 설치된 환경                    -> run_eval.py --only cons
"""
import gc
import os
import re

import pandas as pd
import torch

from . import config
from .detect import CharacterDetector, any_detected, target_detected


class VQAScorer:
    """t2v_metrics 의 VQAScore(clip-flant5-xl) 래퍼. CONS / p-CONS 계산에 쓴다."""

    def __init__(self, model: str = config.VQA_MODEL, device: str = "cuda"):
        import t2v_metrics    # CONS 를 계산할 때만 필요하므로 여기서 import (DETECT 만 돌릴 땐 설치 안 돼 있어도 됨)
        # cache_dir 을 고정하지 않으면 t2v_metrics 는 실행한 폴더의 ./hf_cache/ 를 보고 없으면 다시 받는다
        self.model = t2v_metrics.VQAScore(model=model, device=device, cache_dir=config.VQA_CACHE_DIR)

    def score(self, path: str, text: str) -> float:
        """이미지 1장과 문장 1개의 일치도 점수 (0~1)"""
        return float(self.model(images=[path], texts=[text]).reshape(-1)[0].cpu())

    def unload(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


def load_keyword(name: str, csv_path: str) -> str:
    """CONS 평가 문구를 찾는다.

    key_entitie_n1.csv 형식: 첫 열은 인덱스, 두 번째 열이 캐릭터 이름(target), keyword 열이 평가 문구.
    이름은 대소문자/공백/하이픈을 무시하고 비교한다 (Spider-Man == Spiderman).
    """
    norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())
    df = pd.read_csv(csv_path, index_col=1)      # 두 번째 열(target)을 이름으로 사용
    m = {}
    for nm, val in df["keyword"].items():
        m.setdefault(norm(nm), val)
    if norm(name) not in m:
        raise KeyError(f"{csv_path} 에 {name} 가 없어요. --keyword 로 직접 주세요")
    return m[norm(name)]


def evaluate(name: str, image_paths: list, keyword: str, prompt: str = None,
             detector_backend: str = "internvl", do_detect: bool = True, do_vqa: bool = True):
    """이미지들에 대해 DETECT / CONS / p-CONS 를 계산한다.

    do_detect=False : DETECT 를 건너뜀 (InternVL3 를 못 쓰는 환경에서 CONS 만 돌릴 때)
    do_vqa=False    : CONS / p-CONS 를 건너뜀 (t2v_metrics 가 없는 환경에서 DETECT 만 돌릴 때)
    반환: (이미지별 DataFrame, 평균 dict)
    """
    prompt = prompt or name          # p-CONS 텍스트: 사용자가 입력한 프롬프트 (기본은 캐릭터 이름)
    df = pd.DataFrame({"image": image_paths})
    mean = dict(n=len(df))

    if do_detect:
        det = CharacterDetector(detector_backend)          # InternVL3-8B
        dets = det.detect_images(image_paths)
        det.unload()
        rs = [dets[os.path.basename(p)] for p in image_paths]
        df["detect_any"] = [int(any_detected(r)) for r in rs]            # 어떤 캐릭터든 검출되면 1 (DETECT 의 기준)
        df["detect_target"] = [int(target_detected(r, name)) for r in rs]  # 대상 캐릭터로 검출되면 1 (참고용)
        df["detect_character"] = [r["character"] for r in rs]
        df["detect_error"] = [int(r["has_error"]) for r in rs]            # 답변 형식 파싱 실패 여부
        mean["DETECT"] = df["detect_any"].mean()

    if do_vqa:
        vqa = VQAScorer()
        df["cons"] = [vqa.score(p, keyword) for p in image_paths]   # 핵심 특징 문구와의 일치도
        df["pcons"] = [vqa.score(p, prompt) for p in image_paths]   # 입력 프롬프트와의 일치도
        vqa.unload()
        mean["CONS"], mean["pCONS"] = df["cons"].mean(), df["pcons"].mean()
    return df, mean