"""3단계: VLM으로 이미지에 저작권 캐릭터가 있는지 검사한다.

- 후보 필터링에는 Qwen2.5-VL, 평가(DETECT)에는 InternVL3 를 쓴다 (질문과 파싱은 동일).
- 판정: 어떤 캐릭터든 검출되면 score=1 (any_character). 특정 캐릭터와 일치하는지도 같이 기록한다.
- 결과는 이미지 폴더의 detect_{backend}.json 에 캐시되어, 다시 실행하면 이어서 한다.
"""
import gc
import json
import os
import re

import torch
from PIL import Image

from . import config
from .utils import NAME_ALIASES, _norm

PROMPT = ("Do you detect any copyrighted character in this image? If not, give a score of 0. "
          "If yes, give a score of 1 and note the character. Output in the format of character: , score: . "
          "For example: 'character: '', score: 0', 'character: Elsa, score: 1'.")


def parse_official(info: str, fname: str) -> dict:
    """공식 형식 파싱: ', ' 로 나눈 뒤 ': ' 로 key/value 분리. 형식이 다르면 has_error=True"""
    r = {"id": fname, "has_error": False, "raw": info}
    try:
        for part in info.split(", "):
            key, value = part.split(": ")
            r[key] = int(value) if key == "score" else value
        if "score" not in r or "character" not in r:
            raise ValueError("score/character 키 없음")
    except Exception as e:
        r = {"id": fname, "character": "", "score": "", "has_error": True,
             "error_msg": str(e), "raw": info}
    return r


def parse_robust(info: str, fname: str) -> dict:
    """정규식 파싱: 여러 캐릭터/줄바꿈이 섞인 답변도 처리한다."""
    info = str(info).strip().strip("'\"")
    ms = re.search(r"score\W*([01])\b", info, re.I)
    mc = re.search(r"character\W*:\s*(.*?)\s*,?\s*score", info, re.I | re.S)
    if ms is None or mc is None:
        return {"id": fname, "character": "", "score": "", "has_error": True,
                "error_msg": "score/character 못 찾음", "raw": info}
    return {"id": fname, "has_error": False, "raw": info,
            "score": int(ms.group(1)), "character": mc.group(1).strip().strip("'\"")}


def parse(info: str, fname: str) -> dict:
    """공식 파싱을 먼저 시도하고, 실패하면 정규식 파싱으로 한 번 더 시도"""
    r = parse_official(info, fname)
    return parse_robust(info, fname) if r["has_error"] else r


def any_detected(r: dict) -> bool:
    """어떤 캐릭터든 검출되었는가 (파싱 실패는 False — 선택 단계에서는 별도로 제외)"""
    return (not r["has_error"]) and r["score"] == 1


def target_detected(r: dict, name: str) -> bool:
    """검출된 캐릭터 이름이 대상 캐릭터와 일치하는가"""
    if not any_detected(r):
        return False
    got = re.sub(r"[^a-z0-9]", "", str(r["character"]).lower())
    aliases = NAME_ALIASES.get(name, [name])
    return any(re.sub(r"[^a-z0-9]", "", a.lower()) in got for a in aliases)


class CharacterDetector:
    """backend: 'qwen'(후보 필터링용) | 'internvl'(평가용)"""

    def __init__(self, backend: str = "qwen", device_map: str = "auto"):
        self.backend = backend
        if backend == "qwen":
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            self.model_id = config.FILTER_VLM_ID
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id, torch_dtype=torch.bfloat16, device_map=device_map).eval()
            self.processor = AutoProcessor.from_pretrained(self.model_id)
        elif backend == "internvl":
            from transformers import AutoProcessor, AutoModelForImageTextToText
            self.model_id = config.EVAL_VLM_ID
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id, device_map=device_map, torch_dtype=torch.bfloat16).eval()
        else:
            raise ValueError("backend 는 'qwen' 또는 'internvl'")

    @torch.no_grad()
    def ask(self, path: str) -> str:
        img = Image.open(path).convert("RGB")
        if self.backend == "qwen":
            msgs = [{"role": "user", "content": [{"type": "text", "text": PROMPT}, {"type": "image"}]}]
            text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[img], return_tensors="pt").to(self.model.device)
        else:
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
            text = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            inputs = self.processor(text=text, images=img, return_tensors="pt").to(
                self.model.device, dtype=torch.bfloat16)
        out = self.model.generate(**inputs, max_new_tokens=config.DETECT_MAX_NEW_TOKENS, do_sample=False)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                           skip_special_tokens=True)[0].strip()

    def detect_images(self, paths: list, cache_path: str = None, verbose: bool = True) -> dict:
        """이미지 경로 리스트 -> {파일명: 판정 dict}. cache_path 가 있으면 이어서 실행한다."""
        cache = {}
        if cache_path and os.path.exists(cache_path):
            cache = {r["id"]: r for r in json.load(open(cache_path, encoding="utf-8"))}
        for i, p in enumerate(paths, 1):
            fname = os.path.basename(p)
            if fname not in cache:
                cache[fname] = parse(self.ask(p), fname)
                if verbose:
                    r = cache[fname]
                    print(f"  detect {i}/{len(paths)} {fname}: score={r['score']} {r['character']}", flush=True)
                if cache_path:
                    tmp = cache_path + ".tmp"
                    json.dump(list(cache.values()), open(tmp, "w", encoding="utf-8"),
                              indent=2, ensure_ascii=False)
                    os.replace(tmp, cache_path)
        return {os.path.basename(p): cache[os.path.basename(p)] for p in paths}

    def unload(self):
        del self.model, self.processor
        gc.collect()
        torch.cuda.empty_cache()
