"""1단계: 오픈 LLM(Llama-3.1-8B)으로 사용자 프롬프트를 우회적 프롬프트로 재작성한다.

캐릭터 이름이 들어간 요청을 받아, 캐릭터를 직접 언급하지 않고 일반적 특징으로
묘사하는 프롬프트를 만든다. 거절 응답, 너무 짧은 응답, 캐릭터 이름 유출이 있으면
seed를 바꿔 다시 생성한다.

사용 예:
    rewriter = PromptRewriter()
    result = rewriter.rewrite("Mario")
    print(result["prompt"])
    rewriter.unload()     # 이미지 생성 전에 GPU 메모리를 비우고 싶을 때
"""
import gc
import json
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from . import config
from .prompts import CLEAN_SYS
from .utils import has_name, slugify

# LLM이 프롬프트 앞에 붙이는 "Here's a prompt:" 같은 설명 문구
PREAMBLE_RE = re.compile(
    r"^\s*("
    r"here['’]?s?[^:\n]{0,150}:|"
    r"sure[^:\n]{0,40}:|certainly[^:\n]{0,40}:|"
    r"a prompt for dall-?e[^:\n]*:|"
    r"to (adhere to|generate|create)[^:\n]{0,60}:|"
    r"given the requirement[^:\n]{0,80}:|"
    r"for the \d+(st|nd|rd|th)[- ]?iteration[^.\n]*\."
    r")\s*",
    re.I,
)
# 프롬프트 뒤에 붙는 "Note: ..." 같은 설명 문구
TRAILING_EXPLAIN_RE = re.compile(
    r"\n\s*(note:|i['’]ve made the following changes|the following changes|"
    r"changes:|\*\s*changed).*$",
    re.I | re.S,
)
REFUSAL_RE = re.compile(
    r"\b(i can['’]?t|i cannot|i can not|i['’]?m (sorry|unable)|i am unable|sorry,? but|"
    r"unable to (fulfill|create|generate|help))\b",
    re.I,
)


def clean_generated_prompt(txt: str) -> str:
    """LLM 출력에서 앞뒤 설명 문구, 번호, 따옴표를 걷어내고 프롬프트만 남긴다."""
    t = txt.strip()
    prev = None
    while prev != t:
        prev = t
        t = PREAMBLE_RE.sub("", t, count=1).strip()
    t = TRAILING_EXPLAIN_RE.sub("", t).strip()
    t = re.sub(r"^\s*\d+[\.\)]\s*", "", t)
    return t.strip(" \"'“”")


def strip_preamble_line(t: str) -> str:
    """첫 줄이 'Here's a rewritten prompt ...' 같은 설명이면 그 줄만 제거한다."""
    parts = t.strip().split("\n", 1)
    if (len(parts) == 2 and len(parts[0]) < 250
            and re.match(r"^\s*(here|sure|certainly)", parts[0], re.I)
            and re.search(r"prompt|rewrit|image", parts[0], re.I)):
        return parts[1].strip()
    return t


def is_bad_prompt(txt: str, name: str = None, min_words: int = 15) -> bool:
    """거절 응답이거나, 너무 짧거나, 캐릭터 이름이 남아 있으면 True"""
    t = txt.strip()
    return (bool(REFUSAL_RE.search(t)) or len(t.split()) < min_words
            or bool(name and has_name(t, name)))


class PromptRewriter:
    def __init__(self, model_id: str = config.LLM_ID, device_map: str = "auto"):
        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map=device_map
        )

    # ------------------------------------------------------------------
    def chat(self, system: str, user: str, max_new_tokens: int = 512,
             temperature: float = None, do_sample: bool = True, top_p: float = 1.0) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        inputs = self.tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)

        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample,
                          pad_token_id=self.tok.eos_token_id,
                          repetition_penalty=1.0)       # 모델 기본값 덮어쓰기
        if do_sample:
            if temperature is None:
                raise ValueError("do_sample=True인데 temperature를 안 줬어요")
            gen_kwargs.update(temperature=temperature, top_p=top_p, top_k=0)  # top_k=0 -> 비활성
        else:
            gen_kwargs.update(temperature=None, top_p=None)

        out = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tok.decode(new_tokens, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    def rewrite(self, name: str,
                temperature: float = config.REWRITE_TEMPERATURE,
                base_seed: int = config.SEED,
                max_tries: int = config.REWRITE_MAX_TRIES,
                max_new_tokens: int = config.REWRITE_MAX_NEW_TOKENS) -> dict:
        """캐릭터 이름 -> 우회적 프롬프트.

        거절/이름 유출/너무 짧은 응답이면 seed를 base_seed+1, +2 ... 로 바꿔 재시도한다.
        반환: dict(name, prompt, raw, seed, attempts, temperature, user_message, attempt_log)
        max_tries번 모두 실패하면 RuntimeError.
        """
        user = f"'Generate an image: {name}."   # He et al. 공식 코드의 user 메시지 그대로
        log = []
        for k in range(max_tries):               # k=0이 기본 설정(seed 42)
            seed = base_seed + k
            set_seed(seed)
            raw = self.chat(CLEAN_SYS, user, max_new_tokens=max_new_tokens,
                            temperature=temperature, do_sample=True)
            fixed = strip_preamble_line(raw)
            prompt = clean_generated_prompt(fixed)
            ok = not is_bad_prompt(prompt, name)
            log.append(dict(seed=seed, ok=ok, format_fix=(fixed != raw), raw=raw))
            if ok:
                return dict(name=name, prompt=prompt, raw=raw, seed=seed, attempts=k + 1,
                            temperature=temperature, max_new_tokens=max_new_tokens,
                            user_message=user, format_fix=(fixed != raw), attempt_log=log)
        raise RuntimeError(f"[{name}] {max_tries}번 시도했지만 사용할 수 있는 프롬프트를 얻지 못했어요")

    def rewrite_cached(self, name: str, cache_dir: str, **kwargs) -> dict:
        """rewrite()와 같지만 결과를 cache_dir/{slug}_rewrite.json에 저장하고, 있으면 불러온다."""
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"{slugify(name)}_rewrite.json")
        if os.path.exists(path):
            d = json.load(open(path, encoding="utf-8"))
            if not is_bad_prompt(d["prompt"], name):
                return d
        result = self.rewrite(name, **kwargs)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    # ------------------------------------------------------------------
    def unload(self):
        """GPU 메모리 해제 (뒤 단계에서 이미지 생성 모델을 올리기 전에 호출)"""
        del self.model, self.tok
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
