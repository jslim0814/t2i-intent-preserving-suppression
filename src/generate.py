"""2단계: 재구성한 프롬프트를 고정하고 초기 노이즈(seed)만 바꿔 후보 이미지를 생성한다.

프롬프트와 생성 제외 키워드(네거티브 프롬프트)는 모든 후보에서 동일하고,
seed만 달라지므로 후보들이 모두 사용자의 원래 요청 의도를 유지한다.

사용 예:
    gen = CandidateGenerator()
    paths = gen.generate(prompt, negative, out_dir="outputs/mario/candidates")
    gen.unload()
"""
import gc
import json
import os
import time
import traceback

import torch
from diffusers import DiffusionPipeline

from . import config


class CandidateGenerator:
    def __init__(self, model_id: str = config.T2I_ID, device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self.pipe = DiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, variant="fp16"
        ).to(device)
        self.pipe.set_progress_bar_config(disable=True)

    def n_tokens(self, text: str):
        """CLIP 토크나이저 기준 토큰 수 (77 초과 시 뒷부분이 잘림)"""
        try:
            return len(self.pipe.tokenizer(text).input_ids)
        except Exception:
            return None

    def generate(self, prompt: str, negative_prompt: str, out_dir: str,
                 seeds=config.CANDIDATE_SEEDS,
                 steps: int = config.NUM_INFERENCE_STEPS,
                 guidance_scale: float = config.GUIDANCE_SCALE,
                 verbose: bool = True) -> list:
        """seed마다 이미지를 1장씩 생성해 out_dir/seed{seed}.png 로 저장한다.

        - 이미 있는 파일은 건너뛴다 (중단 후 다시 실행하면 이어서 생성)
        - 한 장이 실패해도 나머지는 계속 생성한다
        - out_dir/meta.json 에 프롬프트, 네거티브 프롬프트, 설정, 파일 목록을 저장한다
        반환: 생성(또는 이미 존재)된 이미지 경로 리스트
        """
        os.makedirs(out_dir, exist_ok=True)
        nt = self.n_tokens(prompt)
        if nt is not None and nt > config.CLIP_MAX_TOKENS and verbose:
            print(f"[경고] 프롬프트가 {nt} 토큰이라 CLIP 입력({config.CLIP_MAX_TOKENS})을 넘어 뒷부분이 잘립니다")

        paths, images_meta = [], []
        for j, seed in enumerate(seeds, 1):
            fname = f"seed{seed}.png"
            path = os.path.join(out_dir, fname)
            if not os.path.exists(path):
                t0 = time.time()
                try:
                    gen = torch.Generator(self.device).manual_seed(seed)
                    kwargs = dict(prompt=prompt, num_inference_steps=steps,
                                  guidance_scale=guidance_scale, generator=gen)
                    if negative_prompt:
                        kwargs["negative_prompt"] = negative_prompt
                    img = self.pipe(**kwargs).images[0]
                    img.save(path + ".tmp", format="PNG")
                    os.replace(path + ".tmp", path)      # 저장 도중 끊겨도 깨진 파일이 남지 않게
                    del img
                    if verbose:
                        print(f"  candidate {j}/{len(seeds)} seed{seed} ({time.time() - t0:.1f}s)", flush=True)
                except Exception:
                    print(f"!!! seed{seed} 생성 실패\n{traceback.format_exc()}", flush=True)
                    gc.collect()
                    continue
                finally:
                    torch.cuda.empty_cache()
            paths.append(path)
            images_meta.append(dict(file=fname, seed=seed))

        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(dict(model_id=self.model_id, prompt=prompt, negative_prompt=negative_prompt,
                           n_clip_tokens=nt, truncated=(nt is not None and nt > config.CLIP_MAX_TOKENS),
                           steps=steps, guidance_scale=guidance_scale,
                           expected=len(list(seeds)), done=len(paths), images=images_meta),
                      f, ensure_ascii=False, indent=2)
        return paths

    def unload(self):
        """GPU 메모리 해제"""
        del self.pipe
        gc.collect()
        torch.cuda.empty_cache()
