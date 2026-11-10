"""캐릭터 이름 하나를 넣으면 최종 이미지까지 한 번에 만든다.

  캐릭터 이름 -> (1) 우회적 프롬프트 재작성 -> (2) 생성 제외 키워드(네거티브 프롬프트)
              -> (3) 후보 이미지 50장 생성 -> (4) VLM 필터 -> (5) 저작권 위험도 기반 선택 -> 최종 이미지

GPU 메모리를 아끼려고 단계마다 모델을 올렸다가 내린다.
"""
import json
import os
import shutil

from . import config
from . import keywords as kw
from .utils import slugify


def run(name: str = config.CHARACTER, out_root: str = "outputs", seeds=config.CANDIDATE_SEEDS,
        keyword_dir=None, verbose: bool = True) -> dict:
    from .rewrite import PromptRewriter
    from .generate import CandidateGenerator
    from .detect import CharacterDetector
    from .select import OriginalityScorer, select_final

    slug = slugify(name)
    out_dir = os.path.join(out_root, slug)
    cand_dir = os.path.join(out_dir, "candidates")
    os.makedirs(cand_dir, exist_ok=True)

    # (1) 프롬프트 재작성
    rw = PromptRewriter()
    rewrite = rw.rewrite_cached(name, out_dir)
    prompt = rewrite["prompt"]

    # (2) 생성 제외 키워드: data/keywords/final/{slug}.json 이 없으면 처음이라고 보고 새로 만든다
    #     (LLM 후보 50개 -> LAION 샤드 스캔 -> 공출현 top-5 재랭킹. 스캔은 오래 걸리지만 이어서 실행 가능)
    final_dir = keyword_dir or (config.KEYWORD_DIR / "final")
    if not os.path.exists(os.path.join(final_dir, f"{slug}.json")):
        cands = kw.generate_candidates(rw, name)
        rw.unload()
        kw.run_scan(name)
        kw.rank_keywords(name, candidates=cands, final_dir=final_dir)
    else:
        rw.unload()
    negative = kw.load_negative_prompt(name, final_dir=final_dir)
    if verbose:
        print(f"[prompt] {prompt}\n[negative] {negative}")

    # (3) 후보 이미지 생성 (프롬프트 고정, seed만 변경)
    gen = CandidateGenerator()
    paths = gen.generate(prompt, negative, cand_dir, seeds=seeds, verbose=verbose)
    gen.unload()

    # (4) VLM 필터
    det = CharacterDetector("qwen")
    detections = det.detect_images(paths, cache_path=os.path.join(out_dir, "detect_filter.json"),
                                   verbose=verbose)
    det.unload()

    # (5) 저작권 위험도 기반 선택
    scorer = OriginalityScorer()
    sel = select_final(paths, detections, scorer)
    scorer.unload()

    final_path = os.path.join(out_dir, "final.png")
    shutil.copy(sel["selected_path"], final_path)
    sel["table"].to_csv(os.path.join(out_dir, "selection_scores.csv"), index=False)
    summary = dict(character=name, prompt=prompt, negative_prompt=negative,
                   selected=sel["selected"], risk_score=sel["score"],
                   n_generated=len(paths), n_candidates=sel["n_candidates"],
                   fallback=sel["fallback"], final_image=final_path)
    json.dump(summary, open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    if verbose:
        print(f"[final] {final_path} (선택: {sel['selected']}, 후보 {sel['n_candidates']}/{len(paths)}장)")
    return summary
