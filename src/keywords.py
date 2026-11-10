"""생성 제외 키워드(네거티브 프롬프트) 구성.

캐릭터와 LAION 캡션에서 자주 함께 나오는 키워드 상위 5개를 뽑아, 캐릭터명과 함께
생성 제외 키워드로 쓴다. 세 단계로 이뤄진다.

  1) generate_candidates : LLM이 캐릭터와 연관된 키워드 후보 50개를 낸다
  2) run_scan            : Re-LAION-2B-en 캡션 중 캐릭터명이 든 캡션을 샤드별로 모은다 (오래 걸림)
  3) rank_keywords       : 모아 둔 캡션에서 후보 키워드와 캐릭터명의 동시 출현 빈도를 세어
                           상위 5개를 뽑고 data/keywords/final/{slug}.json 에 저장한다

캐릭터 하나(기본값 config.CHARACTER = "mario")를 기준으로 동작한다. 키워드 후보를 바꾸거나
다른 LLM을 써도 2)를 다시 돌릴 필요 없이 3)만 다시 하면 된다.

이미 final/{slug}.json 이 있으면 load_negative_prompt() 로 바로 불러 쓰면 되고,
1)~3)은 새 캐릭터의 키워드를 처음 만들 때만 필요하다.
"""
import collections
import glob
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from . import config
from .prompts import KEYWORD_TMPL
from .utils import build_negative_prompt, has_name, slugify


# ======================================================================
# 1) LLM 키워드 후보
# ======================================================================
def parse_keywords(raw: str) -> list:
    """'a, b, c' 또는 줄바꿈/번호 형식의 LLM 출력을 키워드 리스트로"""
    out = []
    for p in re.split(r",|\n", raw):
        k = re.sub(r"^\s*\d+[\.\)]\s*", "", p).strip().strip(" .\"'").lower()
        if k and not k.endswith(":"):
            out.append(k)
    return out


def filter_name_kws(kws: list, name: str) -> list:
    """중복과 캐릭터 이름이 든 키워드를 제거"""
    return [k for k in dict.fromkeys(kws) if not has_name(k, name)]


def generate_candidates(rewriter, name: str = config.CHARACTER, out_dir=None) -> list:
    """LLM(PromptRewriter)에게 캐릭터 연관 키워드 후보를 요청한다 (greedy 디코딩).

    out_dir를 주면 {slug}.json 으로 저장하고, 이미 있으면 불러온다.
    """
    out_dir = out_dir or (config.KEYWORD_DIR / "candidates")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{slugify(name)}.json")
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))["keywords"]

    raw = rewriter.chat("You are a helpful assistant.",
                        KEYWORD_TMPL.format(name=name),
                        max_new_tokens=config.KEYWORD_MAX_NEW_TOKENS, do_sample=False)
    if name in raw:                                   # He et al. 공식 코드와 같은 후처리
        raw = raw.replace(name, "").replace(", ,", ",").lstrip(" ,")
    kws = filter_name_kws(parse_keywords(raw), name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(name=name, llm=config.LLM_TAG, raw=raw, keywords=kws),
                  f, ensure_ascii=False, indent=2)
    return kws


# ======================================================================
# 2) LAION 캡션 스캔
# ======================================================================
def _get_fs(path: str):
    if path.startswith("datasets/"):                  # HF 경로일 때만 HfFileSystem 사용
        from huggingface_hub import HfFileSystem
        return HfFileSystem()
    return None                                       # 로컬 파일


def hits_dir(name: str = config.CHARACTER, scan_dir=config.SCAN_DIR) -> str:
    return os.path.join(scan_dir, "hits", slugify(name))


def meta_dir(name: str = config.CHARACTER, scan_dir=config.SCAN_DIR) -> str:
    return os.path.join(scan_dir, "meta", slugify(name))


def scan_shard(shard: str, name: str = config.CHARACTER, scan_dir=config.SCAN_DIR):
    """샤드 하나를 읽어 캐릭터명이 든 캡션을 hits/{slug}/{shard}.tsv 에 저장한다.

    meta/{slug}/{shard}.json 이 있으면 이미 끝난 샤드로 보고 건너뛴다. meta는 맨 마지막에
    쓰므로, 중간에 끊겨도 다시 실행하면 그 샤드부터 이어서 한다.
    반환: (샤드 파일명, 걸린 시간, 메시지)
    """
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    fn = os.path.basename(shard)
    mdir = meta_dir(name, scan_dir)
    meta_path = os.path.join(mdir, f"{fn}.json")
    if os.path.exists(meta_path):                      # 이미 끝난 샤드
        return fn, 0.0, "skip"

    t = time.time()
    tbl = None
    for attempt in range(3):
        try:
            tbl = pq.read_table(shard, columns=config.LAION_COLS, filesystem=_get_fs(shard))
            break
        except Exception as e:
            print(f"retry {fn} ({attempt + 1}/3): {e}", flush=True)
            time.sleep(10)
    if tbl is None:
        return fn, time.time() - t, "FAILED"

    key = name.lower()
    cap_col = config.LAION_COLS[0]
    cap = pc.fill_null(tbl.column(cap_col), "")
    sub = tbl.filter(pc.match_substring(cap, key, ignore_case=True))   # 부분문자열 기준 (상위집합)
    caps = sub.column(cap_col).to_pylist()
    sims = sub.column("similarity").to_pylist()
    puns = sub.column("punsafe").to_pylist()

    name_re = re.compile(rf"\b{re.escape(key)}\b", re.I)
    n_word = 0
    hdir = hits_dir(name, scan_dir)
    os.makedirs(hdir, exist_ok=True)
    os.makedirs(mdir, exist_ok=True)
    hp = os.path.join(hdir, f"{fn}.tsv")
    with open(hp + ".tmp", "w", encoding="utf-8") as f:
        for c, sm, pu in zip(caps, sims, puns):
            c = (c or "").replace("\t", " ").replace("\r", " ").replace("\n", " ")
            n_word += bool(name_re.search(c))
            f.write(f"{sm}\t{pu}\t{c}\n")
    os.replace(hp + ".tmp", hp)

    meta = {"shard": fn, "n_rows": tbl.num_rows, "n_sub": len(caps), "n_word": n_word}
    with open(meta_path + ".tmp", "w") as f:
        json.dump(meta, f)
    os.replace(meta_path + ".tmp", meta_path)          # meta가 있으면 그 샤드는 완료
    return fn, time.time() - t, f"rows {tbl.num_rows:,} | {key}(word) {n_word:,} | {key}(sub) {len(caps):,}"


def list_shards(repo: str = config.LAION_REPO) -> list:
    from huggingface_hub import HfFileSystem
    return sorted(HfFileSystem().glob(f"{repo}/**/*.parquet"))


def run_scan(name: str = config.CHARACTER, limit: int = None, shards: list = None,
             workers: int = config.SCAN_WORKERS, scan_dir=config.SCAN_DIR):
    """모든 샤드에서 캐릭터명이 든 캡션을 모은다. limit=3 이면 샤드 3개만 시험한다."""
    shards = shards if shards is not None else list_shards()
    if limit:
        shards = shards[:limit]
    print(f"{len(shards)} shards, workers={workers}, character={name}", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(scan_shard, s, name, scan_dir) for s in shards]
        for fu in as_completed(futs):
            fn, sec, msg = fu.result()
            done += 1
            print(f"[{done}/{len(shards)}] {fn[:12]} {sec:.0f}s {msg}", flush=True)


# ======================================================================
# 3) 재랭킹 -> 최종 생성 제외 키워드
# ======================================================================
def _clean_candidates(raw_kws: list, name: str) -> list:
    seen, out = set(), []
    for k in raw_kws:
        k = re.sub(r"^\s*\d+[\.\)]\s*", "", str(k)).strip().lower().strip(" .\"'")
        if not k or k in seen or has_name(k, name):    # 캐릭터명이 든 키워드는 제외
            continue
        seen.add(k)
        out.append(k)
    return out


def rank_keywords(name: str = config.CHARACTER, candidates: list = None, mode: str = config.KEYWORD_MATCH,
                  topk: int = config.KEYWORD_TOPK, scan_dir=config.SCAN_DIR,
                  final_dir=None, llm: str = config.LLM_TAG, verbose: bool = True) -> dict:
    """스캔해 둔 캡션에서 후보 키워드가 캐릭터명과 함께 나온 횟수를 세어 상위 topk를 고른다.

    candidates를 안 주면 data/keywords/candidates/{slug}.json 에서 읽는다.
    결과는 final_dir/{slug}.json 에 저장하고 dict로도 돌려준다.
    """
    final_dir = final_dir or (config.KEYWORD_DIR / "final")
    if candidates is None:
        cpath = os.path.join(config.KEYWORD_DIR, "candidates", f"{slugify(name)}.json")
        candidates = json.load(open(cpath, encoding="utf-8"))["keywords"]
    kws = _clean_candidates(candidates, name)
    pats = {k: re.compile(rf"\b{re.escape(k)}\b", re.I) for k in kws}
    name_re = re.compile(rf"\b{re.escape(name.lower())}\b", re.I)

    files = sorted(glob.glob(os.path.join(hits_dir(name, scan_dir), "*.tsv")))
    if not files:
        raise FileNotFoundError(f"{name}: 스캔 결과가 없어요. run_scan()을 먼저 실행하세요")
    co, n = collections.Counter(), 0
    for p in files:
        with open(p, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t", 2)
                if len(parts) < 3:
                    continue
                c = parts[2]
                if mode == "word" and not name_re.search(c):
                    continue                           # word 모드: 캐릭터명이 독립 단어인 캡션만
                n += 1
                for k, pat in pats.items():
                    if pat.search(c):
                        co[k] += 1

    ranked = sorted(kws, key=lambda k: -co[k])         # 동률은 LLM이 낸 순서 유지
    top = ranked[:topk]
    result = {"name": name, "llm": llm, "match": mode,
              "shards_used": len(files), "n_captions": n,
              "counts": {k: co[k] for k in ranked},
              "top_keywords": top,
              "negative_prompt": build_negative_prompt(name.lower(), top),   # scan_mario.py와 같이 소문자 이름
              "corpus": config.LAION_REPO}
    os.makedirs(final_dir, exist_ok=True)
    out = os.path.join(final_dir, f"{slugify(name)}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    if verbose:
        print(f"[{name} | {llm} | {mode}] 샤드 {len(files)}개, 캡션 {n:,}개, 후보 {len(kws)}개")
        for i, k in enumerate(ranked[:topk + 5], 1):
            print(f"{i:>3}. {k:28s} {co[k]:>8,}")
        print(f"negative prompt: {result['negative_prompt']}\nsaved: {out}")
    return result


def load_negative_prompt(name: str = config.CHARACTER, final_dir=None) -> str:
    """저장된 최종 생성 제외 키워드 문자열을 불러온다 ('캐릭터명, 키워드1, ..., 키워드5')."""
    final_dir = final_dir or (config.KEYWORD_DIR / "final")
    path = os.path.join(final_dir, f"{slugify(name)}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} 가 없어요. keywords.generate_candidates / run_scan / rank_keywords 로 먼저 만들어야 해요")
    d = json.load(open(path, encoding="utf-8"))
    return d.get("negative_prompt") or build_negative_prompt(name, d["top_keywords"])
