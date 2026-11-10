"""전체 파이프라인 설정값. 실험에 쓴 값을 그대로 기본값으로 둔다."""
from pathlib import Path

# ---- 모델 ID (처음 실행할 때 Hugging Face에서 자동으로 내려받음) ----
LLM_ID = "meta-llama/Llama-3.1-8B-Instruct"          # 프롬프트 재작성 (접근 승인 필요)
T2I_ID = "playgroundai/playground-v2.5-1024px-aesthetic"  # 후보 이미지 생성

# ---- 프롬프트 재작성 (rewrite.py) ----
SEED = 42
REWRITE_TEMPERATURE = 0.7
REWRITE_MAX_NEW_TOKENS = 512
REWRITE_MAX_TRIES = 5          # 거절/이름 유출 시 seed를 바꿔 재시도하는 최대 횟수

# ---- 후보 이미지 생성 (generate.py) ----
NUM_CANDIDATES = 50
CANDIDATE_SEEDS = list(range(SEED, SEED + NUM_CANDIDATES))   # 42 ~ 91, 초기 노이즈만 달라짐
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 3
CLIP_MAX_TOKENS = 77           # 이 길이를 넘는 프롬프트는 CLIP 입력에서 잘림

# ---- 생성 제외 키워드 (keywords.py) ----

ROOT = Path(__file__).resolve().parents[1]
KEYWORD_DIR = ROOT / "data" / "keywords"          # candidates/ (LLM 후보), final/ (LAION 재랭킹 결과)
SCAN_DIR = ROOT / "work" / "laion_scan"           # LAION 스캔 중간 산출물 (용량이 크므로 배포 시 제외)

CHARACTER = "mario"                               # 기본 대상 캐릭터 (예시)
LLM_TAG = "llama31_8b"                            # 키워드 후보를 낸 LLM의 이름표
KEYWORD_CANDIDATES = 50                           # LLM에게 요청하는 후보 키워드 수
KEYWORD_MAX_NEW_TOKENS = 512
KEYWORD_TOPK = 5                                  # 최종 생성 제외 키워드 수
KEYWORD_MATCH = "word"                            # "word": 캐릭터명이 독립 단어인 캡션만 집계 / "sub": 부분문자열

LAION_REPO = "datasets/laion/relaion2B-en-research-safe"
LAION_COLS = ["caption", "similarity", "punsafe"]
SCAN_WORKERS = 1                                  # 메모리/속도 확인 후 2~3으로

# ---- 후보 이미지 필터링 (detect.py) ----
FILTER_VLM_ID = "Qwen/Qwen2.5-VL-7B-Instruct"        # 후보 이미지 검사용 (선택 과정에서 사용)
EVAL_VLM_ID = "OpenGVLab/InternVL3-8B-hf"             # DETECT 평가용 (필터와 다른 모델로 편향 방지)
DETECT_MAX_NEW_TOKENS = 300

# ---- 저작권 위험도 기반 선택 (select.py) ----
_LOCAL_CLIP = Path.home() / "models" / "clip-vit-base-patch32"   # 직접 받아 둔 폴더가 있으면 그걸 사용
CLIP_ID = str(_LOCAL_CLIP) if _LOCAL_CLIP.exists() else "openai/clip-vit-base-patch32"   # 후보 간 평균 CLIP 거리 계산용

# ---- 평가 (metrics.py) ----
VQA_MODEL = "clip-flant5-xl"                          # CONS / p-CONS (t2v_metrics VQAScore)
# t2v_metrics 의 기본 캐시는 "./hf_cache/" (실행한 폴더 기준)라서 위치를 고정해 두지 않으면 매번 다시 받는다
VQA_CACHE_DIR = str(Path.home() / ".cache" / "huggingface" / "hub")