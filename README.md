# 저작권 캐릭터 생성 억제 + 사용자 의도 보존 프롬프트 재구성

캐릭터 이름 하나를 넣으면 **우회적 프롬프트 재작성 → 후보 이미지 50장 생성 → VLM 필터 → 저작권 위험도 기반 선택 → 최종 이미지**까지 한 번에 실행한다.

## 설치
```bash
pip install -r requirements.txt
huggingface-cli login      # Llama-3.1-8B-Instruct 는 접근 승인 필요
```
모델은 처음 실행할 때 Hugging Face 에서 자동으로 내려받는다 (GPU 24GB 기준, 단계마다 모델을 내려서 메모리를 아낀다).

## 실행 (프로젝트 루트에서)
```bash
python -m scripts.run_pipeline --name Mario
# -> outputs/mario/final.png, result.json, selection_scores.csv, candidates/seed42~91.png
```
`data/keywords/final/mario.json` 이 없으면 처음 실행으로 보고 생성 제외 키워드를 먼저 만든다
(LLM 후보 50개 -> Re-LAION 스캔 -> 공출현 top-5). 스캔은 오래 걸리지만 끊겨도 이어서 실행된다.

## 평가
```bash
python -m scripts.run_eval --name Mario --images outputs/mario/final.png --keyword Plumber
python -m scripts.run_eval --name Mario --images outputs/mario/candidates   # 후보 전체 평균 = Standard 기준선
```
- DETECT: InternVL3 로 검출된 비율 (낮을수록 좋음) / CONS: 핵심 특징 일치도 / p-CONS: 입력 프롬프트 일치도 (높을수록 좋음)
- `--keyword` 대신 `data/eval/key_entitie_n1.csv` (기존 CopyCat 파일 그대로) 를 넣어 두면 keyword 열을 자동으로 읽는다.

## 구성
| 파일 | 역할 |
|---|---|
| `src/rewrite.py` | Llama-3.1-8B 로 우회적 프롬프트 재작성 |
| `src/keywords.py` | LAION 공출현 키워드 top-5 -> 생성 제외 키워드 |
| `src/generate.py` | Playground v2.5, 프롬프트 고정 + seed 42~91 후보 50장 |
| `src/detect.py` | Qwen2.5-VL(필터) / InternVL3(평가) 캐릭터 검출 |
| `src/select.py` | 필터 통과 후보 중 평균 CLIP 거리 최저 이미지 선택 (전부 검출되면 전체에서 선택) |
| `src/metrics.py` | DETECT, CONS, p-CONS |
| `src/pipeline.py` | 전체 파이프라인 |
