# On-Device 작업 핸드오프 (다른 PC에서 이어서)

> 이 문서 하나로 작업 재개 가능. 브랜치 **`feat/ondevice-galaxy`**.
> 스펙 원본: 루트 `KoDialect_OnDevice_BRIEF.md`. 사용자 면접메모: `ondevice/MEMO.md`(로컬 전용, git 미포함).

## 0. 한 줄 목표
KoDialect `sft_merged` 0.5B → GGUF → **Galaxy S25 Ultra 실측**으로 BRIEF §8.3 각주("실속도는 GGUF Q4_K_M 경로")를 측정으로 닫는다. 가치 = 측정 깊이(Pareto·prefill/decode 분해·부호뒤집힘), "폰에서 돌렸다"가 아님.

## 1. 지금까지 한 일 (커밋 2개, `ondevice/` + `docs/ONDEVICE_*` 만)
- `d2f87d0` — 측정 하니스 전체(serve/demo/bench/eval/report/quantize + docs + tests).
- `cbadc46` — **validity gate**(채점 비교 성립 증명: 입력패리티+평가셋동일성+디코딩결정성+출력token동치).

상태: **빌드 완료, 순수 파이썬 경로 검증 완료. 폰 실측은 아직 안 함**(모든 수치 PREDICTED).
오프라인 테스트 `7 passed, 1 skipped`(=라이브 동치), ruff clean.

## 2. 재개 환경
- 데스크탑 파이썬 env: **uv 관리 `./.venv`** (`$(pwd)/.venv/bin/python`). 구 conda env `balaenoptera`는 2026-08-04에 제거됨.
  재생성: `uv venv --python 3.11 .venv && make install-torch && make install && make install-unsloth`.
- 폰: Termux + llama.cpp(직접 빌드). llama.cpp 는 루트에 클론돼 있을 수 있음(`llama.cpp/`, gitignore).
- CI 게이트: `ruff check` + `ruff format --check` (커밋 전 `ruff format .` 필수).

## 3. 검증된 핵심 인터페이스 (재사용처)
- prompt 템플릿 SSOT: `src/ko_dialect/data/template.py`
  - system: `당신은 한국어 방언 변환 전문가입니다.`
  - user(dia→std): `다음 {지역} 사투리를 표준어로 바꿔줘:\n{source}`
  - **수동 ChatML == `apply_chat_template` 바이트 동일** (테스트로 고정).
- 평가셋 로더: `ko_dialect.evaluation.generation.load_eval_samples(path, split, target_do, direction, n)` → `{source, reference}`. dia→std는 source=`dialect`, reference=`standard`. 데이터: `outputs/dialect_raw_new` valid, 컬럼 `do`/`dialect`/`standard`/`is_identical`, 지역 = gangwondo/gyeongsangdo.
- 메트릭: `ko_dialect.evaluation.metrics.compute_chrf(outs, refs)`, `reconstruction_bleu(outs, refs)` — 둘 다 `(list[str], list[str])`.
- export: `ko_dialect.export.{convert_to_gguf, quantize_gguf, merge_lora_and_save}` (llama.cpp 바이너리 호출). `quantize/sweep.py`가 래핑.
- Pareto: `report/pareto.py`가 leaderboard.pareto_frontier 패턴을 mixed-direction(↑↓)으로 자체 구현.

## 4. validity gate (폰 측정 전 반드시 통과 — README §검증게이트)
1. 입력 ChatML 바이트 패리티 ✅
2. 평가셋 동일성: `make_eval_prompts.py` 결정적 first-n + 지역별 SHA256 `id_hash` + `data/eval_manifest.json`. fp16 레퍼런스(`desktop_ref.py`)도 같은 셋 채점.
3. 디코딩 결정성: `eval/decoding.py` SSOT(greedy temp0/top_k1/seed0/stop) → gen_capture(폰)·desktop_ref(데스크탑) 공유. 데모는 손 안 댐(체감용).
4. 출력 token-id 동치: `eval/check_token_equivalence.py` — desktop llama.cpp vs forwarded phone llama.cpp `/tokenize`+`/completion` ids 동일 assert. env-gated 테스트로도 존재.

`score_offline.py`는 fp16 레퍼런스 대비 `ΔchrF / Δrecon_bleu`(양자화 비용)를 자동 출력.

## 5. 다음 할 일 (순서, 상세는 README 액션플랜)
0. (데스크탑) `quantize/sweep.py`로 variant 생성 → `make_eval_prompts.py`(id_hash 기록) → `desktop_ref.py` + `score_offline.py --variant desktop_fp16`(fp16 기준선).
1. (폰) `serve/launch_server.sh`로 시동 확인(★).
2. (노트북) `adb forward` → `demo/index.html` 복붙→tok/s 녹화.
3. (폰) `bench/run_bench.sh` variant 스윕 + `mem_logger.sh` → `bench/logger.py`로 §6 JSONL. (노트북) `gen_capture.py`+`score_offline.py` 품질.
4. `report/pareto.py` + `plots.py` → `docs/ONDEVICE_RESULTS.md` 채우고 PREDICTED→MEASURED.

## 6. 미해결 / 주의
- **llama.cpp 응답 키 미확정**: `/tokenize`·`/completion`의 `tokens`가 build에 따라 `int` 또는 `{"id":..}`. gen_capture/check_token_equivalence는 둘 다 파싱하지만, **실제 폰 빌드 응답 1회 확인 후 키 확정** 필요.
- **부호뒤집힘 가설(PREDICTED)**: 모바일 ARM에서 Q4 ≥ Q8/fp16 decode 예상(memory-bandwidth bound). 2060에선 NF4 0.5×였음. 맞든 틀리든 MEASURED로 기록.
- **KV-quant silent fallback**(MEMO.md 면접포인트): `-fa` 필수 + K/V 타입 대칭. 비대칭이면 경고 없이 느린 non-fused 경로. `launch_server.sh`가 사전 차단 중 → DECISIONS.md에 카드화하면 좋음(아직 별도 카드 미작성).
- Tier2(KV cache 관리 실측)는 후속 — 미수행분은 "예측+근거"로만.

## 7. 재개 첫 명령
```bash
git checkout feat/ondevice-galaxy
PY=.venv/bin/python
$PY -m pytest ondevice/tests/test_ondevice.py -q     # 7 passed, 1 skipped 확인
cat ondevice/README.md                                # 액션플랜
```
