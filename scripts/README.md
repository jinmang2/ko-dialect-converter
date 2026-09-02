# scripts/ — 무엇을 언제 실행하나

35개 스크립트가 있어서 이름만 봐서는 순서를 알 수 없다. **파이프라인 순서대로** 정리한 지도다.
각 줄의 `→`는 산출물. 전체 설계 맥락은 [`../docs/DESIGN_DECISIONS.md`](../docs/DESIGN_DECISIONS.md),
5지역 확장 절차는 [`../docs/DATA_EXPANSION_RUNBOOK.md`](../docs/DATA_EXPANSION_RUNBOOK.md).

> **CLI 스타일이 두 가지다.** `stage1/2/3`은 **Hydra**(`key=value`, `--` 없음),
> 나머지는 **fire**(`--key value`). 헷갈리면 이 표의 예시를 그대로 쓸 것.

---

## 0. 데이터 확보

| 스크립트 | 용도 |
|---|---|
| **`aihub_fetch.py`** | AI-Hub 조회·계획·다운로드·압축해제·대조. **여기서 시작.** 음성(수십~150GB) 오다운로드 가드 내장 |
| `download_from_aihub.sh` | (구) 라벨 전체 일괄 다운로드 래퍼. 용량 표시·가드가 없어 `aihub_fetch.py` 권장 |
| `unzip.sh` | (구) 아카이브 압축 해제. `aihub_fetch.py extract`가 멱등·인코딩 교정까지 한다 |

```bash
python scripts/aihub_fetch.py plan --datasetkey 71558          # 라벨 vs 음성, 용량, filekey
python scripts/aihub_fetch.py fetch --version v2_2022          # 한 세대 라벨 전량 + 압축해제
python scripts/aihub_fetch.py extract --version v1_2020        # 압축해제만 (멱등)
python scripts/aihub_fetch.py diff_trees --a <old> --b <new>   # 재다운로드본이 같은지 sha256 대조
```

**목적지를 손으로 타이핑하지 말 것.** `src/ko_dialect/corpora.py`(7개 데이터셋 레지스트리)가
`데이터셋 → 버전 → 디렉토리`를 결정한다: `data/aihub/v1_2020`(구포맷) / `data/aihub/v2_2022`(신포맷).
`prepare_data.py`는 **트리 단위로** 파서를 고르므로, 두 포맷이 한 트리에 섞이면 잘못된 파서가
`do="unknown"`을 만들고 stage0이 조용히 전량 폐기한다(에러 없음).

## 1. 전처리 → raw Arrow

| 스크립트 | 용도 |
|---|---|
| **`prepare_data.py`** | 라벨 JSON → `outputs/dialect_raw_{new,old}`. old 코퍼스는 `--use_old_format` 필수 |
| `resplit_speaker_disjoint.py` | **화자 분리 재분할**(opt-in). AI-Hub 기본 split은 valid 화자의 18.1%가 train에도 있다 → `_spk` 데이터셋 별도 생성 |
| `derive_prosody_markers.py` | 기존 raw에 `prosody_marker` 컬럼만 추가(재빌드 없이) |

## 2. 게이트 — 빌드 직후 반드시

| 스크립트 | 무엇을 잡나 |
|---|---|
| **`audit_raw_signal.py`** | 어절 신호 / 길이 분포 / ≤5어절 비중 / F0 커버리지. 지역이 `unknown`이면 stage0이 조용히 전량 폐기하므로 여기서 걸러야 함 |
| **`audit_corpus_fields.py`** | `fields` = 미사용 필드(stt·grammarType·intent·domain·speaker) 전수 + ASR 후처리 가능성 / `leakage` = 화자 누출 |
| `audit_ger_errors.py` | ASR이 무엇을 틀리는지 유형 분류(어미·어휘·구 재작성·숫자·띄어쓰기). GER이 0.5B LM이 풀 수 있는 태스크인지 판정 |
| `audit_lexical_gap.py` | 지역별 어휘 집중도(TTR·상위N 커버). **사전/룩업으로 될 지역인지** 판정. 지역당 어절 예산을 통일해야 결론이 표본에 오염되지 않음 |

```bash
python scripts/audit_raw_signal.py --raw_dataset_path outputs/dialect_raw_new
python scripts/audit_corpus_fields.py leakage --raw_dataset_path outputs/dialect_raw_new
```

## 3. 학습 파이프라인

| 단계 | 스크립트 | CLI | 예시 |
|---|---|---|---|
| Stage 0 | `stage0_build_datasets.py` | fire | `--raw_dataset_path outputs/dialect_raw_new --prosody_mode eojeol` |
| Stage 1 | `stage1_sft.py` | **Hydra** | `data.sft_dataset_path=outputs/datasets/sft` |
| Stage 2 | `stage2_train_classifier.py` | **Hydra** | `data.cls_dataset_path=outputs/datasets/classifier` |
| Stage 3 | `stage3_grpo.py` | **Hydra** | 직접 실행 대신 **`run_grpo.sh`** 사용 (2060용 bnb/env 픽스 포함) |
| 보조 | `smoke_grpo.py` | fire | 5스텝 스모크. GRPO가 터질 때 먼저 이걸로 |
| 보조 | `build_sft_pair.py` | fire | 프로소디 A/B용 대조 데이터셋 쌍 |
| 보조 | `merge_sft_lora.py` | fire | LoRA 병합 → 독립 모델 |

```bash
scripts/run_grpo.sh training.max_steps=200 logger=wandb    # ← Stage 3 진입점
SMOKE_4BIT=0 SMOKE_N=64 SMOKE_STEPS=5 python scripts/smoke_grpo.py
```

## 4. 평가

| 스크립트 | 언제 |
|---|---|
| **`eval.py`** | 통합 진입점(config-driven). `config` / `single` / `leaderboard` 서브커맨드 |
| **`eval_leaderboard.py`** | 크로스런 랭킹 — per-region + OVERALL, Pareto, Koehn 유의성. ⚠️ **std2dia 전용** — `direction == "std2dia"`로 필터한다 |
| **`eval_dia2std.py`** | **배포 방향(방언→표준어) 평가.** 온디바이스 태스크가 dia2std이므로 제품 성능은 여기서 본다. 헤드라인은 `gain_over_copy`(복사 베이스라인 대비 순기여) — raw chrF는 지역별 바닥이 18.9~79.9로 달라 지역 간 비교가 성립하지 않음 |
| `evaluate.py` | 단일 모델 TDR/DFS/어절정확도. `eval.py single`의 백엔드 |
| `compare_sft_grpo.py` | SFT vs 특정 GRPO 1건 심층 비교(계층 버킷 + 정성 샘플) |
| `eval_grpo_checkpoints.py` | 한 런 **내부** 체크포인트 스윕 — 과최적화 지점 찾기 |
| `eval_prosody_ab.py` | 프로소디 A/B 전용 하니스 |
| `inspect_classifier.py` | stage2 분류기 정성/정량 점검 |
| `grpo_run_report.py` | 완료된 GRPO 런들의 trainer-state 건강도 |
| `report.py` | `outputs/eval_logs/*.json` → `docs/RESULTS.md` 재생성 |

## 5. 서빙 · 양자화 · 배포

| 스크립트 | 용도 |
|---|---|
| **`serve_compare.py`** | 베이스 1개만 VRAM에 올리고 어댑터 핫스왑 — SFT/여러 GRPO arm 동시 비교(6GB 안전) |
| `serve_sft_vllm.py` | vLLM 오프라인 배치 정성 확인 |
| `bench_serving.py` | 지연/처리량 p50·p95 + 품질 |
| `bench_train.py` | 학습 처리량 unsloth vs DeepSpeed |
| `quantize_eval.py` | 양자화 trade-off(품질×크기×지연) |
| `ptq_quantize.py` | AWQ/GPTQ PTQ |
| `export_gguf.py` | GGUF 변환·양자화·서빙(llama.cpp) |
| `push_to_hub.py` | 증거 포함 모델 카드로 HF Hub 배포 |

## 6. 진단

| 스크립트 | 용도 |
|---|---|
| `wandb_doctor.py` | W&B에 metric이 안 올라올 때 (`.claude/skills/wandb-doctor`) |

---

## 자주 밟는 함정

- **Hydra/fire 혼동** — stage1/2/3만 `key=value`. 나머지는 `--key value`.
- **old 포맷에 `--use_old_format`을 빼먹음** → `do="unknown"` → stage0에서 조용히 전량 폐기.
  §2 게이트가 이걸 잡는다.
- **GRPO를 `stage3_grpo.py`로 직접 실행** → bitsandbytes `libnvJitLink.so.13` 로드 실패.
  `run_grpo.sh`가 `LD_LIBRARY_PATH`를 세팅한다.
- **커밋 전 `ruff format .`** — CI가 `ruff check` + `ruff format --check` 둘 다 본다.
