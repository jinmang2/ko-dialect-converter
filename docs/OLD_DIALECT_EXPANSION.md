# old_dialect 활용 · 5개 도 확장 · 음성 도입 검토

세 갈래(코드 정리 / old 데이터 / 음성)를 톺아본 결과와 다음 단계 런북.

---

## 1. 무엇이 고쳐졌나 — old_dialect 전사 정제(복원된 로직)

`scripts/prepare_data.py`는 old 포맷의 **구조 파싱**(`_process_old_single_json`,
`REGION_MAP`로 5개 도 매핑)은 이미 갖고 있었지만, 옛 노트북
(`notebooks/260531_preprocessing.ipynb`)에 있던 **텍스트 정제 단계가 누락**돼 있었다.
그래서 AI-Hub 주석 태그가 학습쌍에 그대로 새고 있었다.

### AI-Hub 한국어 방언 발화 주석 규약과 처리 규칙
(전라도/Training 300파일·80,003발화 스캔 기준, 빈도순)

| 태그 | 의미 | 처리 |
|---|---|---|
| `(방언)/(표준)` 잔존 `)/(` | 미해소 이중 전사 | std는 둘째·dia는 첫째 토큰 채택 |
| `(( ))` | 청취 불가 | 마커 제거 |
| `( ... )` | 단일 괄호 불확실 주석 | 제거 |
| `{ ... }` | 비언어음 `{laughing}` | 제거 |
| `& ... &` | 개인정보 익명화 `&company-name&` | 토큰 제거(걸린 조사 잔존은 허용 — 양쪽 동일) |
| `#word` | 방언 감탄사 마커 `#오메` | `#`만 제거, 단어 유지 |
| `~` (아~) | 발화 늘임 | **유지**(운율, 양쪽 동일) |
| 고아 `(`/`)` | 이중전사가 발화 경계를 가로지른 잔여물 | 짝 없는 괄호문자 제거 |

구현: `clean_old_transcript(text, side)` (`scripts/prepare_data.py`).
`standard_form`/`dialect_form`과 `eojeolList` 텍스트 양쪽에 적용, 정제 후 빈
발화는 드롭. 신규(139-1) 코퍼스는 이미 깨끗해 **old 경로에만** 적용된다.

### 검증 (5개 도, 각 200파일)
| 도 | 발화수 | 정제후 태그잔여 | 드롭 |
|---|---|---|---|
| 강원도 | 65,131 | 0 | 36 |
| 경상도 | 56,533 | 0 | 10 |
| 전라도 | 53,283 | 0 | 127 |
| 제주도 | 110,092 | 0 | 1,712 |
| 충청도 | 63,588 | 0 | 51 |

태그 잔여율 6.4% → 0.0%. 제주 드롭이 많은 건 가장 이질적인 방언이라 청취불가 주석이
많은 자연스러운 결과. 테스트: `tests/test_prepare_data_old_cleaning.py` (12개, 실데이터 예시 기반).

> 스키마 메모: old의 `dialect_eojeol_map`은 `idx` 키, new는 `standard_idx`/`dialect_idx`.
> 다운스트림 소비자(`rewards/edit.py`, `evaluation/metrics.py`)는 `standard`/`dialect`
> 텍스트만 읽으므로 **불일치는 무해**(수정 불필요).

---

## 1b. 라벨 파이프라인 5개 도 지원 (완료된 코드 변경)

이전엔 라벨 맵이 **세 곳에 복붙**돼 있었고(`data/dataset.py`의 `DIALECT_LABELS`,
`rewards/style.py`·`evaluation/metrics.py`의 `DO_TO_LABEL`) 강원/경상 2개 도만 알았다.
그래서 5개 도 raw를 만들어도 stage0가 나머지 3개 도를 **조용히 필터링**해 버렸다
(`num_labels=3` 하드코딩).

변경:
- **`src/ko_dialect/data/labels.py`** 신설 — 단일 진실원천(SSOT). `DIALECT_LABELS`
  (standard=0, gangwon=1, gyeongsang=2, **jeolla=3, jeju=4, chungcheong=5**),
  `DO_TO_LABEL`, `SUPPORTED_DO`, `num_labels_for()`. id는 **append-only**(기존 체크포인트
  정렬 보존).
- 라벨 맵은 원래 **4곳**에 복붙돼 있었다: `dataset.py`(DIALECT_LABELS), `style.py`·
  `metrics.py`(DO_TO_LABEL), **`models/classifier.py`(LABEL2ID/ID2LABEL)**. 넷 다 이제
  `labels.py`에서 파생(중복 제거 — Track 1 정리 겸함).
- `stage2_train_classifier.py`: `num_labels = num_labels_for(ds["train"]["label"])`로
  **데이터 주도**. 강원/경상만이면 3-class(기존 그대로), 5개 도면 6-class. → 기존 3-class
  체크포인트 하위호환.
- **`classifier.py` 미묘 버그 수정**: `TextCNNConfig`가 num_labels=6인데 id2label은
  3-entry 기본값을 받으면 transformers가 `len(id2label)`에서 num_labels=3을 역산해
  헤드를 **조용히 3으로 만들어버림**. 이제 id2label/label2id를 num_labels에 맞춰
  슬라이스 → 항상 일치. 회귀 테스트 추가(`test_textcnn_*_six_class_*`).
- 검증: 실데이터에서 6-class 분류기 데이터셋 빌드 확인(라벨 {0..5}, `num_labels_for`→6),
  `TextCNNConfig(num_labels=6)` 헤드 (B,6) 확인. 테스트 `test_data_labels.py`(6) +
  `test_models_classifier.py`(8). 전체 **229개 통과**, `ruff check .` clean.

## 2. 5개 도로 확장하는 런북

### (1) old 5개 도 raw 데이터 — ✅ 생성 완료 (CPU)
```bash
python scripts/prepare_data.py --data_path raw_data/old_dialect \
    --use_old_format True --output_dir outputs --verbose True
```
산출: `outputs/dialect_raw_old` (+manifest, source_format="old").
**실측**: train 10,230,204 / valid 1,337,267, 5개 도 전부, 태그잔여 0.
도별 train — 제주 2.73M, 경상 2.09M, 전라 1.99M, 충청 1.85M, 강원 1.57M.

### (2)~(4) GPU/대용량 — 사용자 머신에서 실행
```bash
# (2) stage0: SFT/GRPO/분류기 데이터셋 (region-agnostic, 이제 5개 도 통과)
python scripts/stage0_build_datasets.py \
    --raw_dataset_path outputs/dialect_raw_old \
    --output_dir outputs/datasets_old
#   old는 prosody 없음 → prosody_mode 기본(none) 유지.
#   ⚠️ 11.5M rows → SFT 양방향 23M. cls_standard_cap_ratio / cls_max_per_label로
#      classifier 클래스 불균형(standard 압도) 조정 권장.

# (3) 분류기 재학습 — 6-class 자동. (기존 3-class 체크포인트와 호환 안 됨: 새 모델)
python scripts/stage2_train_classifier.py

# (4) 평가 — eval_leaderboard가 신규 도를 자동 포함(_resolve_regions가 data-driven)
python scripts/eval_leaderboard.py --all_regions --n 150
```

신규 도 학습 시 점검:
- **데이터 품질/불균형**: old는 자유발화라 near-standard 행이 많아
  `cls_min_norm_levenshtein`(0.1)/`filter_identical` 통과 후 도별 dialect 표본이 적다.
  `cls_standard_cap_ratio`·`class_weighting`으로 보정.
- **혼합 학습**(강원/경상 new + 5도 old) 시 도메인 시프트 주의 — 도별 leaderboard로 회귀 확인.

---

## 3. 코드/스크립트 정리 — 발견 (별도 리뷰 권장)

- **평가 진입점 중복**: `scripts/eval.py`가 이미 `single/leaderboard/prosody_ab/checkpoints/config`
  서브커맨드로 통합 layer 역할. 그런데 `evaluate.py`(단일), `eval_leaderboard.py`,
  `report.py`, `eval_grpo_checkpoints.py`, `eval_prosody_ab.py`가 독립 진입점으로 공존.
  → 문서를 `eval.py` 한 front door로 좁히면 표면 감소(eval.py가 이미 이들 main을 forward).
- **"미참조" 7개 ≠ 死코드**: `wandb_doctor`(→ `.claude/skills/wandb-doctor`),
  `derive_prosody_markers`/`build_sft_pair`(prosody A/B), `inspect_classifier`,
  `ptq_quantize`, `serve_sft_vllm`, `grpo_run_report`는 스킬/메모리/워크플로에서 쓰임.
  **삭제 금지**, README/Makefile 참조만 보강 권장.
- 이 정리는 eval 표면을 바꾸는 다중파일 리팩토링이라, 정제 작업과 분리해 별도
  `/code-review` → 리팩토링 PR로 진행하는 게 안전.

---

## 4. 음성 모델 도입 — 검토 (설계 단계)

- **현황**: `raw_data`에 오디오 0개. old_dialect는 JSON+txt(전사)만, new도 JSON만.
  즉 음성 도입은 **AI-Hub 원천 오디오 다운로드가 선행**(대용량, 6GB RTX2060엔 부담).
- 현재 prosody는 오디오 없이 JSON의 F0/intonation 메타에서 마커를 뽑는 방식(`data/prosody.py`).
- 의미 있는 갈래:
  - **(a) STT front-end**: 방언 음성 → ASR(Whisper 계열 한국어/방언 파인튜닝) → 기존 표준화 모델.
    파이프라인 확장. ASR 방언 견고성이 관건.
  - **(b) 실제 오디오 prosody**: 현 JSON 추정 마커를 실제 F0 추출로 대체/검증.
    `IDEAS.md`의 운율 마커 토큰화와 직접 연결.
  - **(c) TTS back-end**: 표준→방언 변환 후 방언 합성. 별개 큰 프로젝트.
- **권장**: 텍스트 5개 도 확장(2장)을 먼저 끝내 커버리지·품질을 확보한 뒤,
  음성은 (b) 소규모 PoC(한 도, 소량 오디오로 JSON-추정 vs 실측 F0 일치도 측정)부터.
