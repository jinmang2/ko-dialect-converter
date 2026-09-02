# DATA_ANALYSIS.md — 코퍼스 실측 + 평가 타당성 감사

이 문서는 추정이 아니라 **실측**이다. 모든 수치는 아래 명령으로 재현된다.

```bash
uv run python scripts/analyze_data.py all          # 전 섹션
uv run python scripts/analyze_data.py regions      # §1
uv run python scripts/analyze_data.py columns      # §2
uv run python scripts/analyze_data.py edits --do gangwondo   # §3
uv run python scripts/analyze_data.py evalset --n 150        # §4
```

측정 시점 데이터: `outputs/dialect_raw_new` (원본), `outputs/datasets/grpo` (리더보드 입력).

---

## 1. 코퍼스 구성 — 라벨은 5개 도, 데이터는 2개 도

| 계층 | 강원 | 경상 | 전라 | 제주 | 충청 |
|---|---|---|---|---|---|
| `labels.py` 라벨 id | 1 | 2 | 3 | 4 | 5 |
| `prepare_data.py` 지역 매핑 | ✅ | ✅ | ✅ | ✅ | ✅ |
| `raw_data/` 원본 | ✅ | ✅ | ❌ | ❌ | ❌ |
| 빌드된 데이터셋 | ✅ | ✅ | 0행 | 0행 | 0행 |

`raw_data/`에 있는 것은 **139-1(중·노년층 한국어 방언, 강원도·경상도) 20GB 하나**뿐이다.
전라/제주/충청은 라벨 id만 예약돼 있다 (append-only 규칙이라 예약 자체는 안전하다).

**주의 — 경로 불일치.** `prepare_data.py`의 기본 `data_path`는 `raw_data/new_dialect`,
old 경로는 `raw_data/old_dialect`인데 디스크에는 `raw_data/korean_dialect/139-1…`만 있다.
기본값으로 실행하면 즉시 실패한다. 정제 규칙 주석이 "300-file scan of 전라도/Training"에서
도출됐다고 적혀 있으므로 전라도 원본은 다른 작업 PC에 있을 가능성이 크다.

### `is_identical` — 절반 가까이가 학습 신호가 없다

방언 문장이 표준어와 **글자까지 동일한** 행이다. 전이(transfer)에 대해 아무것도 가르쳐
주지 않으므로 모든 학습/평가 경로가 이 행을 버린다. 버리는 양이 오차 수준이 아니다.

| split | 전체 | usable (identical 제외) |
|---|---|---|
| train | 614,844 | 357,116 (58.1%) |
| valid | 76,676 | 43,100 (56.2%) |

지역별로도 비대칭이다 — 강원도는 절반 이상이 사라진다.

| 지역 (valid) | 전체 | identical | usable |
|---|---|---|---|
| gyeongsangdo | 47,389 | 17,893 (37.8%) | 29,496 |
| gangwondo | 29,287 | 15,683 (53.5%) | 13,604 |

### `speech_kind`가 `is_identical`의 거의 전부를 설명한다

| speech_kind | n (valid) | identical 비율 |
|---|---|---|
| `read` (읽기) | 23,876 | **4.5%** |
| `talk` (자유대화) | 15,716 | 58.3% |
| `say` (따라말하기) | 37,084 | **62.9%** |

읽기 발화는 방언이 거의 항상 표준어와 다르고, 따라말하기/자유대화는 대다수가 동일하다.
즉 `is_identical`은 무작위 결측이 아니라 **수집 방식의 함수**다. 필터링·층화·오차 분석에서
`speech_kind`를 무시하면 안 되는 이유다.

---

## 2. 채워져 있지만 아무도 읽지 않는 컬럼

`prepare_data.py`가 채우고 나서 `src/`·`scripts/`의 어떤 코드도 소비하지 않는다.

| 컬럼 | 결측 (valid) | 내용 |
|---|---|---|
| `speech_kind` | 0.0% | `say` 48.4% / `read` 31.1% / `talk` 20.5% |
| `intent` | 0.3% | 8종 — `DES` 36.7%, `REP` 32.7%, `EXP` 18.4%, `PRO`/`INT`/`DIR` … |
| `emotion` | 0.2% | 5종 — `irrelevant` 66.6%, `negative` 16.4%, `positive` 16.0% |

`dialect_eojeol_map`은 **43.2%가 비어 있다**. `eojeol_accuracy`가 이 맵에 의존하므로,
그 지표는 사실상 데이터의 절반 조금 넘는 부분에서만 계산된다 — 지표를 읽을 때 감안해야 한다.

---

## 3. 변환 난이도 — copy bias가 데이터에 내장돼 있다

usable 쌍만 봐도 바뀌는 양이 작다.

| 지역 (valid, usable) | 표준어 어절 p50 | 바뀐 어절 mean | 바뀐 비율 mean | 1어절만 다른 쌍 |
|---|---|---|---|---|
| gangwondo (13,604) | 14 | 4.33 | 32.6% | **23.8%** |
| gyeongsangdo (29,496) | 12 | 3.36 | 25.4% | **28.5%** |

길이는 사실상 보존된다: 문자 수 Δ(방언−표준어) 평균이 강원 **−0.07자**, 경상 **+0.04자**.

이건 `copy_margin`(= chrF(gen,gold) − chrF(gen,source))이 왜 필요한지에 대한 **데이터 근거**다.
문장의 3/4가 그대로인데 chrF/BLEU는 그 3/4까지 점수로 쳐 준다 — 소스를 그대로 복사해도
높은 점수가 나온다는 DIA-REFINE(arXiv:2511.06680)의 지적이 이 코퍼스에서 그대로 성립한다.

---

## 4. 평가 타당성 — 발견 2건

### 4.1 first-n 선택이 평가셋을 쉬운 쪽으로 왜곡한다

모든 평가 경로가 `select(range(n))`, 즉 셔플 없는 **first-n**으로 표본을 고른다.
의도된 선택이었다 — 시드 없이 재현되므로 on-device validity gate가 "폰 채점과 데스크탑
채점이 같은 행을 봤다"를 증명할 수 있다. 다만 그 대가를 **측정한 적이 없었다.**

리더보드가 실제로 읽는 `outputs/datasets/grpo` valid, n=150:

| 난이도 버킷 | 강원 모집단 | 강원 head-150 | 경상 모집단 | 경상 head-150 |
|---|---|---|---|---|
| 1어절 | 23.8% | **60.7%** | 28.5% | 40.0% |
| 2–3어절 | 20.0% | 30.0% | 35.4% | 38.7% |
| 4–6어절 | 37.3% | **5.3%** | 29.2% | 14.0% |
| 7어절+ | 18.9% | 4.0% | 6.8% | 7.3% |
| **총변동거리(TVD)** | — | **46.9%** | — | **15.2%** |

강원도 평가셋은 가장 쉬운 버킷이 2.5배 과대표집, 중간 난이도(4–6어절)는 7배 과소표집이다.
더 나쁜 것은 **왜곡 크기가 지역마다 다르다는 점**(TVD 46.9% vs 15.2%)이다. 지역별 점수를
가중 평균해 OVERALL을 내는 현재 집계는 지역 간 왜곡이 같다고 가정하는데, 그렇지 않다.
게다가 §3의 copy bias와 방향이 겹친다 — first-n은 하필 "복사만 해도 거의 맞는" 행을 더 뽑는다.

**대응(구현됨, opt-in).** `ko_dialect.evaluation.sampling.stratified_indices`는 시드도 RNG도
쓰지 않고 모집단의 난이도 분포를 맞춘다. 버킷별로 모집단 비율만큼 배분(최대잔여법으로 합이
정확히 n)한 뒤 각 버킷의 앞쪽 행을 취하고 원래 순서로 되돌린다 → **결정성 유지**, validity
gate의 재현 해시도 그대로 성립.

| | 강원 TVD | 경상 TVD |
|---|---|---|
| head-150 (현행 기본값) | 46.9% | 15.2% |
| stratified-150 | **0.2%** | **0.3%** |

`load_eval_samples(..., strategy="stratified")`로 쓸 수 있다. **기본값은 여전히 `"head"`** —
바꾸면 `RESULTS.md`와 리더보드의 기존 수치가 전부 무효가 되므로, 재실행을 동반한 의식적인
결정으로 처리해야 할 사항이다 (§5).

### 4.2 `reconstruction_bleu`가 학습되지 않은 프롬프트 포맷으로 측정된다

`reconstruction_bleu`는 CLAUDE.md와 `leaderboard.py`가 지정한 **주 랭킹 지표**이고,
"proxy로 게임할 수 없는 독립 신호"라는 근거로 채택됐다. 그런데 역방향 생성에 쓰는 프롬프트가
학습 템플릿과 다르다.

학습/추론 SSOT (`src/ko_dialect/data/template.py`):

```
<|im_start|>system
당신은 한국어 방언 변환 전문가입니다.<|im_end|>
<|im_start|>user
다음 {지역} 사투리를 표준어로 바꿔줘:
{source}<|im_end|>
<|im_start|>assistant
```

역방향 패스 (`leaderboard.py: REVERSE_PROMPT`):

```
다음 방언 문장을 표준어로 바꿔줘.
방언: {dialect}
표준어:
```

ChatML 래퍼도, system 프롬프트도, 지역 정보도 없는 평문이다. instruction-tuned 모델을
학습된 적 없는 포맷으로 프롬프트하는 셈이라 두 가지 문제가 생긴다.

1. 재구성 품질이 체계적으로 낮게 측정된다.
2. 더 심각한 쪽 — 열화 폭이 **각 arm의 포맷 외 일반화 능력**에 좌우된다. 이는 측정하려는
   방언 전이 품질과 무관한 변수이므로, 랭킹 신호에 관련 없는 잡음이 섞인다.

on-device validity gate는 이걸 잡지 못한다. 게이트는 "폰 채점 == 데스크탑 채점"의 **동일성**만
보장하고, 프롬프트가 학습 분포와 맞는지(**타당성**)는 다루지 않는다. §4.1과 같은 구조의 갭이다.

---

## 5. 미결 결정사항

두 건 모두 고치는 순간 기존에 보고된 수치가 무효가 되므로, 재실행 계획과 함께 결정해야 한다.

| # | 항목 | 선택지 |
|---|---|---|
| D1 | 평가셋 선택 기본값 | (a) `head` 유지 + 신규 실험만 `stratified` (b) `stratified`로 전환 + 전 arm 재실행 (c) 두 수치 병기 |
| D2 | 역방향 프롬프트 | (a) 현행 유지 (기존 수치와의 연속성) (b) `template.py` SSOT 사용 + 재실행 (c) 두 포맷 모두 측정해 포맷 민감도를 지표로 보고 |

D2 (c)는 부수적으로 흥미로운 결과를 준다 — 포맷 민감도 자체가 소형 모델의 견고성 지표다.

## 6. 후속 연구 후보 (근거 있는 것만)

- **`speech_kind` 층화** — §1에서 이 컬럼이 `is_identical`을 거의 결정한다. 평가셋을 난이도뿐
  아니라 `speech_kind`로도 층화하면 "읽기에서만 잘한다" 같은 실패 모드가 드러난다.
  걸림돌: `outputs/datasets/grpo`에 이 컬럼이 없다 → 데이터셋 빌드 시 보존 필요.
- **`intent`(8종) 조건부 성능** — 발화 의도별 오차 분해. 컬럼이 이미 99.7% 채워져 있어 비용이 낮다.
- **`dialect_eojeol_map` 43.2% 결측 원인 규명** — `eojeol_accuracy`의 유효 표본이 절반뿐인 이유.
- **전라/제주/충청 확장** — 라벨·파서는 준비 완료. 필요한 것은 원본 데이터와 `prepare_data.py`
  경로 정리뿐이다 (`docs/OLD_DIALECT_EXPANSION.md` 런북 참조).
