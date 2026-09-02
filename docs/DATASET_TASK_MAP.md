# DATASET_TASK_MAP.md — 이 데이터셋으로 실제로 할 수 있는 것

> 작성 2026-08-04. **보상/알고리즘 설계는 [`RL_AND_PRODUCT_AGENDA.md`](RL_AND_PRODUCT_AGENDA.md)에
> 있고 여기서 반복하지 않는다.** 이 문서는 *데이터에 무엇이 들어 있는지*를 측정으로 확정하고,
> 거기서 열리는 **태스크**와 **산업 대응물**을 매핑한다.
> 데이터 확보 절차는 [`DATA_EXPANSION_RUNBOOK.md`](DATA_EXPANSION_RUNBOOK.md).

---

## 1. 새로 열린 자산 (측정치, `scripts/audit_corpus_fields.py`)

139-2 제주 2인발화 샘플 **188파일 / 1,601문장** 전수 스캔. 아래 필드는 **전부 100% 존재**하는데
지금까지 파이프라인이 버리고 있었다 → 오늘 `prepare_data.py`가 컬럼으로 보존하도록 고쳤다.

| 필드 | 내용 | 측정 |
|---|---|---|
| **`stt_hypothesis`** ★ | **Naver Clova Speech**의 실제 ASR 출력, 타임스탬프 정렬 | 정렬 성공 **1,584/1,601 = 98.9%** |
| `grammar_type` | DEC/YNI/WHI/IMP/PRO | DEC 1353 · **YNI 129 · WHI 78** · PRO 23 · IMP 18 |
| `intent` | 대화행위 | REP 808 · DES 331 · EXP 217 · INT 167 · ETC/PRO/DIR |
| `emotion` | 감성 | irrelevant 1230 · neg 184 · pos 151 · neutral 36 |
| `domain` | 발화 주제 | 가족·자연·식·건강·주·풍속·농경·의 (8종) |
| `speaker_*` | 성별·출생연도·직업 | 48화자, **f 371 : m 5**(샘플 한정) · 1970s 302 / 1960s 64 / 1950s 10 |
| `audio.recordDuration` | 발화 길이 | mean 67.9s (음성 없이 발화속도 계산 가능) |

**그리고 결함 하나**: `outputs/dialect_raw_new`에서 **valid 화자 623명 중 113명(18.1%)이 train에도
존재**. 지금까지의 held-out 점수는 그만큼 부풀려져 있다. → `scripts/resplit_speaker_disjoint.py`로
화자 공유 0인 `outputs/dialect_raw_new_spk`(train 622,327 / valid 69,193)를 만들어 뒀다(opt-in).

> ⚠️ 현재 2지역(71517) raw 원본은 디스크에 없어 `stt` 등 신규 컬럼이 비어 있다.
> 강원·경상에도 적용하려면 **71517 라벨 4.2GB 재다운로드 후 재빌드**가 필요하다.

---

## 2. 이 데이터로 가능한 태스크 (실무성 × 차별성 순)

### T1. ASR 후처리 교정 — **가장 중요한 재프레이밍** ★★★
`stt_hypothesis → standard`. 지금까지의 태스크(`gold 방언 전사 → 표준`)는 **입력이 추론 시점에
존재하지 않는다**(사람이 전사해 줘야 하니까). 반면 ASR 가설은 시스템이 실제로 내놓는 문자열이다.

측정된 근거 — 상용 ASR이 방언에서 무너진다:
```
STT  : 아니 옛날에는 기계 하실 때는 … 기계가 저분하니 난 요즘에 하은 기계기계로 다 하나 막 좋수다
방언 : 아니 옛날에는 기계 엇일 때는 … 기계가 좋아부난예 난 이 요즘예 호꼼 기계잇이난 … 막 좋수다
표준 : 아니 옛날에는 기계 없을 때는 … 기계가 좋아서요 나는 요즘에 조금 기계있으니까 … 아주 좋습니다
```
STT~방언 유사도 0.808 vs STT~표준 0.778, **STT가 방언 쪽에 더 가까운 비율 59.0%** — 즉 ASR은
방언을 방언대로(그것도 틀리게) 받아쓰고, 표준화는 아무도 안 해준다. 그 간극이 태스크다.

- 산업 명칭: **generative error correction (GER) / post-ASR correction / ASR 정규화**
- 근거: GER 개념 [2307.04172](https://arxiv.org/abs/2307.04172) · 희귀어+음성문맥
  [2505.17410](https://arxiv.org/abs/2505.17410) · 잡음강건 [2509.04392](https://arxiv.org/abs/2509.04392) ·
  서베이 [2508.07285](https://arxiv.org/abs/2508.07285)
- **반대 근거도 반드시 읽을 것**: 저자원 ASR에서 LLM 교정이 정말 되는지 오염(contamination)을
  통제해 회의적으로 검증한 West Frisian 연구 [2605.19711](https://arxiv.org/abs/2605.19711).
  우리도 **"교정이 실제로 이득인가"를 먼저 재는 실험**부터 해야 한다(무조건 이득 가정 금지).
- 왜 우리에게 유리한가: 0.5B 온디바이스 후처리기는 상용 ASR API 뒤에 붙이는 **얇은 레이어**라
  교체·배포가 쉽고, [`RL_AND_PRODUCT_AGENDA.md`](RL_AND_PRODUCT_AGENDA.md) §5의 온프렘 논리와 맞는다.

### T2. 문장유형 계층 평가 — 단문 실패의 진단 도구 ★★
`grammar_type`으로 DEC/YNI/WHI/IMP를 나눠 점수를 낸다. 오늘 데모에서 실패한 문장은 대부분
**의문문**(`밥 뭇나?`, `마카 다 모였소?`)이었는데, 지금까지는 그걸 **분해해서 볼 수단이 없었다**.
의문·명령문은 합쳐도 15% 미만이라 OVERALL 평균에 묻힌다.

### T3. 화자 분리 평가 재보정 ★★
`_spk` 데이터셋으로 리더보드를 한 번 더 돌려 **기존 점수와의 차이**를 보고한다. 차이가 크면
지금까지의 SFT/GRPO 비교 결론 일부가 흔들린다 — 그걸 아는 것 자체가 성과다.

### T4. 도메인 시프트 평가 ★★
`domain` 8종으로 train/eval을 나눠 **미학습 도메인 일반화**를 잰다. 실전에서 가장 자주 깨지는 축인데
지금 프레임에는 없다.

### T5. 인구통계 공정성 ★
`speaker_gender` / `speaker_birth_year`별 성능 격차. 샘플에서 성별이 **f 371 : m 5**로 극단적으로
치우쳐 있었다(제주 2인발화 한정이라 전수 확인 필요). 고령층 대상 서비스라면 연령대별 격차는
그 자체로 리스크 항목이다.

### T6. 대화 단위 확장 ★
`talk_*`(2인발화)는 `speaker_id`로 화자가 구분되므로 **멀티턴 문맥**이 있다. 문장 단위 변환이
안정되면 "직전 턴을 문맥으로 주면 좋아지는가"가 자연스러운 다음 질문이다.
목표지향 대화에서의 ASR 오류 처리 맥락: [2501.06129](https://arxiv.org/abs/2501.06129).

### T7. intent/emotion 보조 태스크 ★
멀티태스크 보조 신호 또는 라우팅 피처. 우선순위는 낮지만 라벨이 공짜다.

### T8. 음성 직접 경로
기존 [`SPEECH_MODULE_DESIGN.md`](SPEECH_MODULE_DESIGN.md). 원천데이터 150GB가 필요해 이 PC 범위 밖.
저자원 방언 ASR에서 운율을 함께 쓰는 최신 사례: CantoASR [2511.04139](https://arxiv.org/abs/2511.04139),
악센트 강건성의 화자수/시간/다양성 효과 [2506.04364](https://arxiv.org/abs/2506.04364).

---

## 3. 산업 대응물 — "비슷한 실무 태스크가 뭐냐"

| 우리 태스크 | 산업에서 부르는 이름 | 어디에 쓰이나 |
|---|---|---|
| `stt_hypothesis → standard` | post-ASR correction / GER, ITN(inverse text normalization) | 콜센터 전사 정제, 회의록, 의료 받아쓰기 |
| 방언→표준 정규화 | text normalization / canonicalization | 검색 질의 정규화, 의도분류 전처리 |
| abstention/router | confidence gating, guardrail, cascade routing | 소형→대형 모델 캐스케이드, 비용 최적화 |
| 문장유형 계층 평가 | slice-based evaluation | 모델 리스크 관리, 회귀 감시 |
| 화자 분리 split | group-aware / leakage-free split | 의료·음성 ML의 기본 요건 |
| 인구통계 격차 | fairness slice / subgroup robustness | 규제 대응, 접근성 |
| 온디바이스 소형 특화 | edge inference, on-prem NLP | 개인정보 반출 불가 도메인 |

**요약**: "방언 번역기"는 좁게 들리지만, `ASR 출력 정제 + 개입 여부 판단 + 슬라이스별 품질 관리`로
기술하면 **음성 파이프라인을 굴리는 회사의 일상 업무**와 같은 문제다.

---

## 4. 방법론 이식 (사용자 다른 레포에서 검증된 것)

`~/workspace_jera`와 `~/agentic_memory`에 이미 정착한 규범 중 이 프로젝트에 바로 붙는 것:

| 이식 대상 | 출처 | 이 레포에 적용하면 |
|---|---|---|
| **goodput(SLO 달성 req/s)**, TTFT/TPOT 분해, warmup 제외, ISL/OSL 통제, Poisson vs constant 도착 | jera `serving-bench-methodology-llm-2026-07.md` | `bench_serving.py`는 지금 p50/p95·tok/s만 본다. **goodput이 빠져 있다** — 온디바이스 SLA 서사에 직접 필요 |
| 통합 bench aggregation + **회귀 추적** | jera `bench-eval-stack-validation-2026-07.md` | 리더보드는 있으나 **회귀 게이트가 없다**. 실험이 늘수록 필수 |
| LLM-judge 2026 best practice | 같은 문서 | [`RL_AND_PRODUCT_AGENDA.md`](RL_AND_PRODUCT_AGENDA.md) Layer J(오프라인 판정)의 실행 규격 |
| **lineage pinning** — 갈라지는 상수마다 출처를 명시하고 전환 가능·테스트 가능하게 | agmem README | 이미 부분 적용됨(`PROSODY_MARKER_POLICY_V1/V2`). **보상 정책에도 확장**하면 arm 간 비교가 재현 가능해진다 |
| **적대적 검증(LLM 호출 0)** — 코드라인 인용 + 결정적 재현으로 주장을 반박 | agmem 감사 프로세스 | 오늘 v1 신호 판정을 뒤집을 때 쓴 방식과 동일. `docs/AUDIT.md` 관행에 명문화 |
| 산출물은 한 디렉터리에만, 원본은 append-only | jera `CLAUDE.md` | 이 레포는 `outputs/` 관행이 이미 있음 — `data/`(원본) 불가침을 명문화하면 완성 |

> agmem의 핵심 교훈 하나는 그대로 가져올 만하다: **"논문에 충실"이라는 목표는 단일 지점으로
> 존재하지 않는다.** 재현 대상이 논문인지 공개 코드인지 벤치마크 하니스인지를 *먼저 고정*해야 한다.

---

## 5. 실험 아젠다 v2 — 추가분만

[`RL_AND_PRODUCT_AGENDA.md`](RL_AND_PRODUCT_AGENDA.md) §6의 8개는 유효하고, 오늘 발견으로 **아래가 추가**된다.

| # | 실험 | 비용 | 선행 |
|---|---|---|---|
| **A** | **GER 베이스라인**: `stt_hypothesis → standard`를 SFT로 학습하고, ①무보정 STT ②기존 dia→std 모델에 STT를 그냥 넣은 것 과 비교 | 중 | 71517 재다운로드(4.2GB) |
| **B** | **GER 이득 검증(회의적)**: [2605.19711] 방식으로 "교정이 실제로 WER/chrF를 개선하는가"를 오염 통제하에 확인. **개선이 없으면 T1을 접는다** | 낮 | A |
| **C** | **화자분리 재평가**: `_spk`로 리더보드 재실행 → 기존 결론과의 차이 보고 | 낮 | 없음(지금 가능) |
| **D** | **문장유형 슬라이스**: 리더보드에 grammar_type별 컬럼 추가 | 낮 | 71517 재빌드 |
| **E** | **도메인 홀드아웃**: domain 2종을 eval 전용으로 빼고 일반화 측정 | 중 | 71517 재빌드 |
| **F** | **goodput 계측**: `bench_serving.py`에 SLO 기반 goodput + ISL/OSL 통제 추가 | 낮 | 없음(지금 가능) |
| **G** | **공정성 슬라이스**: 성별·연령대별 격차 리포팅 | 낮 | 71517 재빌드 |

**지금 당장 가능한 것은 C와 F** — 둘 다 재다운로드가 필요 없고, 각각 *지표 신뢰도*와
*운영 서사*라는 서로 다른 약점을 때린다.

---

## 6. 연구 포지셔닝 — 이 일이 어디에 놓이나

포트폴리오에서 "혼자 만든 파이프라인"이 아니라 **기존 연구 지형 안의 한 점**으로 말할 수 있어야 한다.

### 6.1 직접 선행연구가 이미 있다
이 레포가 `A1`으로 인용해 온 **DIA-REFINE([2511.06680](https://arxiv.org/abs/2511.06680))의 실제 제목은
*"Steering LLMs toward Korean Local Speech: Iterative Refinement Framework for Faithful Dialect
Translation"*** — 즉 **같은 과제(한국어 방언 번역)의 직접 선행연구**다. TDR/DFS 지표를 빌려 쓰는
관계를 넘어, **동일 과제에서의 비교 대상**으로 명시하는 편이 정직하고 강하다.
→ 할 일: 이들의 설정(데이터·지표·모델 규모)과 우리 설정의 차이를 표로 정리. 겹치면 재현 비교,
안 겹치면 "왜 다른 선택을 했는가"가 곧 기여 서술이 된다.

### 6.2 벤치마크 좌표
- **DialectBench** ([2403.11009](https://arxiv.org/abs/2403.11009), ACL 2024) — 방언·변종·근접언어를
  묶은 NLP 벤치마크. 방언 연구의 공통 좌표계이므로, 우리 과제를 그 축(방언 식별 / 변종 간 이해)
  위에 놓으면 "AI-Hub 데이터로 뭔가 했다"보다 훨씬 잘 전달된다.
- **SD-QA** — 구어 방언 QA에 **한국어 포함**. 음성 경로로 확장할 때의 비교 지점.
- **KITE** ([2510.15558](https://arxiv.org/abs/2510.15558)) — 한국어 instruction-following 벤치마크.
  라우터/도구사용으로 확장할 때 한국어 일반 능력 회귀를 재는 용도.

### 6.3 화자 분리는 우리만의 깐깐함이 아니다
§1에서 발견한 18.1% 화자 누출은 음성 계열에서 **표준 관행**으로 다뤄지는 문제다. 화자가 양쪽에
있으면 모델이 과제 대신 화자 재식별을 학습해 점수가 낙관적으로 부풀고, 저자 단위 과제에서
**5~10%p 부풀림**이 보고돼 있다. 저자원 ASR의 분할 전략을 직접 다룬 연구도 있다
([2208.12888](https://arxiv.org/abs/2208.12888)).
→ 즉 이건 "내가 깐깐하게 굴었다"가 아니라 **기존 평가 규범을 뒤늦게 맞춘 것**이며, 그렇게
서술해야 한다. 동시에 화자 재식별 위험은 프라이버시 논거와도 이어진다
([2606.07210](https://arxiv.org/abs/2606.07210), 음성 익명화의 재식별 위험 대규모 분석) —
[`RL_AND_PRODUCT_AGENDA.md`](RL_AND_PRODUCT_AGENDA.md) §5의 온프렘 포지셔닝 보강 근거.

### 6.4 포트폴리오로 말할 때의 한 줄
> "AI-Hub 방언 코퍼스로 0.5B 모델을 튜닝했다"가 아니라,
> **"직접 선행연구가 있는 과제에서, 프록시 보상의 과최적화와 화자 누출이라는 두 개의 평가 함정을
> 측정으로 찾아내고 각각을 보상 설계와 분할 정책으로 막았다"**.
> 모델 성능이 아니라 **평가를 신뢰할 수 있게 만든 과정**이 이 프로젝트의 산출물이다.

---

## 7. 참고문헌 (2026-08-04 ID·제목 검증)

| 주제 | 문헌 |
|---|---|
| GER 개념 | [2307.04172](https://arxiv.org/abs/2307.04172) Can Generative LLMs Perform ASR Error Correction? |
| GER 희귀어·음성문맥 | [2505.17410](https://arxiv.org/abs/2505.17410) |
| 잡음강건 GER | [2509.04392](https://arxiv.org/abs/2509.04392) Denoising GER |
| ASR 정제 서베이 | [2508.07285](https://arxiv.org/abs/2508.07285) Non-Intrusive ASR Refinement |
| **저자원 GER 회의적 검증** | [2605.19711](https://arxiv.org/abs/2605.19711) West Frisian, contamination-aware |
| 목표지향 대화의 ASR 오류 처리 | [2501.06129](https://arxiv.org/abs/2501.06129) |
| 저자원 방언 ASR + 운율 | [2511.04139](https://arxiv.org/abs/2511.04139) CantoASR |
| 악센트 강건성 요인 | [2506.04364](https://arxiv.org/abs/2506.04364) |
| **직접 선행연구(한국어 방언)** | [2511.06680](https://arxiv.org/abs/2511.06680) Steering LLMs toward Korean Local Speech (= 이 레포의 DIA-REFINE) |
| 방언 벤치마크 좌표 | [2403.11009](https://arxiv.org/abs/2403.11009) DialectBench (ACL 2024) |
| 한국어 instruction 벤치 | [2510.15558](https://arxiv.org/abs/2510.15558) KITE |
| 저자원 ASR 분할 전략 | [2208.12888](https://arxiv.org/abs/2208.12888) |
| 화자 재식별 위험 | [2606.07210](https://arxiv.org/abs/2606.07210) |

> §2~§5의 태스크 우선순위·산업 매핑·이식 판단은 **이 레포의 자체 설계**이며 논문 권고가 아니다.
> 인용을 붙이지 말 것 ([`REFERENCES.md`](REFERENCES.md) 규범).
