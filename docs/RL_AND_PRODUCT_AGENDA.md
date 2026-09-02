# RL_AND_PRODUCT_AGENDA.md — 보상 설계 · tool-use · 실전 포지셔닝 사전 검토

> 작성 2026-08-04. "이거 회사 입장에선 toy 아닌가"에 대한 정면 답변 + 다음 라운드 실험 설계.
> 인용은 전부 **오늘 검색으로 ID·저자·제목 확인**했다([`REFERENCES.md`](REFERENCES.md) 규범).
> 관련: [`DESIGN_DECISIONS.md`](DESIGN_DECISIONS.md) · [`DATA_EXPANSION_RUNBOOK.md`](DATA_EXPANSION_RUNBOOK.md)

---

## 0. 세 줄 요약

1. **지금 하는 건 이미 hybrid RLVR이다.** 보상 4개 중 3개(content/edit/length)가 gold 기준
   결정적 프로그램이고, 학습된 성분은 style(TextCNN) 하나뿐. 그리고 **지난번 reward hacking은
   정확히 그 하나에서 터졌다** — 이건 우연이 아니라 RLVR 이론이 예측하는 그대로다.
2. 다음 라운드의 핵심 원칙: **루프 안은 검증 가능한 것만, 판단이 필요한 건 루프 밖 오프라인 게이트로.**
3. toy를 벗는 열쇠는 모델 크기가 아니라 **"언제 개입하지 않을지"를 학습시키는 것**(abstention/router).
   이게 tool-use 학습의 가장 현실적인 진입점이면서, 오늘 데모가 실패한 지점과 정확히 같다.

---

## 1. 사실 확인 — "지금 RLVR 하고 있는 건가?"

**부분적으로 그렇다.** RLVR = 학습된 보상 모델 대신 **결정적 검증 함수**로 보상을 주는 방식
(Tülu 3, Lambert et al., [arXiv:2411.15124](https://arxiv.org/abs/2411.15124)에서 명명 → DeepSeek-R1로 확산).

현재 `configs/training/grpo.yaml`의 보상 구성:

| 보상 | 가중치 | 구현 | 검증 가능? |
|---|---|---|---|
| `content` | 1.0 | `chrF(gen, gold)` — sacrebleu, 결정적 | ✅ gold 기준 프로그램 |
| `edit` | 0.5 | `dialect_eojeol_map`(주석된 정답) 기준 제거/등장 비율 | ✅ 주석 기반 정답 |
| `length` | 0.2 | gold 길이 대비 soft overlong penalty (DAPO 계열) | ✅ 규칙 |
| `style` | 0.5 | **TextCNN 분류기** | ❌ **학습된 RM — 유일한 비검증 성분** |

**핵심 통찰(포트폴리오 1급 소재)**: 500-step 실험에서 *보상은 오르는데 chrF/BLEU가 −9~11pt* 였던
그 실패는, 검증 가능한 세 축이 아니라 **유일하게 검증 불가능한 축**에서 발생했다.
RLVR 문헌이 학습된 verifier의 실패 양상으로 지목하는 것과 정확히 일치한다 — 최근 연구는 이를
**verifier failure**(학습 verifier가 인정한 걸 레퍼런스 판정단은 거부)와 **rubric-design
limitation**으로 분리한다: *Reward Hacking in Rubric-Based Reinforcement Learning*
(Mahmoud et al., [arXiv:2605.12474](https://arxiv.org/abs/2605.12474)), 후속 재현 연구
[arXiv:2606.04923](https://arxiv.org/abs/2606.04923).

> 즉 서사가 바뀐다. "GRPO 돌려봤다"가 아니라
> **"hybrid RLVR을 설계했고, 비검증 성분이 정확히 예측된 방식으로 무너지는 걸 측정으로 잡았다."**

---

## 2. 보상 설계 로드맵 — 3층 구조

### Layer V — 검증 가능(루프 **안**). 여기를 두껍게 만든다.

| # | 이름 | 정의 | 왜 |
|---|---|---|---|
| **V1** ★ | **identity / abstention verifier** | gold == source인 행에서 **출력 == 입력 정확 일치**면 1, 아니면 0 (binary) | 오늘 데모의 환각(`일 다 했스요? → …밥 먹고 나와요`)·과교정(`마카→만치`)을 직격. v2 데이터의 **44%가 is_identical**이라 신호량이 충분하다 |
| **V2** | **eojeol exact-hit verifier** | `dialect_eojeol_map`의 각 gold 어절을 정확히 산출했는가 (어절별 binary → 집계) | 현 `r_edit`은 soft(0.7 제거 + 0.3 등장)라 부분점수가 샌다. hard 버전을 **추가**해 A/B |
| **V3** | **hallucination verifier** | 출력 토큰 중 (소스 ∪ gold 어절 ∪ 표준어 사전)에 없는 비율에 페널티 | `창녕 → 창(^)(` 같은 디코딩 붕괴를 프로그램으로 잡는다. 사전은 코퍼스 표준어 어휘로 부트스트랩 |
| V4 | format/length | 이미 있음 | 유지 |

### Layer M — 학습된 모델(루프 안, **최소화**)
- `style`(TextCNN)은 **유지하되 비중을 더 낮추거나 아예 빼는 arm**을 실험한다.
- 근거: 약한 verifier의 proxy 이득은 레퍼런스 판정으로 **전이되지 않는다**([arXiv:2605.12474]).
  그리고 동급 크기 LLM autorater는 style-aware 지표보다 못하다 — *Mind the Style Gap*
  (Pauli, Augenstein & Assent, EMNLP Findings 2025, [arXiv:2502.15022](https://arxiv.org/abs/2502.15022)).
- **가설**: V1~V3를 넣으면 style 가중치를 0.5 → 0.2로 낮춰도 dialectness가 유지될 것이다. 검증 대상.

### Layer J — 판정단(루프 **밖**, 오프라인 게이트 전용)
- 루브릭 기반 보상 자체는 검증 불가 영역으로 RLVR을 확장하는 최신 흐름이다 — *Rubrics as Rewards*
  (Gunjal et al., [arXiv:2507.17746](https://arxiv.org/abs/2507.17746)), 루브릭 자동생성은
  *OpenRubrics* ([arXiv:2510.07743](https://arxiv.org/abs/2510.07743)).
- **그러나 우리는 루프 안에 넣지 않는다.** 이유 두 가지: ① 6GB에서 비용이 안 맞고 ② 우리가 이미
  겪은 실패(비검증 성분의 over-optimization)를 그대로 반복하는 길이다.
- 대신 **최종 선택(model selection)에서만** 서로 다른 계열의 판정자 여러 개를 쓴다
  (cross-family panel — [arXiv:2605.12474]의 방법론). 기존 "proxy-independent 선택" 원칙의 자연스러운 확장.

> **설계 원칙 한 줄**: *루프 안은 검증 가능한 것만. 판단이 필요한 건 루프 밖에서, 서로 다른 계열의
> 판정자로.* — 이 한 줄이 DR-E/DR-F 카드의 다음 버전이 된다.

---

## 3. 알고리즘 레시피 검토

| 기법 | 현재 | 판단 |
|---|---|---|
| GRPO ([arXiv:2402.03300](https://arxiv.org/abs/2402.03300)) | 사용 중 | 유지 |
| DAPO clip-higher + truncated mask ([arXiv:2503.14476](https://arxiv.org/abs/2503.14476)) | `epsilon_high=0.28`, `mask_truncated_completions=true` | 이미 채택 ✓ |
| MO-GRPO ([arXiv:2509.22047](https://arxiv.org/abs/2509.22047)) | `aggregation=mo_grpo` arm | 유지 |
| **Dr.GRPO** (Liu et al., [arXiv:2503.20783](https://arxiv.org/abs/2503.20783)) | 미적용 | **우선 검토.** length·std 정규화 제거. 우리는 `max_new_tokens=64`라 length bias는 작지만, **std 정규화 제거**는 보상 축마다 분산이 다른 우리 상황과 직접 관련. ⚠️ **MO-GRPO도 정규화를 건드리므로 중복·상충 위험** — 반드시 단독 arm으로 분리해 교호작용을 측정할 것 |
| GSPO (Zheng et al., [arXiv:2507.18071](https://arxiv.org/abs/2507.18071)) | 미적용 | 우선순위 낮음. sequence-level importance ratio는 긴 시퀀스·MoE에서 효과가 크고, 우리는 64토큰 dense 0.5B라 이득이 작을 것으로 예상. 여유 있을 때 |

**우선순위**: 보상 층(§2) ≫ Dr.GRPO 정규화 정리 > GSPO.
알고리즘 튜닝보다 보상 설계가 이 프로젝트에서 훨씬 큰 레버다(이미 측정으로 확인된 사실).

---

## 4. tool-use / agentic 방향 — "단순 converter" 탈피

### 4.1 재프레이밍
모델을 **번역기**가 아니라 **"언제·어떻게 개입할지 판단하는 컴포넌트"**로 본다.
운영에서 중요한 건 변환 품질보다 **과교정률**과 **기권 정확도**인 경우가 많다.

### 4.2 제품 맥락은 데이터가 이미 말해주고 있다
AI-Hub 코퍼스는 **중·노년층 방언 화자**다. 자연스러운 실전 시나리오는
**고령층 대상 음성 상담/민원/의료접수 파이프라인의 방언 정규화 전처리기**다.
이건 접근성 문제이지 장난감이 아니고, 아래 §5의 프라이버시·온프렘 논리와도 맞아떨어진다.

### 4.3 두 갈래

**(A) 우리 모델을 *tool로 제공*** — 쉽고 빠름
큰 에이전트가 `normalize_dialect(text, region?)`를 호출하고 우리 모델이 그 툴이 된다.
필요한 것: 낮은 지연·비용(**이미 ondevice 브랜치에 GGUF/측정 하니스 있음**) + **신뢰 가능한 기권**.

**(B) 모델이 *tool을 쓰도록 학습*** — 본인 관심사, 더 어려움
- 액션 공간 예: `passthrough` / `normalize(region)` / `lookup(방언어휘사전)` / `ask_region` / `abstain`
- **이 데이터셋의 희소한 장점**: 툴 사용의 성공/실패를 **프로그램으로 판정**할 수 있다.
  지역 라벨(정답 존재), `is_identical`(개입이 필요했는지 정답), `dialect_eojeol_map`(무엇을
  바꿔야 했는지 정답). 즉 **tool-use RLVR을 붙일 조건이 이미 갖춰져 있다.**
- 참고: Search-R1 ([arXiv:2503.09516](https://arxiv.org/abs/2503.09516)),
  R1-Searcher ([arXiv:2503.05592](https://arxiv.org/abs/2503.05592)),
  Tool-R1 ([arXiv:2509.12867](https://arxiv.org/abs/2509.12867)),
  ToolRM — tool-calling 전용 outcome reward model ([arXiv:2509.11963](https://arxiv.org/abs/2509.11963)).
- ⚠️ **정직한 경고**: 위 연구들은 대부분 3B~7B+ 기반이다. **0.5B로 멀티턴 툴 사용은 현실적으로
  매우 어렵다.** 그대로 따라 하면 실패한다.

### 4.4 그래서 첫 스텝은 이것 — **Router / Abstention 학습** ★
멀티턴을 포기하고 **단일 결정**부터 간다. 입력을 받아 `{그대로 통과 | 정규화 | 지역질의}`를 고른다.

- 보상: **전부 검증 가능** (§2의 V1이 그대로 이 태스크의 보상이다)
- 얻는 것 세 가지가 한 번에:
  1. 오늘 데모의 단문 환각/과교정 문제를 **직접** 해결
  2. tool-use 서사의 정당한 진입점 (0.5B가 감당 가능한 스코프)
  3. 운영에서 가장 중요한 지표(과교정률)를 학습 목표로 승격
- 확장 경로: 단일 결정이 되면 → `lookup` 툴 1개 추가 → 그 다음에야 멀티턴.

---

## 5. Closed API 대비 포지셔닝 (정직하게)

| 축 | GPT/Claude 등 | 우리(0.5B 특화) |
|---|---|---|
| 범용 품질 | **압도적 우위** | 진다. 인정하고 시작해야 논의가 성립 |
| 지연 · 단가 | API 왕복 + 토큰 과금 | GGUF 온디바이스, 과금 0 — **측정 하니스 이미 있음** |
| 프라이버시 · 온프렘 | 외부 전송 필요 | **의료·공공 음성은 외부 반출 불가한 경우가 있다 → 결정적 차별점** |
| 오프라인 · 엣지 | 불가 | 갤럭시 실측 진행 중 |
| 도메인 정확도 | 방언 어휘에서 자주 틀림(추정) | 특화 학습. **단, 아직 측정 안 함 — §5.1 참조** |
| 평가 방법론 | — | reward hacking을 측정으로 잡아낸 프레임 자체가 **재사용 가능한 자산** |

> **결론**: 서사는 "GPT보다 잘한다"가 아니라
> **"제약(6GB·온디바이스·프라이버시) 하에서 검증 가능하게 운용한다"** 여야 한다.
> 기업이 사는 건 대개 후자다.

### 5.1 실전 운용에 필요한데 **지금 없는 것**

| # | 항목 | 왜 필요 |
|---|---|---|
| **P1** ★ | **frontier baseline 측정** | 같은 held-out 평가셋을 GPT/Claude에 물려 우리 위치를 숫자로 박는다. **이게 없으면 "toy"라는 인상을 절대 못 벗는다.** 결과가 나쁘게 나와도 상관없다 — 어디서 이기고 어디서 지는지가 포지셔닝의 근거가 된다 |
| P2 | 과교정률(over-correction rate) SLA | 운영 품질의 실질 지표. §4.4 라우터의 목표 지표와 동일 |
| P3 | 지역 오분류 로깅 · drift 모니터링 | 5지역 확장 후 필수 |
| P4 | A/B 롤아웃 | `serve_compare.py`가 이미 다중 어댑터 서빙을 함 — 골격 존재 |

---

## 6. 실험 우선순위 (비용 대비 서사 가치)

| 순위 | 실험 | 비용 | 얻는 것 |
|---|---|---|---|
| 1 | **V1 identity/abstention verifier** 추가 후 GRPO 재실행 | 낮음 (보상 함수 1개) | 오늘 데모 실패 해결 + "검증 가능 성분을 두껍게" 서사 |
| 2 | **frontier baseline 측정** (P1) | 낮음 (API 호출) | toy 프레임 탈출. **1순위와 병행 가능** |
| 3 | **Router 학습** (§4.4) | 중간 | tool-use 진입점 + 운영 지표 승격 |
| 4 | V2/V3 verifier + style 가중치 0.5→0.2 arm | 중간 | "비검증 성분 없이도 되는가" 정면 검증 |
| 5 | 5지역 확장([`DATA_EXPANSION_RUNBOOK.md`](DATA_EXPANSION_RUNBOOK.md)) | 높음 (타 PC) | 커버리지 + 6-class |
| 6 | Dr.GRPO 정규화 arm | 중간 | 알고리즘 깊이 (MO-GRPO 교호작용 주의) |
| 7 | 오프라인 루브릭 판정단 게이트 | 중간 | 선택 신뢰도 |
| 8 | GSPO | 높음 | 우선순위 낮음 |

> 1·2번을 먼저 하는 이유: **둘 다 저비용이고, 각각 "기술적 깊이"와 "실전 감각"이라는 서로 다른
> 약점을 정확히 때린다.** 5지역 확장은 커버리지 서사엔 좋지만 위 두 개보다 설득력 기여가 낮다.

---

## 7. 참고문헌 (2026-08-04 ID·저자·제목 검증 완료)

| 주제 | 문헌 | 이 레포에서의 용도 |
|---|---|---|
| RLVR 명명 | Tülu 3 — [2411.15124](https://arxiv.org/abs/2411.15124) | §1 "지금 하는 게 RLVR인가" 정의 근거 |
| 루브릭 보상 | Rubrics as Rewards, Gunjal et al. — [2507.17746](https://arxiv.org/abs/2507.17746) | Layer J (오프라인 전용) |
| 루브릭 자동생성 | OpenRubrics — [2510.07743](https://arxiv.org/abs/2510.07743) | Layer J 확장 |
| **비검증 보상의 해킹** | Mahmoud et al. — [2605.12474](https://arxiv.org/abs/2605.12474) | §1·§2의 핵심 근거. verifier failure vs design limitation 분리, cross-family 판정단 |
| 위 재현 연구 | [2606.04923](https://arxiv.org/abs/2606.04923) | 탐지 방법 참고 |
| tool-use RL | Search-R1 [2503.09516](https://arxiv.org/abs/2503.09516) · R1-Searcher [2503.05592](https://arxiv.org/abs/2503.05592) · Tool-R1 [2509.12867](https://arxiv.org/abs/2509.12867) | §4 갈래 (B) |
| tool-call 보상모델 | ToolRM — [2509.11963](https://arxiv.org/abs/2509.11963) | §4 보상 설계 참고 |
| GRPO 정규화 편향 | Dr.GRPO, Liu et al. — [2503.20783](https://arxiv.org/abs/2503.20783) | §3 |
| sequence-level RL | GSPO, Zheng et al. — [2507.18071](https://arxiv.org/abs/2507.18071) | §3 (후순위) |
| 스타일 지표 한계 | Pauli, Augenstein & Assent — [2502.15022](https://arxiv.org/abs/2502.15022) | 기존 A3, Layer M 근거 |

> ⚠️ §2의 V1~V3 보상 설계, §4.4 라우터 스코프, §5 포지셔닝은 **이 레포의 자체 설계**이며 특정
> 논문의 권고가 아니다. 인용을 붙이지 말 것.
