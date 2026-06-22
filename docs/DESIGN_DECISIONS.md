# DESIGN_DECISIONS.md — 의사결정 중심 설계·구현·논문 학습 가이드

KoDialect(한국어 방언→표준어 변환, Qwen2.5-0.5B, RTX 2060 6GB)의 **모든 설계 결정**을
`문제 → 선택지 → 선택 → 근거(논문+실측) → 코드`의 의사결정 카드로 정리한 문서입니다.
포트폴리오 발표·면접 대비 + 논문/구현 학습 교재 겸용.

- 논문 원문 검증·인용 정정: [`REFERENCES.md`](REFERENCES.md)
- 코드 감사·재현성: [`AUDIT.md`](AUDIT.md)
- 실험 결과·재현 커맨드: [`EXPERIMENTS.md`](EXPERIMENTS.md)

**의사결정 카드 읽는 법**: 각 카드는 `DR-<그룹><번호>`로 식별. 그룹 = A(인프라) · B(데이터) ·
C(운율) · D(모델·학습) · E(보상) · F(평가). ★ = 이 프로젝트의 시그니처 결정(면접 핵심).

---

## 0. 한눈에 — 의사결정의 배경

| 항목 | 값 | 결정에 준 제약 |
|---|---|---|
| 과제 | 방언→표준 번역 (양방향) | style↔content 트레이드오프(TST) |
| 모델 | Qwen2.5-0.5B-Instruct | 작아서 보상해킹·포화에 취약 |
| HW | RTX 2060, Turing SM7.5, 6GB | **BF16 불가(fp16 only)**, 4bit 커널 느림 |
| 방법 | SFT → TextCNN 분류기(보상) → GRPO | 분류기가 hackable → 가드 설계 필요 |
| 데이터 | AI-Hub 방언(강원·경상 공개) | 5지역은 미공개 → 확장 보류 |

**파이프라인**:
```
raw JSON ─prepare_data→ Arrow ─stage0→ {sft, grpo, classifier} 데이터셋
  Stage1 SFT(QLoRA) → sft_merged
  Stage2 TextCNN 분류기 → GRPO 스타일 보상 모델
  Stage3 GRPO(보상 = style + 충실도 가드들) → grpo_*
  평가: proxy-independent 선택 + Pareto + Koehn 유의성
```

---

## 1. 마스터 의사결정 흐름 (큰 인과 사슬)

이 프로젝트의 의사결정은 **"제약 → 순진한 선택 → 실측으로 실패 발견 → 재조정"** 의 정직한
루프입니다. 포트폴리오의 핵심 서사:

```
[제약] 0.5B + 6GB + fp16
   │
   ├─→ (D) 작은 모델이라 큰 LM 보상 대신 TextCNN 분류기 보상 채택
   │        └─→ 문제: 분류기는 hackable (max-pool 포화)
   │
   ├─→ (E) 순진하게 style 보상만 강하게 → GRPO
   │        └─→ [실측] 500-step: 분류기 보상↑ but chrF/BLEU −9~11pt (over-optimization!)
   │              ├─→ (E) KL beta 0.04→0.1 (SFT 앵커 강화)
   │              ├─→ (E) style 디모트 + content/edit/overcorrection/length 가드 추가
   │              └─→ (F) 선택을 proxy-independent 신호로만 (J-score/TDR 금지)
   │
   └─→ (F) style↔content는 음의 상관 → 단일 랭크 강요 X → Pareto + Koehn 유의성
            └─→ [결론] GRPO 우위는 방언 의존적 (강원=SFT, 경상=GRPO, grpo_500 robust)
```

이 한 장이 면접에서 "왜 그렇게 했나"의 80%를 설명합니다.

---

## 2. 의사결정 카드

### A. 하드웨어 · 인프라

**DR-A1 ★ · fp16 강제, BF16 전면 금지**
- 문제: RTX 2060(Turing SM7.5)은 native BF16이 없음 → bf16 요청 시 조용히 성능 저하/크래시.
- 선택지: ① bf16(최신 기본값) ② fp16 ③ fp32(OOM).
- 선택: **fp16**, 그리고 bf16 요청을 코드에서 fp16으로 강등.
- 근거: 하드웨어 사실(SM<8.0). 실측으로 bnb 4bit 학습이 `_amp_foreach_..._unscale BFloat16` 에러.
- 코드: `models/loading.py:86-107 resolve_dtype` (SM<8 → fp16 + 경고), 모든 config `bf16: false`, `sft_trainer.py:195-198`(떠도는 bf16 파라미터 캐스팅).

**DR-A2 ★ · 학습 백엔드 = unsloth 16-bit LoRA (4-bit QLoRA 아님)**
- 문제: 6GB에서 GRPO가 돌아야 함. 4bit가 메모리엔 좋아 보임.
- 선택지: ① hf full-FT(OOM) ② bnb 4bit QLoRA ③ unsloth 16-bit LoRA.
- 선택: **unsloth 16-bit LoRA**.
- 근거(실측): 2060에서 4bit는 ~146s/step(빠른 4bit dequant 커널 없음), **16-bit LoRA는 ~3.4s/step(~44×)**, peak VRAM ~2.7GB.
- 코드: `loading.py:140-166 _load_unsloth`, `configs/training/grpo.yaml`(backend/load_in_4bit 주석에 측정치). 메모리 [[grpo-2060-setup]].
- 교훈: "메모리 절약 ≠ 빠름". 커널 지원이 없으면 양자화가 손해.

**DR-A3 · 플러그형 백엔드 팩토리 (단일 진입점)**
- 문제: SFT·GRPO가 각자 양자화/LoRA를 따로 다루면 드리프트.
- 선택: `load_backbone(BackendConfig)` 한 곳 — unsloth/bnb/hf/loftq/awq/gptq/qat 7종 디스패치.
- 코드: `loading.py:110-128`. (awq/gptq=사전양자화+LoRA, qat=의도적 스텁 `268-291`)

**DR-A4 · config-driven (Hydra/YAML), 하드코딩 금지**
- 문제: 하이퍼파라미터가 코드에 박히면 실험 재현·교체 불가.
- 선택: 모든 설정 YAML(`configs/`), 보상·템플릿·백엔드까지 레지스트리로 교체.
- 근거: 재현성. 코드: `configs/**`, 레지스트리(DR-E11, DR-D-template).

### B. 데이터 품질 (보상의 토대)

**DR-B1 ★ · 라벨 단일 출처 + append-only id**
- 문제: 라벨맵이 분류기/보상/평가 3곳에 흩어져 silent drift.
- 선택: `labels.py` 한 곳, `standard=0` 고정, 지역은 **뒤에 추가만**(재번호 금지).
- 근거: 재번호하면 학습된 분류기 체크포인트 출력 뉴런이 오정렬됨.
- 코드: `data/labels.py:15-29`. 재export: `dataset.py:13-15`, 소비: `classifier.py:11`, `rewards/style.py:6`, `metrics.py:10`.

**DR-B2 · 데이터 기반 클래스 수 (`num_labels_for`)**
- 문제: 2지역=3클래스, 5지역=6클래스. 전역 상수를 손으로 바꾸면 실수.
- 선택: 학습 라벨의 `max+1`로 자동 결정. 기존 3클래스 체크포인트는 같은 데이터면 그대로.
- 코드: `labels.py:32-39`, `stage2_train_classifier.py`(호출). ⚠️ sparse id면 phantom 클래스(현재 연속이라 안전).

**DR-B3 ★ · Levenshtein ≥ 0.1 필터 (DIA-REFINE)**
- 문제: 표준과 거의 같은 "방언" 샘플이 보상에게 "near-standard도 방언" 이라고 가르침 → 보상해킹 씨앗.
- 선택: 정규화 문자 Levenshtein(표준, 방언) ≥ 0.1 인 것만 admit.
- 근거(논문): DIA-REFINE §3.1 "form-level divergence 보장"(원문 확인). True Attempt vs False Success 구분.
- 코드: `data/filtering.py:32-62 norm_levenshtein`, `dataset.py:335-340`.

**DR-B4 · standard 클래스 de-pollution (마커 기반)**
- 문제: AI-Hub "표준" 전사에 고정밀 방언 어미(`카노`,`드래요`)가 섞여 표준 경계를 흐림.
- 선택: 그런 마커가 있는 label-0 문장 드롭. 강원은 표준과 겹쳐서 마커를 **극소수만** 보수적으로.
- 코드: `filtering.py:69-95`(GYEONGSANG 15개/GANGWON 2개), `dataset.py:327-330`.
- 교훈: 도메인 지식(어미)을 코드에 명시적으로.

**DR-B5 · 라벨 충돌 제거**
- 문제: 같은 문자열이 두 라벨로 → 학습 불가능한 모순, `P(dia)−P(std)` 보상 오염.
- 선택: 충돌 텍스트는 **양쪽 다** 드롭(어느 게 맞는지 모름).
- 코드: `dataset.py:261-275 _drop_label_collisions`.

**DR-B6 · is_identical 필터 + graceful fallback**
- 문제: 표준==방언인 복사 행이 학습 노이즈.
- 선택: 드롭. 단, 컬럼 없으면 `standard==dialect`로 폴백(KeyError 방지).
- 코드: `dataset.py:78-80,332`(우리가 `.get` 폴백으로 보강).

**DR-B7 · 클래스 불균형 제어 (train만 다운샘플)**
- 문제: standard가 압도 → 보상 모델이 다수 클래스로 치우침.
- 선택: `standard_cap_ratio`(최대 방언클래스 ×N), `max_per_label`. **valid는 손대지 않음**(정직한 평가 위해 실제 분포 유지).
- 코드: `dataset.py:219-258 _downsample_rows`(seed 고정), `:348` train만.

**DR-B8 · double-pointer 어절 정렬 (N:M 매핑)**
- 문제: 어절맵(GRPO edit 보상 근거)을 표준↔방언 단어열에서 정확히 정렬해야 함. 축약(`가 버리고→가삐고`) 존재.
- 선택: 양쪽 포인터로 1:1은 unroll, N:M은 통째 항목, 동일어절은 포인터 싱크만.
- 코드: `prepare_data.py:231-317`. 정렬 틀리면 `r_edit` 보상이 틀림.

**DR-B9 · MapReduce 병렬 + jsonl 스트리밍**
- 문제: 33G raw를 메모리에 다 못 올림.
- 선택: 워커가 PID별 jsonl에 스트리밍 write → 커널 레벨 병합 → Arrow.
- 코드: `prepare_data.py:405-543`. orjson auto-fallback(`29-60`), SIGINT 워커 무시(`438-439`).

**DR-B10 · old/new 포맷 분리 + 복구한 태그 클리너**
- 문제: 구형 AI-Hub 전사에 `(방언)/(표준)`,`((불명))`,`{비언어}`,`&PII&`,`#` 태그 잔존(6.4%).
- 선택: old 경로에만 `clean_old_transcript` 적용(신규는 이미 깨끗). 태그 잔여 6.4%→0%.
- 코드: `prepare_data.py:120-144`. 메모리 [[old-dialect-transcript-cleaning]].

**DR-B11 · output_mode: text vs structured**
- 문제: 템플릿/loss-mask를 바꿀 때마다 데이터 재빌드하면 비쌈.
- 선택: `text`(빌드 때 렌더, 빠름) / `structured`(학습 때 렌더, 템플릿 교체 자유) 둘 다 지원.
- 코드: `dataset.py:46-51,111-123`, `sft_trainer.py:135-172`.

**DR-B12 · 양방향 학습 + 정직한 KNOWN ISSUE**
- 문제: std2dia/dia2std 둘 다 학습. 단, prosody 모드에서 dia2std의 source가 마커를 달아 OOD.
- 선택: 양방향 유지하되 **한계를 주석으로 명시**(평가하는 std2dia는 영향 없음).
- 코드: `dataset.py:100-109`(KNOWN ISSUE 주석).
- 교훈: 알려진 결함을 숨기지 않고 문서화 = 신뢰성.

### C. 운율 / 마커 (K-ToBI)

**DR-C1 · prosody v1 동결 + v2 declination-aware**
- 문제: v1 대칭 ±0.15 밴드가 한국어 자연 F0 하강(median −0.21)을 falling tone으로 오태깅(~60% DOWN), `<WAVE>`는 없는 필드 의존해 죽은 클래스.
- 선택: v1을 **재현용으로 동결**, v2는 비대칭 임계(up +0.05/down −0.40) + `<WAVE>`를 변동계수로. 614k행 튜닝(UP19/DOWN23/WAVE6/KEEP52%).
- 근거(논문): K-ToBI IP 경계톤(Jun 2000): UP=H%, DOWN=L%, WAVE=contour, KEEP=level.
- 코드: `prosody.py:16-49,88-121`.

**DR-C2 · 어절 마커는 1:1 정렬일 때만 적용**
- 문제: 세그먼트:단어가 N:M(~23%)이면 재구성이 단어를 빠뜨릴 위험.
- 선택: 정확히 1:1일 때만 부착, 아니면 원문 그대로(SFT target 무결성 보장 — 마커 제거 시 gold 복원).
- 코드: `prosody.py:183-206`.

**DR-C3 · 운율의 정체: 충실도 정규화 (실측 결론)**
- 문제: 운율 마커가 방언성을 올리나?
- 선택/결론: A/B 실측 — recon_bleu +3.9/+6.4(p<.001) 유의하게↑ but 방언 축은 flat → **방언성 레버가 아니라 충실도 정규화**. v2도 같은 결론.
- 근거: 자체 실측(EXPERIMENTS §2). 메모리 [[prosody-sft-ab-verdict]].

### D. 모델 · 학습

**DR-D1 ★ · 보상 모델 = TextCNN (큰 LM 아님)**
- 문제: GRPO 스타일 보상을 무엇으로? LLM judge는 비싸고 6GB에 안 맞음.
- 선택지: ① LLM-judge ② 회귀 헤드 ③ TextCNN 분류기.
- 선택: **TextCNN**(Kim 2014) — 단일 forward, 길이 무관(max-pool), 싸고 macro-F1 0.95.
- 근거(논문): Kim 2014; Mind the Style Gap(동급 LLM 오토레이터 < style-aware 지표).
- 코드: `models/classifier.py:75-140`. ⚠️ **트레이드오프**: max-pool이 반복 trigger로 포화 가능 → 보상해킹의 메커니즘적 원인(DR-E2로 연결).

**DR-D2 · eval OOM 가드 (hidden 무시 + logits 축소)**
- 문제: HF Trainer eval이 비-loss 출력을 전체 eval셋에 누적 → GPU 폭발.
- 선택: 분류기는 `keys_to_ignore_at_inference=["hidden_states"]`; SFT는 logits `(B,T,vocab)→(B,T,2)`(argmax+엔트로피)로 축소.
- 코드: `classifier.py:26`, `sft_trainer.py:104-132`.

**DR-D3 · save_merged_16bit (역사적 GRPO 로딩 버그 수정)**
- 문제: SFT가 어댑터 디렉토리만 저장하면 GRPO가 **raw base**를 로드(학습 무효).
- 선택: SFT 후 LoRA를 base에 merge해 standalone 16bit 저장 → GRPO가 학습 가중치 로드.
- 코드: `loading.py:310-329`, `sft_trainer.py:307-311`.

**DR-D4 · 분류기 선택 지표 = macro-F1 (accuracy 아님)**
- 문제: standard가 다수라 accuracy는 쏠림.
- 선택: macro-F1(클래스 균형 분리). DIA-REFINE 헤드라인 지표와 일치. early-stop on(patience=5).
- 코드: `classifier_trainer.py:28-36`. class_weights balanced(`43-67`).

**DR-D5 · SFT 기본값: early-stop off (의도적, 결정 대기)**
- 문제: SFT에 best-checkpoint 복원을 기본으로 켤까?
- 선택: **off**(기존 런 재현성 보존). 켜는 경로·방향 가드는 준비됨.
- 코드: `sft_trainer.py:64-71,224-246`. (AUDIT M6 — 새 런 품질 원하면 config 토글)

### E. 보상 설계 (이 프로젝트의 심장)

**DR-E1 ★ · GRPO 채택 (PPO 아님)**
- 문제: 6GB에서 RL. PPO는 critic(value net)이 VRAM 2배.
- 선택: GRPO — critic 없이 그룹 상대 advantage `Â=(r−mean)/std`. KL-to-ref 항.
- 근거(논문): DeepSeekMath/DeepSeek-R1(arXiv:2402.03300). critic 제거 ≈ VRAM 절반.
- 코드: `training/grpo_trainer.py`, `configs/training/grpo.yaml`.

**DR-E2 ★ · over-optimization 대응: beta 0.04→0.1 (실측 후)**
- 문제: 500-step 후 분류기 보상↑(TDR이 gold 초과)인데 chrF/BLEU vs gold −9~11pt.
- 선택: KL beta(SFT 앵커) 0.04→0.1로 상향.
- 근거(실측): 자체 500-step 런. 메모리 [[grpo-reward-over-optimization]].
- 코드: `configs/training/grpo.yaml:beta`.

**DR-E3 ★ · 보상 재조정: style 디모트 + 충실도 앵커**
- 문제: style 단독은 "표준에서 멀어져라"는 무한 push → 정책이 과장/오지역 방언으로 게임.
- 선택: style 가중 낮추고 content(chrF vs gold) + edit(정확한 gold 어절) 앵커가 지배. length는 verbosity 가드.
- 코드: `configs/training/grpo.yaml rewards` 주석, `registry.py:144-148 DEFAULT_REWARDS`.

**DR-E4 · DAPO Clip-Higher (epsilon_high)**
- 문제: 대칭 PPO 클립이 희귀 정답 토큰 확률 상승을 억눌러 entropy collapse.
- 선택: 비대칭 클립 `[1−ε_low, 1+ε_high]`, ε_high(0.28)>ε_low(0.2).
- 근거(논문): DAPO(arXiv:2503.14476). 코드: `grpo.yaml`, `grpo_trainer.py`(우리가 ε_high≥ε 가드 추가).

**DR-E5 · DAPO soft overlong / length 보상 (verbosity hacking 차단)**
- 문제: GRPO가 출력을 패딩해 r_edit p_hit을 우연히 올리는 verbosity bias.
- 선택: gold 길이 기준 soft 패널티(자유밴드→선형감쇠→0). over-length만 벌점. + `mask_truncated_completions`.
- 근거(논문): DAPO overlong shaping. 코드: `rewards/length.py:6-53`.

**DR-E6 ★ · 보상 집계 3종: weighted / mo_grpo / hm**
- 문제: 여러 보상을 어떻게 스칼라 advantage로?
- 선택지/선택:
  - `weighted`(Arm0): TRL 기본 가중합→그룹정규화. 단 넓은 range 보상이 silent 지배.
  - `mo_grpo`(Arm1): **축별 z-norm 후 합** → 고분산 축 지배 제거.
  - `hm`: 가중 조화평균(floor) → 한 축 0이면 전체 0(축 trade-off 불가).
- 근거(논문): MO-GRPO(arXiv:2509.22047, 원문 Eq.5 일치 확인).
- 코드: `grpo_trainer.py:201-263`, `mo_grpo_trainer.py`.

**DR-E7 · normalize_rewards (공통 [0,1] 기저)**
- 문제: r_style ∈[-1,1], 나머지 [0,1]. 가중합 전이라 넓은 range가 지배.
- 선택: 모든 보상을 `REWARD_OUTPUT_RANGES`로 [0,1] 리스케일 → weight만 믹싱 노브.
- 코드: `rewards/_utils.py:50-68`, `registry.py:40-51,189-191`.

**DR-E8 · style 보상 = p(target)−p(standard), 방향 플립**
- 선택: 분류기 softmax에서 `p(지역)−p(표준)`. dia2std면 부호 반전.
- 근거: Thank-you-BART sc_loss. 코드: `rewards/style.py:11-42`. (우리가 미지지역 하드 lookup으로 보강)

**DR-E9 ★ · copy_margin (copy-bias 제거)**
- 문제: 강원처럼 gold≈source면 plain chrF가 **복사**를 보상.
- 선택: `chrF(gen,gold)−chrF(gen,source)`. echo는 ~0.5로 붕괴. `chrF(gen,source)`는 gen-의존이라 GRPO 그룹정규화에서 **소거 안 됨**(진짜 패널티). 반면 r_content의 copy-baseline은 상수라 소거 → 일부러 생략.
- 근거(논문): DIA-REFINE(False Success: BLEU 52.5/DFS −0.67), Mind the Style Gap.
- 코드: `rewards/content.py:10-65`, `metrics.py:157-165`.

**DR-E10 · 과교정·collateral 가드 (overcorrection, edit precision/recall)**
- 문제: copy_margin은 gold 초과 드리프트를 막지 못함; r_edit recall은 비-타겟 망가뜨려 게임 가능.
- 선택: overcorrection(gold 편집예산 초과 벌점) + edit_precision(비-타겟 보존) + edit_recall(정확 gold형 등장).
- 코드: `rewards/content.py:68-99`, `rewards/edit.py`.

**DR-E11 · 보상 레지스트리 (config 한 줄로 교체)**
- 선택: `@register_reward` 데코레이터, `{name,weight}` specs → `(funcs,weights)`. 미지 이름 즉시 에러.
- 코드: `rewards/registry.py`.

**DR-E12 · 보상해킹 가드: 구두점 정규화**
- 문제: `갔어,`≠`갔어`로 보면 남아있는 표준어가 "제거됨"으로 오집계 → 보상 부풀림.
- 선택: 양끝 구두점 정규화 후 매칭.
- 코드: `rewards/_utils.py:35-47`.

**DR-E13 ★ · reconstruction 보상 (Dual-RL, proxy-independent, 비용게이트)**
- 문제: 모든 보상이 분류기/eojeol_map에 의존하면 같은 경로로 해킹됨.
- 선택: 역번역 사이클 일관성 `BLEU(backtranslate(gen), source)` — **분류기·eojeol_map 안 씀** → 같은 해킹 경로로 게임 불가. 단 스텝마다 생성 1회 추가 → 비용게이트(미연결, smoke 필요).
- 근거(논문): Dual-RL(Luo et al. 2019). 코드: `rewards/reconstruction.py`.

### F. 평가 · 선택 (정직성의 핵심)

**DR-F1 ★ · proxy-independent 선택만 (J-score/TDR/DFS 금지)**
- 문제: 학습 보상과 겹치는 지표로 체크포인트를 고르면 over-optimization 재유입(순환성).
- 선택: 선택은 `reconstruction_bleu`(역번역, 분류기 무관) + 정성 게이트만. J-score/TDR/DFS는 monitoring only.
- 근거: **우리 추론**(MO-GRPO 논문 주장 아님) — DIA-REFINE False Success + 자체 실측 over-optimization + 도메인(강원 gold≈표준).
- 코드: `metrics.py:14-31 ADR`, `leaderboard.py:8-11`, `metric_registry.py`(group=selection/monitoring).

**DR-F2 · copy_margin을 content 축으로 (chrF 직접 아님)**
- 근거(논문): Mind the Style Gap §4(copy-bias). 코드: `metrics.py:157-165`, `metric_registry`(copy_margin↑).

**DR-F3 ★ · Pareto 프런티어 (단일 랭크 강요 X)**
- 문제: style↔content는 음의 상관(트레이드오프) → 하나의 숫자로 우열 못 매김.
- 선택: (copy_margin↑ × reconstruction_bleu↑) Pareto 프런티어 보고. ★로 표시.
- 근거(논문): TST 트레이드오프 — 서베이 arXiv:2010.12742(Hu 2022), 음의 상관 arXiv:2312.14708(Mukherjee 2022), 직접보상 arXiv:2010.12771.
- 코드: `leaderboard.py:150-171 pareto_frontier`.

**DR-F4 · harmonic_joint = 우리 통계(요약만)**
- 결정: 조화평균은 불균형을 더 세게 벌점 → 요약용. **선택엔 안 씀**(순환성). 원문(2010.12771)은 가중합/산술평균이라 harmonic은 **우리 것**(PDF §2.3 직접 확인).
- 코드: `leaderboard.py:174-188`.

**DR-F5 ★ · Koehn 2004 paired bootstrap 유의성**
- 문제: 150문장 점추정은 샘플링 노이즈. "37.1 vs 38.4가 유의한가?"
- 선택: 같은 인덱스를 양 시스템에 동시 적용해 B회 resample, 양측 p=2·min(win,1−win). 신뢰 ≥~300문장.
- 근거(논문): Koehn 2004 EMNLP. 코드: `significance.py:99-144`.

**DR-F6 · chrF 우선 (BLEU보다), sacreBLEU 표준화**
- 근거(논문): chrF(Popović 2015)는 문자 n-gram이라 **교착어 한국어**(`먹었니`vs`먹었어`)에서 부분점수 → 적합. sacreBLEU(Post 2018)로 토크나이즈 표준화.
- 코드: `metrics.py:126-154`.

**DR-F7 · metric_registry = 지표 방향 단일 출처**
- 문제: "copy_margin 낮으면 좋은가?" 반복 혼동.
- 선택: 모든 지표에 ↑/↓·그룹·의미·주의 등록. `orient()`로 호출부가 방향 신경 안 씀.
- 코드: `metric_registry.py`.

**DR-F8 · report.py anti-drift + multi 스키마**
- 문제: 손으로 숫자 베끼면 stale.
- 선택: `eval_logs/*.json`→`RESULTS.md` 자동 생성. multi/v1 인식 + 옛 flat supersede.
- 코드: `evaluation/report.py`, `scripts/report.py`(우리가 H1 + supersede 수정).

---

## 3. 핵심 서사 — "보상해킹과의 싸움" (포트폴리오 하이라이트)

면접에서 이 한 흐름을 말할 수 있으면 깊이가 증명됩니다:

1. **가설**: 작은 모델 → LLM-judge 대신 싼 TextCNN 분류기를 GRPO 보상으로. (DR-D1)
2. **함정**: 분류기는 hackable(max-pool 포화) — 출력을 과장하면 보상이 오름. (DR-D1 트레이드오프)
3. **증거**: 500-step 실측 → 보상↑ 인데 chrF/BLEU −9~11pt. 즉 정책이 **gold에서 멀어지며 프록시만 게임**. (DR-E2)
4. **대응 3종**:
   - KL 앵커 강화(beta 0.04→0.1) — SFT 근처 유지. (DR-E2)
   - 보상 재설계 — style 디모트 + content/edit/overcorrection/length 가드 + copy_margin. (DR-E3,E5,E9,E10)
   - 집계 개선 — MO-GRPO로 고분산 축 지배 제거. (DR-E6)
5. **메타 원칙**: 선택을 **proxy-independent 신호로만**(reconstruction_bleu) — 평가가 보상과 같은 프록시면 over-optimization을 못 잡음. (DR-F1)
6. **정직한 결론**: GRPO 우위는 **방언 의존적**(강원=SFT, 경상=GRPO), `grpo_500`이 가장 robust. Pareto+Koehn로 "단일 승자 없음"을 정량 증명. (DR-F3,F5)

**한 줄**: "작은 GPU에서 reward-hacking을 측정으로 발견하고, 보상·집계·선택 세 층위에서 막은 프로젝트."

---

## 4. 논문 Appendix (전부, 충실히)

원문 검증·인용 정정 전체는 [`REFERENCES.md`](REFERENCES.md). 여기선 **무엇을 가져왔나** 중심 요약.

| 논문 | id | 핵심 주장 | 이 repo에서 | 카드 |
|---|---|---|---|---|
| **DIA-REFINE** (Park et al., LREC 2026) | 2511.06680 | TDR/DFS 지표, n-gram이 source-copy 보상(False Success) | Levenshtein 필터, copy_margin, TDR/DFS | B3,E9,F1 |
| **MO-GRPO** (Ichihara et al. 2025) | 2509.22047 | 축별 z-norm-then-sum로 고분산 축 지배 제거 | Arm1 집계 | E6 |
| **Mind the Style Gap** (Pauli, Augenstein & Assent, EMNLP'25) | 2502.15022 | 지표가 style을 content에서 분리 못함; 동급 LLM judge<style-aware 지표 | copy_margin 동기, TextCNN 보상 정당화 | D1,E9,F2 |
| **TST Direct Rewards** (Liu/Neubig/Wieting, NAACL'21) | 2010.12771 | 직접보상 TST(style 분류기+SIM 콘텐츠), 가중합/산술평균 선택 | 보상 분리의 조상; harmonic은 우리 것 | E3,F4 |
| **TST Review** (Hu et al., KDD'22) | 2010.12742 | TST 서베이/19종 벤치, style↔content 긴장 | Pareto 정당화(서베이) | F3 |
| **Polarity-aware Denoising** (Mukherjee et al. 2022) | 2312.14708 | style-content 음의 상관 실증 | "음의 상관"의 정확한 출처 | F3 |
| **Dual-RL** (Luo et al., IJCAI'19) | 1905.10060 | 사이클 일관성/역번역 콘텐츠 신호 | reconstruction 보상 + reconstruction_bleu 선택 | E13,F1 |
| **Koehn 2004** (EMNLP) | — | paired bootstrap 유의성(≥300) | leaderboard 유의성 | F5 |
| **chrF** (Popović 2015 / WMT) | — | 문자 n-gram F, 교착어 적합 | content/copy_margin | F6 |
| **BLEU/sacreBLEU** (Papineni'02 / Post'18) | 1804.08771 | n-gram 정밀도+BP / 토크나이즈 표준화 | reconstruction_bleu, 재현성 | F6 |
| **TextCNN** (Kim 2014, EMNLP) | 1408.5882 | conv+max-pool 문장분류 | 보상 모델 아키텍처 | D1 |
| **K-ToBI** (Jun 2000) | — | 한국어 IP 경계톤(H%/L%...) | 운율 마커 스킴 | C1 |
| **GRPO** (DeepSeekMath/R1) | 2402.03300 | critic 없는 그룹 상대 advantage | Stage3 알고리즘 | E1 |
| **DAPO** (Yu et al. 2025) | 2503.14476 | Clip-Higher, soft overlong | ε_high, length 보상 | E4,E5 |
| **LoRA** (Hu et al. ICLR'22) | 2106.09685 | 저랭크 어댑터(base frozen) | r16/α32 어댑터 | A2,A3 |
| **QLoRA/NF4** (Dettmers NeurIPS'23) | 2305.14314 | 4bit NF4 + LoRA | bnb 백엔드 | A2 |
| **AWQ / GPTQ** | 2306.00978 / 2210.17323 | 활성인지 / 2차정보 PTQ | export 옵션 | A3 |

> ⚠️ 인용 정정 4건(Mind the Style Gap 저자, LLM-judge 주장, 2010.12742 서베이, harmonic 귀속)과
> MO-GRPO "proxy-independent 선택=우리 추론"은 [`REFERENCES.md`](REFERENCES.md)에 상세.

---

## 5. 코드 ↔ 결정 ↔ 논문 교차 인덱스

| 파일 | 결정 | 논문 |
|---|---|---|
| `data/labels.py` | B1,B2 | — |
| `data/filtering.py` | B3,B4,B5 | DIA-REFINE |
| `data/dataset.py` | B6,B7,B11,B12 | DIA-REFINE |
| `data/prosody.py` | C1,C2,C3 | K-ToBI(Jun 2000) |
| `data/template.py` | (DO_NAME 5지역=H2 수정) | — |
| `scripts/prepare_data.py` | B8,B9,B10 | — |
| `models/classifier.py` | D1,D2 | TextCNN |
| `models/loading.py` | A1,A2,A3,D3 | LoRA, QLoRA, AWQ/GPTQ |
| `training/sft_trainer.py` | D2,D3,D5,B11 | — |
| `training/classifier_trainer.py` | D4 | — |
| `training/grpo_trainer.py` | E1,E2,E4,E5,E6,E7 | GRPO, DAPO, MO-GRPO |
| `training/mo_grpo_trainer.py` | E6 | MO-GRPO |
| `rewards/style.py` | E8 | DIA-REFINE |
| `rewards/content.py` | E9,E10 | DIA-REFINE, Mind the Style Gap |
| `rewards/edit.py` | E10 | Dual-RL |
| `rewards/length.py` | E5 | DAPO |
| `rewards/reconstruction.py` | E13 | Dual-RL |
| `rewards/registry.py` `_utils.py` | E7,E11,E12 | — |
| `evaluation/metrics.py` | F1,F2,F6 | DIA-REFINE, Mind the Style Gap, Dual-RL |
| `evaluation/significance.py` | F5 | Koehn 2004 |
| `evaluation/leaderboard.py` | F1,F3,F4 | MO-GRPO, TST trade-off |
| `evaluation/metric_registry.py` | F7 | — |
| `evaluation/report.py` | F8 | — |

---

## 6. 면접 대비 Q&A (포트폴리오용)

**Q. 왜 GRPO인가요? PPO 아니고?**
A. 6GB에서 PPO의 critic(value net)이 VRAM을 2배로 쓰는데, GRPO는 그룹 상대 advantage로 critic을 없애 ≈절반만 씁니다. (DR-E1)

**Q. 보상 모델로 왜 LLM-judge가 아니라 작은 CNN을?**
A. 비용/메모리 + Mind the Style Gap이 "동급 LLM 오토레이터가 style-aware 지표보다 못함"을 보입니다. TextCNN은 macro-F1 0.95로 충분히 분리하고 쌉니다. 다만 max-pool 포화로 hackable한 트레이드오프가 있어, 그걸 가드로 막았습니다. (DR-D1)

**Q. reward hacking을 어떻게 발견하고 막았나요?**
A. 500-step 실측에서 보상↑인데 chrF/BLEU −9~11pt로 떨어지는 걸 측정. (1) KL 앵커 강화, (2) style 디모트+content/edit/overcorrection/length 가드+copy_margin, (3) MO-GRPO 집계, (4) 선택을 proxy-independent(reconstruction_bleu)로. (3장 서사)

**Q. 평가에서 가장 신경 쓴 점은?**
A. 학습 보상과 겹치는 지표(J-score/TDR)로 선택하면 over-optimization을 못 잡습니다. 그래서 분류기를 안 쓰는 reconstruction_bleu로만 선택하고, style↔content 트레이드오프는 Pareto+Koehn 유의성으로 "단일 승자 없음"을 정량화했습니다. (DR-F1,F3,F5)

**Q. 하드웨어 제약이 설계에 어떻게 들어갔나요?**
A. Turing은 BF16이 없고 4bit 커널이 느려서 fp16 + 16-bit LoRA(4bit 대비 44×)를 택했고, 이걸 `resolve_dtype`로 코드에서 강제합니다. (DR-A1,A2)

**Q. copy_margin이 정확히 뭔가요? 왜 필요?**
A. `chrF(gen,gold)−chrF(gen,source)`. 강원처럼 gold≈source면 plain chrF가 입력 복사를 보상하는데, source 유사도를 빼면 echo가 0.5로 붕괴해 진짜 변환만 보상됩니다. (DR-E9)

---

*이 문서는 코드 26유닛 정독 + 6리뷰어 감사 + 논문 원문 검증(REFERENCES.md)을 의사결정 관점으로 재구성한 것입니다. 각 카드의 file:line은 작성 시점 기준 — 변경 시 검증 요망.*
