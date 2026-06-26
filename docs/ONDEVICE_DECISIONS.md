# On-Device 의사결정 카드 (KoDialect → Galaxy S25 Ultra)

> 가치는 "폰에서 돌렸다"가 아니라 **측정 기반 경량화/추론 의사결정**이다.
> (Pareto · prefill/decode 분해 · 부호-뒤집힘 서사). BRIEF §7 대응.

| ID | 결정 | 근거 | 대안 / 폐기한 것 |
|----|------|------|------------------|
| **O1** | **측정 선행, 세일즈 금지** | 0.5B가 "돌아간다"는 자명. 차별점은 Pareto·분해·부호뒤집힘 같은 **측정 깊이**. | "폰에서 LLM 구동!" 마케팅 — 임팩트 약함(§10). |
| **O2** | **k-quant only (Tier0/1)** | GGUF k-quant 가 모바일 ARM에 최적화된 llama.cpp 커널(NEON/dotprod/i8mm)을 가짐. | bnb 4bit(NF4) — 데스크탑 §8.3에서 Turing에 dequant 커널 없어 0.5× 느렸음. 모바일 런타임도 부적합. |
| **O3** | **품질 채점 분리: 폰=생성, 데스크탑=채점** | 단일 출처(기존 `evaluation/metrics.py`)로 재현성 유지. 폰에 평가환경 안 만듦. | 폰에서 chrF 계산 — 토크나이저/버전 drift 위험. `adb forward`로 데스크탑이 폰 서버를 구동(생성은 on-device). |
| **O4** | **Tier2 = KV cache 관리 실측** | "긴 sys prompt면 KV cache 어떻게 관리?" 면접 답변과 1:1. prefix 재사용(TTFT) + cache 양자화(메모리/품질) + GQA 구조. | "자동차 RAG 긴-context NPU prefill 데모" — 측정 불가 시나리오라 폐기. 서버 PagedAttention(멀티유저)과 온디바이스(단일유저) 구분 서술. |
| **O5** | **thermal 곡선을 차별점으로** | S25U sustained 약함(Wildlife Extreme 1분 후 ~20% throttle) → 연속추론 곡선이 자동차(장시간) 서사. | 단발 측정만 — throttle 서사 못 살림. (선택 항목) |

## prompt 패리티 (측정 정합성의 핵심)

폰 `/completion` 은 **raw prompt**를 받으므로, 학습/데스크탑 평가와 **바이트 동일한 Qwen2.5 ChatML**을
클라이언트에서 구성한다. 단일 출처는 `src/ko_dialect/data/template.py`:

```
<|im_start|>system
당신은 한국어 방언 변환 전문가입니다.<|im_end|>
<|im_start|>user
다음 {지역} 사투리를 표준어로 바꿔줘:
{방언문장}<|im_end|>
<|im_start|>assistant
```

- `demo/index.html` `buildPrompt()` 와 `eval/make_eval_prompts.py` `_chatml()` 가 이 문자열을 공유.
- 검증: 위 수동 ChatML == `tokenizer.apply_chat_template(..., add_generation_prompt=True)` (MATCH: True).
- 디코딩도 데스크탑과 동일: greedy(temperature=0), seed=0, stop=`["<|im_end|>","<|im_start|>"]`.

## 핵심 가설 (BRIEF §3 Tier1) — MEASURED 로 닫을 것

**부호 뒤집힘**: 데스크탑 RTX 2060(Turing)에선 NF4가 fp16보다 느렸다(dequant 커널 부재).
모바일 ARM에선 llama.cpp k-quant 커널 + decode가 **memory-bandwidth bound**라
**Q4 < fp16/Q8 (더 빠를 것)** 으로 예측(PREDICTED). 결과가 뒤집히면
"커널 지원이 속도를 지배"(§8.4)를 **반대 하드웨어에서 재입증**. 맞든 틀리든 기록.
