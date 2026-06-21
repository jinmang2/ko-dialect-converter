# 음성 도입 기술 딥리서치 (2024–2026) — KoDialect / 6GB RTX 2060 제약

> 텍스트 old_dialect 5개 도 확장은 폐기(품질 2~4% 신호). 음성은 AI-Hub 원천 오디오를
> 받을 수 있으므로 그쪽으로 전환. 본 문서는 2024–2026 최신 + **6GB Turing(fp16, BF16 불가)**
> 에 올라가는 경량 모델 위주 조사.

---

## 0. 핵심 재구성 — 아키텍처를 먼저 고른다

방언 음성을 다루는 두 가지 구조:

- **(A) 2-stage**: 방언 ASR(오디오→방언 텍스트) → 기존 GRPO 텍스트 모델(방언→표준).
  기존 자산(SFT/GRPO/leaderboard) 재사용 가능. 단 오차 전파(ASR 오류가 텍스트 단계로).
- **(B) Direct / speech-translation**: **방언 오디오 → 표준 텍스트**를 한 모델로.
  - 선례가 강함: **Swiss German STT**는 스위스독일어 음성 → *표준 독일어 텍스트*를
    "전사+번역 1-step" speech-translation으로 정식화([arXiv:2412.15726]). Darija→표준아랍어
    Whisper FT(BLEU 0.65), **BanglaDialecto = end-to-end 지역 음성 표준화**([arXiv:2411.10879]).
  - **AI-Hub 방언 발화는 (오디오, dialect_form, standard_form) 트리플** 제공 →
    Whisper를 `오디오 → standard_form`으로 바로 파인튜닝하면 **프로젝트 전체가 단일 모델로 축약**
    되고, old 텍스트 데이터 품질 문제를 통째로 우회.

> **권고**: (B)를 1차 실험으로. (A)는 기존 GRPO 텍스트 모델을 살리고 싶을 때의 백업.

---

## 1. 경량 ASR 지형 2024–2026 (백본 후보)

| 모델 | 파라미터 | VRAM(추론) | 한국어 | Turing/fp16 | 비고 |
|---|---|---|---|---|---|
| **Whisper large-v3-turbo** | 809M | fp16 ~6GB / **int8 1.6GB** | ✅ 99개 | ✅ | 속도·정확도 sweet spot, 디코더 4층 |
| **faster-whisper(CTranslate2)** | (turbo/large 변환) | int8 ~1.6GB | ✅ | ✅ fp16+int8 | PyTorch 대비 4× 빠름·VRAM 50%↓, **2060 배포 1순위** |
| distil-whisper large-v3 | 756M | ~5GB | ❌ 영어전용 | ✅ | 한국어 불가 → 제외 |
| Parakeet TDT 0.6/1.1B | 0.6–1.1B | ~4GB | △(RNNT-multilingual 변형만 ko) | ✅ NeMo | RTFx>2000 초고속, 한국어는 별도 multilingual 체크포인트 |
| Canary-1B-v2 | 1B | 보통 | ❌ (25개 유럽어, **한국어 없음**) | — | 한국어 미지원 → 제외 |
| **Qwen3-ASR 0.6B / 1.7B** (2026) | 0.6/1.7B | 1.7B fp16 ~3.4GB | ✅(1.7B 명시, 52개) | ✅ | 오픈 SOTA, Qwen3-Omni 기반. 1.7B이 6GB에 들어감 |
| Moonshine | 27M | 매우 낮음 | ❌(영어 edge) | ✅ | 초경량 on-device, 한국어 약함 |
| Samba-ASR (SSM/Mamba, 2025) | — | — | — | — | 신흥 구조, 한국어 검증 미흡 |
| wav2vec2 XLS-R | 0.3–1B | 중 | (FT 필요) | ✅ | encoder-only, **방언 식별/파인튜닝**에 적합 |

**Turing 메모**: 위 후보 전부 fp16로 동작(BF16 불필요). faster-whisper/CTranslate2는 SM7.5에서
fp16·int8 모두 정상. → CLAUDE.md의 "BF16 금지" 제약과 충돌 없음.

**1순위 백본**: `Whisper large-v3-turbo` (한국어 O, 809M, LoRA FT가 6GB에 들어감),
배포는 `faster-whisper int8`(~1.6GB). 차순위로 신상 `Qwen3-ASR-1.7B`.

---

## 2. 한국어 현실 (가장 중요한 레슨)

- Whisper 사전학습이 **~83% 영어** → 바닐라 한국어 성능 약함. 한국어 파인튜닝이 큰 이득
  (KsponSpeech FT에서 CER **−4.5%p**, ~1000h로 유의미 개선) [eksss pss-15-3-83].
- **한국어 토크나이저가 단일 최대 레버**: ENERZAi *EZWhisper*가 커스텀 한국어 토크나이저만으로
  Whisper-Small CER **18.05%→6.45%** [edge-ai-vision 2025-11].
- **극단 압축 가능**: EZWhisper 1.58-bit QAT, **70MB 모델이 Whisper-large-v3를 한국어에서 능가**,
  484MB는 large 대비 CER 절반. → 2060은 물론 CPU/edge 배포까지 현실적.
- 방언: Jeju 방언 Whisper FT 시도 존재; **한국어 방언 식별 wav2vec2 XLS-R** 연구 있음
  [eksss pss-17-3-83] → 텍스트 TextCNN 스타일 분류기를 **오디오 기반 방언 분류기**로
  대체/보강하는 선택지.

---

## 3. 6GB 2060에서의 파인튜닝 (기존 QLoRA 노하우 재사용)

- **Whisper-large PEFT/LoRA가 8GB 미만에서 풀파인튜닝급 성능** [Vaibhavs10/fast-whisper-finetuning].
  turbo(809M)+LoRA는 6GB fp16에 여유. 극한이면 QLoRA(느리지만 거의 무손실).
- 기존 프로젝트의 Unsloth/PEFT·bnb·fp16 셋업과 동일 근육 → 진입장벽 낮음.
- 레시피: AI-Hub(오디오→`standard_form`)로 Whisper-turbo+LoRA를 SFT처럼 단계 학습.

---

## 4. 실제 오디오 기반 운율 (IDEAS.md JSON 마커 대체/검증)

현재 `data/prosody.py`의 `<UP>/<DOWN>/<KEEP>` 마커는 AI-Hub JSON의 intonation 메타에서 추정.
오디오를 받으면 **실측 F0**로 검증/재생성 가능:

| F0 추출기 | 파라미터 | 특징 |
|---|---|---|
| **PESTO** | **0.13M** | 초경량·실시간, 2060/CPU 적합 |
| **FCPE** | 10.6M | Lynx-Net(depthwise-sep conv), 빠르고 noise-robust |
| CREPE / torchcrepe | 22M | 표준 CNN 피치 트래커 |
| RMVPE | 90M | 가장 robust(SNR 전구간)하나 무거움 |

- ProMode(2025, [arXiv:2508.09389]) = 음향+텍스트 조건 운율 모델.
- 활용: 기존 마커를 PESTO/FCPE 실측 F0와 대조 검증 → IDEAS.md 운율 토큰화의 신뢰도 확보.
  (prosody-sft-ab가 recon_bleu만 올리고 dialectness는 평탄했던 원인 진단에도 도움)

---

## 5. (선택) TTS 백엔드

- **Kokoro-82M** (Apache, 한국어 지원, <0.3s, edge) — 경량 기본값 1순위.
- F5-TTS — voice cloning/표현력 우수하나 diffusion이라 무거움.
- **한국어 사투리 TTS는 오픈 기성품 부재** → 직접 FT 필요(연구 영역).

---

## 6. 음성번역 파운데이션 (Whisper 대안/교사 모델)

- **SeamlessM4T-v2-large**: 오디오→텍스트 번역 ~100개 언어, UnitY2. 강력하나 ~2.3B+로
  **배포 타깃으로는 무거움**. 강한 baseline/distillation 교사로만 권장.

---

## 7. KoDialect 구체 권고 (실행 순서)

1. **AI-Hub 방언 발화 오디오 다운로드** (지역당 3000+시간, 화자 2000+; JSON/전사는 이미 보유).
   `scripts/download_from_aihub.sh` 확장.
2. **1차 실험 — Direct 표준화 ASR**: `Whisper-large-v3-turbo + LoRA`로
   **방언 오디오 → `standard_form`** 직접 매핑(Swiss-German 방식). 배포는 `faster-whisper int8`
   (~1.6GB, 2060에 큰 여유).
   - ⚠️ **한국어 토크나이저 품질이 최대 변수**(ENERZAi 교훈): CER 면밀 평가, 필요시 한국어
     토크나이저 보강.
3. **2차 — 오디오 운율**: `PESTO`/`FCPE`로 IDEAS.md 마커 검증·재생성(2060에서 가볍게).
4. **방언 식별**: TextCNN 스타일 보상을 `wav2vec2 XLS-R` 오디오 방언 분류기로 대체/보강(한국어 선례 有).
5. **평가**: 기존 reconstruction/chrF·leaderboard 프레임 재사용(이제 audio→standard 기준).

**제약 요약**: 위 전부 fp16로 Turing 동작. turbo/int8·PESTO·FCPE는 6GB 내 충분, turbo+LoRA FT도
6GB 적합. **Canary(한국어X)·풀 SeamlessM4T/large는 배포 타깃에서 제외.**

---

## 출처 (2024–2026 중심)

- ASR 경량 SOTA: Canary-1B-v2 & Parakeet-TDT-0.6B-v3 [arXiv:2509.14128]; Northflank 2026 STT 벤치;
  Samba-ASR [arXiv:2501.02832]; Moonshine [arXiv:2509.02523].
- Qwen3-ASR (0.6B/1.7B, 한국어 지원) [arXiv:2601.21337], [HF: Qwen/Qwen3-ASR-1.7B].
- faster-whisper/CTranslate2 int8, turbo 1.6GB [HF: Zoont/faster-whisper-large-v3-turbo-int8-ct2].
- 한국어 Whisper: KsponSpeech FT [eksss pss-15-3-83 / pss-15-3-75]; ENERZAi 저비트 EZWhisper
  [edge-ai-vision 2025-11]; TTS-합성 데이터 self-refining [arXiv:2506.11130].
- 방언→표준 speech translation: Swiss German [arXiv:2412.15726]; Darija→MSA Whisper FT;
  BanglaDialecto end-to-end 표준화 [arXiv:2411.10879].
- 한국어 방언 식별 wav2vec2 XLS-R [eksss pss-17-3-83].
- Whisper LoRA/PEFT <8GB [github: Vaibhavs10/fast-whisper-finetuning]; QLoRA 소비자 GPU [arXiv:2509.12229].
- 운율 F0: RMVPE [arXiv:2306.15412]; CREPE; FCPE; PESTO; ProMode [arXiv:2508.09389].
- TTS: Kokoro-82M; F5-TTS.
- 데이터: AI-Hub 한국어 방언 발화(경상도 dataSetSn=119 등), 5개 도·3000+h·표준/방언 페어.
