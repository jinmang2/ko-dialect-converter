# 음성(speech) 모듈 설계 — 구현 사양

> 배경/후보 비교는 `docs/SPEECH_RESEARCH.md`(2024–2026 딥리서치) 참조. 본 문서는 그것을
> **이 레포에 바로 구현 가능한 결정**으로 좁힌다. 6GB RTX 2060(fp16, BF16 불가) 제약 고정.
> 코드 골격은 `src/ko_dialect/speech/`(US-002)에 최소 stub로 존재.

---

## 1. 아키텍처 결정 (1차 확정)

**채택: Direct 표준화 ASR** — `방언 오디오 → standard_form 텍스트`를 단일 모델로.
- 근거: AI-Hub 방언 발화가 `(오디오, dialect_form, standard_form)` 트리플을 제공 →
  Whisper 디코더 타깃을 `standard_form`으로 두면 전사+표준화가 1-step. 선례 Swiss German→
  표준독일어([arXiv:2412.15726]), BanglaDialecto([arXiv:2411.10879]).
- **이점**: 폐기한 텍스트 경로(old_dialect 품질 2~4%)를 우회. 텍스트 GRPO 모델 없이도 동작.
- **백업(2-stage)**: `방언 오디오 → dialect_form`(순수 ASR) → 기존 GRPO 텍스트 모델.
  기존 SFT/GRPO 자산을 살리고 싶을 때만. 오차 전파 단점.
- 백본: **Whisper-large-v3-turbo(809M, 한국어 O)** + LoRA. 차순위 Qwen3-ASR-1.7B.
  **Canary는 한국어 미지원 → 제외.**

> 한 raw 샘플로 **두 타깃**(standard / dialect) 학습쌍을 만들 수 있어, 기존
> `both_directions` SFT 설계와 같은 정신으로 direct/2-stage를 한 데이터에서 모두 실험 가능.

---

## 2. 데이터 파이프라인 (기존 자산 재사용)

### 2.1 오디오↔텍스트 페어링 — 이미 파싱된 타임스탬프 활용
`scripts/prepare_data.py`가 **이미** 문장/세그먼트 시간을 파싱한다:
- NEW 포맷: `sentence["startTime"]/["endTime"]`, `segment["_start_seconds"]`(`t2s()`),
  `data["fileName"]` → 원본 오디오 파일명. `id = f"{fileName}_{sentenceId}"`.
- OLD 포맷: `utterance["start"]/["end"]`(앞 조사에서 확인), `id`.

→ **신규 작업은 "오디오 슬라이싱"뿐**: `fileName`으로 오디오 로드 →
`[startTime, endTime]` 구간을 잘라 발화 단위 wav 생성 → 기존 raw 샘플(`id`,`do`,`split`,
`standard`,`dialect`)과 `id`로 조인. 즉 **텍스트 전처리는 그대로 두고 audio manifest만 추가**.

설계: `scripts/prepare_audio.py`(신규, 추후) 또는 `prepare_data.py`에 `--with_audio` 옵션.
산출: 각 raw 샘플에 `audio_path`(슬라이스 wav) 또는 `(source_wav, start_s, end_s)` lazy 슬라이스
컬럼 추가 → HF `datasets` `Audio` feature로 캐스팅.

### 2.2 오디오 포맷
- 16kHz mono PCM(Whisper 표준). torchaudio/soundfile로 리샘플.
- lazy 슬라이스 권장(원본 보관, 학습 시 on-the-fly cut) → 디스크 폭증 방지
  (old 텍스트가 11.5M rows였듯 오디오 풀슬라이스는 TB급).

### 2.3 split / 품질 게이트
- 기존 `split`(train/valid) 재사용.
- direct 학습은 `standard_form`이 타깃이라 **old 텍스트 품질 문제(near-identity)와 무관** —
  오디오→표준은 발화가 표준어와 같아도 유효한 학습쌍. (단 dialectness 학습은 약함 → 평가에서 구분)
- new(139-1, 낭독)와 old(자유발화) 오디오는 **별도 데이터셋**(포맷·성격 상이, 혼합 금지는
  텍스트와 동일 원칙). 우선순위는 데이터 품질 좋은 제주·강원(조사 결과) 또는 new 낭독.

---

## 3. 학습 레시피 (6GB RTX 2060, fp16)

| 항목 | 값 | 비고 |
|---|---|---|
| 백본 | `openai/whisper-large-v3-turbo` | 809M, 디코더 4층, 한국어 O |
| 방식 | LoRA (PEFT) | full FT 불요; turbo+LoRA <6GB ([Vaibhavs10/fast-whisper-finetuning]) |
| LoRA 타깃 | `q_proj,v_proj`(+`k_proj,out_proj` 여유 시) | rank 16~32, alpha 2×rank |
| precision | **fp16**(`fp16=True`) | **BF16 금지**(Turing). 기존 CLAUDE.md 규약 |
| batch | 8~16 (grad-accum로 유효 32~64) | 30s 청크 기준 6GB에 맞춤 |
| 타깃 텍스트 | `standard_form`(direct) / `dialect_form`(2-stage) | 동일 데이터, 옵션 스위치 |
| 옵티마 | 기존 SFT와 동일(AdamW, cosine) | `configs/training/` 패턴 재사용 |

**⚠️ 최대 리스크 — 한국어 토크나이저**: Whisper는 ~83% 영어 학습이라 한국어 CER이 약하고,
ENERZAi가 **커스텀 한국어 토크나이저만으로 CER 18.05→6.45%**를 보였다([edge-ai-vision 2025-11]).
→ (a) 먼저 바닐라 turbo 한국어 CER 베이스라인 측정, (b) 필요 시 한국어 토크나이저/사전 보강 검토.
표준화 타깃이라 출력 토크나이즈가 특히 중요.

---

## 4. 서빙

- **faster-whisper(CTranslate2) int8** — turbo를 ~1.6GB로 적재, PyTorch 대비 4× 빠름, fp16/int8
  모두 Turing 정상. LoRA 어댑터는 머지 후 CT2 변환(merge→ct2-converter).
- 기존 `scripts/serve_*` 패턴과 합류 가능(별도 `scripts/serve_asr.py` 추후).

---

## 5. 운율(prosody) 통합

`data/prosody.py`의 `<UP>/<DOWN>/<KEEP>` 마커는 현재 AI-Hub JSON intonation 메타에서 **추정**.
오디오 확보 시 **실측 F0로 검증/재생성**:
- **PESTO(0.13M)** 1순위(초경량, 실시간, 2060/CPU), 보조 **FCPE(10.6M)**.
- 절차: 발화 슬라이스 → F0 시계열 → 기존 `summarize_intonation`/`prosody_marker` 규칙으로
  재라벨 → JSON-추정 마커와 일치도 측정. (규칙 함수는 순수 파이썬이나, `ko_dialect.data`
  패키지 init이 torch를 끌어와 호출 시점엔 torch가 로드됨 — import 시점은 가볍게 유지.)
  ([[prosody-sft-ab-verdict]]가 recon_bleu만 올리고
  dialectness 평탄했던 원인 진단에 기여).
- 산출 마커는 direct ASR 타깃에 부착 가능(`standard_form + <UP>` 등, 실험적).

---

## 6. 평가 (기존 프레임 재사용)

- `ko_dialect/evaluation/` + `eval_leaderboard.py` 재활용. 신규 지표:
  - **CER/WER**(Whisper Normalizer 적용 후) — ASR 정확도.
  - **reconstruction-style**: 출력(표준) → (기존 reverse) 비교, copy_margin 아날로그
    (표준 출력이 입력 방언을 얼마나 표준화했는지 vs 단순 전사).
  - per-region + OVERALL, Koehn paired-bootstrap 유의성(기존 `significance.py`).
- 오디오 평가셋은 valid split 슬라이스에서 구성, `n_samples` 기존 관례(≥300 권장).

---

## 7. 레포 통합 지점

```
src/ko_dialect/speech/         # US-002 골격 (config + ASR/prosody 인터페이스, lazy import)
  __init__.py
  config.py                    # SpeechConfig (6GB 기본값)
  asr.py                       # DialectStandardizerASR (faster-whisper lazy)
  prosody_audio.py             # AudioProsodyExtractor (PESTO/FCPE lazy)
configs/speech/default.yaml    # SpeechConfig 미러 (eval/default.yaml 패턴)
scripts/prepare_audio.py       # (추후) 오디오 슬라이싱·매니페스트
scripts/train_asr.py           # (추후) Whisper-turbo+LoRA 학습
scripts/serve_asr.py           # (추후) faster-whisper int8 서빙
pyproject [project.optional-dependencies] speech = [faster-whisper, torchaudio, soundfile]
```
- 의존성은 **opt-in extra `speech`**로 분리(core/CI lean 유지 — 기존 track/eval/vllm extras 패턴).

---

## 8. 단계별 로드맵 & 리스크

| 단계 | 내용 | 차단요인 |
|---|---|---|
| P0 (지금) | 설계 + 골격(US-001/002), CPU 테스트 | 없음 |
| P1 | AI-Hub 오디오 다운로드 + 슬라이싱 매니페스트 | **수동/인증 다운로드(대용량)** — 사용자 |
| P2 | turbo+LoRA direct 학습(소규모 지역부터), 바닐라 CER 베이스라인 | **GPU** — 사용자 |
| P3 | faster-whisper int8 서빙 + leaderboard 평가 | P2 |
| P4 | PESTO/FCPE 운율 검증, (선택) 마커 부착 실험 | P1 |

**리스크**: ①한국어 토크나이저 CER(최대) ②오디오 디스크 용량(lazy 슬라이스로 완화)
③new vs old 오디오 성격 차 ④direct 모델의 dialectness 학습 약화(평가로 모니터).

## 출처
`docs/SPEECH_RESEARCH.md` 및 그 출처 목록(Whisper-turbo/faster-whisper, Qwen3-ASR,
ENERZAi 저비트 한국어, Swiss German·BanglaDialecto direct 표준화, PESTO/FCPE/RMVPE, Kokoro).
