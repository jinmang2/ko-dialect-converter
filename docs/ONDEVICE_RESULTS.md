# On-Device 실측 결과 (KoDialect → Galaxy S25 Ultra)

> ⚠️ **스켈레톤** — 폰 실측 전. 모든 수치는 `PREDICTED`. 실측 후 `MEASURED` 로 교체.
> 표 수치는 항상 `llama-bench` median (≥5회, 워밍업 1회 버림). 데모 수치는 체감/녹화용.

## 1. 환경

| 항목 | 값 |
|---|---|
| Device | Galaxy S25 Ultra (SM-S938N) · Snapdragon 8 Elite (SM8750-AC) |
| Runtime | llama.cpp (Termux, ARM NEON/dotprod/i8mm) · CPU |
| Model | Qwen2.5-0.5B-Instruct, QLoRA SFT → merge (`sft_merged`) → GGUF |
| Threads | 6 (perf cores) |
| 평가셋 | 강원/경상 각 n=150, 고정 인덱스 (데스크탑과 동일, dia→std) |

## 2. variant × (size / speed / quality) — BRIEF §9 6축

| variant | on-disk MB | decode tok/s | prefill tok/s | TTFT ms | peak RSS MB | chrF | recon_bleu | tag |
|---------|-----------:|-------------:|--------------:|--------:|------------:|-----:|-----------:|-----|
| Q8_0    | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | PREDICTED |
| Q5_K_M  | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | PREDICTED |
| Q4_K_M  | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | PREDICTED |
| Q4_K_S  | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | PREDICTED |
| Q3_K_M  | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | PREDICTED (옵션) |
| IQ4_XS  | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | PREDICTED (옵션) |

> 채움 명령: `python ondevice/report/pareto.py` (표 + 프런티어) ·
> `python ondevice/report/plots.py` (Pareto 산점도 + prefill/decode bar).

## 3. Pareto 프런티어

- 축: `decode_tok_s↑ × recon_bleu↑ × ondisk_mb↓` (지배당하지 않은 variant 집합).
- 그림: `ondevice/report/figs/pareto.{png,svg}`, `ondevice/report/figs/prefill_decode.{png,svg}`.
- _프런티어 결과 TBD._

## 4. 부호 뒤집힘 가설 (핵심)

- **예측(PREDICTED)**: 모바일 ARM은 decode가 memory-bandwidth bound → **Q4 ≥ Q8/fp16 decode 속도**.
- **데스크탑 대조(MEASURED, §8.3)**: RTX 2060 Turing 에선 NF4가 fp16의 ~0.5× (dequant 커널 부재).
- **폰 실측 결과**: _TBD — 맞든 틀리든 기록. 뒤집히면 "커널 지원이 속도를 지배"를 반대 HW에서 재입증._

## 5. 품질 동치성 (양자화 손실)

- 데스크탑 fp16 출력 vs 폰 GGUF 출력, 동일 고정셋·동일 디코딩(greedy/seed=0).
- chrF/recon_bleu Δ: _TBD (양자화로 품질 얼마 깎였나 1줄)._

## 6. Tier2 (후속, 미수행) — KV cache 관리

> 미수행분은 "예측 + 근거"로만. 한 것처럼 쓰지 않음 (§10).

- **Exp A** prefix cache ON/OFF → 긴 sys prompt TTFT 절감(amortization). _TBD._
- **Exp B** KV cache `f16 / q8_0 / q4_0` (`-fa` on, K/V 타입 매칭) → peak RSS × max ctx × 품질.
  킬러: 문헌은 "KV quant 손실 작다"지만 **0.6B급 소형은 저하 큼** → 본인 0.5B에서 실측 확정. _TBD._
- **Exp C** KV/token vs context length 곡선 (GQA 7× 효과). 앵커: ≈12 KB/tok → 4K≈50MB, 32K≈0.39GB. _TBD._

## 7. 재현 (1줄)

```bash
# (폰) 시동: ondevice/serve/launch_server.sh -m models/kodialect-Q4_K_M.gguf
# (폰) 속도: ondevice/bench/run_bench.sh -m models/kodialect-Q4_K_M.gguf -v Q4_K_M
# (데스크탑) 품질: python ondevice/eval/gen_capture.py --variant Q4_K_M && \
#                  python ondevice/eval/score_offline.py --variant Q4_K_M
# (데스크탑) 리포트: python ondevice/report/pareto.py && python ondevice/report/plots.py
```
