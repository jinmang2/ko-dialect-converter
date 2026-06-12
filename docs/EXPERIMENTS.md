# KoDialect — Experiment Results

Consolidated findings for the standard→dialect conversion model. Each section links to
the reproducing command and the raw result artifacts under `outputs/eval_logs/`.

Pipeline: Stage0 data → Stage1 SFT (QLoRA) → Stage2 TextCNN classifier (GRPO reward) →
Stage3 GRPO. Base `Qwen/Qwen2.5-0.5B-Instruct`, RTX 2060 (Turing, fp16-only).

---

## 1. Does GRPO beat SFT? — cross-run leaderboard

`python scripts/eval_leaderboard.py --all_regions --n 150`
→ `outputs/eval_logs/leaderboard_overall_*.{json,md}`

Ranked by **reconstruction_bleu↑** (proxy-independent; MO-GRPO arXiv:2509.22047). Pareto
frontier over (copy_margin↑ × reconstruction_bleu↑); significance = paired bootstrap vs
SFT (Koehn 2004). All metrics carry a direction arrow — **copy_margin is higher-is-better**
(negative = the output copies the standard input).

| scope | Pareto frontier (★) | note |
|---|---|---|
| gangwondo (gold≈standard) | SFT, grpo_arm2, grpo_500 | SFT #1 on recon; its copying *looks* good where gold≈source |
| gyeongsangdo (distinct) | **grpo_v2, grpo_500** | **SFT off-frontier (dominated)**; GRPO copy_margin crosses positive |
| OVERALL (weighted) | SFT, grpo_500 | grpo_500 recon deficit vs SFT **not significant** (p=.80) |

**Verdict: GRPO's advantage is dialect-dependent — there is no single winner.** On the
subtle Gangwon dialect SFT leads; on the distinct Gyeongsang dialect GRPO clearly wins
(SFT is dominated). **`grpo_500` is the most robust single model** — on the Pareto
frontier in all three scopes, strongest copy_margin/DFS, recon deficit never significant.
The newer MO-GRPO arms (arm1/arm2) are dominated by grpo_500 overall.

Decision tooling: `serve_compare.py` (host SFT+arms together, hot-swap adapters) and
`push_to_hub.py` (ship a chosen model with an evidence-rich card). Don't force one
choice — inspect the frontier per region.

See memory `grpo-crossrun-leaderboard`, `grpo-mo-grpo-arm1-verdict`,
`grpo-reward-over-optimization`.

---

## 2. Does prosody supervision help? — prosody-SFT A/B

`python scripts/eval_prosody_ab.py --control outputs/sft_control_s15 --prosody outputs/sft_prosody_s15 --target_do <region>`
→ `outputs/eval_logs/prosody_ab_*.log`

Matched 15k/1ep SFT arms, identical rows; only difference = F0 markers (`<UP>/<DOWN>/<KEEP>`)
on the dialect side. Markers stripped before scoring. n=150 each.

| metric | gangwondo (ctrl→pros) | gyeongsangdo (ctrl→pros) |
|---|---|---|
| **reconstruction_bleu↑** | 45.5 → **49.4 (+3.9)** ▲ p<.001 | 50.8 → **57.2 (+6.4)** ▲ p<.001 |
| copy_margin↑ | −7.44 → −7.59 | −8.89 → −9.07 |
| tdr↑ / dfs↑ / eojeol↑ | flat | flat |

**Verdict: prosody supervision is a content-fidelity regularizer, not a dialectness lever.**
It significantly raises reconstruction_bleu in both regions while every dialect-conversion
axis stays flat. Predicting F0 markers makes the model translate more faithfully, not more
dialectally.

Caveats: recon_bleu↑ with copy_margin slightly↓ is the copy-bias signature (a source-faithful
output back-translates more easily) — the small copy_margin move suggests it is a minor
contributor, but the qualitative gate is needed to confirm. First signal only: 15k/1ep,
n=150, no `<WAVE>` marker (existing F0 summary lacks f0_range), base+adapter eval.

See memory `prosody-sft-ab-verdict`.

---

## Evaluation methodology (references)

- **DIA-REFINE** arXiv:2511.06680 — TDR + DFS; n-gram metrics reward source-copying → copy_margin.
- **MO-GRPO** arXiv:2509.22047 — reward/proxy circularity; select on proxy-independent signals.
- **Mind the Style Gap** arXiv:2502.15022 — copy_margin / J-score limits.
- **Koehn 2004** (EMNLP) — paired bootstrap significance for chrF/BLEU.
- **TST trade-off** arXiv:2010.12771 / 2010.12742 — style vs content negatively correlated → Pareto + harmonic joint.

---

## 3. Quantization (PTQ) — measured

`python scripts/quantize_eval.py --model_path outputs/sft_merged --target_do gangwondo`
→ `outputs/eval_logs/quantize_tradeoff_*.json`. Trade-off summary is the CPU-tested
`evaluation.serving.quantization_tradeoff` (size%, speedup, chrf_drop vs the fp16 baseline).

sft_merged, gangwon (n=150, confirms an n=24 first signal):

| variant | size | p50 latency | chrF | copy_margin |
|---|---|---|---|---|
| fp16 | 953 MB | 488 ms | 68.5 | −5.09 |
| **bnb 4-bit (NF4)** | **238 MB (25%)** | **985 ms (0.50× — slower)** | 70.7 | −5.72 |

**Verdict: 4-bit PTQ is a memory win, not a latency win on RTX 2060.** It cuts the
footprint 4× with no quality loss (chrF is non-negative at both n=24 and n=150), but is ~2×
*slower* — bitsandbytes 4-bit dequant overhead dominates for a 0.5B model without optimized
Turing kernels. For actual speed, use the **GGUF Q4_K_M** path (llama.cpp,
`scripts/export_gguf.py` + `bench_serving.py`). Escalate to QAT (`training.backend=qat`)
only if PTQ shows real quality loss — it does not here.

---

## 4. Training optimization — DeepSpeed vs unsloth

`scripts/bench_train.py` times a short SFT run → steady-state tokens/sec + peak VRAM.
DeepSpeed ZeRO-2 + optimizer CPU-offload config in `configs/training/deepspeed_zero2_offload.json`
(use `backend=bnb/hf` — unsloth patches the model and does not compose with the engine).

Baseline (unsloth, bs=1, seq=256, 20 steps): **286 tok/s, 1.12 steps/s, peak VRAM 1284 MB**.

**Verdict on this RTX 2060 box: unsloth is the only training path that actually runs.**
The head-to-head arms are *environment*-blocked, not just slower:
- **DeepSpeed** installs but cannot initialize — no CUDA toolkit (`nvcc` absent), which its
  JIT op builder requires. Ready for a machine with the toolkit / multiple GPUs.
- **Vanilla bnb 4-bit LoRA training** trips the Turing `_amp_foreach_..._unscale` BF16 error
  (fp16 grad-scaler vs BF16 adapter params) — the documented reason this project trains with
  unsloth's 16-bit LoRA rather than a 4-bit backbone.

So on a single 6 GB Turing GPU unsloth wins by default. The DeepSpeed config + bench
(`training.bench.compare_throughput`) remain for a CUDA-toolkit/multi-GPU environment.

---

## Prosody marker scheme (v2, K-ToBI-grounded)

`src/ko_dialect/data/prosody.py` markers map to Korean Intonation-Phrase boundary tones
(Jun 2000, *K-ToBI*): `<UP>`=H% / `<DOWN>`=L% (steep fall beyond declination) /
`<WAVE>`=complex contour / `<KEEP>`=level. v2 is declination-aware (Korean F0 declines
naturally) and keys `<WAVE>` on the F0 coefficient of variation (`f0_std/f0_mean`), which
the stored summary carries, so all four classes fire — real 614k-row distribution
**UP 19.4% / KEEP 51.9% / DOWN 22.8% / WAVE 5.7%** (v1 was 60% `<DOWN>` with a dead
`<WAVE>`). Per-eojeol markers (Phase 2) recover word-level F0 by slicing the sentence series
at each segment's time window. v1 frozen + reproducible (`prosody_marker(p, version=1)`).
