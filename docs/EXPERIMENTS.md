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

## 3. Next: quantization (PTQ / QAT) — planned

Backends already scaffolded in `models/loading.py` (`bnb`, `loftq`, `awq`, `gptq`, `qat`)
and `scripts/{ptq_quantize,export_gguf}.py`, but not yet measured. Plan: quantize the
chosen model, then quantify the quality/latency/size trade-off with the same leaderboard +
`bench_serving.py` harness. PTQ first (cheap), escalate to QAT only if PTQ degrades too much.
