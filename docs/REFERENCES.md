# REFERENCES.md — Papers, Techniques & Options Dossier

This file is the **grounding dossier** for KoDialect: every paper, technique, and
non-obvious option the code relies on, cross-referenced to where it is used and
**verified against the primary source** (June 2026).

How to read each entry:
- **Cited in** — `file:line` where the code/docs invoke it, and *what claim it grounds*.
- **What the source actually says** — verified from arXiv / ACL Anthology / official docs.
- **Verification note** — confirmed vs. unconfirmed vs. **corrected** (citation errors we found).

Scope: **core (non-speech) pipeline**. Speech-pivot papers are parked in
[Appendix C](#appendix-c--parked-speech-papers-not-yet-verified) (the speech module is
set aside; those are listed, not yet verified).

> ⚠️ **Citation fixes found during this audit** are collected in
> [§ Citation discrepancies](#citation-discrepancies-action-list). The code currently
> mis-attributes two references; this dossier carries the corrected attribution.

---

## A. Papers grounding the core code

### A1. DIA-REFINE — TDR / DFS metrics, copy-bias, data filter

- **Title:** *Steering LLMs toward Korean Local Speech: Iterative Refinement Framework for Faithful Dialect Translation* (DIA-REFINE)
- **Authors / venue:** Keunhyeung Park, Seunguk Yu, Youngbin Kim — LREC 2026 — **arXiv:2511.06680** — https://arxiv.org/abs/2511.06680
- **Framework:** an iterative translation → verification (external dialect classifier) → feedback loop; ICL examples help further.
- **Cited in:**
  - `src/ko_dialect/data/filtering.py:9` — the *True Attempt* vs *False Success* distinction; the **normalized-Levenshtein ≥ 0.1** admission rule for dialect samples.
  - `src/ko_dialect/data/dataset.py:292` — keep the same data boundary in the classifier set.
  - `src/ko_dialect/evaluation/metrics.py:47` (`compute_tdr`) and `:76` (`compute_dfs`).
  - `src/ko_dialect/evaluation/metric_registry.py:52,59`; `leaderboard.py:312`; `README.md:159`.
- **What the source says (verified):**
  - **TDR (Target Dialect Rate)** — classifier-based success rate: fraction of outputs tagged as the target dialect (§5.2). Matches `compute_tdr` exactly.
  - **DFS (Dialect Fidelity Score)** — log-ratio `log((1 + cos(h, ref) + ε) / (1 + cos(h, src) + ε))` (§5.2). Matches `compute_dfs` exactly.
  - **Copy-bias** — a source-copying baseline can score **BLEU 52.54 / DFS −0.672** (Table 1): high n-gram score, negative fidelity. This is *the* motivation for `copy_margin = chrF(gen,gold) − chrF(gen,source)`.
  - **Data filter** — Levenshtein ≥ 0.1 threshold confirmed from §3.1.
- **Verification note:** ✅ All repo claims confirmed against the paper.

### A2. MO-GRPO — per-objective reward normalization

- **Title:** *MO-GRPO: Mitigating Reward Hacking of Group Relative Policy Optimization on Multi-Objective Problems*
- **Authors / venue:** Ichihara, Jinnai, Morimura, Sakamoto, Mitsuhashi, Uchibe (CyberAgentAILab), 2025 — **arXiv:2509.22047** — https://arxiv.org/abs/2509.22047
- **Cited in:**
  - `src/ko_dialect/training/mo_grpo_trainer.py:3,47` — the aggregation method (Arm 1).
  - `src/ko_dialect/training/grpo_trainer.py:69`; `configs/training/grpo.yaml:45`; `configs/experiment/grpo_arm1.yaml:2`.
  - `src/ko_dialect/evaluation/metrics.py:18,188`; `leaderboard.py:8`; selection rationale across eval scripts.
- **What the source says (verified):**
  - **Eq. 5 normalization** — per-objective z-score *within the group*, then sum across objectives. The repo's "per-objective z-norm then sum" description is **accurate**.
  - §5.3 evaluates on a held-out GPT-Eval metric to show the method does not overfit the training rewards.
- **Verification note:** ⚠️ **Partial.** The aggregation claim is confirmed. The repo's other claim — *"select checkpoints on proxy-independent signals only"* — is **NOT an explicit recommendation in the paper**. It is an **editorial inference** the repo draws from MO-GRPO's reward-hacking analysis. Keep it, but label it as our inference, not Ichihara et al.'s prescription.

### A3. Mind the Style Gap — copy_margin / J-score limits

- **Title:** *Mind the Style Gap: **Meta-Evaluation of Style and Attribute Transfer Metrics***
- **Authors / venue:** **Amalie Brogaard Pauli, Isabelle Augenstein, Ira Assent** — **EMNLP Findings 2025** — **arXiv:2502.15022** — https://arxiv.org/abs/2502.15022
- **Cited in:**
  - `src/ko_dialect/rewards/content.py:50,97` and `fluency.py:35` — copy-bias / over-stylization rationale.
  - `src/ko_dialect/evaluation/metrics.py:30,155,263` — `copy_margin` motivation.
  - `metric_registry.py:45`; eval scripts; `CLAUDE.md:80`.
- **What the source actually says (verified directly):** Widely-used style-transfer metrics correlate *surprisingly well* with human judgment, **yet fail to abstract style changes from content** when scoring content preservation (they conflate style edits with content loss). The high correlations are largely an artifact of existing test sets; the authors build a harder test set with high content-preservation variance, argue content metrics must be **style-aware**, and show **small specialized LMs out-perform comparably-sized LLMs as autoraters**.
- **Verification note:** ❌ **Two corrections.**
  1. **Author attribution is wrong in the code.** Comments cite *"Hallinan et al. 2025"*; the paper is **Pauli, Augenstein & Assent**. → fix `content.py:97`, `fluency.py:34`, `CLAUDE.md`.
  2. The claim *"LLM-judge ≤ human on content"* is a **mischaracterization**. The paper's actual finding is *"same-size LLM autoraters underperform style-aware content metrics / smaller specialized LMs"* — not a blanket human-superiority claim.

### A4. On Learning Text Style Transfer with Direct Rewards

- **Title:** *On Learning Text Style Transfer with Direct Rewards*
- **Authors / venue:** **Yixin Liu, Graham Neubig, John Wieting** — **NAACL 2021**, pp. 4262–4273 — **arXiv:2010.12771** — https://arxiv.org/abs/2010.12771 · PDF https://aclanthology.org/2021.naacl-main.337.pdf
- **Cited in:** `src/ko_dialect/evaluation/leaderboard.py:46` (two orthogonal axes), `:177` (`harmonic_joint`).
- **What the source says:** Direct-reward TST training using a **style-classifier reward + a semantic-similarity content reward** (no parallel data required) — the conceptual ancestor of this repo's style + content reward split.
- **Verification note:** ⚠️ The repo attributes a **"harmonic-mean joint style↔content objective" to §3** — this specific *combination formula* (harmonic vs. product vs. weighted sum) is **unverified**. The `harmonic_joint` helper is best described as *our own* summary statistic inspired by this line of work. **TODO:** read §3 of the PDF and either confirm or soften the comment in `leaderboard.py:177`.

### A5. Text Style Transfer: A Review and Experimental Evaluation (the "TST trade-off" cite)

- **Title:** *Text Style Transfer: A Review and Experimental Evaluation*
- **Authors / venue:** **Zhiqiang Hu, Roy Ka-Wei Lee, Charu C. Aggarwal, Aston Zhang** — **KDD Explorations, Vol. 24(1), 2022**, pp. 14–45 — **arXiv:2010.12742** — https://arxiv.org/abs/2010.12742
- **Cited in:** `src/ko_dialect/evaluation/leaderboard.py:47,157` — "style vs content negatively correlated → Pareto frontier".
- **What the source actually is (verified directly):** A **survey + reproducibility benchmark of 19 TST algorithms**. It discusses the style/content tension inherent to TST but is *not* a dedicated study of negative correlation.
- **Verification note:** ❌ **Correction.** The crisp *"negatively correlated → Pareto"* framing is **the repo's own**, not a result of this survey. For the **empirical negative-correlation claim**, cite **Mukherjee, Kasner & Dušek, "Balancing the Style-Content Trade-Off in Sentiment Transfer Using Polarity-Aware Denoising" — arXiv:2312.14708** (note: *sentiment* transfer specifically). Keep 2010.12742 as a general TST review reference.

### A6. Dual-RL — cycle-consistency, the basis of `reconstruction_bleu`

- **Title:** *A Dual Reinforcement Learning Framework for Unsupervised Text Style Transfer*
- **Authors / venue:** Fuli Luo, Peng Li, Jie Zhou, Pengcheng Yang, Baobao Chang, Zhifang Sui, Xu Sun — **IJCAI 2019** (pp. 5116–5122) — **arXiv:1905.10060** — https://arxiv.org/abs/1905.10060 · code https://github.com/luofuli/DualRL
- **Cited in:** `src/ko_dialect/rewards/reconstruction.py:25,54` (reconstruction reward); `src/ko_dialect/evaluation/metrics.py:25` (selection metric).
- **What the source says:** Train forward `f` (dialect→standard) and backward `g` (standard→dialect) jointly; each uses the other as a reward via **cycle-consistency / back-translation** — `sim(g(f(x)), x)` is the content-preservation reward for `f`. No parallel data needed. `R(f,x) = λ_content·sim(g(f(x)),x) + λ_style·P(target|f(x))`, optimized with REINFORCE.
- **Repo use:** `reconstruction_bleu` is the **content-preservation leg** of Dual-RL and is **proxy-independent** (no hackable classifier), so it is the **primary checkpoint-selection metric** (memory: `grpo-reward-over-optimization`).
- **Verification note:** ✅ Confirmed.

### A7. Koehn 2004 — paired bootstrap significance

- **Title:** *Statistical Significance Tests for Machine Translation Evaluation*
- **Authors / venue:** Philipp Koehn — **EMNLP 2004**, pp. 388–395 — https://aclanthology.org/W04-3250/
- **Cited in:** `src/ko_dialect/evaluation/significance.py` (`paired_bootstrap`), `scripts/eval_leaderboard.py:141`, `scripts/eval_prosody_ab.py:9,126`, `model_card.py:125`.
- **What the source says:** Draw B≈1000 pseudo-test-sets of size N by resampling sentence indices **with replacement, jointly for both systems** ("paired"); compute the metric for A and B on each; two-sided p ≈ 2·min(win-frac, loss-frac). Reliable at **N ≥ ~300** sentences. De-facto standard; `sacrebleu --paired-bs` implements exactly this.
- **Verification note:** ✅ Confirmed. **Action:** keep `eval_leaderboard.py --n` at **≥ 300 per region** for trustworthy p-values.

### A8. chrF — Popović 2015

- **Title:** *chrF: character n-gram F-score for automatic MT evaluation*
- **Authors / venue:** Maja Popović — **WMT 2015**, pp. 392–395 — https://aclanthology.org/W15-3049/ (DOI 10.18653/v1/W15-3049)
- **Cited in:** `copy_margin` and chrF columns in `evaluation/metrics.py` / `leaderboard.py`.
- **What the source says:** `F = (1+β²)·P·R / (β²·P + R)` over character n-grams (n=6). β=1 balanced; sacreBLEU `chrF2` uses β=2 (recall-weighted). **Character-level credit is key for Korean** (agglutinative): 먹었니 vs 먹었어 = 0 word overlap for BLEU, but chrF gives partial credit — a far better adequacy proxy for dialect↔standard.
- **Verification note:** ✅ Confirmed.

### A9. BLEU (Papineni 2002) + sacreBLEU (Post 2018)

- **BLEU:** Papineni, Roukos, Ward, Zhu (IBM) — **ACL 2002**, pp. 311–318 — https://aclanthology.org/P02-1040/ — modified n-gram precision (n=1..4, geometric mean) × brevity penalty `BP=exp(1−r/c)`. **Corpus-level**; segment-level BLEU is not meaningful.
- **sacreBLEU:** Matt Post — *A Call for Clarity in Reporting BLEU Scores* — **WMT 2018** — https://aclanthology.org/W18-6319/ · arXiv:1804.08771 — standardizes tokenization (unreported choices cause up to ~1.8 BLEU drift) and emits a reproducible version signature; exposes `--paired-bs` (Koehn 2004).
- **Cited in:** `reconstruction_bleu` in `metrics.py`/`leaderboard.py`; BLEU is secondary to chrF because copy artifacts inflate it (see A1 copy-bias).
- **Verification note:** ✅ Confirmed. **Action:** log the sacreBLEU signature string with every reported number.

### A10. TextCNN — Kim 2014

- **Title:** *Convolutional Neural Networks for Sentence Classification*
- **Authors / venue:** Yoon Kim — **EMNLP 2014**, pp. 1746–1751 — https://aclanthology.org/D14-1181/ · arXiv:1408.5882 · code https://github.com/yoonkim/CNN_sentence
- **Cited in:** `src/ko_dialect/models/classifier.py:76`.
- **What the source says (original defaults):** embedding → parallel conv filters of widths {3,4,5} (100 maps each) → **max-over-time pooling** → dropout(0.5) → softmax. Cheap, single forward pass, variable-length via pooling; competitive on 7 benchmarks.
- **Repo use:** TextCNN serves **two roles** — (a) data-cleaning gate in `prepare_data.py`, and (b) the **GRPO style reward** `p(target_do) − p(standard)` in `rewards/style.py` (macro-F1 ≈ 0.95 after cleaning). ⚠️ Max-over-time pooling is **saturable by repeated trigger n-grams** → this is *the mechanism* behind GRPO reward over-optimization (reward↑ while chrF↓; memory `grpo-reward-over-optimization`).
- **Verification note:** ✅ Confirmed (architecture is the original Kim 2014; the repo adapts hyperparams — check `models/classifier.py` for the actual filter sizes used here).

### A11. K-ToBI — Jun 2000

- **Title:** *K-ToBI (Korean ToBI) Labelling Conventions, v3.0*
- **Author / venue:** Sun-Ah Jun (UCLA) — *Speech Sciences* 7 (2000) 143–169 (v3.1: UCLA WPP 99, 149–173) — https://linguistics.ucla.edu/people/jun/ktobi/k-tobi.html
- **Cited in:** `src/ko_dialect/data/prosody.py:40` (boundary-tone marker scheme), `README.md:135`, `docs/EXPERIMENTS.md:160`.
- **What the source says:** Korean intonation = Accentual Phrase (AP) + Intonation Phrase (IP), on tonal/orthographic/break-index/misc tiers. **IP boundary tones:** H% (question rise), L% (declarative fall), LH%, HL%, … Break indices 0–3 (3 = IP edge). **Declination:** F0 drifts down across an IP, resets at IP boundaries.
- **Repo use:** prosody-marker SFT derives sentence-type tokens — `<UP>`=H%, `<DOWN>`=L%, `<WAVE>`=contour (via F0 CoV), `<KEEP>`=level. Verdict (memory `prosody-sft-ab-verdict`): raises recon_bleu significantly but is a *fidelity regularizer*, not a dialectness lever.
- **Verification note:** ✅ Confirmed. Jun 2014 ("Intonational Phonology of Seoul Korean Revisited") updates AP counts but Jun 2000 remains correct for the IP boundary inventory.

---

## B. Techniques & options (grounded in code, often without an explicit citation)

### B1. GRPO — Group Relative Policy Optimization

- **Source:** DeepSeekMath — **arXiv:2402.03300** — https://arxiv.org/abs/2402.03300 (clearest objective: DeepSeek-R1 — arXiv:2501.12948).
- **Idea:** No critic/value net — estimate the advantage baseline from a group of G sampled completions: `Â_i = (r_i − mean(r)) / std(r)`. Objective = clipped surrogate − `β·D_KL(π_θ‖π_ref)`. Dropping the critic ~halves PPO's VRAM (decisive on RTX 2060).
- **Cited in:** `training/grpo_trainer.py`, `configs/training/grpo.yaml` (`num_generations`, `epsilon`, `beta`).

### B2. KL-to-SFT anchor (`beta`)

- The `β·D_KL(π_θ‖π_ref)` term keeps the policy near the **SFT reference**. Higher β → more conservative, less reward-hacking, lower ceiling gains; lower β → more exploration, more over-optimization risk.
- **Cited in:** `configs/training/grpo.yaml` `beta: 0.1` — *raised 0.04→0.1* after the 500-step run over-optimized the classifier (TDR past gold) while chrF/BLEU vs gold dropped ~9–11 pts. Pairs with `reconstruction_bleu` selection as the two anti-over-optimization guards.

### B3. DAPO — Clip-Higher + soft overlong penalty

- **Source:** *DAPO: An Open-Source LLM RL System at Scale* — Yu et al. (ByteDance Seed) — **arXiv:2503.14476** — https://arxiv.org/abs/2503.14476.
- **Clip-Higher:** asymmetric PPO clip `clip(r, 1−ε_low, 1+ε_high)` with `ε_high > ε_low` → lets rare correct tokens grow, preventing entropy collapse. **Repo:** `epsilon: 0.2`, `epsilon_high: 0.28` in `configs/training/grpo.yaml`.
- **Soft overlong / length penalty:** linear ramp near `L_max` instead of a hard −1 cliff (no gradient at the boundary). **Repo:** the `length` reward + `mask_truncated_completions: true` counter verbosity hacking (completions pinned at `max_new_tokens`).
- **Dynamic sampling / token-level loss** (other DAPO ideas) — not currently wired in; candidate future work.

### B4. LoRA — Low-Rank Adaptation

- **Source:** Hu et al. — **arXiv:2106.09685** — ICLR 2022 — https://arxiv.org/abs/2106.09685.
- **Idea:** freeze W₀; learn `h = W₀x + (α/r)·BAx` (B=0 at init → starts from pretrained). Only A,B carry gradients/optimizer state; merges at inference (zero added latency).
- **Cited in:** `configs/training/grpo.yaml` `lora_r: 16`, `lora_alpha: 32`, `lora_dropout: 0.05`; `apply_lora: true` (8.8M / 512M params trained). `r`=capacity, `α/r`=effective scale, dropout=regularization.

### B5. QLoRA + NF4

- **Source:** Dettmers et al. — **arXiv:2305.14314** — NeurIPS 2023 — https://arxiv.org/abs/2305.14314.
- **Idea:** train LoRA on a **4-bit NF4** frozen base. NF4 = quantile bins of a normal distribution (info-optimal for ~normal weights); **double quantization** (quantize the scales, ~0.5 bit/param saved); **paged optimizers** (page Adam state to CPU under VRAM pressure).
- **Cited in:** `configs/training/grpo.yaml` `load_in_4bit` (default **false** — see B6), `bnb_4bit_quant_type="nf4"`, `bnb_4bit_compute_dtype=float16` (**never bfloat16 on Turing**).

### B6. Unsloth & bitsandbytes — and why 16-bit LoRA on Turing

- **Unsloth** — optimized Triton/CUDA LoRA kernels for small GPUs (software, no canonical paper).
- **bitsandbytes** — provides the 4-bit NF4 / 8-bit quant used by QLoRA (software).
- **2060-specific finding (measured, `configs/training/grpo.yaml`):** on Turing (SM 7.5) there is **no fast 4-bit dequant kernel**, so 4-bit QLoRA runs ~146 s/step vs **~3.4 s/step for 16-bit LoRA (~44×)** at ~2.7 GB peak. → default `backend: unsloth`, `load_in_4bit: false`, `dtype: fp16`. Flip to 4-bit only on Ampere+.
- **Verification note:** ⚠️ The "44×" and the dequant-kernel reasoning are this repo's **own measurements**, not a cited paper. A subagent proposed `arXiv:2604.02556` / `arXiv:2601.14277` as sources — **treated as unverified/likely-spurious and intentionally NOT cited.**

### B7. Export / PTQ options — AWQ, GPTQ, GGUF

- **AWQ** — *Activation-aware Weight Quantization* — **arXiv:2306.00978** — https://arxiv.org/abs/2306.00978 — protects ~1% salient weights via activation-aware scaling; fast 4-bit inference.
- **GPTQ** — **arXiv:2210.17323** — https://arxiv.org/abs/2210.17323 — one-shot PTQ using approximate second-order (Hessian) info.
- **GGUF / llama.cpp** — file format + CPU/edge inference engine (software). **Q4_K_M** = 4-bit K-quant, *medium* tier (mixed bit-widths per tensor for a quality/size balance). This is the project's export target.
- **Cited in:** `src/ko_dialect/export/gguf.py`, `scripts/ptq_quantize.py`, `scripts/export_gguf.py`, `configs/export.yaml`; `backend` enum in `grpo.yaml` lists `awq|gptq` as alternatives.

### B8. Evaluation concepts — Pareto frontier & harmonic mean

- **Pareto frontier** — for two competing axes (dialectness `copy_margin` ↑ vs fidelity `reconstruction_bleu` ↑), a run is Pareto-optimal if no other run beats it on both. The repo **reports the frontier instead of forcing one rank** (`leaderboard.py:148` `pareto_frontier`, `★` flag). Grounded by the style↔content tension (A4/A5).
- **Harmonic mean** — `harmonic_joint` (`leaderboard.py:172`) penalizes imbalance harder than arithmetic/geometric mean (one near-zero axis tanks the score). Used as a **summary** only — **never** for checkpoint selection (same circularity risk as J-score; A2).

---

## Citation discrepancies (action list)

These are **factual citation errors** found during the audit. The dossier above carries the
corrected attribution; the source files still need fixing (proposed, not yet applied):

| # | Where | Current (wrong) | Correct |
|---|---|---|---|
| 1 | `rewards/content.py:97`, `rewards/fluency.py:34`, `CLAUDE.md` | "Mind the Style Gap" = **Hallinan et al. 2025** | **Pauli, Augenstein & Assent**, EMNLP Findings 2025 (arXiv:2502.15022) |
| 2 | `CLAUDE.md:80`, comments | "LLM-judge ≤ human on content" | "same-size LLM autoraters underperform style-aware content metrics" |
| 3 | `leaderboard.py:47,157`, `CLAUDE.md:82` | 2010.12742 ⇒ "negatively correlated" | 2010.12742 is a **survey** (Hu et al. 2022); cite **Mukherjee & Dušek 2022, arXiv:2312.14708** for the negative-correlation claim |
| 4 | `leaderboard.py:177` | "harmonic-mean joint objective (2010.12771 §3)" | combination formula **unverified** — confirm against PDF §3 or soften to "our summary statistic" |

Other non-blocking actions: keep `eval_leaderboard.py --n ≥ 300` (A7); log sacreBLEU signature (A9).

---

## Source index

| Ref | ID / URL | Status |
|---|---|---|
| DIA-REFINE | arXiv:2511.06680 | ✅ verified |
| MO-GRPO | arXiv:2509.22047 | ⚠️ aggregation ✅, selection-claim = our inference |
| Mind the Style Gap | arXiv:2502.15022 | ❌ author + finding corrected |
| TST Direct Rewards | arXiv:2010.12771 | ⚠️ joint-formula unverified |
| TST Review (survey) | arXiv:2010.12742 | ❌ reframed; add arXiv:2312.14708 |
| Dual-RL | arXiv:1905.10060 | ✅ verified |
| Koehn 2004 | aclanthology.org/W04-3250 | ✅ verified |
| chrF | aclanthology.org/W15-3049 | ✅ verified |
| BLEU / sacreBLEU | P02-1040 / W18-6319 | ✅ verified |
| TextCNN | arXiv:1408.5882 | ✅ verified |
| K-ToBI | Jun 2000, UCLA | ✅ verified |
| GRPO | arXiv:2402.03300 / 2501.12948 | ✅ verified |
| DAPO | arXiv:2503.14476 | ✅ verified |
| LoRA | arXiv:2106.09685 | ✅ verified |
| QLoRA / NF4 | arXiv:2305.14314 | ✅ verified |
| AWQ | arXiv:2306.00978 | ✅ verified |
| GPTQ | arXiv:2210.17323 | ✅ verified |
| GGUF / Unsloth / bnb | software (no paper) | ✅ (no fake arXiv id) |

---

## Appendix C — Parked speech papers (NOT yet verified)

The speech module is set aside; these are listed for completeness and have **not** been
verified in this pass. Cited only in `docs/SPEECH_RESEARCH.md` / `docs/SPEECH_MODULE_DESIGN.md`.

| Topic | id |
|---|---|
| RMVPE (F0) | arXiv:2306.15412 |
| ProMode (prosody) | arXiv:2508.09389 |
| BanglaDialecto (e2e dialect normalization) | arXiv:2411.10879 |
| Swiss-German speech translation | arXiv:2412.15726 |
| Samba-ASR | arXiv:2501.02832 |
| Moonshine (ASR) | arXiv:2509.02523 |
| TTS self-refining data | arXiv:2506.11130 |
| QLoRA on consumer GPU | arXiv:2509.12229 |
| Canary / Parakeet ASR | arXiv:2509.14128 |
| Qwen3-ASR | arXiv:2601.21337 |
