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
- **Verification note:** ✅ **Resolved (PDF read directly).** §2.3 / Eq.10: the paper combines its four rewards by a **weighted sum** for training, and selects checkpoints by the **arithmetic mean** of style accuracy + BLEU (a *proxy-dependent* rule we deliberately avoid). **No harmonic mean appears anywhere** ("harmonic" count = 0). → `harmonic_joint` is **entirely our own** statistic; the prior "harmonic-mean joint objective (§3)" attribution was wrong and has been removed from `leaderboard.py:177`. The paper remains the conceptual ancestor of the style+content reward split, nothing more.

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

### A12. HyPoradise — LLM-based generative error correction (grounds the GER task)

- **Title:** *HyPoradise: An Open Baseline for Generative Speech Recognition with Large Language Models*
- **Authors / venue:** Chen Chen, Yuchen Hu, Chao-Han Huck Yang, Sabato Marco Siniscalchi, Pin-Yu Chen, Eng Siong Chng — **NeurIPS 2023, Datasets & Benchmarks Track** — **arXiv:2309.15701** — https://arxiv.org/abs/2309.15701
- **Cited in:**
  - `src/ko_dialect/data/dataset.py::build_ger_dataset` — that "ASR hypothesis → correct transcript" is an established task with LLMs, not an invention of this repo.
  - `src/ko_dialect/data/template.py` (`stt2std` direction), `scripts/audit_ger_errors.py`.
- **What the source says (verified from the arXiv abstract):**
  - A dataset of **>334,000 pairs of N-best hypotheses and accurate transcriptions** across common speech domains.
  - The framing is a *paradigm shift from LM rescoring*, which "can only select one candidate hypothesis as the output transcription"; generative correction can "correct those tokens that are missing in the N-best list".
- **⚠️ Difference from our setup — do not overclaim.** HyPoradise conditions on an **N-best list**;
  the AI-Hub corpus stores a **single 1-best Naver Clova hypothesis** per utterance
  (`stt_hypothesis`). Our variant is therefore strictly *thinner* in input information than
  the benchmark's, so HyPoradise's WER reductions are **not** a prediction of ours. It grounds
  the task's existence and framing, nothing quantitative.
- **Verification note:** ✅ Bibliographic details and the N-best claim confirmed against the arXiv abstract (2026-08-06). No quantitative claim from this paper is used in the repo.

### A13. I-measure — scoring against a "do-nothing" baseline (grounds `gain_over_copy`)

- **Title:** *Towards a standard evaluation method for grammatical error detection and correction*
- **Authors / venue:** Mariano Felice, Ted Briscoe — **NAACL-HLT 2015**, pp. 578–587 — https://aclanthology.org/N15-1060/
- **Cited in:**
  - `src/ko_dialect/evaluation/metrics.py::compute_gain_over_copy` / `compute_copy_baseline`
  - `src/ko_dialect/evaluation/metric_registry.py` (`gain_over_copy`, `copy_baseline`), `scripts/eval_dia2std.py`
- **The problem it names:** F-score style metrics cannot distinguish a system that *does nothing*
  from one that only makes wrong corrections — both score 0 — so there is no way to ask whether a
  system improved on the input at all. This repo hit the same wall from the other side: measured
  dia2std, raw chrF reads 76.9 on Gangwon and 75.4 on Chungcheong, which look respectable until
  you notice the unchanged input already scores 76.8 and 79.9 (docs/CORPUS_ANALYSIS_5REGION.md §6).
- **What the source says (verified from the authors' project page, https://ilexir.co.uk/i-measure/):**
  > "The I-measure is positive if the text quality has improved, negative if it has deteriorated
  > and zero if it remains the same, either because no change has been effected or because the
  > positive impact of errors successfully corrected by the system is equal to the negative
  > impact of new errors introduced."
- **⚠️ We do not implement the I-measure.** It is a token-level three-way alignment (source /
  hypothesis / reference) for GEC. `gain_over_copy` is a corpus-chrF difference,
  `chrF(gen,gold) − chrF(source,gold)`. What is borrowed is the **principle** — score relative to
  the unchanged input, with negative meaning "worse than doing nothing" — not the metric.
- **Verification note:** ⚠️ **Partially verified.** Bibliography confirmed via ACL Anthology
  (N15-1060) and the quoted semantics via the authors' own I-measure page. The primary PDF could
  not be text-extracted in this environment, so no section- or table-level claim is made from it.

### A14. Alphabet-level pivot NMT for Korean dialects — external support for the dia2std priority

- **Title:** *Low-Resourced Alphabet-Level Pivot-Based Neural Machine Translation for Translating Korean Dialects*
- **Venue:** *Applied Sciences* 15(17):9459 (2025) — https://doi.org/10.3390/app15179459
- **Why it matters here:** it treats **dialect → standard normalization as the reusable component** —
  a pivot whose output is then fed to an off-the-shelf translator or LLM. That is exactly the
  direction this repo deploys (`dia2std`) and, until 2026-08-05, the one GRPO never optimised
  (docs/CORPUS_ANALYSIS_5REGION.md §7). Independent support for the L4 reprioritisation.
- **What the abstract claims:** a two-stage pivot; the normalizer is a **minGRU encoder + GRU
  decoder seq2seq with alphabet-level (jamo) tokenization**, and jamo tokenization is reported
  "more effective for Korean dialect normalization than other widely used sub-word tokenizations".
- **⚠️ Not actionable for this repo, and do not cite it as if it were.** That result is for a small
  seq2seq **trained from scratch**; we fine-tune a pretrained Qwen2.5-0.5B whose BPE vocabulary
  cannot be swapped without discarding the pretraining that makes a 0.5B model viable at all. It is
  a *hypothesis generator* for why Jeju degrades (dialect differences are often sub-syllabic, and
  syllable-level BPE may fragment them), not a prescription.
- **Verification note:** ⚠️ **Unverified against the primary source.** MDPI returns HTTP 403 to this
  environment; the claims above come from the publisher's abstract via search. No quantitative
  result from this paper is used anywhere in the repo.

### A15. Prosodic labels in seq2seq synthesis — why the prosody-marker A/B came out flat

- **Title:** *Prosodic Prominence and Boundaries in Sequence-to-Sequence Speech Synthesis*
- **Authors / venue:** Antti Suni, Sofoklis Kakouros, Martti Vainio, Juraj Šimko — **Speech Prosody 2020** — **arXiv:2006.15967** — https://arxiv.org/abs/2006.15967
- **Cited in:** `src/ko_dialect/data/prosody.py`, `docs/EXPERIMENTS.md` §2 (prosody-marker A/B verdict).
- **What the source says (verified from the arXiv abstract):** augmenting text input with
  automatically extracted **word-prominence and phrase-boundary labels** "significantly improves the
  output in terms of faithfulness of f0 and energy contours"; the system still falls short on local
  prosodic events needing longer-range semantics. The gains are **acoustic-prosodic realisation**.
- **Why this is the right frame for our null result:** the measured verdict here (memory
  `prosody-sft-ab-verdict`) was that sentence-level prosody markers **raise `recon_bleu`
  significantly (+3.9 / +6.4, p<.001) while the dialectness axes stay flat** — i.e. a fidelity
  regularizer, not a dialectness lever. That is what this literature predicts: prosody labels carry
  boundary/prominence information, not lexical identity, and dialectness in our corpus is
  overwhelmingly lexical (`하영→많이`, `경→그렇게`; docs/CORPUS_ANALYSIS_5REGION.md §10).
  The flat result is the expected outcome, not a failed experiment.
- **⚠️ Domain gap:** that paper conditions a TTS acoustic model; we prepend discrete markers to text
  for a text-to-text LM. The shared claim is only about *what prosody labels carry*.
- **Verification note:** ✅ Bibliography and the quoted claims confirmed against the arXiv abstract (2026-08-06).

### A16. Round-trip RL for low-resource MT — the unused reward we already implement

- **Title:** *Improving Low-Resource Machine Translation via Round-Trip Reinforcement Learning*
- **Authors / venue:** Ahmed Attia, Alham Fikri Aji — **arXiv:2601.12535** (cs.CL, Jan 2026) — https://arxiv.org/abs/2601.12535
- **What the source says (verified from the arXiv abstract):** self-supervised fine-tuning that
  translates English → low-resource language → English and uses **chrF++ + BLEU on the
  reconstructed English as the reward**. It **does not require reference translations**.
  Evaluated on **NLLB 600M and 1.3B**; consistent improvements on Central Aymara, Friulian,
  Wolof, Dyula, Bhojpuri, Russian. No numeric table is quoted here — the abstract gives none.
- **Why it matters here:** this repo already implements the same idea as
  `rewards/reconstruction.py::make_reconstruction_reward` (grounded in Dual-RL, §A6) and
  registers it as `"reconstruction"` — **but no experiment arm uses it**; arms 1 and 2 run
  style + copy_margin + edit_precision + edit_recall, with reconstruction and fluency parked on
  VRAM cost. The 600M result is the closest external evidence that the axis is viable at our
  0.5B scale.
- **⚠️ Caveat specific to `dia2std`:** the back-translator for that arm would be
  standard → dialect, a direction this model is *also* weak at (docs/CORPUS_ANALYSIS_5REGION.md
  §6). A weak back-translator makes the reward noisy, so enable it on std2dia first or use a
  fixed external back-translator.
- **Verification note:** ✅ Bibliography and method confirmed against the arXiv abstract (2026-08-06).

### A17. MT-R1-Zero — rule-metric mixed reward, GRPO for translation

- **Title:** *MT-R1-Zero: Advancing LLM-based Machine Translation via R1-Zero-like Reinforcement Learning*
- **Venue:** **arXiv:2504.10160** (2025) — https://arxiv.org/pdf/2504.10160
- **What it reports:** R1-Zero-style RL for MT **without a supervised warm start**, using a
  *rule-metric mixed reward* (format checks combined with translation-quality metrics) under
  GRPO; the authors report that pure RL can rival SFT for LLM-based translation.
- **Why it matters here:** our SFT budget is hardware-bound (≈3.5 h per arm on the 6 GB card
  after the eval fix), so "RL without an SFT warm start" is a directly relevant alternative
  shape for the pipeline. It also matches the direction this repo already leans: mixing a
  cheap deterministic signal with a metric-based one.
- **Verification note:** ⚠️ **Abstract-level only.** Read via search summary; not verified
  against the PDF in this environment. No quantitative claim is used in the repo.

### A18. RLVR — verifiable rewards over learned reward models

- **What it is:** the 2025–2026 framing in which the reward comes from a **deterministic
  verifier** rather than a learned reward model; GRPO is the optimiser most commonly paired
  with it. Survey/collection: https://github.com/opendilab/awesome-RLVR
- **Why it matters here — the concrete consequence:** this corpus ships a verifier.
  `dialect_eojeol_map` is a per-word gold edit list covering **99.7% of trainable rows**
  (docs/CORPUS_ANALYSIS_5REGION.md §3.2), so "did the output convert `하영`→`많이`" is an exact
  string check, not a proxy. `r_edit_precision` / `r_edit_recall` already consume it. The
  TextCNN style reward, by contrast, is a *learned* proxy this project has already measured as
  hackable (reward ↑ while chrF ↓) and whose margin is thinnest exactly where it matters —
  Chungcheong sits closest to standard (copy floor 79.9 chrF). Hence `style` is weighted down
  to 0.25 in `configs/experiment/grpo_dia2std.yaml` in favour of the edit axes.
- **⚠️ Not a paper citation.** RLVR is a paradigm label, not a single result; the repo makes no
  claim attributable to any one RLVR paper.
- **Verification note:** ⚠️ Paradigm-level context only, no quantitative claim.

### A19. Pref-GRPO — group normalization amplifies tiny reward gaps into fake advantages

- **Title:** *Pref-GRPO: Pairwise Preference Reward-based GRPO for Stable Text-to-Image Reinforcement Learning*
- **Authors:** Yibin Wang, Zhimin Li, Yuhang Zang, Yujie Zhou, Jiazi Bu, Chunyu Wang, Qinglin Lu, Cheng Jin, Jiaqi Wang — **arXiv:2508.20751**
- **What the source says (verified from the abstract):** with pointwise reward models,
  "minimal score differences between images are **amplified after normalization**, creating
  **illusory advantages** that drive the model to over-optimize for trivial gains, ultimately
  destabilizing" generation. Their fix replaces score maximisation with pairwise **preference
  fitting** (win rate) inside each group.
- **Why it matters here:** it names the mechanism behind the low-headroom measurement in
  docs/CORPUS_ANALYSIS_5REGION.md §12.1. 28.5% of our GRPO prompts have a gold within 1 eojeol
  of the source, so every completion scores nearly the same — and `(r − mean)/std` with a tiny
  `std` turns that noise into a large advantage. It is worse than wasted compute: it is a
  reward-hacking vector. **And this repo runs MO-GRPO (A2), whose per-objective unit-variance
  normalisation inside the group is exactly the amplifying operation.** Hence
  `build_grpo_dataset(low_headroom_ratio=…)` is a stability measure, not an efficiency tweak.
- **⚠️ Domain gap:** text-to-image with a pointwise reward model. The *mechanism* is
  architecture-independent; their quantitative results are **not** claimed to transfer. Our
  evidence is the §12.1 distribution, nothing more.
- **Verification note:** ✅ Bibliography and quoted claims confirmed against the arXiv abstract (2026-08-06).

### A20. Dialect-to-standard normalization is *character transduction*, not translation

- **Title:** *Dialect-to-Standard Normalization: A Large-Scale Multilingual Evaluation*
- **Authors / venue:** Olli Kuparinen, Aleksandra Miletić, Yves Scherrer — **Findings of EMNLP 2023** — https://aclanthology.org/2023.findings-emnlp.923/
- **What the source says (verified from the abstract + ACL page):** it introduces
  dialect-to-standard normalization — "mapping phonetic transcriptions from different dialects
  to the orthographic norm of the standard variety" — as "**a distinct sentence-level character
  transduction task**", evaluated across Finnish, Norwegian, Swiss German and Slovene. A
  **character-level Transformer trained on sliding windows of three words** is best for Finnish,
  Swiss German and Slovene; pre-trained **byT5** on full sentences wins for Norwegian.
- **Why it matters here — this is the closest published framing of our deployed task.** Two
  independent findings converge with A14: the winning granularity is **sub-word (character /
  byte)**, and the winning context is **narrow (3 words)**, not whole sentences. Our setup is the
  opposite on both axes — Qwen2.5 BPE over full sentences. That is a hypothesis for why Jeju
  collapses, and an argument that the on-device normalizer does not have to be a general LLM.
- **⚠️ Constraint, not a prescription:** the project is committed to a pretrained decoder LM for
  GGUF export; swapping to a char-level or byte-level model discards the pretraining that makes
  0.5B viable. Treat as a design alternative to evaluate, not a change to make.
- **Verification note:** ✅ Confirmed against the ACL Anthology abstract (2026-08-06). No numeric result is used.

### A21. LLM + morphological rules for dialect normalization without parallel data

- **Title:** *Dialect Normalization using Large Language Models and Morphological Rules*
- **Authors / venue:** Antonios Dimakis, John Pavlopoulos, Antonios Anastasopoulos — **Findings of ACL 2025** — https://aclanthology.org/2025.findings-acl.1215/
- **What the source says (verified):** combines rule-based linguistically informed
  transformations **and** few-shot LLM prompting, "does not require parallel data", applied to
  **Greek dialects** on regional proverbs, with human evaluation and downstream experiments.
- **Why it matters here — mostly as a contrast that reframes our position.** Its central
  contribution is working *without* parallel data. We have **1,303,105 parallel train rows**
  with per-word gold edit maps. On the axis this literature optimises for, we are not
  low-resource at all; our scarcity is **compute** (6 GB) and **model capacity** (0.5B). Methods
  designed to manufacture supervision (round-trip A16, no-parallel-data A21) therefore solve a
  problem we do not have, and their transfer to us is weak by construction.
- **Verification note:** ✅ Confirmed against the ACL Anthology page (2026-08-06). No numeric result is used.

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
| 4 | `leaderboard.py:177` | "harmonic-mean joint objective (2010.12771 §3)" | ✅ **resolved** — PDF read: paper uses weighted-sum training + arithmetic-mean selection, no harmonic mean. `harmonic_joint` relabeled as entirely ours. |

Other non-blocking actions: keep `eval_leaderboard.py --n ≥ 300` (A7); log sacreBLEU signature (A9).

---

## Source index

| Ref | ID / URL | Status |
|---|---|---|
| DIA-REFINE | arXiv:2511.06680 | ✅ verified |
| MO-GRPO | arXiv:2509.22047 | ⚠️ aggregation ✅, selection-claim = our inference |
| Mind the Style Gap | arXiv:2502.15022 | ❌ author + finding corrected |
| TST Direct Rewards | arXiv:2010.12771 | ✅ verified; harmonic is ours (paper = weighted-sum/arithmetic-mean) |
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
