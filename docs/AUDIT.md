# AUDIT.md — Code review, reproducibility & completeness audit

Full line-by-line audit of the core (non-speech) pipeline, the reproducibility of the
recorded experiments, and how much is actually implemented. Conducted by 6 parallel
reviewers over data / SFT+classifier / GRPO+rewards / evaluation / export-serving, plus a
dedicated reproducibility pass.

**Method:** static read + `py_compile` (all files clean) + cheap CPU checks (pure helpers
exercised on toy data) + artifact cross-checks against `outputs/eval_logs/*.json`.
**Not done:** no training, no GPU generation, no HF upload. LSP (`ty`) and `ruff` were not
on the reviewers' PATH in-sandbox; the project runs in conda env `balaenoptera`.

## Bottom line

- **Completeness: high.** Every core module is fully implemented. The only non-implemented
  pieces are *intentional* stubs / cost-gated paths (listed below) — no accidental dead ends.
- **Correctness: solid.** 0 CRITICAL. **3 HIGH**, ~12 MEDIUM, ~12 LOW. The fp16-only RTX-2060
  invariant is clean everywhere; `push_to_hub` dry-run default is safe.
- **Reproducibility: code+inputs+artifacts all present; experiments are "reproducible-with-GPU."**
  Quoted numbers in the docs **match their artifacts**. BUT the results page (`docs/RESULTS.md`)
  is generated from **stale artifacts** and the current leaderboard format no longer feeds it
  (HIGH-1) — the headline finding.

---

## Resolution status (updated post-audit)

| Item | Status | Commit / note |
|---|---|---|
| **H1** report.py multi/v1 schema | ✅ **fixed** | classify+render multi schema; verified report.py now emits OVERALL+regions; +3 tests |
| **H2** DO_NAME 5-region | ✅ **fixed** | all 5 Korean names; +2 tests (no English-code leak) |
| **H3** speedrun seed | ✅ **fixed** | `random.Random(seed)` |
| Citation errors (REFERENCES) | ✅ **fixed** | Mind-the-Style-Gap author + claim, 2010.12742 reframe, MO-GRPO attribution |
| harmonic_joint §3 attribution | ✅ **resolved** | PDF read: paper uses weighted-sum/arithmetic-mean, no harmonic → relabeled as ours |
| MO-GRPO scale_rewards docstring (M8) | ✅ **fixed** | corrected (no-op in TRL branch, not a double-scale guard) |
| Proxy-independent selection rationale | ✅ **documented** | metrics.py ADR now cites our measured over-optimization + domain + DIA-REFINE |
| **M2** is_identical `.get` | ✅ **fixed** | dataset.py (2 spots) |
| **M10** quantize_eval CUDA guard | ✅ **fixed** | SystemExit on CPU |
| **M11** serve_compare local_files_only | ✅ **fixed** | applied to model load |
| **M13** SFT(0) KeyError guard | ✅ **fixed** | `.get` + skip-branch |
| R1/R2 doc-sync, other MEDIUM/LOW | ⬜ open | see tables below |

---

## HIGH findings (fix first)

### H1 — `report.py` silently drops the CURRENT leaderboard format → RESULTS.md is stale
- **Where:** `scripts/report.py:42-58`; `src/ko_dialect/evaluation/report.py:25` (`classify_artifact`); writer `scripts/eval_leaderboard.py:330-339`.
- **What:** `eval_leaderboard.py` (both single- and multi-region runs) writes
  `schema:"ko_dialect.leaderboard_multi/v1"` with data nested under `per_region`/`overall`
  and **no top-level `ranking`/`rows`**. `classify_artifact` recognizes a leaderboard *only*
  via `"ranking" in data and "rows" in data`, so it returns `None` for every current run.
- **Empirically confirmed:** `report.py` runs OK and logs "Recognised 4 artifact(s)", but it
  ingests only **stale** flat `leaderboard/v2` files (`leaderboard_gangwondo_*.json`, June-12);
  the current `leaderboard_overall_20260612_182816.json` is dropped. So `docs/RESULTS.md` shows
  an old single-region table and **omits the current multi-region/OVERALL leaderboard**. A fresh
  `report.py` after any new leaderboard run would show **no leaderboard at all**.
- **Why the reproducibility pass said "byte-identical":** it *is* byte-identical — because
  `report.py` deterministically re-renders the same stale subset every time. Identical ≠ current.
- **Fix:** in `report.py`/`evaluation/report.py`, detect `schema=="ko_dialect.leaderboard_multi/v1"`
  and iterate `per_region`+`overall` (each already carries `ranking`/`rows`/`target_do`).

### H2 — `template.py` `DO_NAME` covers only 2 of 5 regions → English region codes leak into prompts
- **Where:** `src/ko_dialect/data/template.py:8-11,21`.
- **What:** `DO_NAME = {"gangwondo":"강원도","gyeongsangdo":"경상도"}`; `format_user` does
  `DO_NAME.get(do, do)`. For the old_dialect 5-region corpus, a jeolla row yields the prompt
  `다음 문장을 jeollado 사투리로 바꿔줘:` — an English token mid-Korean-prompt. This silently
  degrades SFT *and* GRPO prompt quality for exactly the three regions the `OLD_DIALECT_EXPANSION`
  effort adds.
- **Fix:** populate all five (`jeollado→전라도, jejudo→제주도, chungcheongdo→충청도`) or derive
  `DO_NAME` by inverting `REGION_MAP` (`prepare_data.py:149`), which already has the Korean names.

### H3 — `prepare_data.py` speedrun sampling is unseeded → non-reproducible smoke builds
- **Where:** `scripts/prepare_data.py:584` — `files = random.sample(files, k=...)` on the global RNG.
- **What:** every speedrun draws a different file subset, so a "speedrun" dataset cannot be
  regenerated identically — undermining the repro path most likely to be re-run during iteration.
  (The full non-speedrun build is deterministic because it consumes all files.)
- **Fix:** add `seed: int = 42`, use `random.Random(seed).sample(...)`, record it in the manifest.

---

## Reproducibility of recorded experiments

All four experiments in `docs/EXPERIMENTS.md` / `docs/RESULTS.md`: scripts import cleanly under
`balaenoptera`; all referenced model/adapter dirs exist with real weights
(`outputs/{sft_merged,sft_control_s15,sft_prosody_s15,sft_prosody_v2,grpo_500,grpo_arm1,grpo_arm2,grpo_v2,classifier_clean}`);
cited eval-log artifacts exist; spot-checked numbers match.

| # | Experiment | Script | Inputs | Artifacts | Numbers match | Verdict |
|---|---|---|---|---|---|---|
| 1 | GRPO-vs-SFT leaderboard | ✅ | ✅ | ✅ `leaderboard_overall_20260612_182816.{json,md}` | ✅ Pareto + `p=0.802` + recon table exact | **reproducible-with-GPU** |
| 2 | Prosody-SFT A/B (v1+v2) | ✅ | ✅ | ✅ `prosody_ab_*.log`, `prosody_ab_v2.log` | ✅ +3.9/+6.4 (v1), +2.47/+4.80 (v2) exact | **reproducible-with-GPU** |
| 3 | Quantization PTQ | ✅ | ✅ `sft_merged` | ✅ `quantize_tradeoff_*.json` | ⚠️ backed but two docs cite different artifacts | **reproducible-with-GPU** (doc-sync) |
| 4 | Train opt (DeepSpeed vs unsloth) | ✅ | ✅ | ⚠️ unsloth+bnb logs only; no DeepSpeed log | ⚠️ unsloth exact; DeepSpeed claim unbacked | **reproducible-with-GPU** (partial) |

"reproducible-with-GPU" = code + inputs + cited artifacts all present and consistent; the only
thing missing is a GPU to re-run the forward/training passes (cannot be done in this sandbox).

**Spot-checks that matched the artifacts:** leaderboard recon BLEU (SFT 38.406 / arm2 37.059 /
grpo_500 37.052 / v2 35.835 / arm1 35.410); `grpo_500` p=0.802 not-significant; prosody v1
45.464→49.358 / 50.763→57.198; v2 paired +2.20 (p=0.008) / +5.65 (p<0.001); unsloth
286.1 tok/s, 1.117 steps/s, 1284.4 MB; bnb blocker = the BF16 `_amp_foreach_..._unscale` error.

### Reproducibility caveats (doc honesty)
- **R1 (MEDIUM) — Quantization doc-sync.** `EXPERIMENTS.md` §3 (n=150: 488/985 ms, chrF 68.5/70.7)
  and `RESULTS.md` (vram-run: 720.9/907.25 ms, chrF 72.9/76.2) cite **different** latency/chrF for
  the same experiment — both trace to real artifacts, but `report.py` keeps only the newest file
  per scope (`quantize_tradeoff_vram.json`). The verdict direction (4-bit ≈ 2× memory win, latency
  *loss*) holds in every artifact. Fix: pin both docs to one artifact, or have `report.py` prefer
  the n=150 run.
- **R2 (MEDIUM) — Exp-4 DeepSpeed claim unbacked.** No DeepSpeed init-failure artifact exists, and
  the unsloth log shows `CUDA Toolkit: 13.0` present — which does **not** corroborate the
  "nvcc absent" narrative in `EXPERIMENTS.md §4`. Re-verify or soften. Only 1 of 3 arms (unsloth)
  has measured numbers; bnb is evidenced, DeepSpeed is not.

---

## Completeness map (what's implemented vs intentionally not)

**Fully implemented:** all of `data/*`, `models/*`, `training/*` (sft/classifier/grpo/mo_grpo),
all 10 `rewards/*`, all of `evaluation/*`, and the script entry points. No TODO/FIXME/pass-only
stubs in the core path. GRPO dataset present (357,116 train rows) with every column the rewards consume.

**Intentional stubs / not-wired / env-gated (by design, documented):**
| Item | Where | Status |
|---|---|---|
| `qat` backend | `models/loading.py:268-291` | `NotImplementedError` scaffold; not in default path |
| `fluency`, `reconstruction` rewards | `rewards/registry.py` | Registered but not wired into any shipped config (need ref_model; cost-gated) |
| vLLM serving/training | `serve_sft_vllm.py`, `configs` `use_vllm` | Aspirational; blocked by TRL 1.5.1 ↔ vLLM version range on this env |
| AWQ / GPTQ PTQ | `scripts/ptq_quantize.py` | Behind ImportError guards; need `autoawq`/`gptqmodel` + GPU |
| GGUF export | `export/gguf.py`, `export_gguf.py` | Correct wrapper; needs a built llama.cpp binary |
| DeepSpeed ZeRO-2 | `configs/training/deepspeed_zero2_offload.json` | For a CUDA-toolkit/multi-GPU box; doesn't init on this 2060 |

---

## MEDIUM findings

| ID | Where | Issue | Fix |
|---|---|---|---|
| M1 | `prepare_data.py:557,580` | default `data_path` is `.../data` but corpus is in `raw_data/`; `**/*.json` glob can mix new+old formats under one `use_old_format` flag (wrong parser → silent empty) | default to `raw_data/new_dialect`/require arg; assert homogeneous input subtree |
| M2 | `data/dataset.py:78,155` | `x["is_identical"]` direct index → opaque `KeyError` in a `datasets` worker if a future source lacks it (classifier path already hedges with `.get` at :326) | use `sample.get("is_identical", x["standard"]==x["dialect"])` consistently |
| M3 | `data/prosody.py:142` (`_RE_ORPHAN`) | strips `[ ]` too, but the cleaning convention header documents only `( ){ }& # ~` | document `[ ]` removal or drop it from the regex |
| M4 | `models/classifier.py:131-136` | class-weight tensor cast to `logits.dtype` (fp16 under default) couples weight precision to activations | cast weights to `float32` |
| M5 | `configs/training/classifier.yaml:11-12` | `eval_steps`/`save_steps` are dead under `eval_strategy: epoch` (misleading) | comment that they apply only with `strategy: steps` |
| M6 | `configs/training/sft.yaml:29-32` | no `load_best_model_at_end`/early-stop despite an eval set → `save_model` keeps *last*, not best (intentional, but a reader expectation gap vs CLAUDE.md) | confirm default; document |
| M7 | `scripts/stage2_train_classifier.py:33-52` | mixes `.get(default)` and direct `cfg` access; no `from_omegaconf` → new trainer fields need editing this script too (shotgun seam) | uniform access / adopt a `ClassifierTrainerConfig` |
| M8 | `training/mo_grpo_trainer.py:16-24` (& `grpo_trainer.py:254-260`) | docstring claims `scale_rewards="none"` prevents "double-scaling" in the `normalize_then_sum` branch; TRL 1.5.1 source shows that branch **ignores** `scale_rewards` and always batch-normalizes → reasoning wrong (behavior correct) | reword docstring; note the extra batch-normalize beyond the pure paper |
| M9 | `training/grpo_trainer.py:205-213` | `aggregation="hm"` clamps axes to [0,1] but doesn't enforce `normalize_rewards=True` → silent distortion if misconfigured (no shipped config uses `hm`) | raise/auto-set when `hm` + not normalized |
| M10 | `scripts/quantize_eval.py:131-136` | on CPU, `size_mb` falls back to disk size → the trade-off table mixes units and is meaningless without CUDA | require CUDA (`SystemExit`) for honest VRAM/latency |
| M11 | `scripts/serve_compare.py:75 vs 81` | `local_files_only=True` on tokenizer but not on model → asymmetric offline failure | apply consistently to model loads |
| M12 | `scripts/wandb_doctor.py:185-196,372,447` | depends on private wandb internals (`sdk.internal.datastore`, `_service_api`) → raw `ImportError` across versions | wrap with a self-explanatory version note |
| M13 | `scripts/eval_grpo_checkpoints.py:289-290` | unconditional `all_outs["SFT(0)"]` → latent `KeyError` that fires only after the expensive GPU sweep | `all_outs.get("SFT(0)")` guard |

(R1, R2 above are also MEDIUM.)

---

## LOW findings (selected)

- `rewards/style.py:33` — `DO_TO_LABEL.get(str(d), 1)` silently maps unknown region → gangwondo; use `[...]` to surface bad data as jeju/jeolla/chungcheong land.
- `training/grpo_trainer.py:55-56,239` — `epsilon_high` not validated `>= epsilon` (shipped 0.28≥0.2 OK).
- `export/gguf.py:101` — `n_gpu_layers=35` default exceeds the model's 24 layers (llama.cpp clamps; cosmetic); default to `-1`/`24`.
- `evaluation/serving.py:87` — `if v.get("latency_ms_p50")` skips a genuine `0.0`; use `is not None` (matches sibling guards).
- `scripts/bench_train.py:64-66` — dataset slice assumes no packing compression; with `packing=True` the run may hit fewer steps than `max_steps`.
- `rewards/edit.py:124,144-151` — `r_edit_recall` docstring says `num_target_eojeols` but divides by a *set* (duplicate gold forms shrink denominator); acknowledged in note.
- `data/labels.py` — `num_labels_for` over a sparse id set would create phantom classes (fine for current contiguous ids).
- `data/collator.py` — fixed `max_length=128` truncation with no over-length counter.
- `scripts/eval_leaderboard.py:131` — unused `select_by` param (dead arg).
- `scripts/inspect_classifier.py` — `_eojeol_diff` is positional, not Levenshtein (naming).
- `scripts/report.py` — **not actually torch-free** despite its docstring intent: importing `ko_dialect.evaluation.report` triggers `evaluation/__init__.py` → `leaderboard` → `metrics` (`import torch`). On a torch-less CPU box the report fails at import before any logic runs. Make `evaluation/__init__.py` lazy or import the submodule path directly. (Resolves once `pip install -e .` provides torch.)
- `evaluation/metrics.py:133` — `compute_chrf` uses chrF (not chrF++ / no word n-grams). Defensible default; note for paper-grade comparability. Not a bug.

---

## Cross-cutting positives (CPU-verified)

- `significance.paired_bootstrap` (significance.py:99): joint resampling; two-sided `min(1,2·p_one)`;
  clear-better→p≈0/sig, identical→p=1/not-sig, worse→symmetric. Correct (Koehn 2004).
- `leaderboard.pareto_frontier` (≥all ∧ >one) and `harmonic_joint` ([.5,.5]→0.5, any axis≤0→0.0) correct.
- `copy_margin` wiring: `evaluate_all` passes `standard_refs` as source → `chrf − chrf_source`. Correct.
- Direction arrows (`metric_registry`): copy_margin↑, reconstruction_bleu↑, chrf_source↓, jscore
  monitoring-only — all correct. Every metric fn has an empty-input guard.
- **Label-id consistency confirmed:** `DIALECT_LABELS["gangwondo"]==DO_TO_LABEL["gangwondo"]==1`,
  `standard` absent from `DO_TO_LABEL` (can't be a TDR target). The single-source refactor holds.
- **fp16-only invariant clean:** no `bf16=True` anywhere; `resolve_dtype()` downgrades bf16→fp16 on
  SM<8.0 with a warning; SFT casts stray bf16 params to fp16.

---

## Recommended fix order

1. **H1** `report.py` multi/v1 schema — without it the results page is stale/empty. (also resolves R1 selection)
2. **H2** `DO_NAME` 5-region — required before any 5-region SFT/GRPO run is meaningful.
3. **H3** speedrun seed — cheap, restores smoke-build reproducibility.
4. **R2** re-verify/soften the DeepSpeed claim; **M1** prepare_data default path + format guard.
5. MEDIUM doc/robustness items (M2–M13) as a cleanup pass.
6. Citation fixes from `docs/REFERENCES.md` (Mind-the-Style-Gap authorship, etc.) — separate doc pass.

> See `docs/REFERENCES.md` for the papers/techniques dossier and the 4 citation corrections.
