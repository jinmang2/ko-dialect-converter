#!/usr/bin/env python3
"""Qualitatively + quantitatively inspect the stage-2 TextCNN dialect classifier.

Answers "why does macro-F1 plateau and not break 90%?" by separating *model* error
from *data* ceiling:

1. **Metrics** — per-class precision/recall/F1/support, macro & weighted F1, accuracy,
   and a confusion matrix on the held-out valid split.
2. **Misclassifications** — the most confident wrong predictions per confusion cell,
   so you can eyeball whether errors are reasonable or label noise.
3. **Data ceiling** — on the raw dialect pairs: how many "dialect" rows differ from
   their standard form by only 1 eojeol (near-identical, intrinsically ambiguous),
   and the *collision rate* — dialect strings that appear verbatim as a standard
   sentence elsewhere. Collisions are an unlearnable label contradiction and put a
   hard cap on achievable accuracy.

Usage:
    python scripts/inspect_classifier.py \
        --classifier_path outputs/classifier \
        --cls_dataset_path outputs/datasets/classifier \
        --raw_dataset_path outputs/dialect_raw_new \
        --n_errors 30 --out outputs/eval_logs/classifier_report.json
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import fire
import torch
from datasets import load_from_disk
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer

from ko_dialect.evaluation import resolve_best_checkpoint
from ko_dialect.models import TextCNNForSequenceClassification
from ko_dialect.models.classifier import ID2LABEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LABELS = [ID2LABEL[i] for i in range(len(ID2LABEL))]


@torch.no_grad()
def _predict(model, tokenizer, texts, device, max_length=128, batch_size=256):
    """Return (pred_ids, max_prob) for every text, batched."""
    preds: list[int] = []
    confs: list[float] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        enc = tokenizer(
            chunk,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        logits = model(enc["input_ids"].to(device)).logits
        prob = torch.softmax(logits, dim=-1)
        conf, pred = prob.max(dim=-1)
        preds.extend(pred.tolist())
        confs.extend(conf.tolist())
        if start % (batch_size * 20) == 0:
            logger.info("classified %d / %d", start + len(chunk), len(texts))
    return preds, confs


def _eojeol_diff(standard: str, dialect: str) -> int:
    """Number of positionally-differing eojeols between standard and dialect."""
    s, d = standard.split(), dialect.split()
    n = min(len(s), len(d))
    diff = sum(1 for i in range(n) if s[i] != d[i])
    return diff + abs(len(s) - len(d))


def _data_ceiling(raw_path: str, n_show: int = 8) -> dict:
    """Quantify intrinsic label ambiguity in the raw dialect pairs (valid split)."""
    ds = load_from_disk(raw_path)
    valid = ds["valid"] if "valid" in ds else ds[list(ds.keys())[0]]
    diff_hist: Counter[int] = Counter()
    n_dialect = 0
    std_set: set[str] = set()
    dialect_texts: list[str] = []
    near_identical_examples: list[dict] = []

    for row in valid:
        std_set.add(row["standard"])
        if row.get("is_identical"):
            continue
        if row["do"] not in ("gangwondo", "gyeongsangdo"):
            continue
        n_dialect += 1
        dialect_texts.append(row["dialect"])
        d = _eojeol_diff(row["standard"], row["dialect"])
        diff_hist[d] += 1
        if d == 1 and len(near_identical_examples) < n_show:
            near_identical_examples.append(
                {"do": row["do"], "standard": row["standard"], "dialect": row["dialect"]}
            )

    collisions = sum(1 for t in dialect_texts if t in std_set)
    n_total_eojeol_pairs = max(n_dialect, 1)
    return {
        "n_dialect_rows": n_dialect,
        "eojeol_diff_histogram": dict(sorted(diff_hist.items())),
        "pct_differ_by_1_eojeol": round(100 * diff_hist.get(1, 0) / n_total_eojeol_pairs, 2),
        "pct_differ_by_le2_eojeol": round(
            100 * (diff_hist.get(1, 0) + diff_hist.get(2, 0)) / n_total_eojeol_pairs, 2
        ),
        "collision_count": collisions,
        "collision_rate_pct": round(100 * collisions / n_total_eojeol_pairs, 2),
        "near_identical_examples": near_identical_examples,
    }


def main(
    classifier_path: str = "outputs/classifier",
    cls_dataset_path: str = "outputs/datasets/classifier",
    raw_dataset_path: str = "outputs/dialect_raw_new",
    cls_tokenizer_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    split: str = "valid",
    max_length: int = 128,
    batch_size: int = 256,
    n_errors: int = 30,
    out: str | None = None,
) -> None:
    resolved = resolve_best_checkpoint(classifier_path)
    logger.info("Loading classifier from %s", resolved)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TextCNNForSequenceClassification.from_pretrained(resolved).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(cls_tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_from_disk(cls_dataset_path)[split]
    texts, labels = ds["text"], ds["label"]
    logger.info("Evaluating on %d %s examples", len(texts), split)
    preds, confs = _predict(model, tokenizer, texts, device, max_length, batch_size)

    report = classification_report(
        labels, preds, target_names=LABELS, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(labels, preds).tolist()

    # ---- print human-readable summary ----
    print("\n" + "=" * 64)
    print(f"CLASSIFIER REPORT  ({resolved})")
    print("=" * 64)
    print(classification_report(labels, preds, target_names=LABELS, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(f"{'':>12}" + "".join(f"{name:>12}" for name in LABELS))
    for i, lbl in enumerate(LABELS):
        print(f"{lbl:>12}" + "".join(f"{cm[i][j]:>12}" for j in range(len(LABELS))))

    # ---- worst (most confident) misclassifications ----
    wrong = [
        {
            "text": texts[i],
            "true": ID2LABEL[labels[i]],
            "pred": ID2LABEL[preds[i]],
            "conf": confs[i],
        }
        for i in range(len(texts))
        if preds[i] != labels[i]
    ]
    wrong.sort(key=lambda r: r["conf"], reverse=True)
    print(f"\nTop {n_errors} most-confident misclassifications:")
    for r in wrong[:n_errors]:
        print(f"  [{r['true']}->{r['pred']} {r['conf']:.2f}] {r['text'][:90]}")

    # ---- data ceiling ----
    print("\n" + "=" * 64)
    print("DATA CEILING ANALYSIS (raw valid pairs)")
    print("=" * 64)
    ceiling = _data_ceiling(raw_dataset_path)
    print(f"dialect rows (is_identical=False): {ceiling['n_dialect_rows']}")
    print(f"differ from standard by exactly 1 eojeol: {ceiling['pct_differ_by_1_eojeol']}%")
    print(f"differ by <=2 eojeols:                    {ceiling['pct_differ_by_le2_eojeol']}%")
    print(
        f"collisions (dialect text == some standard text): "
        f"{ceiling['collision_count']} ({ceiling['collision_rate_pct']}%)  <-- unlearnable"
    )
    print("eojeol-diff histogram (n_differing_eojeols -> count):")
    print("  " + str(ceiling["eojeol_diff_histogram"]))
    print("examples differing by 1 eojeol:")
    for ex in ceiling["near_identical_examples"]:
        print(f"  [{ex['do']}] std: {ex['standard']}")
        print(f"            dia: {ex['dialect']}")

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": resolved,
            "split": split,
            "n_examples": len(texts),
            "accuracy": report["accuracy"],
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"],
            "per_class": {name: report[name] for name in LABELS},
            "confusion_matrix": cm,
            "labels": LABELS,
            "data_ceiling": ceiling,
            "top_misclassifications": wrong[:n_errors],
        }
        Path(out).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        logger.info("Report written to %s", out)


if __name__ == "__main__":
    fire.Fire(main)
