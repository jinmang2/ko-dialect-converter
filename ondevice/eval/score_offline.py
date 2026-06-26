"""Score phone-captured generations on the desktop (chrF / recon_bleu) and merge
the two quality axes back into the measurement JSONL (brief §6, decision O3).

Reuses the project's own metrics so on-device quality is on the SAME scale as the
desktop leaderboard:
  * chrF       = compute_chrf(outputs, references)             — vs 표준어 gold
  * recon_bleu = reconstruction_bleu(outputs, references)      — did we reconstruct standard

  in : ondevice/eval/outputs/<variant>.jsonl  {idx, do, output}
       ondevice/data/eval_prompts.jsonl       {idx, do, reference, ...}
  out: per-region + overall chrF/recon_bleu; optionally merged into measurements.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from ko_dialect.evaluation.metrics import compute_chrf, reconstruction_bleu  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from schema import read_jsonl, write_jsonl  # noqa: E402


def _load(path: str) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _score(outputs: list[str], references: list[str]) -> dict[str, float]:
    return {
        "chrF": round(compute_chrf(outputs, references), 2),
        "recon_bleu": round(reconstruction_bleu(outputs, references), 2),
    }


def score(
    variant: str,
    outputs: str | None = None,
    prompts: str = "ondevice/data/eval_prompts.jsonl",
    measurements: str | None = "ondevice/bench/logs/measurements.jsonl",
    out_json: str = "ondevice/eval/quality.jsonl",
    merge: bool = True,
) -> None:
    """Score one variant's generations; print per-region + overall; merge into measurements."""
    outputs = outputs or f"ondevice/eval/outputs/{variant}.jsonl"
    gen = _load(outputs)
    ref_rows = _load(prompts)
    # key references by (do, idx) so we align even if order differs
    ref_by_key = {(r["do"], r["idx"]): r["reference"] for r in ref_rows}

    by_region: dict[str, tuple[list[str], list[str]]] = {}
    for g in gen:
        key = (g["do"], g["idx"])
        if key not in ref_by_key:
            continue
        outs, refs = by_region.setdefault(g["do"], ([], []))
        outs.append(g["output"])
        refs.append(ref_by_key[key])

    summary = {"variant": variant, "regions": {}}
    all_outs: list[str] = []
    all_refs: list[str] = []
    for do, (outs, refs) in sorted(by_region.items()):
        summary["regions"][do] = {"n": len(outs), **_score(outs, refs)}
        all_outs += outs
        all_refs += refs
        print(
            f"  {do:14s} n={len(outs):4d}  chrF={summary['regions'][do]['chrF']:6.2f}  "
            f"recon_bleu={summary['regions'][do]['recon_bleu']:6.2f}"
        )
    overall = _score(all_outs, all_refs) if all_outs else {"chrF": None, "recon_bleu": None}
    summary["overall"] = {"n": len(all_outs), **overall}
    print(
        f"  {'OVERALL':14s} n={len(all_outs):4d}  chrF={overall['chrF']}  recon_bleu={overall['recon_bleu']}"
    )

    # append summary
    out_path = Path(out_json)
    prev = read_jsonl(out_path) if out_path.exists() else []
    write_jsonl([*prev, summary], out_path)

    # merge overall chrF/recon_bleu into every measurement row of this variant
    if merge and measurements and Path(measurements).exists():
        rows = read_jsonl(measurements)
        hit = 0
        for r in rows:
            if r.get("variant") == variant:
                r["chrF"] = overall["chrF"]
                r["recon_bleu"] = overall["recon_bleu"]
                hit += 1
        write_jsonl(rows, measurements)
        print(f"merged quality into {hit} measurement row(s) for {variant}")


if __name__ == "__main__":
    fire.Fire(score)
