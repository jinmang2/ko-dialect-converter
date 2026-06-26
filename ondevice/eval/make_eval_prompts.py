"""Build the fixed on-device eval set → ondevice/data/eval_prompts.jsonl (run once, desktop).

Reuses the canonical `load_eval_samples` (same filter + first-n indices as the desktop
leaderboard, brief §5 "기존과 동일 인덱스") and the canonical `format_user` so the phone
prompt is byte-identical to training/eval. Direction is dia→std (the on-device task:
방언 → 표준어). One row per (region, sample):

  {"idx", "do", "source"(방언), "reference"(표준어), "prompt"(Qwen2.5 ChatML)}

The phone server is fed `prompt` verbatim via /completion; `reference` is the desktop
gold for chrF/recon_bleu (eval/score_offline.py).

VALIDITY GATE 1/3: the selection is deterministic (first-n, no shuffle) and a SHA256
content hash per region is printed + written to eval_manifest.json. The phone-quantized
runs and the desktop fp16 reference (desktop_ref.py) score the SAME manifest, so the
"chrF −N pt vs fp16" claim is a same-set comparison. Re-running build must reproduce the
hash; if it changes, the eval set drifted and prior numbers are not comparable.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import fire

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from ko_dialect.data.template import SYSTEM_PROMPT, get_template  # noqa: E402
from ko_dialect.evaluation.generation import load_eval_samples  # noqa: E402

DIRECTION = "dia2std"  # 방언 → 표준어 (on-device task)


def _chatml(user_content: str) -> str:
    """Qwen2.5 ChatML — must match ondevice/demo/index.html buildPrompt() exactly."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _region_hash(samples: list[dict]) -> str:
    """SHA256 over the ordered (idx, source, reference) triples — fingerprints the set.

    Content-based (not row-id based) so it is stable across dataset re-exports as long as
    the selected pairs are identical, and changes the moment selection/order drifts.
    """
    h = hashlib.sha256()
    for i, s in enumerate(samples):
        h.update(f"{i}\x1f{s['source']}\x1f{s['reference']}\x1e".encode())
    return h.hexdigest()


def build(
    raw_dataset_path: str = "outputs/dialect_raw_new",
    split: str = "valid",
    regions: str = "gangwondo,gyeongsangdo",
    n: int = 150,
    out: str = "ondevice/data/eval_prompts.jsonl",
    manifest: str = "ondevice/data/eval_manifest.json",
) -> None:
    """Write the fixed eval prompts for each region + a reproducibility manifest."""
    tmpl = get_template("default")
    rows = []
    manifest_obj: dict[str, object] = {
        "raw_dataset_path": raw_dataset_path,
        "split": split,
        "direction": DIRECTION,
        "n_requested": n,
        "regions": {},
    }
    for do in (r.strip() for r in regions.split(",") if r.strip()):
        samples = load_eval_samples(raw_dataset_path, split, do, DIRECTION, n)
        digest = _region_hash(samples)
        manifest_obj["regions"][do] = {"n": len(samples), "id_hash": digest}
        for i, s in enumerate(samples):
            user = tmpl.format_user(s["source"], do, DIRECTION)
            rows.append(
                {
                    "idx": i,
                    "do": do,
                    "source": s["source"],  # 방언 (input)
                    "reference": s["reference"],  # 표준어 (gold)
                    "prompt": _chatml(user),
                }
            )
        print(f"  {do}: {len(samples)} samples  id_hash={digest[:16]}…")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    Path(manifest).write_text(
        json.dumps(manifest_obj, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {len(rows)} rows → {out_path}")
    print(f"wrote manifest → {manifest}")
    print("  (re-running build must reproduce these id_hash values, else the set drifted)")


if __name__ == "__main__":
    fire.Fire(build)
