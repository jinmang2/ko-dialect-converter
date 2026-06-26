"""Generate the k-quant variant sweep for the on-device Pareto (brief §3 Tier1).

Thin wrapper over the EXISTING export path (`ko_dialect.export`): merge is already done
(sft_merged), so this converts to f16 GGUF once, then quantizes to each k-quant target.
Decision O2: GGUF k-quant only (not bnb 4bit) — llama.cpp ships the ARM-optimal kernels.

  python ondevice/quantize/sweep.py \
      --merged outputs/sft_merged \
      --variants Q8_0,Q5_K_M,Q4_K_M,Q4_K_S \
      --out_dir ondevice/models --llama_cpp_dir llama.cpp

Then `adb push ondevice/models/*.gguf /sdcard/...` to the phone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import fire

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from ko_dialect.export import convert_to_gguf, quantize_gguf  # noqa: E402

# Defaults span the size/quality range the brief wants on the frontier.
# IQ4_XS / Q3_K_M are "if possible" extras (older llama-quantize builds may lack IQ types).
DEFAULT_VARIANTS = "Q8_0,Q5_K_M,Q4_K_M,Q4_K_S"


def sweep(
    merged: str = "outputs/sft_merged",
    variants: str = DEFAULT_VARIANTS,
    out_dir: str = "ondevice/models",
    llama_cpp_dir: str = "llama.cpp",
    name: str = "kodialect",
    f16_gguf: str | None = None,
) -> None:
    """Convert merged HF model → f16 GGUF (once) → each quant variant."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    f16_path = Path(f16_gguf) if f16_gguf else out / f"{name}-f16.gguf"
    if not f16_path.exists():
        print(f"[convert] {merged} → {f16_path}")
        convert_to_gguf(merged, str(f16_path), llama_cpp_dir, "f16")
    else:
        print(f"[convert] reuse existing {f16_path}")

    targets = [v.strip() for v in variants.split(",") if v.strip()]
    made = []
    for q in targets:
        dst = out / f"{name}-{q}.gguf"
        print(f"[quantize] {f16_path.name} → {dst.name} ({q})")
        try:
            quantize_gguf(str(f16_path), str(dst), llama_cpp_dir, q)
            mb = round(dst.stat().st_size / (1024 * 1024), 1)
            made.append((q, dst.name, mb))
            print(f"  ✓ {dst.name}  {mb} MB")
        except Exception as e:  # one unsupported quant type shouldn't abort the sweep
            print(f"  ✗ {q} failed: {e}")

    print("\n# variant sizes (on-disk MB):")
    for q, fn, mb in made:
        print(f"  {q:10s} {mb:8.1f}  {fn}")
    if made:
        print("\n# push to phone, e.g.:")
        print(f"#   adb push {out}/{name}-*.gguf /sdcard/Download/")


if __name__ == "__main__":
    fire.Fire(sweep)
