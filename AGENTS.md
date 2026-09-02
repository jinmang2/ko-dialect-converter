# AGENTS.md — KoDialect AI Agent Guide

This file is for AI coding agents (Claude Code, Cursor, GitHub Copilot, etc.).
It describes how to navigate this repo, run tasks, and contribute safely.
---

## TL;DR for Agents

1. **Read `CLAUDE.md` first** — it has hardware constraints, key commands, and conventions.
2. **Never use `bf16`** — RTX 2060 (Turing, SM 7.5) does not support it.
3. **Run `pytest tests/ -x -q` to verify changes** — GPU tests are skipped in CI.
4. **Linting is required before any PR** — `ruff check . && ruff format --check .`
5. **Do not commit large binary files** — models go to HuggingFace Hub.

---

## Environment Setup

```bash
# Python 3.11+ required
python --version

# Install package + dev deps
pip install -e ".[dev]"

# Install pre-commit hooks (run once after cloning)
pre-commit install

# Verify GPU + quantization stack
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
# Expected for RTX 2060: GeForce RTX 2060, False
```

### Unsloth install note
Unsloth requires a CUDA-specific wheel. Follow the [official install guide](https://github.com/unslothai/unsloth).
For RTX 2060 (CUDA 11.x / 12.x):
```bash
pip install "unsloth[cu121-torch250] @ https://github.com/unslothai/unsloth/..."
# Check unsloth README for the exact wheel URL matching your CUDA version
```

---

## Key Technologies

| Library | Role |
|---|---|
| `unsloth` | 2–5× faster LoRA training, reduced VRAM via custom CUDA kernels |
| `trl` (`SFTTrainer`) | Supervised fine-tuning loop with HF ecosystem integration |
| `peft` | LoRA adapter management (save, load, merge) |
| `bitsandbytes` | 4-bit NF4 quantization for QLoRA backbone |
| `transformers` | Model/tokenizer loading, generation |
| `datasets` | Data loading, caching, map-style preprocessing |
| `evaluate` + `sacrebleu` | BLEU, ChrF scoring |
| `llama.cpp` (external) | GGUF conversion and quantization |

Mark GPU tests with `@pytest.mark.gpu` and skip in CI:
```python
# conftest.py
import pytest, torch

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
```

Run all non-GPU tests:
```bash
pytest tests/ -x -q -m "not gpu"
```

---

## CI Behavior

All CI jobs run on `ubuntu-latest` (no GPU). Jobs:

1. **lint** — `ruff check`, `ruff format --check`
2. **test** — `pytest tests/ -x -q -m "not gpu"`
3. **import-check** — `python -c "import ko_dialect"` (catches missing `__init__` exports)

CI fails → GitHub Action automatically opens an Issue tagged `ci-failure` with the workflow run link.

---

## Code Conventions

- **No hardcoded paths** — use `pathlib.Path` relative to project root or config
- **No print statements in library code** — use `logging.getLogger(__name__)`
- **Type hints on all public functions** — `def load_model(path: str) -> PreTrainedModel:`
- **Docstrings only for non-obvious behavior** — not for wrapper boilerplate
- **YAML configs are the source of truth** for hyperparameters
