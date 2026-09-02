.PHONY: install install-gpu install-all install-vllm install-llama install-webui \
        download-model check check-vllm

# ---------------------------------------------------------------------------
# Install targets — uv owns the environment
# ---------------------------------------------------------------------------
# `uv sync` builds .venv from pyproject.toml + uv.lock, so every machine gets the
# same resolution. Extras are opt-in: see [project.optional-dependencies].
# torch/torchvision come from the cu130 index wired up in [tool.uv.sources].

# Everyday dev env: core stack + pytest/ruff/mypy. No CUDA-only kernels.
install:
	uv sync --extra dev

# Training box: adds unsloth + xformers on top of dev (CUDA-only wheels).
install-gpu:
	uv sync --extra dev --extra gpu

# Everything the repo's scripts can use, except vLLM (which compiles from source).
EXTRAS = --extra dev --extra gpu --extra track --extra datagen --extra serve --extra speech
install-all:
	uv sync $(EXTRAS)

# vLLM is pinned to a git rev in [tool.uv.sources]. VLLM_USE_PRECOMPILED reuses the
# upstream prebuilt kernels instead of a multi-hour local CUDA build.
# Turing = SM 7.5: flash-attn v2 미지원 → 실행 시 VLLM_ATTENTION_BACKEND=TORCH_SDPA (check-vllm 참고).
install-vllm:
	VLLM_USE_PRECOMPILED=1 MAX_JOBS=3 VLLM_TARGET_DEVICE=cuda uv sync $(EXTRAS) --extra vllm

# llama-cpp-python needs a CMAKE flag that pyproject.toml cannot express.
install-llama:
	CMAKE_ARGS="-DGGML_CUDA=on" uv pip install "llama-cpp-python[server]" --no-cache-dir

# open-webui's pins resolve only to yanked mlflow releases, so it cannot share this
# project's dependency resolution — run it as an isolated tool instead.
install-webui:
	uv tool install open-webui

# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

download-model:
	mkdir -p models
	uv run huggingface-cli download google/gemma-3-1b-it-qat-q4_0-gguf \
		gemma-3-1b-it-qat-q4_0.gguf \
		--local-dir models/

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

check:
	uv run python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
	uv run python -c "from llama_cpp import Llama; print('llama-cpp-python OK')"
	uv run python -c "import trl, peft, transformers, datasets; print('trl/peft/transformers/datasets OK')"

check-vllm:
	VLLM_ATTENTION_BACKEND=TORCH_SDPA uv run python -c "import vllm; print('vllm OK')"