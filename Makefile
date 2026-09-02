.PHONY: venv install install-extras install-llama install-unsloth install-vllm \
        lock upgrade download-model check check-vllm test lint fmt

# ---------------------------------------------------------------------------
# Environment (uv project interface — pyproject.toml + uv.lock)
# ---------------------------------------------------------------------------
# The env lives at ./.venv and is reproduced from uv.lock, so another machine gets the
# same resolution instead of whatever PyPI serves that day. torch is pinned to
# 2.11.0+cu130 and sourced from the CUDA-13 wheel index via [tool.uv.sources].
#
#   fresh machine:  make venv && make install
#
# Three packages stay OUT of the lock on purpose:
#   unsloth / vllm      — CUDA-specific git builds; `uv lock` cannot resolve them here
#                         (vllm's sdist wants /usr/local/cuda/bin/nvcc)
#   llama-cpp-python    — needs CMAKE_ARGS at build time, unexpressible in pyproject
# Install those with the dedicated targets below, AFTER `make install`.

venv:
	uv venv --python 3.11 .venv

# Core + dev, exactly as locked. Also installs this package in editable mode.
install:
	uv sync --extra dev

# Optional extras: track (wandb/mlflow) | datagen | serve | eval | speech
# e.g. make install-extras EXTRAS="--extra track --extra eval"
install-extras:
	uv sync --extra dev $(EXTRAS)

lock:
	uv lock

# Re-resolve within the pyproject constraints (torch stays pinned).
upgrade:
	uv lock --upgrade

install-llama:
	CMAKE_ARGS="-DGGML_CUDA=on" uv pip install "llama-cpp-python[server]" --no-cache-dir

install-unsloth:
#     wheel depends on the CUDA version — see https://github.com/unslothai/unsloth#installation
#     unsloth_zoo is a hard runtime import but NOT declared by the git package:
#     `import unsloth` raises "Please install unsloth_zoo" without it.
	uv pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"
	uv pip install unsloth_zoo xformers
	uv run python -c "import unsloth, torch; print('unsloth OK | torch', torch.__version__)"

install-vllm:
#     RTX 2060 = SM 7.5: flash-attn v2 미지원 → TORCH_SDPA 백엔드 사용
#     vLLM이 torch를 올릴 수 있음 → 설치 후 `make check`로 trl/transformers/peft 호환 재확인 필수
	MAX_JOBS=3 VLLM_TARGET_DEVICE=cuda uv pip install git+https://github.com/vllm-project/vllm.git

# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

download-model:
	mkdir -p models
	uv run huggingface-cli download google/gemma-3-1b-it-qat-q4_0-gguf \
		gemma-3-1b-it-qat-q4_0.gguf \
		--local-dir models/

# ---------------------------------------------------------------------------
# Sanity checks / QA  (uv run == the project venv, no activation needed)
# ---------------------------------------------------------------------------

check:
	uv run python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
	uv run python -c "import trl, peft, transformers, datasets; print('trl/peft/transformers/datasets OK')"
	uv run python -c "import ko_dialect; print('ko_dialect OK')"

check-vllm:
	VLLM_ATTENTION_BACKEND=TORCH_SDPA uv run python -c "import vllm; print('vllm OK')"

test:
	uv run pytest tests/ -q -m "not gpu"

lint:
	uv run ruff check .

fmt:
	uv run ruff format .
