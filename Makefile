.PHONY: install-torch install install-llama install-eval install-wandb install-vllm install-all \
        download-model check check-vllm

# ---------------------------------------------------------------------------
# Install targets
# ---------------------------------------------------------------------------

# torch는 vLLM 설치 시 자동으로 따라옴 (cu130)
# fresh 환경에서 vLLM 없이 torch만 필요하면: make install-torch
install-torch:
	uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

install:
# 	uv pip install -e .
	uv pip install -r pyproject.toml

# llama-cpp-python requires CMAKE flag for CUDA — cannot be expressed in pyproject.toml
install-llama:
	CMAKE_ARGS="-DGGML_CUDA=on" uv pip install "llama-cpp-python[server]" --no-cache-dir

install-eval:
# 	uv pip install -e ".[eval]"
	uv pip install -r pyproject.toml --extra eval

install-wandb:
# 	uv pip install -e ".[wandb]"
	uv pip install -r pyproject.toml --extra wandb

install-vllm:
# 	RTX 2060 = SM 7.5: flash-attn v2 미지원 → TORCH_SDPA 백엔드 사용 (conda activate.d/env_vars.sh에 설정)
# 	vLLM이 torch를 업그레이드할 수 있음 → 설치 후 make check로 trl/transformers/peft 호환성 재확인 필수
# 	uv pip install vllm
# 	RAM=$(free -m | awk '/^Mem:/{print int($2/1024)}'); CORES=$(nproc); JOBS=$((RAM/4>0?RAM/4:1)); BEST=$((CORES<JOBS?CORES:JOBS)); echo -e "\n💻 논리 코어: ${CORES}개\n🧠 가용 RAM: ${RAM}GB\n🚀 권장 MAX_JOBS=${BEST}\n"
# 	git+https://github.com/vllm-project/vllm.git@v0.4.2
# 	https://github.com/vllm-project/vllm.git@0a5cbf63
	MAX_JOBS=3 VLLM_TARGET_DEVICE=cuda uv pip install git+https://github.com/vllm-project/vllm.git

install-unsloth:
#     unsloth: install separately — wheel depends on CUDA version
#     see: https://github.com/unslothai/unsloth#installation
#     curl -fsSL https://unsloth.ai/install.sh | sh
#     irm https://unsloth.ai/install.ps1 | iex
	uv pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"
	uv pip install xformers

install-all: install install-llama install-eval install-wandb

# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

download-model:
	mkdir -p models
	huggingface-cli download google/gemma-3-1b-it-qat-q4_0-gguf \
		gemma-3-1b-it-qat-q4_0.gguf \
		--local-dir models/

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

check:
	python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
	python -c "from llama_cpp import Llama; print('llama-cpp-python OK')"
	python -c "import trl, peft, transformers, datasets; print('trl/peft/transformers/datasets OK')"

check-vllm:
	VLLM_ATTENTION_BACKEND=TORCH_SDPA python -c "import vllm; print('vllm OK')"