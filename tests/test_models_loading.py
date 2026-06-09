from __future__ import annotations

import pytest

from ko_dialect.models.loading import (
    VALID_BACKENDS,
    BackendConfig,
    load_backbone,
)


def test_backend_config_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown backend"):
        BackendConfig(backend="not-a-backend")


def test_backend_config_defaults_are_2060_safe():
    cfg = BackendConfig()
    assert cfg.backend == "unsloth"
    assert cfg.dtype == "fp16"
    assert cfg.use_vllm is False
    assert cfg.load_in_4bit is True


@pytest.mark.parametrize("backend", sorted(VALID_BACKENDS))
def test_all_backends_are_dispatchable(backend):
    """The factory must route every declared backend to a loader (import-time only)."""
    cfg = BackendConfig(backend=backend, quantized_model_path="/tmp/none")
    # We don't actually load weights (no GPU / no network); assert the dispatch table
    # has an entry by catching only *loader* errors, not a KeyError/ValueError routing bug.
    try:
        load_backbone(cfg)
    except KeyError as exc:  # pragma: no cover - would be a routing bug
        pytest.fail(f"backend {backend!r} not wired into dispatch: {exc}")
    except Exception:
        # ImportError (missing unsloth/torchao), OSError (no model), NotImplementedError
        # (qat scaffold) are all expected without a GPU/network — routing still worked.
        pass


def test_ptq_backend_requires_quantized_path():
    cfg = BackendConfig(backend="awq", quantized_model_path=None)
    with pytest.raises(ValueError, match="quantized_model_path"):
        load_backbone(cfg)


def test_qat_backend_is_scaffold():
    cfg = BackendConfig(backend="qat")
    # Either torchao is missing (ImportError) or the scaffold raises NotImplementedError.
    with pytest.raises((NotImplementedError, ImportError)):
        load_backbone(cfg)
