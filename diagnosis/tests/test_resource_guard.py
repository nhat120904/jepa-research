import pytest

from scripts import _resource_guard


def test_jepa_wm_droid_is_blocked_on_12_gib_gpu(monkeypatch):
    monkeypatch.delenv("CAI_JEPA_ALLOW_HEAVY_MODEL", raising=False)
    monkeypatch.setattr(_resource_guard, "_cuda_total_memory_gib", lambda: 12.0)

    with pytest.raises(RuntimeError, match="at least 22.0 GiB"):
        _resource_guard.preflight_model_load("jepa_wm_droid", "cuda")


def test_jepa_wm_droid_override_is_explicit(monkeypatch):
    monkeypatch.setenv("CAI_JEPA_ALLOW_HEAVY_MODEL", "1")
    monkeypatch.setattr(_resource_guard, "_cuda_total_memory_gib", lambda: 12.0)

    _resource_guard.preflight_model_load("jepa_wm_droid", "cuda")
