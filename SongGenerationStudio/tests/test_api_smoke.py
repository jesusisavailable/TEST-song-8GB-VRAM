"""
test_api_smoke.py - Offline unit tests for SongGeneration Studio backend helpers.

These tests do NOT require GPU, ffmpeg, httpx, or a running server - they exercise
the upload/VRAM/export guard logic in main.py directly. Run with:

    python -m pytest SongGenerationStudio/tests/test_api_smoke.py -v

Requires only: fastapi, uvicorn, requests (already in the venv) + pytest.
"""
import os
import sys

import pytest

# conftest already put SongGenerationStudio/ on sys.path and chdir'd into it.
import main as app_module
from main import (
    _safe_filename,
    _validate_reference_upload,
    _vram_admission,
    _VALID_EXPORT,
)


# ---------------------------------------------------------------------------
# _safe_filename
# ---------------------------------------------------------------------------
def test_safe_filename_strips_paths():
    assert _safe_filename("song.wav") == "song.wav"
    assert _safe_filename("..\\..\\evil.wav") == "evil.wav"
    assert _safe_filename("..\\\\..\\\\evil.wav") == "evil.wav"
    assert _safe_filename("C:\\Windows\\secret.wav") == "secret.wav"
    assert _safe_filename("a/b/c.mp3") == "c.mp3"


def test_safe_filename_drops_unsafe_chars_and_empty():
    assert _safe_filename("../boom!.wav") == "boom.wav"
    assert _safe_filename("") == "audio"
    assert _safe_filename("....") == "audio"  # no valid chars left


# ---------------------------------------------------------------------------
# _validate_reference_upload
# ---------------------------------------------------------------------------
class FakeFile:
    def __init__(self, filename, headers=None):
        self.filename = filename
        self.headers = headers or {}


def _as_http_exc(exc_info):
    return exc_info.value


def test_validate_rejects_missing_filename():
    with pytest.raises(Exception):
        _validate_reference_upload(FakeFile(""))
    with pytest.raises(Exception):
        _validate_reference_upload(FakeFile(None))


def test_validate_rejects_bad_extension():
    with pytest.raises(Exception) as ei:
        _validate_reference_upload(FakeFile("song.exe"))
    assert ei.value.status_code == 400


def test_validate_oversize_via_content_length():
    big = {"content-length": str((int(app_module.MAX_UPLOAD_MB) + 1) * 1024 * 1024)}
    with pytest.raises(Exception) as ei:
        _validate_reference_upload(FakeFile("song.wav", big))
    assert ei.value.status_code == 413


def test_validate_accepts_valid():
    fname = _validate_reference_upload(FakeFile("my song v2.wav", {"content-length": "1024"}))
    assert fname == "my song v2.wav"


# ---------------------------------------------------------------------------
# _vram_admission
# ---------------------------------------------------------------------------
def test_vram_admission_no_gpu(monkeypatch):
    monkeypatch.setattr(app_module, "refresh_gpu_info", lambda: {"available": False, "gpu": None})
    assert _vram_admission("songgeneration_base") == "GPU not detected"


def test_vram_admission_low_free_memory(monkeypatch):
    # GTX 1070 has 8 GB total; force low free memory so the 8GB base model is rejected.
    fake = {"available": True, "gpu": {"free_gb": 6.0, "total_gb": 8.0}}
    monkeypatch.setattr(app_module, "refresh_gpu_info", lambda: fake)
    assert _vram_admission("songgeneration_base") is not None


def test_vram_admission_sufficient(monkeypatch):
    fake = {"available": True, "gpu": {"free_gb": 12.0, "total_gb": 16.0}}
    monkeypatch.setattr(app_module, "refresh_gpu_info", lambda: fake)
    assert _vram_admission("songgeneration_base") is None


# ---------------------------------------------------------------------------
# Export format allow-list
# ---------------------------------------------------------------------------
def test_valid_export_formats():
    assert "mp3" in _VALID_EXPORT
    assert "flac" in _VALID_EXPORT
    assert "mp4" not in _VALID_EXPORT  # handled by the dedicated /video endpoint


# ---------------------------------------------------------------------------
# Config safety limits surfaced
# ---------------------------------------------------------------------------
def test_config_safety_limits_exist():
    assert hasattr(app_module, "MAX_UPLOAD_MB")
    assert hasattr(app_module, "MAX_REFERENCE_SECONDS")
    assert hasattr(app_module, "TEMP_DIR")


def test_seed_schema_field():
    from schemas import SongRequest, Section
    req = SongRequest(title="t", sections=[Section(type="verse", lyrics="hi")], seed=42)
    assert req.seed == 42