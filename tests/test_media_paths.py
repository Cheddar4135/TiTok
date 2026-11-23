"""MEDIA_ROOT resolution and the media-file check in both dataset copies.

Needs the training environment (torch, decord, ...); skipped otherwise.
"""
import importlib
import os
import sys

import pytest

from conftest import REPO_ROOT


@pytest.fixture(params=["titok", "sft"])
def dataset_module(request):
    tree = os.path.join(REPO_ROOT, request.param)
    for name in [m for m in sys.modules if m == "qwenvl" or m.startswith("qwenvl.")]:
        del sys.modules[name]          # both trees use the package name "qwenvl"
    sys.path.insert(0, tree)
    try:
        return importlib.import_module("qwenvl.data.dataset")
    except ImportError as e:
        pytest.skip(f"training dependencies not installed: {e}")
    finally:
        sys.path.remove(tree)


def test_resolve_relative_path(dataset_module, monkeypatch, tmp_path):
    monkeypatch.setenv("MEDIA_ROOT", str(tmp_path))
    out = dataset_module.resolve_media_path("unav_100/videos/a.mp4")
    assert out == os.path.join(str(tmp_path), "unav_100/videos/a.mp4")


def test_absolute_path_unchanged(dataset_module, tmp_path):
    p = str(tmp_path / "a.mp4")
    assert dataset_module.resolve_media_path(p) == p
    assert dataset_module.resolve_media_path(None) is None


def test_relative_path_needs_media_root(dataset_module, monkeypatch):
    monkeypatch.delenv("MEDIA_ROOT", raising=False)
    with pytest.raises(ValueError, match="MEDIA_ROOT"):
        dataset_module.resolve_media_path("unav_100/videos/a.mp4")


def test_check_media_files(dataset_module, tmp_path, capsys):
    real = tmp_path / "a.mp4"
    real.write_bytes(b"")
    dataset_module.check_media_files([{"video": str(real)}, {"video": str(tmp_path / "missing.mp4")}])
    assert "1/2" in capsys.readouterr().out
    with pytest.raises(FileNotFoundError, match="MEDIA_ROOT"):
        dataset_module.check_media_files([{"video": str(tmp_path / "x.mp4"), "audio": str(tmp_path / "x.wav")}])
