"""Switching download providers must preserve the pinned model, even after a failed fetch."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from lab import download


@pytest.fixture
def model_repo(tmp_path, monkeypatch):
    files = {"config.json": b'{"model_type":"qwen3"}', "tokenizer.json": b'{"test":1}',
             "tokenizer_config.json": b'{"chat_template":"test"}',
             "model.safetensors": b"a tiny pretend weight file"}
    lock = {"model_id": "Qwen/test", "revision": "a" * 40}
    spec = {**lock, "modelscope": {"model_id": "Qwen/test", "revision": "master"},
            "files": {name: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                      for name, data in files.items()}}
    path = tmp_path / "download-lock.json"
    path.write_text(json.dumps(spec))
    monkeypatch.setattr(download, "SOURCES_FILE", path)
    out = tmp_path / "model"
    out.mkdir()
    (out / "model-lock.json").write_text(json.dumps(lock))
    for name, data in files.items():
        if not name.endswith(".safetensors"):
            (out / name).write_bytes(data)
    metadata = [{"Path": name, "Type": "blob", "Size": entry["size"], "Sha256": entry["sha256"]}
                for name, entry in spec["files"].items()]
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        for name in kwargs["allow_patterns"]:
            (Path(kwargs["local_dir"]) / name).write_bytes(files[name])
        return kwargs["local_dir"]

    api = SimpleNamespace(get_model_files=lambda *args, **kwargs: metadata)
    monkeypatch.setitem(sys.modules, "modelscope", SimpleNamespace(snapshot_download=snapshot_download))
    monkeypatch.setitem(sys.modules, "modelscope.hub.api", SimpleNamespace(HubApi=lambda: api))
    # A ModelScope run must not depend on a reachable Hugging Face API.
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    return out, lock, files, metadata, calls


def test_modelscope_completes_only_missing_files_and_keeps_base_lock(model_repo):
    out, lock, files, metadata, calls = model_repo
    before = (out / "model-lock.json").read_bytes()
    download.download_modelscope(lock["model_id"], out)
    assert (out / "model-lock.json").read_bytes() == before
    assert all((out / name).read_bytes() == data for name, data in files.items())
    assert calls[0]["allow_patterns"] == ["model.safetensors"]
    assert json.loads((out / "download-report.json").read_text())["source"] == "modelscope"
    # A complete cache works offline and doesn't ask the SDK to fetch it again.
    download.download_modelscope(lock["model_id"], out)
    assert len(calls) == 1


def test_modelscope_rejects_different_remote_weights_before_download(model_repo):
    out, lock, files, metadata, calls = model_repo
    metadata[-1]["Sha256"] = "0" * 64
    with pytest.raises(ValueError, match="魔搭文件与锁定版本不一致"):
        download.download_modelscope(lock["model_id"], out)
    assert not calls and not (out / "model.safetensors").exists()


def test_modelscope_does_not_overwrite_changed_tokenizer(model_repo):
    out, lock, files, metadata, calls = model_repo
    changed = b'{"test":2}'
    (out / "tokenizer.json").write_bytes(changed)
    with pytest.raises(ValueError, match="本地文件与锁定版本不一致"):
        download.download_modelscope(lock["model_id"], out)
    assert (out / "tokenizer.json").read_bytes() == changed
    assert not calls


@pytest.mark.parametrize("bad_result", [None, b"x" * 26])
def test_modelscope_checks_actual_download_not_just_sdk_return(model_repo, monkeypatch, bad_result):
    out, lock, files, metadata, calls = model_repo

    def failed_download(**kwargs):
        if bad_result is not None:
            (out / "model.safetensors").write_bytes(bad_result)
        return str(out)

    monkeypatch.setitem(sys.modules, "modelscope", SimpleNamespace(snapshot_download=failed_download))
    with pytest.raises(RuntimeError, match="下载结果不完整或校验失败"):
        download.download_modelscope(lock["model_id"], out)
    assert not (out / "download-report.json").exists()


def test_tokenizer_only_skips_weights_and_checks_lock(model_repo):
    out, lock, files, metadata, calls = model_repo
    download.download_modelscope(lock["model_id"], out, tokenizer_only=True)
    assert not calls and not (out / "model.safetensors").exists()
    (out / "model-lock.json").write_text(json.dumps({**lock, "revision": "b" * 40}))
    with pytest.raises(ValueError, match="model-lock"):
        download.download_modelscope(lock["model_id"], out)


def test_huggingface_partial_local_fallback_is_not_success(tmp_path):
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (tmp_path / name).write_text('{}')
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"layer.a": "model-00001.safetensors", "layer.b": "model-00002.safetensors"}}))
    (tmp_path / "model-00001.safetensors").write_bytes(b"one shard")
    with pytest.raises(RuntimeError, match="model-00002.safetensors"):
        download.verify_download(tmp_path)
    download.verify_download(tmp_path, tokenizer_only=True)


def test_offline_check_rejects_equal_size_corruption(model_repo):
    out, lock, files, metadata, calls = model_repo
    for name, data in files.items():
        (out / name).write_bytes(data)
    download.verify_locked_download(out)
    (out / "model.safetensors").write_bytes(b"x" * len(files["model.safetensors"]))
    with pytest.raises(RuntimeError, match="文件损坏"):
        download.verify_locked_download(out)
    assert not calls


def test_mirror_uses_explicit_endpoint_and_keeps_revision(model_repo, monkeypatch):
    out, lock, files, metadata, calls = model_repo
    hub_calls = []
    def snapshot_download(**kwargs):
        hub_calls.append(kwargs)
        for name, data in files.items():
            (Path(kwargs["local_dir"]) / name).write_bytes(data)
    def api(**kwargs):
        raise AssertionError("an existing revision must not be resolved again")
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=api, snapshot_download=snapshot_download))
    monkeypatch.setattr(sys, "argv", ["download", "--source", "hf-mirror", "--model-id", lock["model_id"], "--output", str(out)])
    download.main()
    assert hub_calls[0]["endpoint"] == "https://hf-mirror.com"
    assert hub_calls[0]["revision"] == lock["revision"]
    assert json.loads((out / "model-lock.json").read_text()) == lock


@pytest.mark.parametrize("source", ["huggingface", "hf-mirror"])
def test_first_download_uses_lesson_revision_not_remote_latest(model_repo, tmp_path, monkeypatch, source):
    old_out, lock, files, metadata, calls = model_repo
    out = tmp_path / "fresh-model"
    hub_calls = []

    def snapshot_download(**kwargs):
        hub_calls.append(kwargs)
        for name, data in files.items():
            (Path(kwargs["local_dir"]) / name).write_bytes(data)

    def api(**kwargs):
        raise AssertionError("lesson downloads must not resolve mutable remote HEAD")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=api, snapshot_download=snapshot_download))
    monkeypatch.setattr(sys, "argv", ["download", "--source", source, "--model-id", lock["model_id"], "--output", str(out)])
    download.main()
    assert hub_calls[0]["revision"] == lock["revision"]
    assert json.loads((out / "model-lock.json").read_text()) == lock


def test_old_base_revision_is_not_silently_mixed_into_the_lesson(model_repo, monkeypatch):
    out, lock, files, metadata, calls = model_repo
    old = {**lock, "revision": "b" * 40}
    (out / "model-lock.json").write_text(json.dumps(old))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=None, snapshot_download=None))
    monkeypatch.setattr(sys, "argv", ["download", "--model-id", lock["model_id"], "--output", str(out)])
    with pytest.raises(ValueError, match="教程固定版本不一致"):
        download.main()
    assert json.loads((out / "model-lock.json").read_text()) == old
