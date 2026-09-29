"""Download the pinned model from Hugging Face or byte-matched ModelScope files."""
import argparse
import json
import re
from pathlib import Path

from lab.common import sha, write_json

SOURCES_FILE = Path(__file__).with_name("download-lock.json")


def verify_download(out, tokenizer_only=False):
    """A hub's fallback to an existing local directory does not prove completion."""
    out = Path(out)
    names = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    if not tokenizer_only:
        index = out / "model.safetensors.index.json"
        if index.is_file():
            weights = json.loads(index.read_text())["weight_map"]
            if not weights:
                raise RuntimeError("权重索引为空，下载未完成")
            names += sorted(set(weights.values()))
        else:
            names += ["model.safetensors"]
    missing = [name for name in names if not (out/name).is_file() or (out/name).stat().st_size == 0]
    if missing:
        raise RuntimeError("下载未完成，缺少文件：" + ", ".join(missing)
                           + "。返回已有目录不代表成功；检查网络后重试或使用 --source modelscope。")


def file_matches(path, expected):
    return (path.is_file() and path.stat().st_size == expected["size"]
            and sha(path) == expected["sha256"])


def verify_locked_download(out, tokenizer_only=False):
    """Offline verification against the audited bytes, including every weight shard."""
    out = Path(out)
    spec = json.loads(SOURCES_FILE.read_text())
    lock = {key: spec[key] for key in ("model_id", "revision")}
    if json.loads((out / "model-lock.json").read_text()) != lock:
        raise ValueError("model-lock 与下载校验清单不一致")
    selected = {name: entry for name, entry in spec["files"].items()
                if not tokenizer_only or not name.endswith(".safetensors")}
    for name, entry in selected.items():
        print("校验大小与 SHA-256：" + name, flush=True)
        if not file_matches(out / name, entry):
            raise RuntimeError("下载未完成或文件损坏：" + name)
    verify_download(out, tokenizer_only)
    print(f"校验通过：{len(selected)} 个文件；" + ("仅分词器，不能训练" if tokenizer_only else "完整模型权重已就绪"))


def download_modelscope(model_id, out, tokenizer_only=False):
    """Keep the HF base identity; enforce audited file hashes across providers."""
    spec = json.loads(SOURCES_FILE.read_text())
    if model_id != spec["model_id"]:
        raise ValueError("此魔搭下载清单仅支持 " + spec["model_id"] + "；其他模型请使用 --source huggingface")
    out = Path(out)
    lock = {key: spec[key] for key in ("model_id", "revision")}
    lock_path = out / "model-lock.json"
    if lock_path.exists():
        if json.loads(lock_path.read_text()) != lock:
            raise ValueError("model-lock 与已核对的魔搭文件版本不一致；请保留现有锁，不要混用模型版本")
    elif out.exists() and any(out.iterdir()):
        raise ValueError("unlocked directory is nonempty; choose a new output")
    selected = {name: entry for name, entry in spec["files"].items()
                if not tokenizer_only or not name.endswith(".safetensors")}
    pending = []
    for name, entry in selected.items():
        path = out / name
        if not path.exists():
            pending.append(name)
        elif not file_matches(path, entry):
            raise ValueError(f"本地文件与锁定版本不一致：{path}；请移至其他目录保留后重试，不会自动覆盖")
    if not lock_path.exists():
        write_json(lock_path, lock)
    print(f"下载源：ModelScope；待下载 {len(pending)} 个文件；基座版本：{lock['revision']}", flush=True)
    if pending:
        try:
            from modelscope import snapshot_download
            from modelscope.hub.api import HubApi
        except ImportError as error:
            raise RuntimeError("请先安装魔搭下载依赖：python -m pip install -r requirements-download.txt") from error
        source = spec["modelscope"]
        remote = {f["Path"]: f for f in HubApi().get_model_files(
            source["model_id"], revision=source["revision"], recursive=True, use_cookies=False)}
        for name in pending:
            expected, actual = selected[name], remote.get(name, {})
            if actual.get("Size") != expected["size"] or actual.get("Sha256") != expected["sha256"]:
                raise ValueError(f"魔搭文件与锁定版本不一致：{name}；已停止下载，不能用 master 的新文件替换本次基座")
        snapshot_download(model_id=source["model_id"], revision=source["revision"],
                          local_dir=str(out), cache_dir=str(out/".cache/modelscope"),
                          allow_patterns=pending, max_workers=3)
        for name in pending:
            print("校验 SHA-256：" + name, flush=True)
            if not file_matches(out/name, selected[name]):
                raise RuntimeError("下载结果不完整或校验失败：" + name + "；请保留版本锁并检查网络/文件后重试")
    verify_download(out, tokenizer_only=tokenizer_only)
    write_json(out / "download-report.json", {"source": "modelscope", "base_lock": lock,
               "source_reference": spec["modelscope"], "verified_files": selected})
    print(f"下载校验通过：{len(selected)} 个文件的大小与 SHA-256 匹配锁定版本", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--output", default="models/qwen3-4b")
    p.add_argument("--tokenizer-only", action="store_true")
    p.add_argument("--source", choices=["huggingface", "modelscope", "hf-mirror"], default="huggingface")
    p.add_argument("--check-only", action="store_true", help="offline size/SHA-256 verification of the lesson model")
    a = p.parse_args()
    if a.check_only:
        verify_locked_download(a.output, a.tokenizer_only)
        return
    if a.source == "modelscope":
        download_modelscope(a.model_id, a.output, a.tokenizer_only)
        return
    from huggingface_hub import HfApi, snapshot_download
    endpoint = "https://hf-mirror.com" if a.source == "hf-mirror" else None
    out = Path(a.output)
    spec = json.loads(SOURCES_FILE.read_text())
    pinned_lesson = a.model_id == spec["model_id"]
    lock_path = out / "model-lock.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        if lock["model_id"] != a.model_id:
            raise ValueError("model id differs from existing lock")
        if pinned_lesson and lock["revision"] != spec["revision"]:
            raise ValueError("model-lock 与教程固定版本不一致；请保留旧目录，使用新目录下载本课基座")
    else:
        if out.exists() and any(out.iterdir()):
            raise ValueError("unlocked directory is nonempty; choose a new output")
        revision = (spec["revision"] if pinned_lesson
                    else HfApi(endpoint=endpoint).model_info(a.model_id).sha)
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("expected immutable model commit SHA")
        lock = {"model_id": a.model_id, "revision": revision}
        write_json(lock_path, lock)
    patterns = ["*.json", "*.jinja", "*.txt", "*.model", "README.md", "LICENSE*"]
    if not a.tokenizer_only:
        patterns += ["*.safetensors"]
    snapshot_download(repo_id=a.model_id, revision=lock["revision"],
                      local_dir=str(out), allow_patterns=patterns, endpoint=endpoint)
    verify_download(out, tokenizer_only=a.tokenizer_only)
    if lock == {key: spec[key] for key in ("model_id", "revision")}:
        verify_locked_download(out, a.tokenizer_only)
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
