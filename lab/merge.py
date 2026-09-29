"""Merge into a fresh, unquantized base at the recorded revision."""
import argparse
import json
import shutil
from pathlib import Path

from lab.common import adapter_signature, sha, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/qwen3-4b")
    p.add_argument("--adapter", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", choices=["cpu", "cuda:0"], default="cpu")
    a = p.parse_args()
    out = Path(a.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("output must be empty")
    lock = json.loads((Path(a.model)/"model-lock.json").read_text())
    manifest = json.loads((Path(a.adapter)/"manifest.json").read_text())
    if lock != manifest["contract"]["base_lock"]:
        raise ValueError("adapter/base revision mismatch")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
        device_map={"": a.device}, local_files_only=True, trust_remote_code=False)
    model = PeftModel.from_pretrained(base, a.adapter, is_trainable=False,
                                    torch_device=a.device, device_map={"": a.device})
    merged = model.merge_and_unload(safe_merge=True)
    merged.config.use_cache = True
    merged.save_pretrained(out, safe_serialization=True, max_shard_size="4GB")
    AutoTokenizer.from_pretrained(a.model, local_files_only=True).save_pretrained(out)
    shutil.copyfile(Path(a.model)/"model-lock.json", out/"model-lock.json")
    for file in Path(a.model).glob("LICENSE*"):
        shutil.copyfile(file, out/file.name)
    write_json(out/"merge-manifest.json", {"base_lock": lock, "dtype": "bfloat16",
        "adapter_sha256": adapter_signature(a.adapter),
        "training_manifest": manifest,
        "files": {f.name: sha(f) for f in out.iterdir() if f.is_file()}})
    print(f"merged model written to {out}; keep the independent adapter")


if __name__ == "__main__":
    main()
