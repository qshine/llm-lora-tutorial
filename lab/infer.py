"""Controlled HF generation: original model, prompt baseline, or LoRA adapter."""
import argparse
import importlib.metadata
import json
import time
from pathlib import Path

from lab.common import (adapter_signature, build_messages, digest, read_jsonl, sha, tokenizer_signature,
                        verify_bundle, write_json, write_jsonl)
from lab.data import expand_episode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/qwen3-4b")
    p.add_argument("--data", required=True)
    p.add_argument("--variant", choices=["A", "B", "C"], default="B")
    p.add_argument("--adapter")
    p.add_argument("--merged", action="store_true")
    p.add_argument("--precision", choices=["nf4", "bf16"], default="nf4")
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--mode", choices=["fixed", "rollout"], default="fixed")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--limit", type=int, default=0, help="limit episodes for a quick inspection; 0 means all")
    p.add_argument("--freeze")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if (a.variant == "C") != bool(a.adapter or a.merged) or (a.adapter and a.merged):
        raise ValueError("C requires exactly one adapter or merged model; A/B use only the base")
    if a.max_new_tokens <= 0 or a.max_length <= a.max_new_tokens:
        raise ValueError("invalid context/output budget")
    out = Path(a.output)
    if out.exists() or out.with_suffix(".meta.json").exists():
        raise ValueError("output already exists")
    bundle = verify_bundle(a.data)
    if not bundle["counts"].get(a.split, 0):
        raise ValueError("selected split is empty; prepare a dataset containing the requested split")
    lock = json.loads((Path(a.model)/"model-lock.json").read_text())
    artifact = None
    if a.adapter:
        manifest = json.loads((Path(a.adapter)/"manifest.json").read_text())
        if manifest["contract"]["base_lock"] != lock:
            raise ValueError("adapter/base revision mismatch")
        artifact = adapter_signature(a.adapter)
    if a.merged:
        artifact = json.loads((Path(a.model)/"merge-manifest.json").read_text())["adapter_sha256"]
        if a.precision != "bf16":
            raise ValueError("first merged-model validation uses BF16")
    decoding = {"do_sample": False, "max_new_tokens": a.max_new_tokens,
                "max_length": a.max_length, "precision": a.precision, "seed": 42}
    if a.split == "test":
        if a.limit:
            raise ValueError("final test evaluates every frozen case; do not cherry-pick with --limit")
        if not a.freeze or not bundle["formal"] or bundle["draft"]:
            raise ValueError("test requires reviewed formal data and a frozen candidate")
        freeze = json.loads(Path(a.freeze).read_text())
        if freeze["data_sha256"] != sha(Path(a.data)/"manifest.json") or freeze["base_lock"] != lock:
            raise ValueError("frozen model/data contract changed")
        if freeze["decoding"] != decoding:
            raise ValueError("decoding differs from frozen protocol")
        if freeze["inference_code_sha256"] != sha(__file__):
            raise ValueError("inference code changed after freezing")
        if a.variant == "C" and artifact != freeze["adapter_sha256"]:
            raise ValueError("candidate adapter changed after freezing")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from peft import PeftModel
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("this inference lesson requires a BF16-capable CUDA GPU")
    set_seed(42)
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True, trust_remote_code=False)
    if tokenizer_signature(tok) != bundle["tokenizer_sha256"]:
        raise ValueError("tokenizer mismatch")
    kwargs = dict(torch_dtype=torch.bfloat16, device_map={"": 0},
                  attn_implementation="sdpa", local_files_only=True, trust_remote_code=False)
    if a.precision == "nf4":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(a.model, **kwargs)
    if a.adapter:
        model = PeftModel.from_pretrained(model, a.adapter, is_trainable=False)
    model.eval()
    model.config.use_cache = True
    folder = Path(a.data)
    cards = read_jsonl(folder/"facts.jsonl")
    fewshots = json.loads((folder/"fewshots.json").read_text())
    episodes = read_jsonl(folder/f"{a.split}.episodes.jsonl")
    if a.limit < 0:
        raise ValueError("limit cannot be negative")
    if a.limit:
        episodes = episodes[:a.limit]
    if not episodes:
        raise ValueError("selected split is empty; prepare a dataset containing the requested split")
    results = []
    torch.cuda.reset_peak_memory_stats()
    for ep in episodes:
        generated_history = []
        for case in expand_episode(ep):
            if a.mode == "rollout":
                case["history"] = list(generated_history)
            messages = build_messages(cards, case, a.variant, fewshots)
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tok(text, add_special_tokens=False, return_tensors="pt").to("cuda:0")
            if inputs.input_ids.shape[1] + a.max_new_tokens > a.max_length:
                raise ValueError(f"context budget exceeded at {case['case_id']}; no silent truncation")
            torch.cuda.synchronize()
            started = time.monotonic()
            with torch.inference_mode():
                output = model.generate(**inputs, do_sample=False, max_new_tokens=a.max_new_tokens,
                                        eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            tokens = output[0, inputs.input_ids.shape[1]:].tolist()
            reply = tok.decode(tokens, skip_special_tokens=True).strip()
            results.append({**case, "reply": reply, "prompt_sha256": digest(messages),
                            "seconds": elapsed, "generated_tokens": len(tokens),
                            "ended_with_eos": bool(tokens and tokens[-1] == tok.eos_token_id)})
            generated_history.extend([{"role": "user", "content": case["user"]},
                                      {"role": "assistant", "content": reply}])
    protocol = {"data_sha256": sha(folder/"manifest.json"), "base_lock": lock,
                "decoding": decoding, "split": a.split, "mode": a.mode,
                "limit": a.limit, "prompt_variant": "A" if a.variant == "A" else "B",
                "inference_code_sha256": sha(__file__),
                "gpu": torch.cuda.get_device_name(0),
                "packages": {n: importlib.metadata.version(n) for n in
                    ["torch", "transformers", "peft"] + (["bitsandbytes"] if a.precision == "nf4" else [])}}
    write_jsonl(out, results)
    write_json(out.with_suffix(".meta.json"), {"variant": a.variant, "protocol": protocol,
        "adapter_sha256": artifact, "merged": a.merged,
        "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30})
    print(f"wrote {len(results)} replies to {out}")


if __name__ == "__main__":
    main()
