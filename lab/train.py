"""Single NVIDIA GPU QLoRA lesson. Run `python -m lab.train --help` first."""
import argparse
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

from lab.common import sha, tokenizer_signature, verify_bundle, write_json
from lab.data import read_sft


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/qwen3-4b")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--accum", type=int, default=16)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-draft", action="store_true")
    p.add_argument("--resume")
    p.add_argument("--stop-after", type=int, default=0)
    a = p.parse_args()
    if min(a.rank, a.accum, a.max_length) <= 0 or a.lr <= 0 or a.epochs <= 0:
        raise ValueError("positive training settings required")
    if a.max_steps == 0 or a.max_steps < -1 or a.limit < 0 or a.stop_after < 0:
        raise ValueError("invalid step/limit settings")
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from lab.training import make_trainer, check_supervision
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("main lesson requires a BF16-capable NVIDIA CUDA GPU")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("this lesson uses one process and one GPU")
    folder, out = Path(a.data), Path(a.output)
    bundle = verify_bundle(folder)
    if bundle["draft"] and not a.allow_draft:
        raise ValueError("draft data requires --allow-draft; results are exploratory, not formal claims")
    if not bundle["formal"] and a.max_steps <= 0:
        raise ValueError("epoch experiments require the formal 600-episode science bundle")
    if out.exists() and any(out.iterdir()) and not a.resume:
        raise ValueError("output nonempty: choose new directory or resume")
    lock = json.loads((Path(a.model)/"model-lock.json").read_text())
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True, trust_remote_code=False)
    if tokenizer_signature(tok) != bundle["tokenizer_sha256"]:
        raise ValueError("tokenizer differs from prepared data")
    tok.padding_side = "right"
    # CLI 读取 lab.prepare 从 dataset.json 生成的快照；Notebook 直接在内存中生成同样的 rows。
    rows = read_sft(folder, "train", bundle)
    if a.limit:
        rows = rows[:a.limit]
    val_rows = read_sft(folder, "val", bundle)
    if not rows:
        raise ValueError("training set is empty")
    for row in [*rows, *val_rows]:
        if len(row["input_ids"]) > a.max_length:
            raise ValueError("sample length exceeds training limit; rewrite and prepare again")
    packages = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "trl", "peft", "bitsandbytes", "accelerate", "datasets")}
    config = {k: v for k,v in vars(a).items() if k not in {"resume", "stop_after", "output", "model", "data"}}
    contract = {"config": config, "base_lock": lock, "data_manifest_sha256": sha(folder/"manifest.json"),
                "code": {f: sha(Path(__file__).with_name(f)) for f in ("train.py", "training.py", "common.py")},
                "packages": packages}
    resume_step = 0
    if a.resume:
        previous = json.loads((out/"manifest.json").read_text())
        if previous["contract"] != contract:
            raise ValueError("resume configuration/data/code/environment differs")
        checkpoint = Path(a.resume).resolve()
        if checkpoint.parent != out.resolve():
            raise ValueError("resume checkpoint must belong to this output directory")
        for f in ("trainer_state.json", "optimizer.pt", "scheduler.pt"):
            if not (checkpoint/f).is_file():
                raise ValueError(f"not a resumable training checkpoint: missing {f}")
        resume_step = json.loads((checkpoint/"trainer_state.json").read_text())["global_step"]
    set_seed(a.seed)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, local_files_only=True, trust_remote_code=False, torch_dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=a.rank, lora_alpha=2*a.rank, lora_dropout=0.05,
        bias="none", target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    trainable = {n: p for n,p in model.named_parameters() if p.requires_grad}
    if not trainable or any("lora_" not in n for n in trainable):
        raise ValueError("unexpected trainable parameters")
    initial = {n: p.detach().cpu().clone() for n,p in trainable.items()}
    def dataset(items):
        # D5：只将 Token 与监督掩码交给训练器，case_id/prompt 等保留用于人工核对。
        return Dataset.from_list([{k: row[k] for k in ("input_ids", "completion_mask")} for row in items])
    trainer = make_trainer(model, tok, dataset(rows), dataset(val_rows) if val_rows else None, str(out),
        epochs=a.epochs, max_steps=a.max_steps, accum=a.accum, lr=a.lr,
        max_length=a.max_length, seed=a.seed, stop_after=a.stop_after)
    check_supervision(trainer, rows, tok.eos_token_id)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"contract": contract, "draft": bundle["draft"], "gpu": torch.cuda.get_device_name(0),
                "cuda": torch.version.cuda, "supervised_rows": len(rows),
                "trainable_parameters": sum(p.numel() for p in trainable.values())}
    write_json(out/"manifest.json", manifest)
    (out/"dependencies.txt").write_text("\n".join(sorted(
        f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())) + "\n")
    driver = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
    (out/"nvidia-smi.txt").write_text(driver.stdout + driver.stderr)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    result = trainer.train(resume_from_checkpoint=a.resume)
    if trainer.state.global_step <= resume_step:
        raise RuntimeError("no new optimizer step; inspect resume/max-steps")
    if not math.isfinite(result.training_loss) or any(not torch.isfinite(p).all() for p in trainable.values()):
        raise RuntimeError("non-finite loss or adapter parameters; inspect data and precision")
    changed = [n for n,p in trainable.items() if not torch.equal(initial[n], p.detach().cpu())]
    if not changed:
        raise RuntimeError("no LoRA parameter changed")
    trainer.save_model(str(out/"adapter"))
    tok.save_pretrained(out/"adapter")
    shutil.copyfile(out/"manifest.json", out/"adapter"/"manifest.json")
    trainer.save_state()
    metrics = dict(result.metrics, elapsed_seconds=time.monotonic()-started,
                   peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                   peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                   changed_lora_tensors=len(changed), resumed_from_step=resume_step,
                   final_step=trainer.state.global_step)
    if val_rows:
        metrics.update(trainer.evaluate())
    write_json(out/f"metrics-step-{trainer.state.global_step}.json", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
