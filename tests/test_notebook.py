"""Run the notebook's actual training/save/reload code with a tiny CPU Qwen model."""
import json
import math
from pathlib import Path
import re
import shutil
import time
import gc
import importlib.metadata

import pytest
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

from lab.common import write_json, sha, adapter_signature, tokenizer_signature, build_messages, render_pair
from lab.data import read_dataset, expand_episode, tag_splits, validate
from test_bundle import Tokenizer
from test_training import tiny_model

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/01_gpu_train.ipynb"


def cells():
    return json.loads(NOTEBOOK.read_text())["cells"]


def source(cell_id):
    return "".join(next(cell for cell in cells() if cell["id"] == cell_id)["source"])


def test_notebook_code_is_visible_unexecuted_and_explained_below():
    notebook_cells = cells()
    assert len({cell["id"] for cell in notebook_cells}) == len(notebook_cells)
    for i, cell in enumerate(notebook_cells):
        if cell["cell_type"] != "code":
            continue
        assert cell["outputs"] == [] and cell["execution_count"] is None
        code = "".join(cell["source"])
        compile(code, cell["id"], "exec")
        assert not any(wrapper in code for wrapper in ("subprocess", "train_python", "pip install", "lab.train", "make_trainer"))
        next_cell = notebook_cells[i + 1]
        assert next_cell["cell_type"] == "markdown" and "为什么" in "".join(next_cell["source"])
    # Local client examples are valid Python even though they are Markdown, not GPU cells.
    for cell in notebook_cells:
        if cell["cell_type"] == "markdown":
            for example in re.findall(r"```python\n(.*?)\n```", "".join(cell["source"]), re.S):
                compile(example, cell["id"], "exec")


def test_save_requires_finished_training_not_just_a_directory(tmp_path):
    from types import SimpleNamespace
    with pytest.raises(AssertionError, match="未达到计划步数"):
        exec(source("save"), {"trainer": SimpleNamespace(state=SimpleNamespace(global_step=0)), "MAX_STEPS": 10})
    assert not (tmp_path / "training-complete.json").exists()


def test_notebook_reads_side_by_side_splits_from_the_real_file():
    # 权重检查和分词器加载需要 GPU 主机上的完整模型；这里只执行 D1 读数据的部分。
    code = source("data")
    code = code[code.index("# D1"):]
    env = dict(json=json, DATA_FILE=ROOT / "data/dataset.json", tag_splits=tag_splits, validate=validate)
    exec(code, env)
    assert env["counts"] == {"train": 500, "val": 50, "test": 50} and env["is_draft"]
    assert env["episodes"] == read_dataset(ROOT / "data/dataset.json")[0]


def test_notebook_builds_rows_from_one_file_without_validation_or_future_leakage():
    episodes, facts = read_dataset(ROOT / "data/dataset.json")
    env = dict(episodes=episodes, facts=facts, tokenizer=Tokenizer(), MAX_LENGTH=4096,
               ACCUMULATION=16, EPOCHS=1, math=math,
               Dataset=Dataset, expand_episode=expand_episode,
               build_messages=build_messages, render_pair=render_pair)
    exec(source("prepare-data"), env)
    rows = env["rows"]
    assert len(rows) == 1150
    assert len(env["val_rows"]) == 115 and env["MAX_STEPS"] == 72
    assert all(not row["case_id"].startswith((episodes[-1]["id"], episodes[-2]["id"])) for row in rows)
    first, second = episodes[0]["turns"]
    assert "小学学段：1-2" in rows[0]["prompt"]
    assert rows[0]["completion"] == first["assistant"] + Tokenizer.eos_token
    assert second["user"] not in rows[0]["prompt"]
    assert first["assistant"] in rows[1]["prompt"]
    assert rows[1]["completion"] == second["assistant"] + Tokenizer.eos_token
    assert facts[0]["text"] in rows[0]["prompt"]
    assert facts[0]["source_url"] not in rows[0]["prompt"]
    assert set(env["train_dataset"].column_names) == {"input_ids", "completion_mask"}
    assert set(env["val_dataset"].column_names) == {"input_ids", "completion_mask"}


def test_actual_notebook_training_reload_and_export_cells(tmp_path, monkeypatch):
    tokenizer, base = tiny_model(lora=False)
    from tokenizers.pre_tokenizers import Whitespace
    tokenizer.backend_tokenizer.pre_tokenizer = Whitespace()
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}"
    base.config.max_position_embeddings = 512
    model_dir = tmp_path / "model"
    base.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    lock = {"model_id": "test/tiny-qwen", "revision": "a" * 40}
    write_json(model_dir / "model-lock.json", lock)
    rows = [{"input_ids": [3, 4, 1], "completion_mask": [0, 1, 1]}] * 4
    cases = [{"id": "hello", "question": "hello", "check": "nonempty output"}]
    env = dict(Path=Path, torch=torch, json=json, math=math, gc=gc, shutil=shutil, time=time,
               importlib=importlib, write_json=write_json, sha=sha, adapter_signature=adapter_signature,
               tokenizer_signature=tokenizer_signature,
               LoraConfig=LoraConfig, PeftModel=PeftModel, get_peft_model=get_peft_model,
               prepare_model_for_kbit_training=prepare_model_for_kbit_training,
               AutoModelForCausalLM=AutoModelForCausalLM, SFTTrainer=SFTTrainer,
               ROOT=tmp_path, MODEL_DIR=model_dir,
               RUN_DIR=tmp_path / "run", ADAPTER_DIR=tmp_path / "run/adapter",
               MERGED_DIR=tmp_path / "run/merged-model", MAX_STEPS=2, MAX_LENGTH=512,
               ACCUMULATION=1, RANK=2, EPOCHS=1, EVAL_EVERY=1, model=base, tokenizer=tokenizer,
               rows=rows, train_dataset=Dataset.from_list(rows),
               val_dataset=Dataset.from_list(rows[:2]), is_draft=True,
               dataset={"facts": [], "episodes": [], "note": "tiny model test fixture"},
               base_lock=lock, quantization=None, cases=cases,
               messages_by_id={"hello": [{"role": "user", "content": "hello"}]})
    # Only replace hardware-specific settings. Training, masking, save, reload and merge
    # execute the notebook cells unchanged, using real Torch/TRL/PEFT operations.
    def cpu_config(**kwargs):
        kwargs.update(bf16=False, use_cpu=True, gradient_checkpointing=False,
                      dataloader_pin_memory=False, save_steps=1)
        return SFTConfig(**kwargs)
    env["SFTConfig"] = cpu_config
    original_load = AutoModelForCausalLM.from_pretrained
    def cpu_load(*args, **kwargs):
        kwargs.pop("quantization_config", None)
        kwargs.update(device_map={"": "cpu"}, torch_dtype=torch.float32)
        return original_load(*args, **kwargs)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", cpu_load)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args: "tiny CPU test")
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    version = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "not-installed-cpu-test" if name == "bitsandbytes" else version(name))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    probe = torch.tensor([[3, 4]])
    for name in ("answer-function", "lora", "trainer", "train", "loss-curve", "save"):
        exec(source(name), env)
    with torch.no_grad():
        expected = env["model"].eval()(probe).logits.clone()
    assert (env["RUN_DIR"] / "loss-curve.png").stat().st_size > 0
    assert any("eval_loss" in item for item in env["history"])
    completed = json.loads((env["RUN_DIR"] / "training-complete.json").read_text())
    assert completed["final_step"] == 2 and completed["changed_lora_tensors"] > 0
    assert json.loads((env["RUN_DIR"] / "dataset.json").read_text()) == env["dataset"]
    assert env["manifest"]["contract"]["data_sha256"] == sha(env["RUN_DIR"] / "dataset.json")
    for name in ("release-training", "reload"):
        exec(source(name), env)
    with torch.no_grad():
        assert torch.allclose(expected, env["new_model"].eval()(probe).logits, atol=1e-6)
    for name in ("evaluate", "release-inference", "merge", "check-merged"):
        exec(source(name), env)
    check = json.loads((env["RUN_DIR"] / "inference-check.json").read_text())
    assert check["nonempty_replies"] == 1 and not check["quality_reviewed"]
    assert json.loads((env["RUN_DIR"] / "merged-inference-check.json").read_text())["reply"]
    exported = original_load(env["MERGED_DIR"], local_files_only=True).eval()
    with torch.no_grad():
        assert torch.allclose(expected, exported(probe).logits, atol=1e-5)
    assert not list(tmp_path.glob("*-results.zip"))
