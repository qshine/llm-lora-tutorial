<div align="center">

# LoRA Fine-Tuning and vLLM Deployment: A Hands-On Guide

**A primary-school science companion · QLoRA fine-tuning, merging and vLLM deployment from scratch on one RTX 4090**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3110/)
[![Model: Qwen3-4B](https://img.shields.io/badge/Model-Qwen3--4B--Instruct--2507-8C432E)](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
[![GPU: RTX 4090](https://img.shields.io/badge/GPU-RTX%204090%20·%2024%20GB-76B900?logo=nvidia&logoColor=white)](tasks/plan.en.md)
[![E-book](https://img.shields.io/badge/E--book-Read%20online-A7492D)](https://qshine.github.io/llm-lora-tutorial/en/)

[简体中文](README.md) · **English** · [日本語](README.ja.md) · [한국어](README.ko.md)

[Read online](https://qshine.github.io/llm-lora-tutorial/en/) · [Tutorial](tasks/plan.en.md) · [GPU notebook](notebooks/01_gpu_train.en.ipynb) · [Data guide](data/README.en.md)

</div>

---

## Overview

Written for **developers who don't work on ML and want to learn LoRA fine-tuning from scratch**: backend, frontend, client or data engineers alike, with no machine-learning background required. On a single RTX 4090 you work through:

```text
QLoRA fine-tuning → save and reload → BF16 merge → vLLM as an OpenAI-compatible API → local calls and load testing
```

and learn where the GPU memory goes at each step, where every file ends up, and how to confirm that each step really succeeded. The practice task is answering primary-school students' science questions in Chinese: the conversational style is distinctive, so the change before and after fine-tuning is easy to see.

## Highlights

| | |
|---|---|
| 📖 **Complete e-book** | 9 chapters + 5 appendices, from concepts, data and renting a GPU to deployment, evaluation and troubleshooting; switch between 中文, English, 日本語 and 한국어 in one click |
| 🧪 **Runnable notebook** | Shows `LoraConfig`, `SFTTrainer` and `trainer.train()` directly; every cell is followed by what it does, why, and what you will see |
| 🗂️ **One data file** | `data/dataset.json` holds 60 fact cards and 500 / 50 / 50 training, validation and test conversations, isolated by topic |
| 🚀 **Training through deployment** | Merge and export, vLLM serving, mounting the adapter, streaming with the OpenAI SDK, load testing with `vllm bench serve` |
| ✅ **Guarded by tests** | Commands, appendix code and deployment examples in the tutorial are checked against the real code |

## Quick start

### 1. Read the e-book

Read online: <https://qshine.github.io/llm-lora-tutorial/en/> (switch between 中文 / English / 日本語 / 한국어 in the top-right corner of any page). Or preview it locally:

```bash
./book_review.sh
```

The script prepares the reading environment and opens <http://127.0.0.1:8000>. The first run needs Python 3.11 and network access; if the port is taken, use `./book_review.sh 8001`; press `Ctrl+C` to stop. Reading needs no GPU or model weights.

### 2. Check the data

Check the data format only (no PyTorch needed):

```bash
python3.11 -m lab.data --input data/dataset.json --allow-draft
```

### 3. Train cell by cell on a GPU

Run `python3 tools/build_gpu_zip.py` to produce `llm-lora-tutorial-gpu.zip`, upload it to the GPU host's data disk, and unzip it. Follow chapter 3 of the tutorial to set up the environment, download and verify the full model from ModelScope, then open [01_gpu_train.en.ipynb](notebooks/01_gpu_train.en.ipynb) (the English edition of [01_gpu_train.ipynb](notebooks/01_gpu_train.ipynb); same code, English explanations and comments) and select the **LoRA · Python 3.11** kernel.

The notebook targets an RTX 4090 / 24 GB: base-model answers → LoRA config → 1 epoch of training (about 72 steps) → loss curve → save and reload → human review → BF16 merge, ending with vLLM deployment and local call examples. Default outputs go to `runs/science-001/`:

| Output | Meaning |
|---|---|
| `training-complete.json` | Marks a finished training run |
| `adapter/` | LoRA delta weights |
| `merged-model/` | Full model for vLLM |
| `answer-review.json` | Before/after answers, waiting for human review |

## Data

Only [data/dataset.json](data/dataset.json) is maintained: `facts` holds 60 reference fact cards, and the sibling `train`, `val`, and `test` arrays hold 500 training, 50 validation, and 50 test conversations. Fact cards are inserted into each question's system prompt via `fact_ids` as grounding for the answer; they are not learning targets. The training split expands turn by turn into 1150 answer targets. Sources, categories, and review records all live in the same file.

The [data guide](data/README.en.md) walks through a complete day-and-night example, explaining every field, which ones reach the model, and which are review metadata only. Comments D1–D7 in the notebook mark loading, filtering, context building, tokenization, the trainer entry point, the loss scope, and the save location.

The notebook reads this JSON directly and builds training samples in memory, so there is no need to run `lab.prepare` first. Training automatically keeps a copy of the actual input at `runs/science-001/dataset.json`. Only the command-line experiments need a newly frozen snapshot (see section 4.6 of the tutorial).

> [!NOTE]
> All conversations are synthetic drafts covering grades 1–6, grouped into bands 1–2, 3–4 and 5–6; 50 training topics with 10 paths each, with val/test isolated by topic. Facts, sources and readability still need item-by-item review. There is no guarantee that fine-tuning beats a full-prompt baseline. The dataset stays in Chinese, because the fine-tuning task itself is answering children's science questions in Chinese.

## Project layout

| Path | Responsibility |
|---|---|
| `tasks/plan.md` (`plan.en.md`, …) | Hands-on tutorial (the e-book text) |
| `notebooks/01_gpu_train.ipynb` (`.en.ipynb`) | Main GPU training entry point, chapters 4–5 of the tutorial |
| `data/dataset.json` | The single data source |
| `lab/data.py`, `prepare.py`, `common.py` | Validate source data, expand Q&A, build prompts and training samples |
| `lab/audit.py` | Check supervision targets with the real TRL collator |
| `lab/download.py`, `train.py`, `training.py` | Pin the model revision, train, and resume |
| `lab/infer.py`, `evaluate.py` | Generate comparison answers, blind review, and result statistics |
| `lab/merge.py` | Standalone merge and export |
| `book/`, `mkdocs*.yml` | Multilingual e-book build |
| `tests/` | Data, training, notebook, tutorial and e-book tests |

`models/`, `runs/`, and `data/processed/` hold local artifacts and are not tracked in version control.

## More documentation

- [Hands-on tutorial](tasks/plan.en.md): examples, commands, field meanings, and how to read the results.
- [Learning checklist](tasks/todo.en.md): separates code checks from training experiments still to be done.
- [Verification log](tasks/verification.en.md): checks that have actually been run, and their results.
- [E-book maintenance](book/README.md) (Chinese): building, link checking, and publishing.

## License

This project is released under the [MIT License](LICENSE). The base model [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) is not distributed with this repository; use it under its own license.
