# LoRA Fine-Tuning and vLLM Deployment Lab: A Primary-School Science Companion

[简体中文](README.md) | **English**

Written for developers who don't work on ML and want to learn LoRA fine-tuning from scratch; no machine-learning background is required. On a single RTX 4090 you work through QLoRA fine-tuning → save and reload → BF16 merge → serving with vLLM as an OpenAI-compatible API → local calls and load testing, and learn where the GPU memory goes at each step. The practice task is answering primary-school students' science questions in Chinese.

> The tutorial, e-book, notebook comments, and dataset are written in Chinese. This page is an English overview.

## Open the e-book

Read online: <https://qshine.github.io/llm-lora-tutorial/>

Or preview it locally:

```bash
./book_review.sh
```

The script prepares the reading environment and opens <http://127.0.0.1:8000>. The first run needs Python 3.11 and network access. If the port is taken, use `./book_review.sh 8001`; press `Ctrl+C` to stop. Reading needs no GPU or model weights.

## One data file, train directly

Only [data/dataset.json](data/dataset.json) is maintained: `facts` holds 60 reference fact cards, and the sibling `train`, `val`, and `test` arrays hold 500 training, 50 validation, and 50 test conversations. Fact cards are inserted into each question's system prompt via `fact_ids` as grounding for the answer; they are not learning targets. The training split expands turn by turn into 1150 answer targets. Sources, categories, and review records all live in the same file.

The [data guide](data/README.md) walks through a complete day-and-night example, explaining every field, which ones reach the model, and which are review metadata only. Comments D1–D7 in the notebook mark loading, filtering, context building, tokenization, the trainer entry point, the loss scope, and the save location.

Check the data format only (no PyTorch needed):

```bash
python3.11 -m lab.data --input data/dataset.json --allow-draft
```

The notebook reads this JSON directly and builds training samples in memory, so there is no need to run `lab.prepare` first. Training automatically keeps a copy of the actual input at `runs/science-001/dataset.json`. The old `data/processed/seed-v1` and `seed-v2` are kept for review; only the command-line experiments need a newly frozen snapshot (see section 4.6 of the tutorial).

## Where to read

- [Hands-on tutorial](tasks/plan.md): examples, commands, field meanings, and how to read the results.
- [Learning checklist](tasks/todo.md): separates code checks from training experiments still to be done.
- [Verification log](tasks/verification.md): checks that have actually been run, and their results.
- [E-book maintenance](book/README.md): building, link checking, and publishing.

## Run cell by cell on a GPU

Run `python3 tools/build_gpu_zip.py` to produce `llm-lora-tutorial-gpu.zip`, upload it to the GPU host's data disk, and unzip it. Follow chapter 3 of the tutorial to set up the environment, download and verify the full model from ModelScope, then open [01_gpu_train.ipynb](notebooks/01_gpu_train.ipynb) and select the **LoRA · Python 3.11** kernel.

The notebook targets an RTX 4090 / 24 GB and uses plain Python code to show the base model's answers, the LoRA config, 1 epoch of training (about 72 steps), the loss curve, saving, reloading, human review, and the BF16 merge. Each code cell is followed by an explanation of what it does and why; environment setup and model download are covered in the e-book. It ends with a vLLM deployment example for the GPU terminal and a local call example; chapter 5 of the tutorial also covers KV cache memory estimation, mounting the adapter directly, streaming calls with the OpenAI SDK, and load testing with `vllm bench serve`.

Default outputs go to `runs/science-001/`: `training-complete.json` marks a finished run, `adapter/` is the LoRA delta, and `merged-model/` is the full model for vLLM. Being able to generate answers doesn't mean quality improved; check `answer-review.json` and complete the review.

All conversations are synthetic drafts covering grades 1–6, grouped into bands 1–2, 3–4, and 5–6. There are 50 training topics with 10 paths each, and val/test are isolated by topic; some knowledge answers are reused within a topic. Facts, sources, and readability still need item-by-item review. 4B GPU training, quality evaluation, and vLLM serving have yet to be actually run; there is no guarantee that fine-tuning beats a full-prompt baseline.

## Code map

| File | Responsibility |
|---|---|
| `lab/data.py`, `prepare.py`, `common.py` | Validate source data, expand Q&A, build prompts and training samples |
| `lab/audit.py` | Check supervision targets with the real TRL collator |
| `lab/download.py`, `train.py`, `training.py` | Pin the model revision, train, and resume |
| `lab/infer.py`, `evaluate.py` | Generate comparison answers, blind review, and result statistics |
| `lab/merge.py` | Standalone merge and export |
| `notebooks/01_gpu_train.ipynb`, tutorial chapters 4–5 | Direct training code, output checks, and API call examples |

`models/`, `runs/`, and `data/processed/` hold local artifacts. Generated files are not tracked in version control.
