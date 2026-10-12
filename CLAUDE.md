# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Chinese-language learning lab for SFT + QLoRA on `Qwen/Qwen3-4B-Instruct-2507` (pinned revision), themed as a primary-school science companion. The target reader is a non-ML developer (backend, frontend, client, data, …) learning LoRA fine-tuning from scratch, including GPU training and deployment: QLoRA training, reload, BF16 merge, vLLM serving (merged model or a mounted adapter), API calls, and load testing. The main tutorial is `tasks/plan.md`, which is also rendered as a MkDocs e-book. Docs, comments, and user-facing messages are written in Chinese; keep that convention. Chinese is the default; the reader-facing materials also have English (`en`), Japanese (`ja`) and Korean (`ko`) editions as `*.<lang>.md` siblings (`README`, `tasks/plan`, `tasks/todo`, `tasks/verification`, `data/README`, `book/home`; `GPU_UPLOAD_README.en.md` is English-only) plus `notebooks/01_gpu_train.en.ipynb` (ja/ko link the English notebook). Code blocks in `plan.ja.md`/`plan.ko.md` are byte-identical to `plan.en.md`. When you change a Chinese doc or the notebook, update every edition in the same change. The dataset and `lab/` code stay Chinese-only. The project is MIT-licensed (`LICENSE`).

The real 4B training runs only on a Linux NVIDIA GPU host (RTX 4090 target). Locally, only data checks, tokenization, and tiny random-model CPU tests run.

## Environments (Python 3.11 only, separate venvs per purpose)

- `.venv-data` (`requirements-data.txt`): data validation and tokenization, no PyTorch.
- `.venv-cpu` (`requirements-cpu.txt`): the full test suite, including tiny CPU TRL/PEFT training.
- `.venv-book` (`requirements-book-lock.txt`): the e-book; `./book_review.sh` creates and reuses it.
- GPU host only: `.venv-train` (torch 2.8.0 cu126 installed separately, plus `requirements-train.txt`, `-download.txt`, `-notebook.txt`; registered as the Jupyter kernel "LoRA · Python 3.11") and `.venv-serve` (vLLM, kept isolated so it does not overwrite the training deps).

## Commands

```bash
.venv-cpu/bin/python -m pytest -q                        # all tests (~10 s, no weights needed)
.venv-cpu/bin/python -m pytest -q tests/test_data.py::test_name
python3.11 -m lab.data --input data/dataset.json --allow-draft   # structure check → {'train': 500, 'val': 50, 'test': 50}
./book_review.sh [port]                                  # serve e-book at 127.0.0.1:8000
.venv-book/bin/python -m mkdocs build --strict && .venv-book/bin/python -m book.check site
```

Each `lab/*.py` module is a CLI (`python -m lab.<name> --help`): `download`, `data`, `prepare`, `audit`, `train`, `infer`, `evaluate` (`blind`/`score`/`freeze`), `merge`.

## Architecture

**Single data source.** `data/dataset.json` (schema 4) holds `facts[60]` plus side-by-side `train[500]`, `val[50]`, `test[50]` conversation arrays (split by `group_id` topic so topics and fact cards never leak across splits). Episodes carry no `split` field in the file; `lab/data.py:tag_splits` (used by `read_dataset` and the notebook) adds it in memory from the array name and rejects the old `episodes` layout. Every episode is a synthetic `draft`. `lab/data.py:validate` enforces the schema, the draft gate, and split isolation; `--formal` checks the exact 100/10/10-per-category quotas; review approval is a separate check that `--allow-draft` skips.

**Prompt and supervision boundary (`lab/common.py`).** This module is the core of the repo:
- `expand_episode` (in `lab/data.py`) turns each assistant turn into one sample, with past turns only as history (1150 train rows).
- `build_messages` assembles the system prompt (`ROLE`, or `BASIC` for variant A), then the grade band, then the selected facts (`id/text/kind/scope` only), then history and the user turn. `behavior`, `scenario`, `review`, and `source_url` must never enter the prompt.
- `render_pair` applies the chat template once and produces `input_ids` plus `completion_mask`. The mask covers the current answer plus exactly one `<|im_end|>` EOS. It raises an error instead of truncating or guessing the boundary.

**Two training entry points that must stay in sync:**
1. `notebooks/01_gpu_train.ipynb` is the main path (`01_gpu_train.en.ipynb` is its English edition: identical code, translated markdown/comments/messages; `tests/test_notebook.py` checks both and that their code ASTs match). It reads the JSON directly in memory (comments D1–D7 mark the data-usage points referenced by the tutorial). It computes `MAX_STEPS = ceil(1150 / 16) = 72` (1 epoch), evaluates val loss every 12 steps, plots `loss-curve.png`, then saves, reloads, merges, and writes artifacts under `runs/science-001/`.
2. The CLI path (`lab.prepare` → frozen bundle in `data/processed/<version>/` with `manifest.json` → `lab.train`/`lab.infer`/`lab.evaluate`) is for the advanced experiments: resume, rank comparison, and A/B/C evaluation. `verify_bundle` rejects a bundle if its files or the SHA of `lab/common.py` changed since prepare. **Any edit to `common.py` invalidates existing bundles**, by design.

Shared TRL setup lives in `lab/training.py` (`make_trainer`, `check_supervision`) and is used by both `train.py` and the CPU tests.

**Guardrails are deliberate.** Draft data needs `--allow-draft` and is recorded as `draft` in the manifest; epoch runs need a bundle prepared with `--formal` (the quota check). `evaluate freeze` and test-split inference still reject draft data. Non-empty output directories are refused. Resume compares the full contract (config, data, code, packages). Downloads are verified against `lab/download-lock.json`, with ModelScope and hf-mirror required to be byte-identical to the pinned HF revision. Don't weaken these checks to make something pass.

## Tests that couple docs to code

`tests/test_tutorial.py` parses `tasks/plan.md` and fails when:
- the commands documented there are not accepted by the real argparse parsers;
- Appendix A no longer matches `lab/train.py` and `lab/training.py` byte-for-byte;
- the vLLM and local-call code blocks in the notebook cells `deploy`/`local-call` differ from those in `plan.md`.

Each check also runs on every translated edition (`plan.<lang>.md` + its notebook), and each must keep the same chapters and the same code blocks as `plan.md` (only comments may differ).

So when you change the code, CLI flags, or the deployment snippets, update `plan.md` (and the notebook) in the same change. `tests/test_notebook.py` executes the notebook's actual code cells against a tiny model, including eval and the loss-curve cell, so it needs matplotlib in `.venv-cpu`.

## E-book

`book/build.py` is a MkDocs hook. It splits `tasks/plan.md` into pages at `##` headings, using `SLUGS`, which must stay aligned with the nav of `mkdocs.yml` and every `mkdocs.<lang>.yml` when chapters are added or reordered. After the Chinese build, the hook builds each other edition listed in `book/build.py:LANGUAGES` (`mkdocs.<lang>.yml`, inheriting `mkdocs.yml`, sources `*.<lang>.md`, generated into `.book-<lang>/`) into `site/<lang>/`; the header language menu plus `book/assets/language.js` switch to the same page in another language. It generates `.book/` (don't hand-edit it) and builds `downloads/lora-lab.zip`. The content sources are `tasks/plan.md`, `tasks/todo.md`, `tasks/verification.md`, and `data/README.md`. `.github/workflows/book-pages.yml` builds and publishes to GitHub Pages.

## GPU upload bundle

Build `llm-lora-tutorial-gpu.zip` with `python3 tools/build_gpu_zip.py`. The zip is git-ignored. It contains the notebook, `lab/`, `data/dataset.json`, `requirements-*`, `download-lock.json`, the local tokenizer/config files from `models/qwen3-4b/` (no weights), and `GPU_UPLOAD_README.md`. Rebuild it whenever any of these change. The script refuses a notebook that still has outputs.

## Conventions

- `tasks/todo.md` checkboxes mean "has real evidence", and `tasks/verification.md` records only checks that were actually run. Never mark GPU, vLLM, or quality items done without running them.
- Never bulk-set `review.status="approved"` in the dataset; approval means real human review.
- `models/`, `runs/`, `data/processed/`, and `site/` are local artifacts and are git-ignored.
