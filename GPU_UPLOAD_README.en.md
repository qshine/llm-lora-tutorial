# GPU Upload Bundle: Single-File Data, Cell-by-Cell Training

For: a single RTX 4090 / 24 GB; image 2.7.1-CUDA12.6-Py3.11-Ubuntu22.
Upload to the data disk and unzip it, then open a GPU terminal in the extracted llm-lora-tutorial directory.
Keep new and old projects in different directories, and keep the old model cache and training results.

## 1. Set up a separate training environment

```bash
python3.11 -m venv .venv-train
.venv-train/bin/python -m pip install --upgrade pip
.venv-train/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
.venv-train/bin/python -m pip install -r requirements-train.txt -r requirements-download.txt -r requirements-notebook.txt
.venv-train/bin/python -m pip check
.venv-train/bin/python -m ipykernel install --user --name lora-train --display-name 'LoRA · Python 3.11'
```

If you cannot reach download.pytorch.org (Network is unreachable), replace the third command with the Alibaba Cloud PyTorch mirror, then continue with the remaining commands:

```bash
.venv-train/bin/python -m pip install 'torch==2.8.0+cu126' -f https://mirrors.aliyun.com/pytorch-wheels/cu126/
```

## 2. Download the full weights from ModelScope and verify them

```bash
.venv-train/bin/python -m lab.download --source modelscope --output models/qwen3-4b
.venv-train/bin/python -m lab.download --check-only --output models/qwen3-4b
```

Do not add --tokenizer-only; it must show that all 12 files passed verification.
On failure, keep model-lock.json and .cache and rerun once the network recovers; if a file is corrupted, move it aside as a backup and download it again.
Fallback source: replace --source modelscope with --source hf-mirror; verification works the same way.
This bundle includes the tokenizer and the download verification manifest, but not the roughly 8 GB of full weights.

## 3. Maintain only one copy of the data

The only data source is data/dataset.json: facts holds 60 fact cards; the sibling arrays train, val, and test hold 500, 50, and 50 conversations respectively. Only train is used for training, expanded into 1150 answer targets.
It covers all three primary-school grade bands; all conversations are still synthetic drafts, and the science facts, readability, and observation suggestions need to be checked.
Each field and where it is used is described in data/README.en.md; sources, grouping, and review records are all kept.
The Notebook reads this one JSON directly and builds the context, tokens, and supervision mask in memory.
You do not need lab.prepare or data/processed/. Only generate a frozen snapshot as described in the e-book when an advanced CLI experiment needs one.

```bash
.venv-train/bin/python -m lab.data --input data/dataset.json --allow-draft
```

## 4. Start training

In JupyterLab, open notebooks/01_gpu_train.en.ipynb and select the LoRA · Python 3.11 kernel. (The Chinese notebook notebooks/01_gpu_train.ipynb runs exactly the same code.)
Run it cell by cell in order. Below each code cell is an explanation of what it does; the D1–D7 comments mark where the data is used.
The default RUN_DIR is runs/science-001; for a new experiment change it to science-002 so old results are not overwritten.
After changing the data, you need to reload, re-tokenize, and retrain; old in-memory variables do not change automatically.

By default it runs a full epoch (about 72 optimizer steps). While training, open another terminal and run nvidia-smi -l 2 to watch VRAM usage.

- dataset.json: a copy of the data actually read in this run; for day-to-day work, still edit only data/dataset.json.
- loss-curve.png: the training/validation loss curve.
- training-complete.json: the actual number of training steps, parameter changes, and a record that saving completed.
- adapter/: the trained LoRA delta, which must be paired with the pinned version of the base model.
- answer-review.json: answers before and after training, plus a human review to be filled in.
- merged-model/: the merged full BF16 model, ready for vLLM deployment.

The end of the Notebook gives examples of vLLM deployment in a GPU terminal, a local SSH tunnel, and API tests; mounting the adapter, streaming calls, and load testing are covered in chapter 5 of the e-book.
vLLM runs in a separate .venv-serve; first confirm that the host driver supports CUDA 12.9, and shut down the training kernel to free VRAM.
Keep an eye on the remaining space on the 50 GB data disk; for full training plus deployment, expanding it to 100 GB is recommended.
Save the Notebook and back up the adapter, version locks, data copy, review, and any necessary checkpoints, then stop the instance as needed.
Passing local CPU checks does not mean an actual 4090 training run has passed; answers differing before and after training does not necessarily mean quality improved — compare them by hand.
