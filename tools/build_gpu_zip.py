"""Build llm-lora-tutorial-gpu.zip: the files a GPU host needs, without model weights.

Run from the repository root after changing the notebook, lab/, data, requirements
or download lock:  python3 tools/build_gpu_zip.py
"""
import argparse
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "llm-lora-tutorial"
FILES = [
    "GPU_UPLOAD_README.md", "GPU_UPLOAD_README.en.md", "pyproject.toml",
    "data/README.md", "data/README.en.md", "data/dataset.json",
    "notebooks/01_gpu_train.ipynb", "notebooks/01_gpu_train.en.ipynb", "lab/download-lock.json",
    "requirements-data.txt", "requirements-download.txt",
    "requirements-notebook.txt", "requirements-train.txt",
]
# Tokenizer, config and license from `python -m lab.download --tokenizer-only`; no *.safetensors.
MODEL_FILES = [
    "LICENSE", "README.md", "config.json", "generation_config.json", "merges.txt",
    "model-lock.json", "model.safetensors.index.json", "tokenizer.json",
    "tokenizer_config.json", "vocab.json",
]


def build(root, output):
    for path in sorted((root / "notebooks").glob("01_gpu_train*.ipynb")):
        notebook = json.loads(path.read_text())
        if any(cell.get("outputs") or cell.get("execution_count") for cell in notebook["cells"]
               if cell["cell_type"] == "code"):
            raise ValueError(f"clear notebook outputs before packaging: {path.name}")
    names = [*FILES, *sorted(p.relative_to(root).as_posix() for p in (root / "lab").glob("*.py")),
             *(f"models/qwen3-4b/{name}" for name in MODEL_FILES)]
    missing = [name for name in names if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing (run lab.download --tokenizer-only first?): {missing}")
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(root / name, f"{PREFIX}/{name}")
    return names


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default=str(ROOT / "llm-lora-tutorial-gpu.zip"))
    a = p.parse_args()
    names = build(ROOT, Path(a.output))
    size = Path(a.output).stat().st_size / 2**20
    print(f"{a.output}: {len(names)} files, {size:.2f} MiB")


if __name__ == "__main__":
    main()
