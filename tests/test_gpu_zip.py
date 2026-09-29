"""The GPU upload bundle carries code, data and tokenizer, never weights or notebook outputs."""
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from tools.build_gpu_zip import PREFIX, build

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / "models/qwen3-4b/tokenizer.json").is_file(),
                    reason="run lab.download --tokenizer-only first")
def test_bundle_matches_sources_without_weights(tmp_path):
    output = tmp_path / "gpu.zip"
    names = build(ROOT, output)
    with ZipFile(output) as archive:
        entries = archive.namelist()
        assert entries == [f"{PREFIX}/{name}" for name in names]
        assert not any(name.endswith(".safetensors") for name in entries)
        for name in ("notebooks/01_gpu_train.ipynb", "data/dataset.json", "lab/common.py"):
            assert archive.read(f"{PREFIX}/{name}") == (ROOT / name).read_bytes()


def test_bundle_refuses_executed_notebook(tmp_path):
    notebook = json.loads((ROOT / "notebooks/01_gpu_train.ipynb").read_text())
    next(c for c in notebook["cells"] if c["cell_type"] == "code")["execution_count"] = 1
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "notebooks/01_gpu_train.ipynb").write_text(json.dumps(notebook))
    with pytest.raises(ValueError, match="clear notebook outputs"):
        build(tmp_path, tmp_path / "gpu.zip")
