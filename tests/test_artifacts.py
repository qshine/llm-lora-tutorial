from pathlib import Path

import pytest

from lab.common import adapter_signature, sha, verify_bundle, write_json


def test_changing_adapter_scale_changes_fingerprint(tmp_path):
    (tmp_path/"adapter_model.safetensors").write_bytes(b"small fixture, not weights")
    write_json(tmp_path/"adapter_config.json", {"r": 16, "lora_alpha": 32})
    initial = adapter_signature(tmp_path)
    write_json(tmp_path/"adapter_config.json", {"r": 16, "lora_alpha": 64})
    assert adapter_signature(tmp_path) != initial


def test_modified_prepared_file_is_rejected(tmp_path):
    payload = tmp_path/"train.sft.jsonl"
    payload.write_text('original')
    write_json(tmp_path/"manifest.json", {"files": {payload.name: sha(payload)},
               "prompt_code_sha256": sha(Path('lab/common.py'))})
    verify_bundle(tmp_path)
    payload.write_text('silently changed')
    with pytest.raises(ValueError, match="changed"):
        verify_bundle(tmp_path)
