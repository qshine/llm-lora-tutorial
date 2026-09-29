"""Prepared data stays small without losing training or evaluation inputs."""
import json
import re

import pytest

from lab.common import read_jsonl, verify_bundle
from lab.prepare import prepare
from lab.data import read_dataset


class Tokenizer:
    eos_token = "<|im_end|>"
    eos_token_id = 1
    pad_token_id = 0
    chat_template = "test chat template"
    special_tokens_map = {"eos_token": eos_token}

    def get_vocab(self):
        return {self.eos_token: 1}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(m["role"] + ":" + m["content"] + self.eos_token + "\n" for m in messages)
        return text + ("assistant:" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        tokens = re.findall(r"<\|im_end\|>|[\s\S]", text)
        return {"input_ids": [1 if t == self.eos_token else ord(t) + 2 for t in tokens]}


def test_seed_bundle_has_six_files_and_preserves_the_training_targets(tmp_path):
    episodes, facts = read_dataset("data/dataset.json")
    episodes = [e for e in episodes if e["split"] == "train"][:3]
    manifest = prepare(episodes, facts, Tokenizer(), tmp_path, allow_draft=True)
    assert manifest["schema"] == 2
    assert {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()} == {
        "facts.jsonl", "train.episodes.jsonl", "fewshots.json", "manifest.json",
        "_derived/train.sft.jsonl", "_derived/token-audit.json",
    }
    assert read_jsonl(tmp_path / "train.episodes.jsonl") == episodes
    rows = read_jsonl(tmp_path / "_derived/train.sft.jsonl")
    assert len(rows) == sum(len(e["turns"]) for e in episodes)
    assert rows[0]["completion"] == episodes[0]["turns"][0]["assistant"] + Tokenizer.eos_token
    assert episodes[0]["turns"][0]["assistant"] in rows[1]["prompt"]
    assert episodes[0]["turns"][1]["user"] not in rows[0]["prompt"]
    assert verify_bundle(tmp_path) == manifest
    with pytest.raises(ValueError, match="not empty"):
        prepare(episodes, facts, Tokenizer(), tmp_path, allow_draft=True)


def test_nonempty_validation_and_test_splits_remain_available(tmp_path):
    episodes, facts = read_dataset("data/dataset.json")
    episodes = episodes[:3] + [next(e for e in episodes if e["split"] == split) for split in ("val", "test")]
    manifest = prepare(episodes, facts, Tokenizer(), tmp_path, allow_draft=True)
    assert manifest["counts"] == {"train": 3, "val": 1, "test": 1}
    assert len(read_jsonl(tmp_path / "_derived/val.sft.jsonl")) == 2
    assert read_jsonl(tmp_path / "test.episodes.jsonl") == [episodes[-1]]
    assert not (tmp_path / "_derived/test.sft.jsonl").exists()
    fewshots = json.loads((tmp_path / "fewshots.json").read_text())
    train_ids = {e["id"] for e in episodes if e["split"] == "train"}
    assert all(e["episode_id"] in train_ids for e in fewshots)
    payload = tmp_path / "_derived/train.sft.jsonl"
    payload.write_text(payload.read_text() + "\n")
    with pytest.raises(ValueError, match="data changed"):
        verify_bundle(tmp_path)


@pytest.mark.parametrize("schema,relative", [(1, "train.sft.jsonl"), (2, "_derived/train.sft.jsonl")])
def test_training_reader_supports_old_and_new_bundles(tmp_path, schema, relative):
    from lab.common import write_jsonl
    from lab.data import read_sft
    rows = [{"input_ids": [10, 11, 1], "completion_mask": [0, 1, 1]}]
    write_jsonl(tmp_path / relative, rows)
    manifest = {"schema": schema, "counts": {"train": 1}}
    assert read_sft(tmp_path, "train", manifest) == rows
    assert read_sft(tmp_path, "val", manifest) == []
    manifest["counts"]["val"] = 1
    with pytest.raises(FileNotFoundError):
        read_sft(tmp_path, "val", manifest)


def test_inference_rejects_missing_split_before_loading_a_model(tmp_path, monkeypatch):
    from lab.infer import main
    episodes, facts = read_dataset("data/dataset.json")
    prepare(episodes[:3], facts, Tokenizer(), tmp_path, allow_draft=True)
    monkeypatch.setattr("sys.argv", ["infer", "--data", str(tmp_path), "--output", str(tmp_path / "replies.jsonl")])
    with pytest.raises(ValueError, match="selected split is empty"):
        main()


def test_fewshots_cover_all_grade_bands_using_train_only(tmp_path):
    episodes, facts = read_dataset("data/dataset.json")
    prepare(episodes, facts, Tokenizer(), tmp_path, allow_draft=True)
    fewshots = json.loads((tmp_path / "fewshots.json").read_text())
    assert {row["grade_band"] for row in fewshots} == {"1-2", "3-4", "5-6"}
    assert all(row["episode_id"] in {e["id"] for e in episodes if e["split"] == "train"} for row in fewshots)
