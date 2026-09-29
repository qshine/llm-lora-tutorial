import copy

import pytest

from lab.data import expand_episode, validate, read_dataset, SPLITS
from lab.common import build_messages, write_json


@pytest.fixture
def seed():
    return read_dataset("data/dataset.json")


def test_single_file_cli_accepts_a_custom_dataset(tmp_path, monkeypatch, capsys, seed):
    from lab.data import main
    episodes, facts = seed
    path = tmp_path / "my-dataset.json"
    write_json(path, {"facts": facts, "train": [{k: v for k, v in e.items() if k != "split"} for e in episodes[:3]],
                      "val": [], "test": []})
    monkeypatch.setattr("sys.argv", ["data", "--input", str(path), "--allow-draft"])
    main()
    assert "'train': 3" in capsys.readouterr().out


@pytest.mark.parametrize("payload", [{}, {"facts": {}, "train": [], "val": [], "test": []},
                                     {"facts": [], "train": [], "val": []}, []])
def test_single_file_requires_fact_and_split_arrays(tmp_path, payload):
    path = tmp_path / "dataset.json"
    write_json(path, payload)
    with pytest.raises(ValueError, match="facts, train, val and test"):
        read_dataset(path)


def test_split_comes_only_from_the_top_level_array(tmp_path, seed):
    episodes, facts = seed
    first = {k: v for k, v in episodes[0].items() if k != "split"}
    path = tmp_path / "dataset.json"
    # 旧版 episodes 数组，以及对话自带 split（可能与所在数组矛盾）都要被拒绝。
    write_json(path, {"facts": facts, "episodes": [first], "train": [], "val": [], "test": []})
    with pytest.raises(ValueError, match="old layout"):
        read_dataset(path)
    write_json(path, {"facts": facts, "train": [], "val": [{**first, "split": "train"}], "test": []})
    with pytest.raises(ValueError, match="without its own split"):
        read_dataset(path)


def test_source_file_stores_splits_side_by_side():
    import json
    data = json.loads(open("data/dataset.json", encoding="utf-8").read())
    assert list(data) == ["schema_version", "description", "facts", *SPLITS]
    assert {split: len(data[split]) for split in SPLITS} == {"train": 500, "val": 50, "test": 50}
    assert not any("split" in ep for split in SPLITS for ep in data[split])


def test_seed_shape(seed):
    episodes, facts = seed
    assert len(episodes) == 600 and len(facts) == 60
    assert validate(episodes, facts, allow_draft=True)["train"] == 500


def test_drafts_cannot_enter_formal_training(seed):
    with pytest.raises(ValueError, match="review"):
        validate(*seed)


def test_group_cannot_cross_splits(seed):
    episodes, facts = copy.deepcopy(seed)
    duplicate = copy.deepcopy(episodes[0])
    duplicate.update(id="other", split="test")
    duplicate["turns"][0]["user"] += " Please."
    with pytest.raises(ValueError, match="group"):
        validate(episodes + [duplicate], facts, allow_draft=True)


def test_missing_source_fails(seed):
    episodes, facts = copy.deepcopy(seed)
    episodes[0]["fact_ids"] = ["missing"]
    with pytest.raises(ValueError, match="fact"):
        validate(episodes, facts, allow_draft=True)


def test_exact_duplicate_detected_even_with_new_id(seed):
    episodes, facts = copy.deepcopy(seed)
    duplicate = copy.deepcopy(episodes[0])
    duplicate.update(id="other", group_id="other")
    with pytest.raises(ValueError, match="duplicate"):
        validate(episodes + [duplicate], facts, allow_draft=True)


def test_no_future_reply_in_history(seed):
    ep = seed[0][0]
    cases = list(expand_episode(ep))
    assert cases[0]["history"] == []
    assert cases[1]["history"] == [
        {"role": "user", "content": ep["turns"][0]["user"]},
        {"role": "assistant", "content": ep["turns"][0]["assistant"]},
    ]
    assert len(cases[0]["history"]) == 0  # later turns cannot mutate it


def test_prompt_facts_are_identical_in_baseline_and_adapter(seed):
    ep, cards = seed[0][0], seed[1]
    case = next(expand_episode(ep))
    messages = build_messages(cards, case, variant="B", fewshots=[])
    assert messages == build_messages(cards, case, variant="C", fewshots=[])
    assert messages[-1] == {"role": "user", "content": case["user"]}
    assert messages[0]["role"] == "system"


def test_formal_budget_is_enforced(seed):
    with pytest.raises(ValueError, match="600"):
        validate(seed[0][:3], seed[1], allow_draft=True, formal=True)


def test_science_corpus_contract(seed):
    from collections import Counter
    episodes, facts = seed
    assert validate(episodes, facts, allow_draft=True, formal=True) == {"train": 500, "val": 50, "test": 50}
    assert len({e["group_id"] for e in episodes}) == 60
    train = [e for e in episodes if e["split"] == "train"]
    assert Counter(e["category"] for e in train) == dict.fromkeys(("space", "earth", "life", "physics", "engineering"), 100)
    assert max(Counter(e["grade_band"] for e in train).values()) - min(Counter(e["grade_band"] for e in train).values()) <= 1
    assert all(e["review"]["status"] == "draft" for e in episodes)


def test_grade_enters_prompt_but_teaching_labels_do_not(seed):
    ep = seed[0][0]
    case = next(expand_episode(ep))
    assert case["grade_band"] == ep["grade_band"]
    case["behavior"] = "DO_NOT_LEAK_TEACHING_LABEL"
    prompt = build_messages(seed[1], case)[0]["content"]
    assert f'小学学段：{ep["grade_band"]}' in prompt
    assert "DO_NOT_LEAK_TEACHING_LABEL" not in prompt


def test_invalid_grade_and_fact_family_leak_are_rejected(seed):
    episodes, facts = copy.deepcopy(seed)
    episodes[0]["grade_band"] = "adult"
    with pytest.raises(ValueError, match="grade"):
        validate(episodes, facts, allow_draft=True)
    episodes, facts = copy.deepcopy(seed)
    # Renaming a group must not allow the same reference fact into another split.
    episodes[0].update(group_id="renamed", split="test")
    with pytest.raises(ValueError, match="fact.*split"):
        validate(episodes, facts, allow_draft=True)


def test_chinese_question_limit(seed):
    episodes, facts = copy.deepcopy(seed)
    episodes[0]["turns"][0]["assistant"] = "为什么？怎么做？"
    with pytest.raises(ValueError, match="question"):
        validate(episodes, facts, allow_draft=True)
