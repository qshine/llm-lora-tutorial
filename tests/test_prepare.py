import pytest
from lab.data import validate, read_dataset


def test_reviewed_small_dataset_is_valid_but_not_formal():
    eps, facts = read_dataset("data/dataset.json")
    eps = eps[:3]
    for ep in eps:
        ep["review"] = {"status": "approved", "reviewer": "test fixture only", "reviewer_kind": "human"}
    assert validate(eps, facts) == {"train": 3}
    with pytest.raises(ValueError, match="600"):
        validate(eps, facts, formal=True)


def test_empty_answer_rejected():
    eps, facts = read_dataset("data/dataset.json")
    eps[0]["turns"][0]["assistant"] = " "
    with pytest.raises(ValueError, match="assistant"):
        validate(eps, facts, allow_draft=True)


def test_three_questions_rejected():
    eps, facts = read_dataset("data/dataset.json")
    eps[0]["turns"][0]["assistant"] = "What? Why? How?"
    with pytest.raises(ValueError, match="question"):
        validate(eps, facts, allow_draft=True)
