import pytest
from lab.evaluate import blind_pairs, summarize


def test_pair_by_case_id_not_file_order():
    left = [{"case_id": "a", "episode_id": "one", "reply": "base", "user": "u", "history": [], "fact_ids": ["f"]}]
    right = [{**left[0], "reply": "tuned"}]
    rows, key = blind_pairs(left, right, seed=42)
    assert set([rows[0]["X"], rows[0]["Y"]]) == {"base", "tuned"}
    assert key["a"]["X"] != key["a"]["Y"]
    with pytest.raises(ValueError, match="case"):
        blind_pairs(left, [])


def test_scoring_rejects_missing_or_invalid_ratings():
    with pytest.raises(ValueError, match="rating"):
        summarize([], {})
    key = {"a": {"X": "B", "Y": "C", "episode_id": "one"}}
    with pytest.raises(ValueError, match="rating"):
        summarize([], key)
    with pytest.raises(ValueError, match="rating"):
        summarize([{"case_id": "a", "winner": "Z"}], key)


def test_ties_count_half_and_clusters_are_episodes():
    key = {"a": {"X": "B", "Y": "C", "episode_id": "one"},
           "b": {"X": "C", "Y": "B", "episode_id": "one"}}
    result = summarize([{"case_id": "a", "winner": "Y", "reason": "clearer"},
                        {"case_id": "b", "winner": "tie", "reason": "equivalent"}], key)
    assert result["preference_C"] == .75
    assert result["episodes"] == 1 and result["turns"] == 2
    assert result["ci95_topic_bootstrap"] is None  # one cluster cannot estimate uncertainty


def test_science_review_keeps_grade_and_clusters_topic_variants():
    left = [{"case_id": str(i), "episode_id": f"episode-{i}", "group_id": "same-topic",
             "grade_band": "1-2", "reply": "base", "user": "u", "history": [], "fact_ids": ["f"]}
            for i in range(2)]
    rows, key = blind_pairs(left, [{**r, "reply": "tuned"} for r in left])
    assert all(row["grade_band"] == "1-2" for row in rows)
    for row in rows:
        row.update(winner="tie", reason="equivalent")
    result = summarize(rows, key)
    assert result["episodes"] == 2 and result["topic_groups"] == 1
    assert result["ci95_topic_bootstrap"] is None
