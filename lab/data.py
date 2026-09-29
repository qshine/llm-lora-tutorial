"""Validate complete conversations before splitting into supervised turns."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from lab.common import CATEGORIES, GRADE_BANDS, read_jsonl


SPLITS = ("train", "val", "test")


def tag_splits(data):
    """train/val/test 在文件里平级存放；读入后给每段对话标上所属集合，供校验和筛选。"""
    if not isinstance(data, dict) or not all(isinstance(data.get(key), list) for key in ("facts", *SPLITS)):
        raise ValueError("dataset.json must contain facts, train, val and test arrays")
    if "episodes" in data:
        raise ValueError("old layout: move episodes into top-level train/val/test arrays")
    episodes = []
    for split in SPLITS:
        for ep in data[split]:
            # 集合只由所在数组决定；对话里再写一份 split 容易和位置不一致。
            if not isinstance(ep, dict) or "split" in ep:
                raise ValueError(f"episode in {split} must be an object without its own split field")
            episodes.append({**ep, "split": split})
    return episodes


def read_dataset(path):
    """Read the single editable source: facts plus train/val/test conversations in one JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return tag_splits(data), data["facts"]


def validate(episodes, facts, allow_draft=False, formal=False):
    if not episodes or not facts:
        raise ValueError("empty dataset")
    card_ids = set()
    for card in facts:
        for key in ("id", "category", "topic", "text", "source_url", "checked_at", "scope", "kind", "reuse"):
            if not isinstance(card.get(key), str) or not card[key].strip():
                raise ValueError(f"fact field missing: {key}")
        if card["id"] in card_ids or card["category"] not in CATEGORIES:
            raise ValueError("duplicate fact or unknown category")
        if not card["source_url"].startswith("https://") or card["kind"] not in {"fact", "legend", "current"}:
            raise ValueError("invalid fact source/kind")
        card_ids.add(card["id"])
    seen, groups, fingerprints, fact_splits = set(), {}, set(), {}
    for ep in episodes:
        for key in ("id", "group_id", "topic", "scenario"):
            if not isinstance(ep.get(key), str) or not ep[key].strip():
                raise ValueError(f"episode field missing: {key}")
        if ep["id"] in seen:
            raise ValueError("duplicate episode id")
        seen.add(ep["id"])
        if ep.get("split") not in {"train", "val", "test"} or ep.get("category") not in CATEGORIES:
            raise ValueError("invalid split/category")
        if ep.get("grade_band") not in GRADE_BANDS:
            raise ValueError("invalid grade_band")
        # group_id/split 只控制分组与数据用途，不拼进模型提示词。
        group, split = ep["group_id"], ep["split"]
        if group in groups and groups[group] != split:
            raise ValueError("group crosses splits")
        groups[group] = split
        # fact_ids 必须引用本文件 facts 中存在的 id。
        refs = ep.get("fact_ids", [])
        if not refs or len(set(refs)) != len(refs) or not set(refs) <= card_ids:
            raise ValueError("unknown/duplicate/missing fact reference")
        for ref in refs:
            if ref in fact_splits and fact_splits[ref] != split:
                raise ValueError("fact family crosses splits")
            fact_splits[ref] = split
        review = ep.get("review", {})
        if review.get("status") not in {"draft", "approved"}:
            raise ValueError("invalid review status")
        if not allow_draft and (review["status"] != "approved" or not review.get("reviewer")
                                or review.get("reviewer_kind") not in {"human", "assistant"}):
            raise ValueError("review approval and actual reviewer required")
        if ep.get("provenance", {}).get("type") != "synthetic":
            raise ValueError("this learning dataset requires synthetic provenance")
        turns = ep.get("turns", [])
        if not 2 <= len(turns) <= 4:
            raise ValueError("expected 2-4 assistant turns per episode")
        for turn in turns:
            for key in ("user", "assistant", "behavior"):
                if not isinstance(turn.get(key), str) or not turn[key].strip():
                    raise ValueError(f"invalid turn: {key}")
            if any(token in turn[role] for role in ("user", "assistant")
                   for token in ("<|im_start|>", "<|im_end|>", "<|endoftext|>")):
                raise ValueError("chat control token in raw text")
            if sum(turn["assistant"].count(mark) for mark in ("?", "？")) > 1:
                raise ValueError("more than one question in reference reply")
        fp = ep["grade_band"] + "|" + "|".join(re.sub(r"\W+", "", t["user"].casefold()) for t in turns)
        if fp in fingerprints:
            raise ValueError("duplicate conversation (near duplicates also need manual review)")
        fingerprints.add(fp)
    counts = Counter(ep["split"] for ep in episodes)
    if formal:
        if counts != Counter(train=500, val=50, test=50) or len(facts) != 60:
            raise ValueError("formal science v1 requires 600 episodes (500/50/50) and 60 facts")
        for category in CATEGORIES:
            if Counter(e["split"] for e in episodes if e["category"] == category) != Counter(train=100, val=10, test=10):
                raise ValueError(f"formal category quota mismatch: {category}")
    return dict(counts)


def expand_episode(ep):
    # D2：一轮 assistant 是一个训练目标；history 只含此前已发生的问答。
    history = []
    for i, turn in enumerate(ep["turns"]):
        yield {"case_id": f"{ep['id']}:{i}", "episode_id": ep["id"],
               "group_id": ep["group_id"], "category": ep["category"],
               "grade_band": ep["grade_band"],
               "fact_ids": ep["fact_ids"], "history": list(history), **turn}
        history.extend([{"role": "user", "content": turn["user"]},
                        {"role": "assistant", "content": turn["assistant"]}])


def read_sft(folder, split, manifest):
    """Read a prepared split, including the original schema-1 layout."""
    if not manifest["counts"].get(split, 0):
        return []
    directory = Path(folder) / "_derived" if manifest["schema"] == 2 else Path(folder)
    return read_jsonl(directory / f"{split}.sft.jsonl")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/dataset.json")
    p.add_argument("--allow-draft", action="store_true")
    p.add_argument("--formal", action="store_true")
    a = p.parse_args()
    episodes, facts = read_dataset(a.input)
    print(validate(episodes, facts, a.allow_draft, a.formal))


if __name__ == "__main__":
    main()
