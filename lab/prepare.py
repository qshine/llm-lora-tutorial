"""Freeze a versioned data bundle; draft bundles are restricted to smoke tests."""
import argparse
from pathlib import Path

from lab.common import (GRADE_BANDS, build_messages, render_pair, sha,
                        tokenizer_signature, write_json, write_jsonl)
from lab.data import expand_episode, validate, read_dataset


def prepare(episodes, facts, tokenizer, output, allow_draft=False, formal=False, max_length=4096):
    counts = validate(episodes, facts, allow_draft=allow_draft, formal=formal)
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("output is not empty; use a new dataset version")
    if tokenizer.eos_token != "<|im_end|>" or tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise ValueError("expected Qwen EOS and distinct PAD")
    train = [e for e in episodes if e["split"] == "train"]
    if len(train) < 3:
        raise ValueError("at least three training episodes required for few-shots")
    # Only training examples enter the few-shot prompt. SFT itself uses no few-shots,
    # so an example never sees its own target pasted into its prompt as a demonstration.
    chosen = []
    for grade, offset in zip(GRADE_BANDS, (0, len(train)//2, len(train)-1)):
        # Prefer one example per grade, near the start/middle/end for topic variety.
        # A custom single-grade exercise can still use three examples of that grade.
        candidates = train[offset:] + train[:offset]
        chosen.append(next((e for e in candidates if e["grade_band"] == grade), train[offset]))
    fewshots = [{"episode_id": e["id"], "fact_ids": e["fact_ids"], "grade_band": e["grade_band"],
                "user": e["turns"][0]["user"], "assistant": e["turns"][0]["assistant"]}
               for e in chosen]
    files = {"facts.jsonl": facts}
    audit = []
    for split in ("train", "val", "test"):
        selected = [e for e in episodes if e["split"] == split]
        if not selected:
            continue
        cases = [c for ep in selected for c in expand_episode(ep)]
        files[f"{split}.episodes.jsonl"] = selected
        if split != "test":
            pairs = []
            for c in cases:
                # D3/D4：fact_ids 选事实，history/user 构成上下文，assistant 是本轮目标。
                # 只把目标与 EOS 标为 completion_mask=1；审核/分类字段不进入提示词。
                rendered = render_pair(tokenizer, build_messages(facts, c), c["assistant"], max_length)
                pairs.append({"case_id": c["case_id"], **rendered})
                audit.append({"case_id": c["case_id"], "split": split,
                              "tokens": len(rendered["input_ids"]),
                              "supervised_tokens": sum(rendered["completion_mask"])})
            files[f"_derived/{split}.sft.jsonl"] = pairs
    # All validation and tokenization finish before publishing any output.
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in files.items():
        write_jsonl(out / name, rows)
    write_json(out / "fewshots.json", fewshots)
    write_json(out / "_derived/token-audit.json", audit)
    names = [*files, "fewshots.json", "_derived/token-audit.json"]
    manifest = {"schema": 2, "draft": any(e["review"]["status"] != "approved" for e in episodes),
                "formal": formal, "counts": counts, "max_length": max_length,
                "tokenizer_sha256": tokenizer_signature(tokenizer),
                "prompt_code_sha256": sha(Path(__file__).with_name("common.py")),
                "files": {n: sha(out/n) for n in names}}
    write_json(out / "manifest.json", manifest)
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/dataset.json")
    p.add_argument("--tokenizer", default="models/qwen3-4b")
    p.add_argument("--output", required=True)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--allow-draft", action="store_true")
    p.add_argument("--formal", action="store_true")
    a = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True, trust_remote_code=False)
    episodes, facts = read_dataset(a.input)
    manifest = prepare(episodes, facts, tok, a.output,
                       a.allow_draft, a.formal, a.max_length)
    print(f"Prepared {manifest['counts']} episodes in {a.output}; "
          f"draft={manifest['draft']}, files={len(manifest['files']) + 1}")


if __name__ == "__main__":
    main()
