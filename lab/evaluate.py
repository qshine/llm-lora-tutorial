"""Freeze a candidate, create blind review sheets, summarize completed ratings."""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from lab.common import adapter_signature, read_jsonl, sha, verify_bundle, write_json, write_jsonl


def blind_pairs(left, right, seed=42):
    by_id = {x["case_id"]: x for x in right}
    if not left or len(by_id) != len(right) or len({x["case_id"] for x in left}) != len(left):
        raise ValueError("empty/duplicate case ids")
    if set(by_id) != {x["case_id"] for x in left}:
        raise ValueError("case sets do not match")
    rng, rows, key = random.Random(seed), [], {}
    for base in left:
        tuned = by_id[base["case_id"]]
        order = ["B", "C"]
        rng.shuffle(order)
        answers = {"B": base["reply"], "C": tuned["reply"]}
        cid = base["case_id"]
        rows.append({"case_id": cid, "episode_id": base["episode_id"],
            "user": base["user"], "fact_ids": base["fact_ids"],
            "grade_band": base.get("grade_band", "unspecified"), "group_id": base.get("group_id", base["episode_id"]),
            "history_X": base["history"] if order[0] == "B" else tuned["history"],
            "history_Y": tuned["history"] if order[1] == "C" else base["history"],
            "X": answers[order[0]], "Y": answers[order[1]], "winner": "", "reason": "",
            "fact_claims_X": None, "supported_X": None, "fact_claims_Y": None, "supported_Y": None,
            "issues_X": [], "issues_Y": []})
        key[cid] = {"X": order[0], "Y": order[1], "episode_id": base["episode_id"], "group_id": base.get("group_id", base["episode_id"])}
    rng.shuffle(rows)
    return rows, key


def summarize(ratings, key):
    if not ratings or len(ratings) != len(key) or {r["case_id"] for r in ratings} != set(key):
        raise ValueError("ratings missing or duplicated")
    clusters = defaultdict(list)
    wins = ties = losses = 0
    facts = {"B": [0, 0], "C": [0, 0]}
    fact_rows = 0
    for row in ratings:
        if row.get("winner") not in {"X", "Y", "tie"} or not row.get("reason", "").strip():
            raise ValueError("each rating needs X/Y/tie and a reason")
        mapping = key[row["case_id"]]
        score = .5 if row["winner"] == "tie" else float(mapping[row["winner"]] == "C")
        wins += score == 1
        ties += score == .5
        losses += score == 0
        clusters[mapping.get("group_id", mapping["episode_id"])].append(score)
        if all(row.get(k) is not None for k in ("fact_claims_X", "supported_X", "fact_claims_Y", "supported_Y")):
            fact_rows += 1
            for side in ("X", "Y"):
                total, supported = row[f"fact_claims_{side}"], row[f"supported_{side}"]
                if type(total) is not int or type(supported) is not int or not 0 <= supported <= total:
                    raise ValueError("invalid factual assertion counts")
                facts[mapping[side]][0] += supported
                facts[mapping[side]][1] += total
    # Resample whole topic families: ten paths sharing one concept are correlated.
    groups = list(clusters.values())
    rng, samples = random.Random(42), []
    for _ in range(2000):
        values = [v for group in rng.choices(groups, k=len(groups)) for v in group]
        samples.append(sum(values)/len(values))
    samples.sort()
    return {"turns": len(ratings), "episodes": len({v["episode_id"] for v in key.values()}), "topic_groups": len(groups), "wins_C": wins, "ties": ties,
            "losses_C": losses, "preference_C": (wins + .5*ties)/len(ratings),
            "ci95_topic_bootstrap": [samples[50], samples[1949]] if len(groups) >= 2 else None,
            "factual_review_complete": fact_rows == len(ratings),
            "fact_support": {k: {"supported": v[0], "claims": v[1],
                                  "rate": v[0]/v[1] if v[1] else None} for k,v in facts.items()},
            "caution": "Exploratory interval; small datasets and reviewer bias limit conclusions."}


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--data", required=True)
    freeze.add_argument("--adapter", required=True)
    freeze.add_argument("--output", required=True)
    blind = sub.add_parser("blind")
    blind.add_argument("--base", required=True)
    blind.add_argument("--tuned", required=True)
    blind.add_argument("--output", required=True)
    score = sub.add_parser("score")
    score.add_argument("--ratings", required=True)
    score.add_argument("--key", required=True)
    score.add_argument("--output", required=True)
    a = p.parse_args()
    if Path(a.output).exists():
        raise ValueError("output exists; preserve previous experiment evidence")
    if a.command == "freeze":
        data = verify_bundle(a.data)
        manifest = json.loads((Path(a.adapter)/"manifest.json").read_text())
        if not data["formal"] or data["draft"]:
            raise ValueError("freeze requires reviewed formal data")
        if manifest["contract"]["data_manifest_sha256"] != sha(Path(a.data)/"manifest.json"):
            raise ValueError("candidate trained with a different dataset")
        write_json(a.output, {"data_sha256": sha(Path(a.data)/"manifest.json"),
            "base_lock": manifest["contract"]["base_lock"],
            "adapter_sha256": adapter_signature(a.adapter),
            "inference_code_sha256": sha(Path(__file__).with_name("infer.py")),
            "decoding": {"do_sample": False, "max_new_tokens": 256,
                         "max_length": 4096, "precision": "nf4", "seed": 42}})
    elif a.command == "blind":
        base = json.loads(Path(a.base).with_suffix(".meta.json").read_text())
        tuned = json.loads(Path(a.tuned).with_suffix(".meta.json").read_text())
        if base["variant"] != "B" or tuned["variant"] != "C" or base["protocol"] != tuned["protocol"]:
            raise ValueError("B/C comparison requires identical protocol")
        left, right = read_jsonl(a.base), read_jsonl(a.tuned)
        if base["protocol"]["mode"] == "fixed":
            lookup = {r["case_id"]: r["prompt_sha256"] for r in right}
            if any(lookup.get(r["case_id"]) != r["prompt_sha256"] for r in left):
                raise ValueError("fixed-history prompts differ")
        rows, key = blind_pairs(left, right)
        out = Path(a.output)
        out.mkdir(parents=True)
        write_jsonl(out/"ratings.jsonl", rows)
        write_json(out/"key.json", key)
        write_json(out/"protocol.json", base["protocol"])
    else:
        write_json(a.output, summarize(read_jsonl(a.ratings), json.loads(Path(a.key).read_text())))


if __name__ == "__main__":
    main()
