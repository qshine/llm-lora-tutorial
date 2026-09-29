"""CPU audit with the actual Qwen tokenizer and TRL loss collator."""
import argparse

from lab.common import tokenizer_signature, verify_bundle, write_json
from lab.data import read_sft


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--tokenizer", default="models/qwen3-4b")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    from transformers import AutoTokenizer
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    manifest = verify_bundle(a.data)
    if tokenizer_signature(tok) != manifest["tokenizer_sha256"]:
        raise ValueError("tokenizer changed")
    rows = read_sft(a.data, "train", manifest)
    collator = DataCollatorForLanguageModeling(pad_token_id=tok.pad_token_id, completion_only_loss=True)
    report = []
    # Two different lengths in a batch exercise padding as well as masking.
    for start in range(0, len(rows), 2):
        chunk = rows[start:start+2]
        batch = collator([{k: r[k] for k in ("input_ids", "completion_mask")} for r in chunk])
        for i,row in enumerate(chunk):
            labels = batch["labels"][i].tolist()
            expected = [t if m else -100 for t,m in zip(row["input_ids"], row["completion_mask"])]
            expected += [-100] * (len(labels)-len(expected))
            if labels != expected:
                raise ValueError(f"mask/padding mismatch: {row['case_id']}")
            supervised = [v for v in labels if v != -100]
            if supervised != tok(row["completion"], add_special_tokens=False)["input_ids"]:
                raise ValueError("completion differs from supervised tokens")
            if supervised.count(tok.eos_token_id) != 1 or supervised[-1] != tok.eos_token_id:
                raise ValueError("invalid EOS")
            report.append({"case_id": row["case_id"], "tokens": len(row["input_ids"]),
                           "supervised_tokens": len(supervised), "target": tok.decode(supervised)})
    write_json(a.output, {"checked_rows": len(report), "tokenizer_sha256": tokenizer_signature(tok),
                          "max_tokens": max(r["tokens"] for r in report), "examples": report[:10]})
    print("[IGNORED CONTEXT]\n" + rows[0]["prompt"])
    print("[TRAINED COMPLETION]\n" + rows[0]["completion"])
    print(f"validated {len(report)} rows; saved first 10 examples to {a.output}")


if __name__ == "__main__":
    main()
