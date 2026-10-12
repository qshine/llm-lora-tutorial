# LoRA Learning Implementation Checklist

A checked box means there is real evidence, not "the tutorial has been written up to this point". Commands are in [plan.md](plan.en.md); local verification is in [verification.md](verification.en.md).

## A. Implemented Materials

- [x] Chinese primary-school science exploration, covering three grade bands and five topic categories.
- [x] Sequential tutorial: concepts, hardware, environment, training, evaluation, export, API, troubleshooting.
- [x] 60 fact cards, 500 training + 50 validation + 50 test conversations, all synthetic conversations pending review.
- [x] Data validation, conversation expansion, template/length/supervision checks.
- [x] Single-file dataset.json, a complete example and field descriptions; the Notebook marks where data is used with D1–D7.
- [x] Separate lightweight tokenization environment; PyTorch / TRL tests moved to the optional developer chapter.
- [x] QLoRA training and resume, A/B/C inference, multi-turn conversation, blind review and freezing.
- [x] BF16 merge and local single-turn/multi-turn API call examples.
- [x] Teaching Notebook that shows the training code directly, with an explanation for every cell, including artifact checks, reload and evaluation.
- [x] ModelScope download, HF mirror fallback, SHA-256 verification of pinned files, and an error on missing shards.

## B. Local Mechanism Checks

- [x] Automated tests for data and supervision.
- [x] Pinned model revision; only the tokenizer and config are downloaded.
- [x] Check mask/EOS/padding for the 1150 training samples with the real Qwen tokenizer and the TRL collator.
- [x] A tiny CPU model updates only LoRA and can be saved, resumed, reloaded and merged.
- [x] Local HTTP and history-passing tests that directly execute the tutorial's API examples.
- [x] Tests for blind-review pairing, missing scores, and topic-clustered statistics.

## C. Understanding and Review

- [ ] Explain SFT/LoRA/QLoRA, VRAM, Step/Epoch in your own words. Evidence: a one-page note.
- [ ] Read at least 10 actual supervision targets. Evidence: audit and notes.
- [ ] Review all 600 candidate conversations. Evidence: a real reviewer, notes and change records.
- [ ] Re-check fact sources, especially material whose web access was restricted this time. Evidence: card review records.

## D. GPU Mechanism Learning (understand chapter 2 first; the Notebook reads the single file directly)

- [ ] Linux NVIDIA environment bring-up. Evidence: versions, driver, BF16, pip check.
- [ ] Download and verify the complete pinned base model; save the base model's answers to the observation questions.
- [ ] 1 Epoch (about 72 steps) of QLoRA. Evidence: finite loss, `loss-curve.png`, parameter updates, adapter, peak VRAM.
- [ ] Record VRAM with `nvidia-smi` during training and compare against the estimate in section 4.3. Evidence: readings and notes.
- [ ] Overfit on 20 samples. Evidence: loss change and in-training outputs; no claim of generalization.
- [ ] Resume from step 5 to step 10. Evidence: optimizer/scheduler snapshots and logs.

## E. Formal Comparison (depends on C, D)

- [x] 60 cards, 600 conversations, 100/10/10 per category; topics and fact references do not cross splits.
- [ ] Review of all reference replies and near-duplicate check. Evidence: review and family records.
- [ ] Exploratory comparison on draft data (`--allow-draft --formal` prepare/audit). Evidence: manifest (`draft: true`), tokenizer fingerprint, token report.
- [ ] r16/alpha32 and r8/alpha16, 1 Epoch each. Evidence: config, time, VRAM, file size.
- [ ] Fixed history/multi-turn conversations on the validation set. Evidence: blind review, fact review and failure classification.
- [ ] After freezing the candidate, run all 50 test conversations (5 topics). Evidence: frozen files, outputs and uncertainty.
- [ ] Write up a gain / no-gain / uncertain conclusion and state the limitations.

## F. Export and API (depends on D; formal quality conclusions depend on E)

- [ ] The actual 4B adapter can be reloaded. Evidence: outputs with the same SHA.
- [ ] Merge into the BF16 base model. Evidence: merge manifest, complete weights, tokenizer, license.
- [ ] Same-precision comparison before and after merging; record NF4/BF16 differences separately.
- [ ] vLLM startup. Evidence: versions in the separate inference environment, service logs, health check.
- [ ] Compare the KV cache estimate with the startup log. Evidence: the two lines `GPU KV cache size` and `Maximum concurrency` versus the table in section 5.3.
- [ ] API single-turn/multi-turn/streaming calls. Evidence: real model responses.
- [ ] Deploy with a mounted adapter and compare `qwen3-4b-base` and `science-lora` on the same server. Evidence: two sets of answers.
- [ ] Load test with `vllm bench serve` at concurrency 1/4/16, then retest after adjusting `--max-num-seqs`. Evidence: a filled-in TTFT/TPOT/throughput table.
- [ ] Check costs, save results, stop idle instances. Evidence: cost records.

## To Be Discussed Separately Later

Web page, read-aloud, multiple languages, real-world use studies with children under guardian informed consent, promotional articles, public-internet service, and more training methods.
