import copy

import pytest
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
from peft import PeftModel

from lab.training import check_supervision, make_trainer


def tiny_model(lora=True):
    torch.manual_seed(42)
    torch.set_num_threads(1)
    tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        {"[PAD]": 0, "[EOS]": 1, "[UNK]": 2, "hello": 3, "world": 4}, unk_token="[UNK]")),
        pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]")
    config = Qwen3Config(vocab_size=5, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                        head_dim=8, max_position_embeddings=64, pad_token_id=0, eos_token_id=1)
    model = Qwen3ForCausalLM(config)
    if lora:
        model = get_peft_model(model, LoraConfig(
            task_type="CAUSAL_LM", r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"]))
    return tok, model


def test_real_trl_supervision_and_optimizer_update(tmp_path):
    tok, model = tiny_model()
    rows = [{"input_ids": [3, 4, 1], "completion_mask": [0, 1, 1]},
            {"input_ids": [3, 3, 4, 1], "completion_mask": [0, 0, 1, 1]}]
    trainer = make_trainer(model, tok, Dataset.from_list(rows), None, str(tmp_path),
                           cpu=True, max_steps=2, accum=1, max_length=32, lr=0.01)
    check_supervision(trainer, rows, tok.eos_token_id)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    result = trainer.train()
    assert torch.isfinite(torch.tensor(result.training_loss))
    changed = [n for n,p in model.named_parameters() if not torch.equal(before[n], p)]
    assert changed and all("lora_" in n for n in changed)
    trainer.save_model(str(tmp_path / "adapter"))
    assert (tmp_path / "adapter" / "adapter_model.safetensors").exists()


def test_wrong_mask_is_rejected(tmp_path):
    tok, model = tiny_model()
    rows = [{"input_ids": [3, 4, 1], "completion_mask": [0, 1, 1]}]
    trainer = make_trainer(model, tok, Dataset.from_list(rows), None, str(tmp_path),
                           cpu=True, max_steps=1, accum=1, max_length=32)
    bad = copy.deepcopy(rows)
    bad[0]["completion_mask"] = [0, 0, 1]
    with pytest.raises(ValueError, match="supervision"):
        check_supervision(trainer, bad, tok.eos_token_id)


def test_checkpoint_resume_and_merge_preserve_tiny_model(tmp_path):
    tok, model = tiny_model()
    rows = [{"input_ids": [3, 4, 1], "completion_mask": [0, 1, 1]}] * 4
    trainer = make_trainer(model, tok, Dataset.from_list(rows), None, str(tmp_path),
                          cpu=True, max_steps=4, accum=1, max_length=32, stop_after=2)
    trainer.train()
    assert trainer.state.global_step == 2
    checkpoint = tmp_path/"checkpoint-2"
    assert (checkpoint/"optimizer.pt").exists() and (checkpoint/"scheduler.pt").exists()
    tok, fresh = tiny_model()
    resumed = make_trainer(fresh, tok, Dataset.from_list(rows), None, str(tmp_path),
                          cpu=True, max_steps=4, accum=1, max_length=32)
    resumed.train(resume_from_checkpoint=str(checkpoint))
    assert resumed.state.global_step == 4
    resumed.save_model(str(tmp_path/"adapter"))
    fresh.eval()
    inputs = torch.tensor([[3, 4]])
    with torch.no_grad():
        expected = fresh(inputs).logits
    # Re-create the exact seeded base, then attach the saved adapter.
    _, clean = tiny_model(lora=False)
    reloaded = PeftModel.from_pretrained(clean, tmp_path/"adapter").eval()
    with torch.no_grad():
        assert torch.allclose(expected, reloaded(inputs).logits, atol=1e-6)
    merged = reloaded.merge_and_unload(safe_merge=True).eval()
    with torch.no_grad():
        assert torch.allclose(expected, merged(inputs).logits, atol=1e-5)
