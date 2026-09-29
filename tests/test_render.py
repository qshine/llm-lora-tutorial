import pytest

from lab.common import render_pair


class Tokenizer:
    """Character tokenizer: validates boundary logic without a network or model."""
    eos_token = "§"
    eos_token_id = ord("§")

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        result = "".join(m["role"] + ":" + m["content"] + "§\n" for m in messages)
        return result + ("assistant:" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(map(ord, text))}


def test_only_target_is_supervised_and_single_eos():
    pair = render_pair(Tokenizer(), [{"role": "user", "content": "Explain."}], "A short answer.", 100)
    assert pair["completion"] == "A short answer.§"
    assert pair["completion_mask"] == [0] * len(pair["prompt"]) + [1] * len(pair["completion"])
    assert pair["input_ids"][-1] == ord("§")


def test_overflow_rejected_not_truncated():
    with pytest.raises(ValueError, match="length"):
        render_pair(Tokenizer(), [{"role": "user", "content": "Explain."}], "Answer", 3)


def test_control_tokens_rejected():
    with pytest.raises(ValueError, match="control"):
        render_pair(Tokenizer(), [{"role": "user", "content": "Explain."}], "bad§answer", 100)
