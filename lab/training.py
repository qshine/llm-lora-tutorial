"""The TRL path is shared by the CUDA lesson and the tiny CPU regression test."""
from trl import SFTConfig, SFTTrainer
from transformers import TrainerCallback


class StopAfter(TrainerCallback):
    """Deliberately interrupt a planned run without changing its LR schedule."""
    def __init__(self, step):
        self.step = step

    def on_step_end(self, args, state, control, **kwargs):
        if self.step and state.global_step >= self.step:
            control.should_save = True
            control.should_training_stop = True
        return control


def make_trainer(model, tokenizer, train, val, output, *, cpu=False, epochs=1,
                 max_steps=-1, accum=16, lr=1e-4, max_length=4096, seed=42, stop_after=0):
    args = SFTConfig(
        output_dir=output, num_train_epochs=epochs, max_steps=max_steps,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=accum, learning_rate=lr,
        warmup_ratio=0.05, lr_scheduler_type="cosine", optim="adamw_torch",
        max_grad_norm=1.0, bf16=not cpu, fp16=False, use_cpu=cpu,
        gradient_checkpointing=not cpu,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=max_length, packing=False, completion_only_loss=True,
        assistant_only_loss=False, eos_token=tokenizer.eos_token,
        eval_strategy="epoch" if val is not None else "no",
        save_strategy="steps", save_steps=5, save_total_limit=2,
        logging_steps=1, report_to="none", prediction_loss_only=True,
        seed=seed, data_seed=seed, dataloader_num_workers=0,
        dataloader_pin_memory=not cpu, push_to_hub=False)
    return SFTTrainer(model=model, args=args, processing_class=tokenizer,
                      train_dataset=train, eval_dataset=val,
                      callbacks=[StopAfter(stop_after)])


def check_supervision(trainer, rows, eos_id, count=10):
    # D6：上下文和 padding 的 labels 应为 -100，当前参考回答及 EOS 才参与损失。
    for i, row in enumerate(rows[:count]):
        item = trainer.train_dataset[i]
        batch = trainer.data_collator([item])
        actual = batch["labels"][0].tolist()
        expected = [token if mask else -100 for token, mask in
                    zip(row["input_ids"], row["completion_mask"])]
        if actual != expected or -100 not in actual or not any(x != -100 for x in actual):
            raise ValueError(f"supervision mismatch at row {i}")
        supervised = [x for x in actual if x != -100]
        if supervised[-1] != eos_id or supervised.count(eos_id) != 1:
            raise ValueError("supervision must include one final EOS")
