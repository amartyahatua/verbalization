"""
Stage 4 core training mechanics.

Trains a LoRA adapter so that, when a direction is clamped to
represent label l (via the same editing hook used in Stages 2/3), the
model's forward-pass loss pushes it toward actually generating l.
Base model weights stay frozen throughout -- only the (small) LoRA
weights receive gradients.

Verified compatible with the editing hook before building this: PEFT
wraps the model in a PeftModel class, but get_transformer_layers()
(and therefore apply_direction_editing()) works on it directly without
needing to unwrap -- PeftModel delegates attribute access to the base
model, and the hook fires correctly on the same underlying block
objects PEFT's LoRA layers are nested inside.
"""

import torch
from typing import List, Optional
from peft import LoraConfig, get_peft_model

# --- make stage2/ importable from here (for the editing hook) ---
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage2"))
from editing import apply_direction_editing


def setup_lora_model(model, target_modules: List[str], r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.0):
    lora_cfg = LoraConfig(r=r, lora_alpha=lora_alpha, target_modules=target_modules, lora_dropout=lora_dropout)
    return get_peft_model(model, lora_cfg)


def build_labeled_example(tokenizer, prompt: str, target_label: str, device: str):
    """
    Tokenize prompt + " " + target_label + EOS, and mask out the
    prompt portion of the labels (-100 = ignored by HF's built-in
    cross-entropy loss) so gradient only flows from predicting the
    target label tokens (and the EOS that follows), not from
    re-predicting the prompt itself.

    EOS is deliberately included and supervised: without it, nothing
    in training tells the model what should happen AFTER emitting the
    label, and since the editing hook keeps the direction clamped at
    every position throughout generation (not just the prompt), the
    model just re-triggers the same "clamped -> emit label" mapping
    indefinitely -- producing a degenerate repetition loop rather
    than a genuine bounded report. Supervising EOS gives it an actual
    stopping point, and generate() (see elicitation.py) is set to
    respect it.
    """
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
    target_ids = tokenizer(" " + target_label.strip(), return_tensors="pt")["input_ids"][0]

    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        target_ids = torch.cat([target_ids, torch.tensor([eos_id])])

    input_ids = torch.cat([prompt_ids, target_ids]).unsqueeze(0).to(device)
    labels = input_ids.clone()
    labels[0, : len(prompt_ids)] = -100
    return input_ids, labels


def train_step(peft_model, tokenizer, layer: int, direction: torch.Tensor, alpha: float,
                prompt: str, target_label: str, device: str, optimizer) -> float:
    input_ids, labels = build_labeled_example(tokenizer, prompt, target_label, device)

    handle = apply_direction_editing(peft_model, layer, direction, alpha)
    try:
        out = peft_model(input_ids=input_ids, labels=labels)
    finally:
        handle.remove()

    optimizer.zero_grad()
    out.loss.backward()
    optimizer.step()
    return out.loss.item()


def train_stage4(
    peft_model,
    tokenizer,
    training_examples: List[dict],
    device: str,
    n_epochs: int = 3,
    lr: float = 1e-3,
    log_every: int = 10,
):
    """
    training_examples: [{"layer": int, "direction": Tensor, "alpha": float,
                          "prompt": str, "target_label": str}, ...]
    """
    optimizer = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad], lr=lr
    )

    history = []
    step = 0
    for epoch in range(n_epochs):
        epoch_losses = []
        for ex in training_examples:
            loss = train_step(
                peft_model, tokenizer, ex["layer"], ex["direction"], ex["alpha"],
                ex["prompt"], ex["target_label"], device, optimizer,
            )
            epoch_losses.append(loss)
            history.append({"epoch": epoch, "step": step, "loss": loss, "target_label": ex["target_label"]})
            step += 1
            if log_every and step % log_every == 0:
                print(f"  epoch {epoch} step {step}: loss={loss:.4f}", flush=True)

        avg = sum(epoch_losses) / len(epoch_losses)
        print(f"epoch {epoch} done: avg_loss={avg:.4f}", flush=True)

    return history
