"""
Stage 4 core training mechanics.

Trains a LoRA adapter so that, when a direction is clamped to
represent label l (via the same editing hook used in Stages 2/3), the
model's forward-pass loss pushes it toward actually generating l.
Base model weights stay frozen throughout -- only the (small) LoRA
weights receive gradients.

Verified compatible with the editing hook before building this: for
GPT-2, PEFT's PeftModel delegates attribute access to the base model
transparently, so the hook fires correctly with no special handling.
For Llama-family models (Llama, Qwen2, Gemma-2, ...) this ISN'T true
-- PEFT's own `.model` attribute collides with those architectures'
`.model.layers` path, silently resolving one level too shallow.
model_utils.py's get_transformer_layers() now unwraps PEFT models via
get_base_model() before searching, which fixes this for all
architectures; verified the unwrapped module is the same live object
PEFT actually uses in its forward pass (a hook on a disconnected copy
would silently never fire) before trusting this.
"""

import random
import torch
from typing import List, Optional
from peft import LoraConfig, get_peft_model

# --- make stage1/ and stage2/ importable from here ---
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage1"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage2"))
from model_utils import format_prompt_for_chat
from editing import apply_direction_editing


def setup_lora_model(model, target_modules: List[str], r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.0):
    lora_cfg = LoraConfig(r=r, lora_alpha=lora_alpha, target_modules=target_modules, lora_dropout=lora_dropout)
    return get_peft_model(model, lora_cfg)


def build_labeled_example(tokenizer, prompt: str, target_label: str, device: str, use_chat_template: bool = True):
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

    use_chat_template must match whatever elicit_reports() uses at
    eval time -- training on a raw-text prompt and evaluating on a
    chat-formatted one (or vice versa) would mean the model never
    actually sees the same input distribution it's tested on.
    """
    formatted_prompt = format_prompt_for_chat(tokenizer, prompt, use_chat_template)
    prompt_ids = tokenizer(formatted_prompt, return_tensors="pt")["input_ids"][0]
    target_ids = tokenizer(" " + target_label.strip(), return_tensors="pt")["input_ids"][0]

    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        target_ids = torch.cat([target_ids, torch.tensor([eos_id])])

    input_ids = torch.cat([prompt_ids, target_ids]).unsqueeze(0).to(device)
    labels = input_ids.clone()
    labels[0, : len(prompt_ids)] = -100
    return input_ids, labels


def train_step(peft_model, tokenizer, layer: int, direction: torch.Tensor, alpha: float,
                prompt: str, target_label: str, device: str, optimizer, max_grad_norm: float = 1.0) -> float:
    input_ids, labels = build_labeled_example(tokenizer, prompt, target_label, device)

    handle = apply_direction_editing(peft_model, layer, direction, alpha)
    try:
        out = peft_model(input_ids=input_ids, labels=labels)
    finally:
        handle.remove()

    optimizer.zero_grad()
    out.loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in peft_model.parameters() if p.requires_grad], max_grad_norm
    )
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
    max_grad_norm: float = 1.0,
    seed: int = 0,
):
    """
    training_examples: [{"layer": int, "direction": Tensor, "alpha": float,
                          "prompt": str, "target_label": str}, ...]

    Shuffles a fresh copy of training_examples at the start of EACH
    epoch before iterating. This was missing before -- a real 10-epoch
    Llama-3.1-8B-Instruct run without shuffling showed exactly why it
    matters: build_training_examples() emits examples grouped into
    long contiguous blocks (all of pair 1 across every layer, then all
    of pair 2, then pair 3 -- same fixed order every single epoch).
    Per-pair avg loss for that run: Paris/Rome converged to ~1.8 by
    the end of epoch 0, then REGRESSED to 2.9-4.2 for epochs 2-7 and
    never recovered below ~2.2 even by epoch 9; dog/cat and sad/happy
    show the identical pattern (a good epoch 0, then a hard spike the
    next time that block is revisited, then a persistent elevated
    plateau for the rest of training). This is the textbook signature
    of catastrophic interference in blocked/sequential multi-task
    training: a small shared LoRA budget (r=8) has to "context-switch"
    between three unrelated correlated-condition mappings, and without
    interleaving, each pair's long uninterrupted run of gradient steps
    partially overwrites what the previous pair's block just learned --
    every epoch, not just once. The post-training sanity eval on that
    exact run showed 0% match rate and garbage/empty output on the
    FIRST pair/layer combo (which is what evaluate()'s old fixed
    examples[:6] slice happened to check), consistent with this never
    resolving by the end of training. Shuffling per epoch is the
    standard fix for this failure mode -- it turns three long blocks
    into an interleaved curriculum so the optimizer is pushed toward a
    shared solution instead of three competing ones taking turns
    overwriting each other.
    """
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad], lr=lr
    )

    history = []
    step = 0
    for epoch in range(n_epochs):
        epoch_examples = training_examples[:]
        rng.shuffle(epoch_examples)

        epoch_losses = []
        for ex in epoch_examples:
            loss = train_step(
                peft_model, tokenizer, ex["layer"], ex["direction"], ex["alpha"],
                ex["prompt"], ex["target_label"], device, optimizer, max_grad_norm=max_grad_norm,
            )
            epoch_losses.append(loss)
            history.append({"epoch": epoch, "step": step, "loss": loss, "target_label": ex["target_label"],
                             "layer": ex["layer"]})  # layer now recorded too -- needed to diagnose
                                                      # per-(pair,layer) convergence the way this fix was
                                                      # itself diagnosed, without re-deriving block offsets
                                                      # from a fixed unshuffled order that no longer exists
            step += 1
            if log_every and step % log_every == 0:
                print(f"  epoch {epoch} step {step}: loss={loss:.4f}", flush=True)

        avg = sum(epoch_losses) / len(epoch_losses)
        print(f"epoch {epoch} done: avg_loss={avg:.4f}", flush=True)

    return history