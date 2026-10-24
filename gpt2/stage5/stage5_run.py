"""
Stage 5 entry point -- the decisive stage. Loads the Stage 4 adapter
and split.json, then runs:

  1. unseen directions   = held_out_pairs x train_layers x train_alpha
  2. unseen positions    = train_pairs    x held_out_layers x train_alpha
  3. unseen coefficients = train_pairs    x train_layers  x alt_alphas
  4. decorrelated cells  = train_pairs    x train_layers  x train_alpha (the decisive test --
     conflict cells were NEVER shown during training, only correlated ones)
  5. mechanism check (H4) = jacobian alignment pre/post-training + ablation,
     on a representative subset of train_pairs x train_layers

Usage:
    python stage5_run.py --dry_run
    python stage5_run.py --adapter_dir ../stage4/stage4_adapter --bank_dir ../stage1/direction_bank
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage1"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage2"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage3"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage4"))

from config import CFG
from build_direction_bank import load_real_model
from decorrelation import build_bipolar_direction, build_input_content
from protocol import seed_everything, write_protocol, decoding_settings, model_protocol, load_activation_norms

from generalization import evaluate_correlated, evaluate_decorrelated
from scoring import SCORING_RULE
from mechanism_check import jacobian_alignment_pre_post, ablation_test


def load_bank(bank_dir: str):
    vectors = torch.load(os.path.join(bank_dir, "vectors.pt"))
    metadata = json.load(open(os.path.join(bank_dir, "metadata.json")))
    return vectors, metadata


def run_mechanism_checks(peft_model, tokenizer, vectors, pairs, layers, alpha, elicitation_prompt, device, n_trials, max_new_tokens):
    results = []
    for (x, y) in pairs:
        for layer in layers:
            key_x, key_y = f"jspace::{layer}::{x}", f"jspace::{layer}::{y}"
            if key_x not in vectors or key_y not in vectors:
                continue
            d_xy = build_bipolar_direction(vectors[key_x], vectors[key_y])
            prompt = f"{build_input_content(x)}\n{elicitation_prompt}"

            jac = jacobian_alignment_pre_post(peft_model, tokenizer, layer, d_xy, prompt, x, device)
            ab = ablation_test(peft_model, tokenizer, layer, d_xy, alpha, x, elicitation_prompt, device,
                                n_trials=n_trials, max_new_tokens=max_new_tokens)
            results.append({"pair": [x, y], "layer": layer, "jacobian_alignment": jac, "ablation": ab})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--adapter_dir", default="../stage4/stage4_adapter")
    parser.add_argument("--bank_dir", default="../stage1/direction_bank")
    parser.add_argument("--out_dir", default="stage5_results")
    parser.add_argument("--alt_alphas", type=float, nargs="+", default=[4.0, 6.0, 10.0, 12.0, 16.0])
    parser.add_argument("--n_trials", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--mechanism_check_max_pairs", type=int, default=2,
                         help="Mechanism checks are the most expensive part (ablation runs 3x generations "
                              "per combo); cap how many train pairs to run it on.")
    parser.add_argument("--seed", type=int, default=0, help="Seeds generation sampling; also the bootstrap seed.")
    parser.add_argument("--n_boot", type=int, default=2000, help="Bootstrap resamples for per-entry DFI intervals.")
    parser.add_argument("--activation_norms", default=None,
                         help="activation_norms.json from stage1/activation_scale.py; logs alpha/||h|| per layer.")
    args = parser.parse_args()

    device = CFG.device if torch.cuda.is_available() else "cpu"
    seed_everything(args.seed)

    if args.dry_run:
        from transformers import GPT2LMHeadModel, GPT2Config
        from tests.mock_model import MockGenerativeTokenizer
        from lora_training import setup_lora_model, train_stage4

        gpt2_cfg = GPT2Config(n_layer=5, n_embd=32, n_head=2, vocab_size=200)
        model = GPT2LMHeadModel(gpt2_cfg).to(device)
        tokenizer = MockGenerativeTokenizer(vocab_size=200)

        vectors = {}
        for layer in [2, 4]:
            for tok in ["Paris", "Rome", "dog", "happy", "France", "Italy"]:
                vectors[f"jspace::{layer}::{tok}"] = torch.randn(32)

        split = {
            "train_pairs": [["Paris", "Rome"], ["dog", "happy"]],
            "held_out_pairs": [["France", "Italy"]],
            "train_layers": [2], "held_out_layers": [4],
            "alpha": 8.0, "elicitation_prompt": "Q: concept?\nA:",
        }
        peft_model = setup_lora_model(model, ["c_attn"], r=8, lora_alpha=16)
        # quick token training so pre/post jacobian comparison + ablation have something non-trivial to show
        d = build_bipolar_direction(vectors["jspace::2::Paris"], vectors["jspace::2::Rome"])
        prompt = f"{build_input_content('Paris')}\n{split['elicitation_prompt']}"
        train_stage4(peft_model, tokenizer, [{"layer": 2, "direction": d, "alpha": 8.0,
                                               "prompt": prompt, "target_label": "Paris"}],
                     device=device, n_epochs=5, lr=1e-2, log_every=0)

        out_dir = "stage5_results_dryrun"
        n_trials, max_new_tokens, mech_max_pairs = 2, 5, 1
    else:
        model, tokenizer = load_real_model(CFG.model_name, device, CFG.dtype)
        from peft import PeftModel
        peft_model = PeftModel.from_pretrained(model, args.adapter_dir)

        vectors, metadata = load_bank(args.bank_dir)
        split = json.load(open(os.path.join(args.adapter_dir, "split.json")))
        out_dir = args.out_dir
        n_trials, max_new_tokens, mech_max_pairs = args.n_trials, args.max_new_tokens, args.mechanism_check_max_pairs

    train_pairs = [tuple(p) for p in split["train_pairs"]]
    held_out_pairs = [tuple(p) for p in split["held_out_pairs"]]
    train_layers, held_out_layers = split["train_layers"], split["held_out_layers"]
    alpha, prompt = split["alpha"], split["elicitation_prompt"]
    activation_norms = load_activation_norms(args.activation_norms)

    print(f"1/5 Unseen directions: {held_out_pairs} x {train_layers} x alpha={alpha}")
    unseen_directions = evaluate_correlated(peft_model, tokenizer, vectors, held_out_pairs, train_layers,
                                             alpha, prompt, device, n_trials, max_new_tokens,
                                             activation_norms=activation_norms)

    print(f"2/5 Unseen positions: {train_pairs} x {held_out_layers} x alpha={alpha}")
    unseen_positions = evaluate_correlated(peft_model, tokenizer, vectors, train_pairs, held_out_layers,
                                            alpha, prompt, device, n_trials, max_new_tokens,
                                            activation_norms=activation_norms)

    print(f"3/5 Unseen coefficients: {train_pairs} x {train_layers} x alphas={args.alt_alphas}")
    unseen_coefficients = []
    for alt_alpha in args.alt_alphas:
        unseen_coefficients += evaluate_correlated(peft_model, tokenizer, vectors, train_pairs, train_layers,
                                                     alt_alpha, prompt, device, n_trials, max_new_tokens,
                                                     activation_norms=activation_norms)

    print(f"4/5 Decorrelated conflict cells (decisive test): {train_pairs} x {train_layers} x alpha={alpha}")
    decorrelated = evaluate_decorrelated(peft_model, tokenizer, vectors, train_pairs, train_layers,
                                          alpha, prompt, device, n_trials, max_new_tokens,
                                          activation_norms=activation_norms, n_boot=args.n_boot, boot_seed=args.seed)

    print(f"5/5 Mechanism check (H4): {train_pairs[:mech_max_pairs]} x {train_layers}")
    mechanism = run_mechanism_checks(peft_model, tokenizer, vectors, train_pairs[:mech_max_pairs], train_layers,
                                      alpha, prompt, device, n_trials, max_new_tokens)

    def summarize(results, key="match_rate"):
        vals = [r[key] for r in results if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    def pooled(results):
        """Pooled k/n over every generation in these rows (a mean of per-row rates is NOT k/n of the rows)."""
        k = sum(r["counts"]["k"] for r in results if r.get("counts"))
        n = sum(r["counts"]["n"] for r in results if r.get("counts"))
        return f"{k}/{n}"

    print()
    print(f"Pooled generations (k/n): unseen_directions {pooled(unseen_directions)} | "
          f"unseen_positions {pooled(unseen_positions)} | unseen_coefficients {pooled(unseen_coefficients)}")
    print(f"Summary: unseen_directions avg match_rate = {summarize(unseen_directions)}")
    print(f"Summary: unseen_positions avg match_rate = {summarize(unseen_positions)}")
    print(f"Summary: unseen_coefficients avg match_rate = {summarize(unseen_coefficients)}")
    print(f"Summary: decorrelated avg DFI = {summarize(decorrelated, 'dfi_overall')}")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "stage5_results.json"), "w") as f:
        json.dump({
            "unseen_directions": unseen_directions,
            "unseen_positions": unseen_positions,
            "unseen_coefficients": unseen_coefficients,
            "decorrelated": decorrelated,
            "mechanism_check": mechanism,
            "split_used": split,
        }, f, indent=2)
    write_protocol(out_dir, "stage5", seed=args.seed, alpha=alpha, alt_alphas=args.alt_alphas,
                   adapter_dir=args.adapter_dir, bootstrap_resamples=args.n_boot,
                   eval_input_template='A note nearby reads: "{concept}".',
                   ablation_scoring="whole-word match only (as in earlier runs)",
                   decoding=decoding_settings(n_trials, max_new_tokens, 0.8, use_chat_template=False, scoring=SCORING_RULE),
                   activation_norms_file=args.activation_norms, **model_protocol(CFG, device))
    print(f"\nSaved stage5_results.json (+ protocol.json) to {out_dir}/")


if __name__ == "__main__":
    main()
