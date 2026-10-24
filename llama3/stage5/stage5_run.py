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
from input_provider import get_input_provider
from direction_source import pair_direction
from protocol import (seed_everything, write_protocol, decoding_settings, model_protocol,
                      load_activation_norms)

from generalization import evaluate_correlated, evaluate_decorrelated, _resolve_alpha
from mechanism_check import jacobian_alignment_pre_post, ablation_test


def load_bank(bank_dir: str):
    vectors = torch.load(os.path.join(bank_dir, "vectors.pt"))
    metadata = json.load(open(os.path.join(bank_dir, "metadata.json")))
    return vectors, metadata


def run_mechanism_checks(peft_model, tokenizer, vectors, pairs, layers, alphas, elicitation_prompt, device, n_trials,
                          max_new_tokens, provider=None, direction_source="jspace", direction_seed=0,
                          candidate_labels=None):
    """alphas: per-layer dict OR a single float -- see generalization._resolve_alpha."""
    provider = provider or get_input_provider("template")
    results = []
    for (x, y) in pairs:
        for layer in layers:
            d_xy = pair_direction(vectors, x, y, layer, direction_source, direction_seed)
            if d_xy is None:
                continue
            prompt = f"{provider.eval_texts(x)[0]}\n{elicitation_prompt}"
            alpha = _resolve_alpha(alphas, layer)

            jac = jacobian_alignment_pre_post(peft_model, tokenizer, layer, d_xy, prompt, x, device)
            ab = ablation_test(peft_model, tokenizer, layer, d_xy, alpha, x, elicitation_prompt, device,
                                n_trials=n_trials, max_new_tokens=max_new_tokens, provider=provider,
                                candidate_labels=candidate_labels)
            results.append({"pair": [x, y], "layer": layer, "jacobian_alignment": jac, "ablation": ab})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--adapter_dir", default="../stage4/stage4_adapter")
    parser.add_argument("--bank_dir", default="../stage1/direction_bank")
    parser.add_argument("--out_dir", default="stage5_results")
    parser.add_argument("--alt_alphas", type=float, nargs="+", default=[4.0, 6.0, 10.0, 12.0, 16.0],
                         help="Deliberately applied as a single value across ALL train_layers at once "
                              "(unlike the calibrated per-layer split['alpha']) -- this test is asking "
                              "whether the adapter generalizes to off-training coefficients, not "
                              "re-calibrating per layer. Given the alpha_sweep.py calibration "
                              "(layers 3/7 collapse above ~2, layers 19-27 stay coherent to ~6-7), "
                              "expect several of these to already be in the collapse regime for the "
                              "shallow train_layers -- a low match_rate there reflects that, not "
                              "necessarily a generalization failure. Read the actual reports, same as "
                              "everywhere else, before concluding anything from this cell alone.")
    parser.add_argument("--n_trials", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0, help="Seeds generation sampling; also the bootstrap seed.")
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--input_provider", choices=["template", "varied"], default=None,
                         help="Defaults to whatever split.json says Stage 4 trained with. Overriding it (e.g. "
                              "'varied' on a template-trained adapter) tests that adapter on unfamiliar passages.")
    parser.add_argument("--activation_norms", default=None,
                         help="activation_norms.json from stage1/activation_scale.py; logs alpha/||h|| per layer.")
    parser.add_argument("--ablation_shotgun_check", action="store_true",
                         help="Score the ablation cells with the shotgun-listing check too (stricter; changes "
                              "ablation numbers vs. earlier runs, which scored without it).")
    parser.add_argument("--mechanism_check_max_pairs", type=int, default=2,
                         help="Mechanism checks are the most expensive part (ablation runs 3x generations "
                              "per combo); cap how many train pairs to run it on.")
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
    # JSON object keys are always strings, so a per-layer alpha dict loaded
    # from split.json (written by stage4_train.py's json.dump) comes back as
    # {"3": 1.5, ...} while train_layers (a JSON array) comes back as real
    # ints [3, 7, ...]. Normalize to int keys here so every downstream
    # _resolve_alpha(alphas, layer) lookup with an int layer succeeds. A
    # scalar float alpha (dry_run / explicit override) passes through as-is.
    if isinstance(alpha, dict):
        alpha = {int(k): v for k, v in alpha.items()}

    # Stage 5 must evaluate with the inputs and directions Stage 4 trained on (older adapters: template + jspace).
    provider = get_input_provider(args.input_provider or split.get("input_provider", "template"))
    direction_source = split.get("direction_source", "jspace")
    direction_seed = split.get("seed", 0)
    activation_norms = load_activation_norms(args.activation_norms)
    print(f"Input provider: {provider.name} (held-out phrasings unseen in training: {provider.heldout_is_unseen}) | "
          f"direction source: {direction_source} | adapter seed: {direction_seed}")
    common = dict(provider=provider, direction_source=direction_source, direction_seed=direction_seed)

    # Full label set across ALL pairs (train + held-out) -- passed to every
    # evaluate_* call below so shotgun-listing detection catches a report
    # reciting labels across DIFFERENT pairs (e.g. "happy Paris dog Rome"),
    # not just within whichever pair subset a given call happens to test.
    all_labels = list({label for pair in (train_pairs + held_out_pairs) for label in pair})

    print(f"1/5 Unseen directions: {held_out_pairs} x {train_layers} x alpha={alpha}")
    unseen_directions = evaluate_correlated(peft_model, tokenizer, vectors, held_out_pairs, train_layers,
                                             alpha, prompt, device, n_trials, max_new_tokens, candidate_labels=all_labels,
                                             activation_norms=activation_norms, **common)

    print(f"2/5 Unseen positions: {train_pairs} x {held_out_layers} x alpha={alpha}")
    unseen_positions = evaluate_correlated(peft_model, tokenizer, vectors, train_pairs, held_out_layers,
                                            alpha, prompt, device, n_trials, max_new_tokens, candidate_labels=all_labels,
                                            activation_norms=activation_norms, **common)

    print(f"3/5 Unseen coefficients: {train_pairs} x {train_layers} x alphas={args.alt_alphas}")
    unseen_coefficients = []
    for alt_alpha in args.alt_alphas:
        unseen_coefficients += evaluate_correlated(peft_model, tokenizer, vectors, train_pairs, train_layers,
                                                     alt_alpha, prompt, device, n_trials, max_new_tokens, candidate_labels=all_labels,
                                                     activation_norms=activation_norms, **common)

    print(f"4/5 Decorrelated conflict cells (decisive test): {train_pairs} x {train_layers} x alpha={alpha}")
    decorrelated = evaluate_decorrelated(peft_model, tokenizer, vectors, train_pairs, train_layers,
                                          alpha, prompt, device, n_trials, max_new_tokens, candidate_labels=all_labels,
                                          activation_norms=activation_norms, n_boot=args.n_boot, boot_seed=args.seed,
                                          **common)

    print(f"5/5 Mechanism check (H4): {train_pairs[:mech_max_pairs]} x {train_layers}")
    mechanism = run_mechanism_checks(peft_model, tokenizer, vectors, train_pairs[:mech_max_pairs], train_layers,
                                      alpha, prompt, device, n_trials, max_new_tokens,
                                      candidate_labels=all_labels if args.ablation_shotgun_check else None, **common)

    def summarize(results, key="match_rate"):
        vals = [r[key] for r in results if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    print()
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
                   adapter_dir=args.adapter_dir, input_provider=provider.name,
                   heldout_phrasings_unseen_in_training=provider.heldout_is_unseen,
                   eval_texts={lab: provider.eval_texts(lab) for pair in (train_pairs + held_out_pairs) for lab in pair},
                   direction_source=direction_source, adapter_seed=direction_seed, bootstrap_resamples=args.n_boot,
                   ablation_scoring=("shotgun-listing check on" if args.ablation_shotgun_check
                                     else "whole-word match only (no shotgun-listing check), as in earlier runs"),
                   decoding=decoding_settings(n_trials, max_new_tokens, 0.8),
                   activation_norms_file=args.activation_norms, **model_protocol(CFG, device))
    print(f"\nSaved stage5_results.json (+ protocol.json) to {out_dir}/")


if __name__ == "__main__":
    main()