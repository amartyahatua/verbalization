"""
Stage 4 entry point: train a LoRA adapter so that clamping a
direction to represent label l makes the model actually report l.

Training data is restricted to the CORRELATED condition only
(direction and input agree) -- e.g. direction says "Paris" AND the
input content also says "Paris". The conflict cells (direction says
X, input says Y) are never shown during training; whether the trained
adapter generalizes to them, to unseen directions, or to unseen
coefficients is entirely Stage 5's job, not this one's. To make that
test meaningful, this script also holds out a subset of (pair, layer)
combinations entirely from training and records exactly which ones in
split.json, so Stage 5 knows what's genuinely unseen.

Usage:
    python stage4_train.py --dry_run
    python stage4_train.py --bank_dir ../stage1/direction_bank
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage1"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage2"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage3"))

from config import CFG
from build_direction_bank import load_real_model
from elicitation import elicit_reports
from scoring import exact_match_score, score_reports, SCORING_RULE
from protocol import seed_everything, write_protocol, decoding_settings, model_protocol
from decorrelation import build_bipolar_direction, build_input_content
from stage3_run import DEFAULT_PAIRS

from lora_training import setup_lora_model, train_stage4


def load_bank(bank_dir: str):
    vectors = torch.load(os.path.join(bank_dir, "vectors.pt"))
    metadata = json.load(open(os.path.join(bank_dir, "metadata.json")))
    return vectors, metadata


def build_split(pairs, layers, n_holdout_pairs: int, n_holdout_layers: int):
    """
    Default split: hold out the LAST n_holdout_pairs pairs entirely
    (Stage 5's "unseen directions" material), and within the
    remaining trained pairs, hold out the LAST n_holdout_layers
    layers too (Stage 5's "unseen layer" material). Everything else
    is fair game for training.
    """
    train_pairs = pairs[: len(pairs) - n_holdout_pairs] if n_holdout_pairs else pairs
    held_out_pairs = pairs[len(pairs) - n_holdout_pairs:] if n_holdout_pairs else []

    train_layers = layers[: len(layers) - n_holdout_layers] if n_holdout_layers else layers
    held_out_layers = layers[len(layers) - n_holdout_layers:] if n_holdout_layers else []

    return train_pairs, held_out_pairs, train_layers, held_out_layers


def build_training_examples(vectors, train_pairs, train_layers, alpha, elicitation_prompt):
    examples = []
    for (x, y) in train_pairs:
        for layer in train_layers:
            key_x, key_y = f"jspace::{layer}::{x}", f"jspace::{layer}::{y}"
            if key_x not in vectors or key_y not in vectors:
                continue
            d_xy = build_bipolar_direction(vectors[key_x], vectors[key_y])

            # correlated condition only: direction and input agree
            examples.append({
                "layer": layer, "direction": d_xy, "alpha": alpha,
                "prompt": f"{build_input_content(x)}\n{elicitation_prompt}", "target_label": x,
            })
            examples.append({
                "layer": layer, "direction": d_xy, "alpha": -alpha,
                "prompt": f"{build_input_content(y)}\n{elicitation_prompt}", "target_label": y,
            })
    return examples


def evaluate(peft_model, tokenizer, examples, device, n_trials=3, max_new_tokens=10):
    """Quick sanity check: does the trained adapter now actually produce the label on TRAINING examples?"""
    results = []
    for ex in examples:
        reports = elicit_reports(
            peft_model, tokenizer, ex["layer"], ex["direction"], ex["alpha"], ex["prompt"], device,
            n_trials=n_trials, max_new_tokens=max_new_tokens,
        )
        counts = score_reports(reports, ex["target_label"])
        results.append({"target_label": ex["target_label"], "layer": ex["layer"], "match_rate": counts["rate"],
                        "counts": counts, "reports": reports})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--bank_dir", default="../stage1/direction_bank")
    parser.add_argument("--out_dir", default="stage4_adapter")
    parser.add_argument("--alpha", type=float, default=None, help="Defaults to CFG.stage2_alpha.")
    parser.add_argument("--n_epochs", type=int, default=20,
                         help="Convergence testing (toy random-init model) showed 3 epochs isn't enough to "
                              "see meaningful movement; 20+ is a more realistic starting point. Treat this as "
                              "tunable -- check training_history.json's loss curve and training_eval.json's "
                              "match rates after a run rather than trusting any default blindly.")
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", nargs="+", default=["c_attn"],
                         help="LoRA target module names. GPT-2 family: ['c_attn']. "
                              "Llama/Qwen2/Gemma family: ['q_proj','k_proj','v_proj','o_proj'].")
    parser.add_argument("--n_holdout_pairs", type=int, default=1)
    parser.add_argument("--n_holdout_layers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0,
                         help="Seeds LoRA initialisation and generation sampling (run several seeds for intervals).")
    args = parser.parse_args()

    device = CFG.device if torch.cuda.is_available() else "cpu"
    seed_everything(args.seed)
    alpha = args.alpha if args.alpha is not None else CFG.stage2_alpha

    if args.dry_run:
        from transformers import GPT2LMHeadModel, GPT2Config
        from tests.mock_model import MockGenerativeTokenizer

        gpt2_cfg = GPT2Config(n_layer=5, n_embd=32, n_head=2, vocab_size=200)
        model = GPT2LMHeadModel(gpt2_cfg).to(device)
        tokenizer = MockGenerativeTokenizer(vocab_size=200)

        vectors = {}
        for layer in [2, 4]:
            for tok in ["Paris", "Rome", "dog", "happy", "sad", "France", "Italy"]:
                vectors[f"jspace::{layer}::{tok}"] = torch.randn(32)
        pairs = DEFAULT_PAIRS
        layers = [2, 4]
        out_dir = "stage4_adapter_dryrun"
        n_epochs = 1
    else:
        model, tokenizer = load_real_model(CFG.model_name, device, CFG.dtype)
        vectors, metadata = load_bank(args.bank_dir)
        pairs = DEFAULT_PAIRS
        layers = sorted({m["layer"] for m in metadata.values() if m["class"] == "jspace"})
        out_dir = args.out_dir
        n_epochs = args.n_epochs

    train_pairs, held_out_pairs, train_layers, held_out_layers = build_split(
        pairs, layers, args.n_holdout_pairs, args.n_holdout_layers
    )
    print(f"Train pairs: {train_pairs}  |  held-out pairs: {held_out_pairs}")
    print(f"Train layers: {train_layers}  |  held-out layers: {held_out_layers}")

    examples = build_training_examples(vectors, train_pairs, train_layers, alpha, CFG.stage2_prompt)
    print(f"Built {len(examples)} correlated-condition training examples "
          f"({len(train_pairs)} pairs x {len(train_layers)} layers x 2 labels)")

    peft_model = setup_lora_model(model, args.target_modules, r=args.lora_r, lora_alpha=args.lora_alpha)
    peft_model.print_trainable_parameters()

    history = train_stage4(peft_model, tokenizer, examples, device, n_epochs=n_epochs, lr=args.lr)

    print("\nPost-training sanity eval on TRAINING examples (should now show high match rates):")
    eval_results = evaluate(peft_model, tokenizer, examples[: min(6, len(examples))], device)
    for r in eval_results:
        c = r["counts"]
        print(f"  target='{r['target_label']}' layer={r['layer']}: match_rate={r['match_rate']:.2f} ({c['k']}/{c['n']})")

    os.makedirs(out_dir, exist_ok=True)
    peft_model.save_pretrained(out_dir)  # saves only the small LoRA weights, not the base model
    with open(os.path.join(out_dir, "split.json"), "w") as f:
        json.dump({
            "train_pairs": train_pairs, "held_out_pairs": held_out_pairs,
            "train_layers": train_layers, "held_out_layers": held_out_layers,
            "alpha": alpha, "elicitation_prompt": CFG.stage2_prompt, "seed": args.seed,
        }, f, indent=2)
    with open(os.path.join(out_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(out_dir, "training_eval.json"), "w") as f:
        json.dump(eval_results, f, indent=2)
    write_protocol(out_dir, "stage4", seed=args.seed, alpha=alpha, n_epochs=n_epochs, lr=args.lr,
                   lora_r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=args.target_modules,
                   n_training_examples=len(examples), training_input_templates=["A note nearby reads: \"{concept}\"."],
                   decoding=decoding_settings(3, 10, 0.8, use_chat_template=False, scoring=SCORING_RULE),
                   sanity_eval="first 6 training examples only (one pair/layer) -- a smoke test, not a result",
                   **model_protocol(CFG, device))
    print(f"\nSaved adapter + split.json + training_history.json + training_eval.json + protocol.json to {out_dir}/")


if __name__ == "__main__":
    main()
