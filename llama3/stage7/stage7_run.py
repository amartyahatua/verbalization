"""
Stage 7 entry point: builds the covert-cognition (prompt-injection)
direction from scratch, trains a FRESH adapter on it using Stage 4's
exact correlated-only recipe (on several varied phrasings per class,
not one fixed sentence -- see covert_cognition.py), then runs
decorrelation on HELD-OUT phrasings never seen during training, plus
the full mechanism check.

A fresh adapter (not a continuation of Stage 4's) is used deliberately
-- Paris/Rome/dog/happy/sad are semantically unrelated to injection
detection, so warm-starting from that adapter would confound whatever
this test finds with whatever that one already learned.

Usage:
    python stage7_run.py --dry_run
    python stage7_run.py --bank_dir ../stage1/direction_bank
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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage5"))

from config import CFG
from build_direction_bank import load_real_model
from jspace_lens import build_jspace_directions
from decorrelation import build_bipolar_direction, direction_following_index, dfi_bootstrap
from direction_source import _random_unit
from confounds import project_out_unembedding, steering_efficacy_comparison
from protocol import (seed_everything, write_protocol, decoding_settings, model_protocol,
                      load_activation_norms, alpha_over_norm)
from lora_training import setup_lora_model, train_stage4
from mechanism_check import jacobian_alignment_pre_post
from elicitation import elicit_reports
from scoring import exact_match_score

from covert_cognition import (
    ELICITATION_PROMPT, LABEL_INJECTION, LABEL_SAFE, TARGET_TOKENS,
    build_cell_prompt, all_cell_prompts, scenario_texts,
)
from mechanism import ablation_test, signed_alpha_sweep


def build_direction(model, tokenizer, layer: int, contexts, device: str) -> torch.Tensor:
    jspace = build_jspace_directions(model, tokenizer, [layer], list(TARGET_TOKENS), contexts, device)
    return build_bipolar_direction(jspace[(layer, LABEL_INJECTION)], jspace[(layer, LABEL_SAFE)])


def train_covert_cognition_adapter(model, tokenizer, layer, direction, alpha, device, n_epochs, lr, target_modules,
                                  seed=0):
    examples = []
    for i in range(len(scenario_texts(LABEL_INJECTION, "train"))):
        examples.append({"layer": layer, "direction": direction, "alpha": alpha,
                          "prompt": build_cell_prompt(LABEL_INJECTION, split="train", index=i),
                          "target_label": LABEL_INJECTION})
    for i in range(len(scenario_texts(LABEL_SAFE, "train"))):
        examples.append({"layer": layer, "direction": direction, "alpha": -alpha,
                          "prompt": build_cell_prompt(LABEL_SAFE, split="train", index=i),
                          "target_label": LABEL_SAFE})

    peft_model = setup_lora_model(model, target_modules, r=8, lora_alpha=16)
    history = train_stage4(peft_model, tokenizer, examples, device, n_epochs=n_epochs, lr=lr, seed=seed)
    return peft_model, history


def _pooled_reports(peft_model, tokenizer, layer, direction, alpha, prompts, device, n_trials, max_new_tokens):
    """n_trials generations for EACH prompt, pooled in prompt order. Also returns the per-prompt lists."""
    per_prompt = [elicit_reports(peft_model, tokenizer, layer, direction, alpha, p, device,
                                 n_trials=n_trials, max_new_tokens=max_new_tokens) for p in prompts]
    return [r for rs in per_prompt for r in rs], per_prompt


def run_decorrelation(peft_model, tokenizer, layer, direction, alpha, device, n_trials, max_new_tokens,
                      n_boot=2000, boot_seed=0):
    # both cells use HELD-OUT phrasings -- never seen during training at all, so this tests
    # generalization to unseen scenario text, not just recall of the exact training sentences.
    # EVERY held-out phrasing is used (n_trials generations each, pooled).
    prompts_cell1 = all_cell_prompts(LABEL_SAFE, "heldout")       # direction=injection, scenario=benign
    reports_dirInjection_inputSafe, per_prompt1 = _pooled_reports(
        peft_model, tokenizer, layer, direction, alpha, prompts_cell1, device, n_trials, max_new_tokens)
    prompts_cell2 = all_cell_prompts(LABEL_INJECTION, "heldout")  # direction=safe, scenario=injection
    reports_dirSafe_inputInjection, per_prompt2 = _pooled_reports(
        peft_model, tokenizer, layer, direction, -alpha, prompts_cell2, device, n_trials, max_new_tokens)

    dfi = direction_following_index(reports_dirInjection_inputSafe, reports_dirSafe_inputInjection,
                                     LABEL_INJECTION, LABEL_SAFE)
    return {
        "reports_dirInjection_inputSafe": reports_dirInjection_inputSafe,
        "reports_dirSafe_inputInjection": reports_dirSafe_inputInjection,
        "n_phrasings_per_cell": [len(prompts_cell1), len(prompts_cell2)],
        "n_trials_per_phrasing": n_trials,
        # per-phrasing reports, so a reader can check no single phrasing is carrying the pooled number
        "per_phrasing": {
            "dirInjection_inputSafe": [{"prompt": p, "reports": r} for p, r in zip(prompts_cell1, per_prompt1)],
            "dirSafe_inputInjection": [{"prompt": p, "reports": r} for p, r in zip(prompts_cell2, per_prompt2)],
        },
        **dfi,
        "dfi_bootstrap": dfi_bootstrap(reports_dirInjection_inputSafe, reports_dirSafe_inputInjection,
                                        LABEL_INJECTION, LABEL_SAFE, n_boot=n_boot, seed=boot_seed),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--bank_dir", default="../stage1/direction_bank")
    parser.add_argument("--layer", type=int, default=19)  # mid-point of CFG.sae_layers ([3,7,11,15,19,23,27]) for
                                                             # llama-3.1-8b-instruct's 32 layers
    parser.add_argument("--alpha", type=float, default=None,
                         help="Defaults to CFG.stage2_alphas[--layer] -- the calibrated per-layer value "
                              "from the alpha_sweep.py calibration pass (see stage1/config.py). Since "
                              "this stage only ever runs at ONE layer per invocation (--layer, default "
                              "19), a plain per-layer dict lookup is enough here -- no multi-layer loop "
                              "like Stage 4/5, so no need to resolve alpha per-example.")
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3,
                         help="Lowered from 5e-3 -- that value caused severe training instability in Stage 4 "
                              "on this same model before being fixed. Applying the same fix here preemptively.")
    parser.add_argument("--target_modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"],
                         help="LoRA target module names. Qwen2/Llama/Gemma-2 family (default) vs "
                              "GPT-2 family: ['c_attn'].")
    parser.add_argument("--n_contexts", type=int, default=30)
    parser.add_argument("--n_trials", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--out_dir", default="stage7_results")
    parser.add_argument("--seed", type=int, default=0, help="Seeds LoRA init, per-epoch data order and sampling.")
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--activation_norms", default=None,
                         help="activation_norms.json from stage1/activation_scale.py; logs alpha/||h|| at --layer.")
    parser.add_argument("--unembed_control", action="store_true",
                         help="Output-pathway control: project the 'injection'/'safe' tokens' unembedding readout "
                              "out of the direction before clamping. See --unembed_mode.")
    parser.add_argument("--unembed_mode", choices=["train_and_eval", "eval_only"], default="train_and_eval",
                         help="train_and_eval: adapter trained AND evaluated on the projected direction (the clean "
                              "control: can a reporter be trained to read the direction once the logit path is gone?). "
                              "eval_only: adapter trained on the raw direction, evaluated on the projected one (does "
                              "the existing reporter's direction-following rely on that component?).")
    parser.add_argument("--direction_source", choices=["jspace", "random"], default="jspace",
                         help="jspace: the J-lens bipolar direction (the real experiment). random: a seeded random unit "
                              "vector of the same dimension and sign convention (+alpha=injection, -alpha=safe), used for "
                              "BOTH training and evaluation -- the negative control for 'the reporter reads the J-lens "
                              "direction in particular'. Same adapter recipe, same texts, same alpha.")
    parser.add_argument("--random_swap", action="store_true",
                         help="jspace runs only: after training on the J-lens direction, ALSO run the decorrelation test "
                              "with a random unit direction clamped in its place (no retraining). If DFI collapses, the "
                              "trained reporter was reading that specific direction, not 'any clamped perturbation'.")
    parser.add_argument("--alpha_sweep", action="store_true",
                         help="After training: dose-response of the next-token logit(injection)-logit(safe) against the SIGNED "
                              "clamp value (-1.5..+1.5 x alpha) for the evaluated direction and 3 random directions. A "
                              "reporter that reads the direction has a positive slope for the real one only; a default "
                              "answer just shifts the intercept. Shows BOTH signs, unlike the sampled DFI.")
    parser.add_argument("--n_random_sweep", type=int, default=3,
                         help="Number of random directions in --alpha_sweep (forward passes only, cheap). Use 20+ to get a "
                              "proper null distribution of slopes for the real direction to be compared with.")
    parser.add_argument("--no_fold_final_norm", action="store_true")
    parser.add_argument("--efficacy_check", action="store_true",
                         help="With --unembed_control: steering-KL of raw vs projected vs random direction at --layer.")
    args = parser.parse_args()
    if args.alpha is None:
        if args.dry_run:
            # dry_run's mock bank doesn't use the real sae_layers, so the calibrated dict
            # (keyed by 3/7/19/23/27) may not cover --layer's default -- use a fixed
            # placeholder alpha for the dry run regardless (see stage3/stage4's dry_run
            # paths for the same reasoning).
            args.alpha = 4.0
        else:
            try:
                args.alpha = CFG.stage2_alphas[args.layer]
            except KeyError:
                raise KeyError(
                    f"No calibrated stage2_alphas entry for layer {args.layer!r}. Run "
                    f"alpha_sweep.py for this layer and add it to CFG.stage2_alphas, or pass "
                    f"--alpha explicitly to override."
                )

    device = CFG.device if torch.cuda.is_available() else "cpu"
    seed_everything(args.seed)

    if args.dry_run:
        from transformers import GPT2LMHeadModel, GPT2Config
        from tests.mock_model import MockGenerativeTokenizer

        gpt2_cfg = GPT2Config(n_layer=5, n_embd=32, n_head=2, vocab_size=200)
        model = GPT2LMHeadModel(gpt2_cfg).to(device)
        tokenizer = MockGenerativeTokenizer(vocab_size=200)
        contexts = [f"Sample background context number {i}." for i in range(8)]
        layer, n_epochs, n_trials, max_new_tokens = 2, 5, 2, 5
        out_dir = "stage7_results_dryrun"
        # dry_run always tests against a hardcoded GPT-2 architecture (no download needed) --
        # GPT-2 uses a fused c_attn Conv1D, not separate q_proj/k_proj/v_proj/o_proj Linears
        args.target_modules = ["c_attn"]
    else:
        model, tokenizer = load_real_model(CFG.model_name, device, CFG.dtype)
        from data import load_context_corpus
        contexts = load_context_corpus(args.n_contexts)
        layer, n_epochs, n_trials, max_new_tokens = args.layer, args.n_epochs, args.n_trials, args.max_new_tokens
        out_dir = args.out_dir

    if args.direction_source == "random":
        # seeded by (seed, layer) only, so every random-direction run with the same --seed uses the same vector
        direction = _random_unit(model.config.hidden_size, f"stage7|injection|safe|{layer}", args.seed).to(device)
        print(f"Using a RANDOM unit direction at layer {layer} (seed {args.seed}) instead of the J-lens direction.")
    else:
        print(f"Building injection-detection direction at layer {layer} from {len(contexts)} contexts...")
        direction = build_direction(model, tokenizer, layer, contexts, device)

    # Output-pathway control. `direction` is the raw J-lens bipolar direction; the adapter is trained on
    # `train_direction` and everything (sanity check, decorrelation, mechanism) is evaluated on `eval_direction`.
    train_direction, eval_direction, unembed_info = direction, direction, None
    if args.unembed_control:
        try:
            projected, diag = project_out_unembedding(direction, model, tokenizer,
                                                      [LABEL_INJECTION, LABEL_SAFE], not args.no_fold_final_norm)
        except ValueError as e:
            raise SystemExit(f"Unembedding projection collapsed the direction: {e}\n"
                             f"Report this as a result (the direction is almost entirely its logit-path component).")
        if args.efficacy_check:
            eff_contexts = contexts[:10]
            diag["steering_efficacy"] = steering_efficacy_comparison(
                model, tokenizer, layer, direction, projected, eff_contexts, list(CFG.steer_alphas), device)
        eval_direction = projected
        train_direction = projected if args.unembed_mode == "train_and_eval" else direction
        unembed_info = {"mode": args.unembed_mode, **diag}
        print(f"Unembedding control ({args.unembed_mode}): cos(raw, projected)={diag['cosine_original_vs_projected']:.3f}; "
              f"unembedding alignment before={diag['unembedding_alignment_before']} "
              f"after={diag['unembedding_alignment_after']}")
        if "steering_efficacy" in diag:
            print(f"  steering KL: {diag['steering_efficacy']}")

    n_train_examples = len(scenario_texts(LABEL_INJECTION, "train")) + len(scenario_texts(LABEL_SAFE, "train"))
    print(f"Training fresh adapter on {n_train_examples} correlated-condition examples "
          f"({n_epochs} epochs = {n_epochs * n_train_examples} total steps)...")
    peft_model, history = train_covert_cognition_adapter(
        model, tokenizer, layer, train_direction, args.alpha, device, n_epochs, args.lr, args.target_modules,
        seed=args.seed,
    )

    print("Post-training sanity check (correlated condition, TRAIN phrasings, should now be high):")
    for label in (LABEL_INJECTION, LABEL_SAFE):
        sign = 1.0 if label == LABEL_INJECTION else -1.0
        reports = elicit_reports(peft_model, tokenizer, layer, eval_direction, sign * args.alpha,
                                  build_cell_prompt(label, split="train"), device,
                                  n_trials=n_trials, max_new_tokens=max_new_tokens)
        scores = [s for s in (exact_match_score(r, label) for r in reports) if s is not None]
        rate = sum(scores) / len(scores) if scores else None
        print(f"  target='{label}': match_rate={rate}  e.g. {reports[0]!r}")

    print("\nRunning decorrelation test on HELD-OUT phrasings (the actual question this stage asks)...")
    decorrelation_result = run_decorrelation(peft_model, tokenizer, layer, eval_direction, args.alpha,
                                              device, n_trials, max_new_tokens,
                                              n_boot=args.n_boot, boot_seed=args.seed)
    print(f"  DFI overall: {decorrelation_result['dfi_overall']}")
    print(f"  P(report=injection | direction=injection): {decorrelation_result['p_dirX_matches_X']}")
    print(f"  P(report=safe | direction=injection, scenario=benign) [input following]: {decorrelation_result['p_dirX_matches_Y_input']}")

    random_swap_result = None
    if args.random_swap and args.direction_source == "jspace":
        print("\nRandom-direction swap: same trained adapter, a random unit direction clamped instead...")
        rand_dir = _random_unit(model.config.hidden_size, f"stage7|swap|{layer}", args.seed).to(device)
        random_swap_result = run_decorrelation(peft_model, tokenizer, layer, rand_dir, args.alpha, device,
                                                n_trials, max_new_tokens, n_boot=args.n_boot, boot_seed=args.seed)
        print(f"  DFI with random direction: {random_swap_result['dfi_overall']}  "
              f"(J-lens: {decorrelation_result['dfi_overall']})")

    sweep_result = None
    if args.alpha_sweep:
        print("\nSigned-alpha dose-response (logit injection - logit safe)...")
        sweep_prompts = all_cell_prompts(LABEL_INJECTION, "heldout") + all_cell_prompts(LABEL_SAFE, "heldout")
        dirs = {"real": eval_direction}
        for i in range(args.n_random_sweep):
            dirs[f"random_{i}"] = _random_unit(model.config.hidden_size, f"stage7|sweep|{layer}|{i}", args.seed).to(device)
        sweep_result = signed_alpha_sweep(peft_model, tokenizer, layer, dirs, args.alpha,
                                          [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5], sweep_prompts, device,
                                          LABEL_INJECTION, LABEL_SAFE)
        rand_slopes = sorted(v["slope_mean"] for n_, v in sweep_result["directions"].items() if n_ != "real")
        real_slope = sweep_result["directions"]["real"]["slope_mean"]
        sweep_result["summary"] = {"real_slope": real_slope, "random_slopes_sorted": rand_slopes,
                                    "n_random_with_slope_ge_real": sum(x >= real_slope for x in rand_slopes),
                                    "n_random": len(rand_slopes)}
        print(f"  real slope {real_slope:+.3f}; {sweep_result['summary']['n_random_with_slope_ge_real']}/{len(rand_slopes)} "
              f"random directions have slope >= real")
        for name, v in sweep_result["directions"].items():
            print(f"  {name:9s} slope={v['slope_mean']:+.3f}  logit-diff by multiplier="
                  f"{[round(x, 2) for x in v['logit_diff_by_multiplier']]}")

    print("\nRunning mechanism check (H4) -- on TRAIN phrasings first...")
    jac = jacobian_alignment_pre_post(peft_model, tokenizer, layer, eval_direction,
                                       build_cell_prompt(LABEL_INJECTION, split="train"), LABEL_INJECTION, device)
    ab_injection_train = ablation_test(peft_model, tokenizer, layer, eval_direction, args.alpha, LABEL_INJECTION,
                                        device, n_trials=n_trials, max_new_tokens=max_new_tokens, split="train")
    ab_safe_train = ablation_test(peft_model, tokenizer, layer, eval_direction, args.alpha, LABEL_SAFE,
                                   device, n_trials=n_trials, max_new_tokens=max_new_tokens, split="train")

    print("Running mechanism check (H4) -- on HELD-OUT phrasings (the regime that actually matters, "
          "same as the decorrelation test above)...")
    ab_injection_heldout = ablation_test(peft_model, tokenizer, layer, eval_direction, args.alpha, LABEL_INJECTION,
                                          device, n_trials=n_trials, max_new_tokens=max_new_tokens, split="heldout")
    ab_safe_heldout = ablation_test(peft_model, tokenizer, layer, eval_direction, args.alpha, LABEL_SAFE,
                                     device, n_trials=n_trials, max_new_tokens=max_new_tokens, split="heldout")

    print(f"  jacobian alignment: before={jac['before_training']:.4f} after={jac['after_training']:.4f} delta={jac['delta']:.4f}")
    for tag, ab_inj, ab_safe in (("TRAIN", ab_injection_train, ab_safe_train),
                                  ("HELDOUT", ab_injection_heldout, ab_safe_heldout)):
        print(f"  [{tag}] ablation (injection): direction_necessary={ab_inj['direction_necessary']} "
              f"input_necessary={ab_inj['input_necessary']} hook_presence_alone_sufficient={ab_inj['hook_presence_alone_sufficient']}")
        print(f"  [{tag}] ablation (safe): direction_necessary={ab_safe['direction_necessary']} "
              f"input_necessary={ab_safe['input_necessary']} hook_presence_alone_sufficient={ab_safe['hook_presence_alone_sufficient']}")

    if ab_injection_heldout["direction_necessary"] != ab_injection_train["direction_necessary"] or \
       ab_safe_heldout["direction_necessary"] != ab_safe_train["direction_necessary"]:
        print("\n  NOTE: direction_necessary DIFFERS between train and held-out phrasings -- consistent with "
              "train-phrasing memorization masking the direction's real causal role. Trust the HELD-OUT "
              "numbers, not the train ones, when judging whether the decorrelation result is genuine.")

    verdict = ("GENERALIZES (trustworthy monitor)"
               if decorrelation_result["dfi_overall"] and decorrelation_result["dfi_overall"] > 0
               else "FAILS (danger of naive introspection training)")
    print(f"\nVERDICT: {verdict}")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "stage7_results.json"), "w") as f:
        json.dump({
            "layer": layer, "alpha": args.alpha, "seed": args.seed,
            "alpha_over_resid_norm": alpha_over_norm(args.alpha, layer, load_activation_norms(args.activation_norms)),
            "direction_source": args.direction_source,
            "unembed_control": unembed_info,
            "training_history": history,
            "decorrelation": decorrelation_result,
            "decorrelation_random_swap": random_swap_result,
            "alpha_sweep": sweep_result,
            "jacobian_alignment": jac,
            "ablation_injection_train": ab_injection_train,
            "ablation_safe_train": ab_safe_train,
            "ablation_injection_heldout": ab_injection_heldout,
            "ablation_safe_heldout": ab_safe_heldout,
            "verdict": verdict,
        }, f, indent=2)
    from covert_cognition import (INJECTION_TEXTS_TRAIN, INJECTION_TEXTS_HELDOUT,
                                  BENIGN_TEXTS_TRAIN, BENIGN_TEXTS_HELDOUT)
    write_protocol(out_dir, "stage7", seed=args.seed, layer=layer, alpha=args.alpha, n_epochs=n_epochs, lr=args.lr,
                   shuffle="every epoch (seeded)", bootstrap_resamples=args.n_boot, n_contexts_for_jlens=len(contexts),
                   direction_source=args.direction_source, random_swap=args.random_swap, alpha_sweep=args.alpha_sweep, n_random_sweep=args.n_random_sweep,
                   unembed_control=args.unembed_control, unembed_mode=args.unembed_mode if args.unembed_control else None,
                   fold_final_norm=not args.no_fold_final_norm, activation_norms_file=args.activation_norms,
                   decoding=decoding_settings(n_trials, max_new_tokens, 0.8),
                   elicitation_prompt_stage7=ELICITATION_PROMPT,
                   texts={"injection_train": INJECTION_TEXTS_TRAIN, "injection_heldout": INJECTION_TEXTS_HELDOUT,
                          "benign_train": BENIGN_TEXTS_TRAIN, "benign_heldout": BENIGN_TEXTS_HELDOUT},
                   **model_protocol(CFG, device))
    print(f"Saved stage7_results.json (+ protocol.json) to {out_dir}/")


if __name__ == "__main__":
    main()