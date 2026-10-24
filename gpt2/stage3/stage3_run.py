"""
Stage 3 entry point: for each (X, Y) concept pair and each layer,
build the bipolar J-space direction, run the two conflict cells
(confound (a) handled structurally by decorrelation.build_cell_prompt
-- injection-then-report, no prior model-generated text to leak
from), then repeat with the direction Gram-Schmidt-orthogonalized
against input embeddings + top co-firing SAE features (confound (b)),
and compute a logit-lens baseline report (confound (c)) for
comparison. Saves everything to stage3_results.json.

Usage:
    python stage3_run.py --dry_run
    python stage3_run.py --bank_dir ../stage1/direction_bank

IMPORTANT CAVEAT carried over from Stage 2: H1's calibration work
found that free-text generation on GPT-2-small (and Qwen2-0.5B-
Instruct) doesn't reliably produce reports that literally name a
clamped concept, across an extensive alpha/prompt/model sweep. Since
this stage's direction-following index depends on the SAME exact-
match-on-generated-text mechanism, it's very possible DFI comes back
near-zero/uninformative for the same underlying reason, not because
H3 is false. Worth knowing going in -- run it and look at the actual
report text (same way we diagnosed H1) before concluding anything
about the decorrelation itself.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage1"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage2"))

from config import CFG
from build_direction_bank import load_real_model

from elicitation import elicit_reports
from decorrelation import build_bipolar_direction, build_cell_prompt, direction_following_index, dfi_bootstrap
from confounds import (gram_schmidt_orthogonalize, get_input_embedding_vectors, get_top_cofiring_sae_features,
                       logit_lens_report, project_out_unembedding, steering_efficacy_comparison)
from scoring import SCORING_RULE
from protocol import (seed_everything, write_protocol, decoding_settings, model_protocol,
                      load_activation_norms, alpha_over_norm)


DEFAULT_PAIRS = [("Paris", "Rome"), ("dog", "happy"), ("sad", "happy"), ("France", "Italy")]


def load_bank(bank_dir: str):
    vectors = torch.load(os.path.join(bank_dir, "vectors.pt"))
    metadata = json.load(open(os.path.join(bank_dir, "metadata.json")))
    return vectors, metadata


def run_pair(model, tokenizer, vectors, metadata, layer, x, y, alpha, elicitation_prompt, n_trials, max_new_tokens, device,
             unembed_control=False, unembed_tokens=None, fold_final_norm=True, efficacy_contexts=None,
             activation_norms=None, n_boot=2000, boot_seed=0):
    key_x = f"jspace::{layer}::{x}"
    key_y = f"jspace::{layer}::{y}"
    if key_x not in vectors or key_y not in vectors:
        return None  # this pair/layer combo isn't in the bank -- skip rather than error

    vec_x, vec_y = vectors[key_x], vectors[key_y]
    d_xy = build_bipolar_direction(vec_x, vec_y)

    sae_vectors = {k: v for k, v in vectors.items() if metadata[k]["class"] == "sae" and metadata[k]["layer"] == layer}
    input_embeds = get_input_embedding_vectors(model, tokenizer, [x, y])
    top_cofiring = get_top_cofiring_sae_features(d_xy, sae_vectors, k=5) if sae_vectors else []
    d_orth = gram_schmidt_orthogonalize(d_xy, input_embeds + top_cofiring)

    result = {"layer": layer, "x": x, "y": y, "alpha": alpha,
              "alpha_over_resid_norm": alpha_over_norm(alpha, layer, activation_norms or {})}

    conditions = [("raw", d_xy), ("orthogonalized", d_orth)]

    # Optional output-pathway control: same experiment on the direction with the label tokens' unembedding
    # readout projected out (third condition 'unembed_projected'). Off by default.
    if unembed_control:
        try:
            d_unemb, diag = project_out_unembedding(d_xy, model, tokenizer, unembed_tokens or [x, y], fold_final_norm)
            if efficacy_contexts:
                diag["steering_efficacy"] = steering_efficacy_comparison(
                    model, tokenizer, layer, d_xy, d_unemb, efficacy_contexts, list(CFG.steer_alphas), device)
            conditions.append(("unembed_projected", d_unemb))
            result["unembed_diagnostics"] = diag
        except ValueError as e:
            result["unembed_diagnostics"] = {"collapsed": True, "reason": str(e)}

    for tag, direction in conditions:
        prompt_input_y = build_cell_prompt(y, elicitation_prompt)  # direction=X, input=Y
        prompt_input_x = build_cell_prompt(x, elicitation_prompt)  # direction=Y, input=X

        reports_dirX_inputY = elicit_reports(model, tokenizer, layer, direction, alpha, prompt_input_y, device,
                                              n_trials=n_trials, max_new_tokens=max_new_tokens)
        reports_dirY_inputX = elicit_reports(model, tokenizer, layer, direction, -alpha, prompt_input_x, device,
                                              n_trials=n_trials, max_new_tokens=max_new_tokens)

        dfi = direction_following_index(reports_dirX_inputY, reports_dirY_inputX, x, y)
        result[tag] = {
            "reports_dirX_inputY": reports_dirX_inputY,
            "reports_dirY_inputX": reports_dirY_inputX,
            **dfi,
            "dfi_bootstrap": dfi_bootstrap(reports_dirX_inputY, reports_dirY_inputX, x, y,
                                            n_boot=n_boot, seed=boot_seed),
            "logit_lens_top_tokens": logit_lens_report(model, tokenizer, direction, k=3),
        }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--bank_dir", default="../stage1/direction_bank")
    parser.add_argument("--out_dir", default="stage3_results")
    parser.add_argument("--alpha", type=float, default=None, help="Defaults to CFG.stage2_alpha.")
    parser.add_argument("--n_trials", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--pairs", type=str, default=None,
                         help='Override DEFAULT_PAIRS, e.g. \'[["Paris","Rome"],["dog","happy"]]\'')
    parser.add_argument("--seed", type=int, default=0, help="Seeds torch/python RNGs (generation sampling).")
    parser.add_argument("--n_boot", type=int, default=2000, help="Bootstrap resamples for per-entry DFI intervals.")
    parser.add_argument("--unembed_control", action="store_true",
                         help="Also run every cell on the direction with the label tokens' unembedding readout "
                              "projected out (third condition, 'unembed_projected').")
    parser.add_argument("--unembed_scope", choices=["pair", "all_labels"], default="pair",
                         help="Which tokens' readout to project out: just this pair's labels, or every "
                              "CFG.jlens_target_tokens label (stricter).")
    parser.add_argument("--no_fold_final_norm", action="store_true",
                         help="Project out only the raw unembedding rows, not also gamma-scaled rows.")
    parser.add_argument("--efficacy_check", action="store_true",
                         help="With --unembed_control: also measure steering KL of original vs projected vs random "
                              "direction, to show the projected direction still does something.")
    parser.add_argument("--efficacy_n_contexts", type=int, default=10)
    parser.add_argument("--activation_norms", default=None,
                         help="activation_norms.json from stage1/activation_scale.py; logs alpha/||h|| per layer.")
    args = parser.parse_args()

    device = CFG.device if torch.cuda.is_available() else "cpu"
    seed_everything(args.seed)
    alpha = args.alpha if args.alpha is not None else CFG.stage2_alpha
    pairs = json.loads(args.pairs) if args.pairs else DEFAULT_PAIRS
    pairs = [tuple(p) for p in pairs]

    if args.dry_run:
        from transformers import GPT2LMHeadModel, GPT2Config
        from tests.mock_model import MockGenerativeTokenizer

        gpt2_cfg = GPT2Config(n_layer=5, n_embd=32, n_head=2, vocab_size=200)
        model = GPT2LMHeadModel(gpt2_cfg).to(device)
        model.eval()
        tokenizer = MockGenerativeTokenizer(vocab_size=200)

        d = 32
        vectors = {}
        metadata = {}
        for layer in [2, 4]:
            for tok in ["Paris", "Rome", "dog", "happy", "sad", "France", "Italy"]:
                vectors[f"jspace::{layer}::{tok}"] = torch.randn(d)
                metadata[f"jspace::{layer}::{tok}"] = {"layer": layer, "class": "jspace"}
            for i in range(10):
                vectors[f"sae::{layer}::{i}"] = torch.randn(d)
                metadata[f"sae::{layer}::{i}"] = {"layer": layer, "class": "sae"}

        pairs = pairs[:2]
        out_dir = "stage3_results_dryrun"
        n_trials, max_new_tokens = 2, 6
    else:
        model, tokenizer = load_real_model(CFG.model_name, device, CFG.dtype)
        vectors, metadata = load_bank(args.bank_dir)
        out_dir = args.out_dir
        n_trials, max_new_tokens = args.n_trials, args.max_new_tokens

    efficacy_contexts = None
    if args.unembed_control and args.efficacy_check:
        if args.dry_run:
            efficacy_contexts = [f"Sample background context number {i}, long enough to tokenize." for i in range(3)]
        else:
            from data import load_context_corpus
            efficacy_contexts = load_context_corpus(args.efficacy_n_contexts)
    unembed_tokens = list(CFG.jlens_target_tokens) if args.unembed_scope == "all_labels" else None
    activation_norms = load_activation_norms(args.activation_norms)

    layers = sorted({m["layer"] for m in metadata.values() if m["class"] == "jspace"})
    results = []
    total = len(pairs) * len(layers)
    i = 0
    for (x, y) in pairs:
        for layer in layers:
            i += 1
            r = run_pair(model, tokenizer, vectors, metadata, layer, x, y, alpha,
                         CFG.stage2_prompt, n_trials, max_new_tokens, device,
                         unembed_control=args.unembed_control, unembed_tokens=unembed_tokens,
                         fold_final_norm=not args.no_fold_final_norm, efficacy_contexts=efficacy_contexts,
                         activation_norms=activation_norms, n_boot=args.n_boot, boot_seed=args.seed)
            if r is None:
                print(f"  [{i}/{total}] ({x},{y}) layer={layer}: skipped (not in bank)")
                continue
            results.append(r)
            raw_dfi = r["raw"]["dfi_overall"]
            orth_dfi = r["orthogonalized"]["dfi_overall"]
            unemb = r.get("unembed_projected", {}).get("dfi_overall") if args.unembed_control else "-"
            print(f"  [{i}/{total}] ({x},{y}) layer={layer} alpha={alpha}: "
                  f"raw_dfi={raw_dfi} orth_dfi={orth_dfi} unembed_dfi={unemb} "
                  f"logit_lens={r['raw']['logit_lens_top_tokens']}", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "stage3_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    write_protocol(out_dir, "stage3", seed=args.seed, alpha=alpha, layers=layers,
                   pairs=[list(p) for p in pairs], bootstrap_resamples=args.n_boot,
                   decoding=decoding_settings(n_trials, max_new_tokens, 0.8, use_chat_template=False, scoring=SCORING_RULE),
                   input_template='A note nearby reads: "{concept}".',
                   unembed_control=args.unembed_control, unembed_scope=args.unembed_scope,
                   fold_final_norm=not args.no_fold_final_norm, activation_norms_file=args.activation_norms,
                   **model_protocol(CFG, device))
    print(f"Saved stage3_results.json (+ protocol.json) to {out_dir}/")


if __name__ == "__main__":
    main()
