"""
Stage 2 entry point: for every direction in the Stage 1 bank, clamp
it to alpha, elicit n_trials free-text reports, score them (exact-
match where a target concept is known), compute the three regression
covariates, and save everything for regression.py to consume.

Usage:
    python stage2_elicit.py --dry_run          # tiny real GPT2 architecture, no downloads
    python stage2_elicit.py                    # full run against your Stage 1 bank

Compute note: this runs n_trials generations PER DIRECTION plus a
jacobian_magnitude estimate over stage2_covariate_n_contexts contexts
per direction. For a full 835-direction bank at the default settings
that's 835 x 5 = ~4,175 generations plus 835 x 50 forward+backward
passes -- expect this to take a while; the --dry_run flag and the
config knobs (stage2_n_trials, stage2_covariate_n_contexts) are there
to let you shrink this for a first pass, same pattern as Stage 1.
"""

import argparse
import json
import os

import torch

# --- make stage1/ importable from here (see editing.py for rationale) ---
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage1"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage3"))

from config import CFG
from elicitation import elicit_reports
from scoring import exact_match_score, score_reports, infer_target_concept, export_for_blind_judging
from confounds import project_out_unembedding
from protocol import seed_everything, write_protocol, decoding_settings, model_protocol
from covariates import jacobian_magnitude, unembedding_alignment, build_jspace_basis, workspace_participation


def load_bank(bank_dir: str):
    vectors = torch.load(os.path.join(bank_dir, "vectors.pt"))
    metadata = json.load(open(os.path.join(bank_dir, "metadata.json")))
    return vectors, metadata


def run_stage2(model, tokenizer, vectors, metadata, contexts, device,
               unembed_control=False, fold_final_norm=True, classes=None):
    bank_for_basis = {k: {"class": v["class"], "layer": v["layer"], "vector": vectors[k]} for k, v in metadata.items()}
    layers = sorted({v["layer"] for v in metadata.values()})
    jspace_basis = {l: build_jspace_basis(bank_for_basis, l) for l in layers}

    results = []
    blind_records = []
    items = [(k, m) for k, m in metadata.items() if classes is None or m["class"] in classes]
    n = len(items)

    for i, (key, meta) in enumerate(items):
        direction = vectors[key]
        layer = meta["layer"]
        target_concept = infer_target_concept(key)

        reports = elicit_reports(
            model, tokenizer, layer, direction, CFG.stage2_alpha, CFG.stage2_prompt, device,
            n_trials=CFG.stage2_n_trials, max_new_tokens=CFG.stage2_max_new_tokens,
            temperature=CFG.stage2_temperature,
        )

        jmag = jacobian_magnitude(model, tokenizer, layer, direction, contexts, device)
        ualign = unembedding_alignment(model, direction, tokenizer=tokenizer, target_token=target_concept)
        wpart = workspace_participation(direction, jspace_basis[layer])

        trial_records = []
        for t, report in enumerate(reports):
            em = exact_match_score(report, target_concept)
            trial_records.append({"trial": t, "report": report, "exact_match": em})
            blind_records.append({"direction_key": key, "trial": t, "report": report})

        matches = [r["exact_match"] for r in trial_records if r["exact_match"] is not None]
        match_rate = sum(matches) / len(matches) if matches else None
        counts = score_reports(reports, target_concept)     # exact k/n behind exact_match_rate
        alpha = CFG.stage2_alpha                              # single alpha in this folder (matches the elicit call above)

        # Output-pathway control: same direction with the label tokens' unembedding readout projected out.
        # The projected set is class-agnostic (every CFG.jlens_target_tokens label) so J-lens, SAE, probe and
        # random directions are all stripped of the SAME logit-path component -- the comparison the reviewer
        # asked for (J-lens unembedding_alignment ~0.57 vs ~0.11-0.20 for controls).
        unembed_block = None
        if unembed_control:
            try:
                d_proj, diag = project_out_unembedding(direction, model, tokenizer,
                                                        list(CFG.jlens_target_tokens), fold_final_norm)
                reports_p = elicit_reports(
                    model, tokenizer, layer, d_proj, alpha, CFG.stage2_prompt, device,
                    n_trials=CFG.stage2_n_trials, max_new_tokens=CFG.stage2_max_new_tokens,
                    temperature=CFG.stage2_temperature,
                )
                counts_p = score_reports(reports_p, target_concept)
                unembed_block = {"reports": reports_p, "counts": counts_p,
                                 "exact_match_rate": counts_p["rate"], "diagnostics": diag}
            except ValueError as e:
                unembed_block = {"collapsed": True, "reason": str(e)}

        results.append({
            "direction_key": key,
            "class": meta["class"],
            "layer": layer,
            "target_concept": target_concept,
            "trials": trial_records,
            "exact_match_rate": match_rate,
            "counts": counts,
            "unembed_projected": unembed_block,
            "jacobian_magnitude": jmag,
            "unembedding_alignment": ualign,
            "workspace_participation": wpart,
        })

        rate_str = f"{match_rate:.2f}" if match_rate is not None else "n/a"
        print(f"  [{i + 1}/{n}] {key}: match_rate={rate_str} jmag={jmag:.4f} "
              f"ualign={ualign:.4f} wpart={wpart:.4f}", flush=True)

    return results, blind_records


def save_results(results, blind_records, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "elicitation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    answer_key_path = export_for_blind_judging(
        blind_records, os.path.join(out_dir, "blind_judging.csv")
    )
    print(f"Saved elicitation_results.json, blind_judging.csv (+ {answer_key_path}) to {out_dir}/")


def _build_dry_run_bank(model, sae_by_layer, tokenizer, device, contexts, counterfact):
    """Reuse Stage 1's own (tested) build_bank so the dry-run bank has the exact real structure."""
    from build_direction_bank import build_bank
    layers_backup = CFG.sae_layers
    tokens_backup = CFG.jlens_target_tokens
    CFG.sae_layers = layers_backup[:2]
    CFG.jlens_target_tokens = tokens_backup[:2]
    CFG.n_sae_features_per_layer = 3
    CFG.n_probe_attributes = 1
    CFG.n_random_directions_per_layer = 3
    CFG.n_random_in_span_per_layer = 3
    CFG.causal_eval_n_contexts = 3
    CFG.steer_alphas = [4.0]
    bank = build_bank(model, tokenizer, sae_by_layer, device, contexts, counterfact)
    CFG.sae_layers = layers_backup
    CFG.jlens_target_tokens = tokens_backup

    vectors = {k: v["vector"] for k, v in bank.items()}
    metadata = {k: {"layer": v["layer"], "class": v["class"]} for k, v in bank.items()}
    return vectors, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--bank_dir", default=CFG.output_dir)
    parser.add_argument("--out_dir", default=CFG.stage2_output_dir)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--unembed_control", action="store_true",
                         help="Also elicit on each direction with the unembedding readout of every "
                              "CFG.jlens_target_tokens label projected out (stored under 'unembed_projected').")
    parser.add_argument("--no_fold_final_norm", action="store_true")
    parser.add_argument("--classes", nargs="+", default=None,
                         help="Restrict to these bank classes (jspace sae probe random random_in_span...). "
                              "Default: all. Handy to halve cost on an 8B model: --classes jspace random.")
    args = parser.parse_args()

    device = CFG.device if torch.cuda.is_available() else "cpu"
    seed_everything(args.seed)

    if args.dry_run:
        from transformers import GPT2LMHeadModel, GPT2Config
        from tests.mock_model import MockSAE, MockGenerativeTokenizer, mock_contexts_and_counterfact

        gpt2_cfg = GPT2Config(n_layer=max(CFG.sae_layers[:2]) + 1, n_embd=32, n_head=2, vocab_size=200)
        model = GPT2LMHeadModel(gpt2_cfg).to(device)
        model.eval()
        tokenizer = MockGenerativeTokenizer(vocab_size=200)
        sae_by_layer = {l: MockSAE(32, d_sae=64) for l in CFG.sae_layers[:2]}
        contexts, counterfact = mock_contexts_and_counterfact(n_contexts=10, n_facts=5)

        vectors, metadata = _build_dry_run_bank(model, sae_by_layer, tokenizer, device, contexts, counterfact)
        CFG.stage2_n_trials = 2
        CFG.stage2_max_new_tokens = 5
        CFG.stage2_covariate_n_contexts = 3
        out_dir = "stage2_results_dryrun"
    else:
        from build_direction_bank import load_real_model
        from data import load_context_corpus

        model, tokenizer = load_real_model(CFG.model_name, device, CFG.dtype)
        vectors, metadata = load_bank(args.bank_dir)
        contexts = load_context_corpus(CFG.stage2_covariate_n_contexts)
        out_dir = args.out_dir

    results, blind_records = run_stage2(
        model, tokenizer, vectors, metadata, contexts[: CFG.stage2_covariate_n_contexts], device,
        unembed_control=args.unembed_control, fold_final_norm=not args.no_fold_final_norm, classes=args.classes,
    )
    save_results(results, blind_records, out_dir)
    write_protocol(out_dir, "stage2", seed=args.seed, alpha=CFG.stage2_alpha, classes=args.classes or "all",
                   decoding=decoding_settings(CFG.stage2_n_trials, CFG.stage2_max_new_tokens, CFG.stage2_temperature,
                                             use_chat_template=CFG.use_chat_template),
                   unembed_control=args.unembed_control, fold_final_norm=not args.no_fold_final_norm,
                   n_directions=len(results), **model_protocol(CFG, device))


if __name__ == "__main__":
    main()
