# Protocol, definitions and camera-ready run plan

Everything a reader needs to reconstruct a number in the paper from the code. Each stage also writes a
`protocol.json` next to its results (decoding settings, prompts/texts, seed, alpha, model, code version).

## 1. Scoring and counts

* A generation is **scored 1** if the target label appears as a whole word (case-insensitive). It is **scored 0**
  if it is a degenerate repetition loop (`is_degenerate_repetition`) or names two or more distinct candidate labels
  (`mentions_multiple_labels`, "shotgun listing"). Code: `stage2/scoring.py`.
* Every rate in a results file now has its integers beside it (`counts: {k, n, n_total, n_degenerate, n_multi_label}`).
  `n` is the number of scorable generations; **a reported percentage must be k/n for that n**. If it is not, something
  was averaged across cells of unequal size — say so in the table caption or report the pooled k/n instead.
* Scoring also recovers a hit when the steering hook glues a capitalised label onto the previous word with no space
  ("aboutFrance"); lowercase labels stay strict whole-word. Repetition detection includes glued-substring loops
  ("catcatcat"). Numbers produced before this scoring change are not comparable to numbers produced after it: re-score
  or rerun everything you report with the same code version (`protocol.json` records the commit).
* Alpha is **per layer** (`CFG.stage2_alphas`, e.g. 1.5 at layers 3/7, 6.0-6.5 at layers 19-27). It is stored in
  Stage 4's `split.json` and reused by Stages 5 and 7; always report it as alpha/‖h‖ (see below), never as one number.
* Stage 4 reshuffles the training examples every epoch with the run's `--seed`, so a seed changes both LoRA
  initialisation and example order.
* Rates in the Stage 6 tables are means of per-entry rates (entry = pair x layer), shown with the pooled k/n.
* Decoding: sampling, temperature 0.8, `max_new_tokens` 10 (Stages 3-5, 7) / 30 (Stage 2), chat template applied.
  Generations per item = `--n_trials` (default 3; Stage 2: `stage2_n_trials` = 5), **per input text**.

## 2. Definitions

| Term | Definition (code) |
|---|---|
| **Clamp** | `h' = h - <h,d>d + alpha*d` at every position of one layer (`stage2/editing.py`). `d` is unit norm. |
| **alpha / ‖h‖** | `|alpha|` divided by the mean residual-stream norm at that layer (`stage1/activation_scale.py` → `--activation_norms`). Logged as `alpha_over_resid_norm`. Report this, not bare alpha: the same alpha is a very different intervention at layer 3 and layer 27. |
| **Conflict cells** | (direction=X, input=Y) and (direction=Y, input=X). Direction state is set by the sign of alpha on the bipolar direction `d_XY`. |
| **DFI** | `½[P(report=X ∣ d=X,i=Y) − P(report=Y ∣ d=X,i=Y) + P(report=Y ∣ d=Y,i=X) − P(report=X ∣ d=Y,i=X)]`. Algebraically equal to `Acc_vs_direction − Acc_vs_input` averaged over the two cells (tested in `tests/`). A convenience summary of two numbers that are always reported next to it, not a separate method. |
| **DFI interval (per entry)** | Percentile bootstrap resampling *generations within each conflict cell* (`dfi_bootstrap`). Sampling noise only. |
| **DFI interval (Stage 6)** | Bootstrap resampling *entries* (pair x layer). Which cells were sampled, not training variance. |
| **DFI interval (seeds)** | `stage6/aggregate_seeds.py`: per-seed values, SD, range, hierarchical bootstrap (seeds → entries). Training variance. With 3 seeds, report the per-seed values. |
| **Baseline** (ablation) | Input present, direction clamped to ±alpha. |
| **Direction ablation** | Input present, direction clamped to **alpha = 0**: the hook is *active*, the component along `d` is projected out and set to zero. |
| **Input ablation** | The input sentence/passage is replaced by the **bare elicitation prompt**; the direction stays clamped at ±alpha. Gets `n_trials × n_texts` generations so its n matches the other cells. |
| **No-hook control** | Input present, hook **absent** (`apply_hook=False`): no projection, no clamp. Differs from direction ablation: a model reacting to "something was edited" regardless of value would pass the alpha=0 check but not this one. |
| **Unseen directions** | Held-out *pairs* x training layers. |
| **Unseen layers** (results key `unseen_positions`) | Training pairs x held-out *layers*. The clamp layer is varied; token position never is. |
| **Unseen coefficients** | Training pairs/layers at alphas not used in training. |
| **unembed_projected** | The direction with the label tokens' output readout projected out before clamping (§4). Third condition in Stage 3 beside `raw` and `orthogonalized`. |
| **`workspace_participation`** | For J-lens directions this is 1.0 by construction (the basis is built from those same directions) — definitional, not a finding. Only the other classes' values are informative. |
| **Jacobian alignment** | Directional derivative of the target label's log-prob along `d`. Reported as a diagnostic; not used to support a claim on its own. |

Input texts: `stage3/input_provider.py` (`template`: 4 label-bearing carrier sentences, no held-out phrasing;
`varied`: 4 train + 2 held-out passages per concept that never name it). Stage 7 texts: `stage7/covert_cognition.py`.
The exact elicitation prompt and every training/eval text are written to `protocol.json`.

## 3. Camera-ready run plan (in this order)

Dependencies: step 2 decides the framing of the paper; step 4 multiplies cost, so it goes last.

```bash
# 0. natural activation scale (once)
cd stage1 && python activation_scale.py            # -> activation_norms.json   (use --activation_norms below)

# 1. OUTPUT-PATHWAY CONTROL on the pre-training analyses (Stage 2 + Stage 3), Llama-3.1-8B
cd ../stage2 && python stage2_elicit.py --bank_dir ../stage1/direction_bank --unembed_control --seed 0
cd ../stage3 && python stage3_run.py --bank_dir ../stage1/direction_bank --unembed_control --efficacy_check \
                  --activation_norms ../stage1/activation_norms.json --seed 0
#    stricter variant (project out every label's readout, not just the pair's):  --unembed_scope all_labels

# 2. ... and on the capstone (layer 19)
cd ../stage7
python stage7_run.py --bank_dir ../stage1/direction_bank --layer 19 --unembed_control --unembed_mode train_and_eval --efficacy_check --out_dir stage7_unembed_train_eval
python stage7_run.py --bank_dir ../stage1/direction_bank --layer 19 --unembed_control --unembed_mode eval_only        --out_dir stage7_unembed_eval_only
python stage7_run.py --bank_dir ../stage1/direction_bank --layer 19                                                   --out_dir stage7_raw   # matched baseline, same seed

# 3. "cannot read" vs "was never forced to read": generic concepts with the capstone recipe + random-direction baseline
cd ../stage4
python stage4_train.py --input_provider varied --direction_source jspace --out_dir adapters/varied_jspace
python stage4_train.py --input_provider varied --direction_source random --out_dir adapters/varied_random
python stage4_train.py --input_provider template --direction_source random --out_dir adapters/template_random   # optional
cd ../stage5
python stage5_run.py --adapter_dir ../stage4/adapters/varied_jspace --out_dir stage5_varied_jspace
python stage5_run.py --adapter_dir ../stage4/adapters/varied_random --out_dir stage5_varied_random

# 4. seeds (3), per configuration you will report; then aggregate
for s in 0 1 2; do
  python ../stage4/stage4_train.py --seed $s --input_provider varied --out_dir ../stage4/adapters/varied_jspace_s$s
  python stage5_run.py --seed $s --adapter_dir ../stage4/adapters/varied_jspace_s$s --out_dir stage5_varied_jspace_s$s
done
cd ../stage6 && python aggregate_seeds.py --kind stage5 ../stage5/stage5_varied_jspace_s{0,1,2}/stage5_results.json
```

`stage6_report.py --stage3_condition unembed_projected` gives the Stage 6 table for the projected direction.

## 4. Output-pathway control: what it does and how to read it

**Why.** J-lens directions are gradients of a label's log-probability, so they carry a large component along that
label's unembedding row (`unembedding_alignment` ≈ 0.57 vs ≈ 0.11–0.20 for the control classes). A clamped J-lens
direction could raise the label's logit directly — "verbalization" with no read-out of an internal state. The existing
logit-lens check only *compares* against that route; this control *removes* it.

**What is removed** (`stage3/confounds.py::project_out_unembedding`): the span of, for every label token,
the raw unembedding row `W_U[t]` and the final-norm-scaled row `γ ⊙ W_U[t]` (the pre-norm direction that actually moves
logit t), each for the bare and the space-prefixed token. The result is re-normalised, so the same alpha means the same
clamp magnitude. Diagnostics (`unembed_diagnostics`) record cos(original, projected) and the alignment with each label's
row before/after (≈ 0 after).

**Reading the outcomes.**

* Direction-following **survives** (DFI/P(report=direction) at layers 19/23/27 and the capstone ≈ unchanged) *and*
  `steering_efficacy.projected` is clearly above `random_floor`: the positive results are not coming from the direct
  logit path through those labels. The positive half of the paper gets stronger.
* It **vanishes**: the positive results were largely that path. Report it as a result; the paper becomes a negative-result
  methods paper with a sharper explanation.
* **Check `steering_efficacy` first.** If the projected direction's steering KL falls to the random floor, a null result
  on it says only that the direction was inert once the unembedding part was removed — not that introspection is absent.
* If the direction **collapses** on projection (`collapsed: true`), it was almost entirely its logit-path component.

**Limits to state in the paper.** It removes the *direct* path through the chosen label tokens' readout, not
indirect routes through later layers, and not other tokens' readout (synonyms, multi-token openings) unless
`--unembed_scope all_labels`. In Stage 7 `train_and_eval` answers "can a reporter be trained to read the direction once
the logit path is gone?"; `eval_only` answers "does the already-trained reporter's direction-following depend on that
component?" — they are different questions, report both.


## Stage 7 hardening (capstone)
- Held-out set is now 5 phrasings per class (was 1). Decorrelation and held-out ablations run `--n_trials` generations on EVERY
  phrasing and pool them; per-phrasing reports are stored under `decorrelation.per_phrasing`. n per cell = 5 x n_trials.
  Results from the earlier 1-phrasing runs are not comparable and should not be pooled with these.
- `--direction_source random`: seeded random unit direction (same dim, +alpha=injection, -alpha=safe) used for training AND evaluation.
  Combine with `--unembed_control` to project it the same way as the J-lens direction.
- `--random_swap` (jspace runs): after training on the J-lens direction, repeat the decorrelation test with a random unit
  direction clamped instead, no retraining -> `decorrelation_random_swap` in stage7_results.json.

## Note: reports are written after a leading <|begin_of_text|> (Llama)
Stage 4/7 training targets are tokenized with `add_special_tokens=True`, so adapters learn to emit BOS first and the label second.
`skip_special_tokens=True` in decoding hides this, so sampled reports are unaffected. Any analysis that reads next-token logits
must read them AFTER the BOS (`stage7/mechanism.py::_label_position_logits`); the first `--alpha_sweep` version did not.
