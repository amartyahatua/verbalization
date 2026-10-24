# When Verbalization Is Not Introspection

Code for **"When Verbalization Is Not Introspection: Causal Tests for Internal-State Reporting"** (NeurIPS 2026 workshop submission; the PDF is in [`paper/`](paper/Verbalization_NeurIPS_2026.pdf)).

We ask whether a language model can be trained to faithfully report an internal state that we set by hand. We edit a concept direction in the residual stream, train a small LoRA adapter to say which concept was edited, and then check whether the report follows the **edited direction** or the **prompt text**.

## The idea in one minute

During ordinary training the edit, the prompt and the correct answer all agree, so a model can score 100% by reading the prompt and ignoring the edit. We separate the two:

|  | Prompt says X | Prompt says Y |
|---|---|---|
| **Direction set to X** | agree | **conflict** |
| **Direction set to Y** | **conflict** | agree |

The adapter is trained only on the agreeing cells. At test time we look only at the conflict cells. The **direction-following index (DFI)** is +1 if the report follows the direction, −1 if it follows the prompt, and 0 for a constant answer.

We also run causal checks: remove the direction (direction ablation), remove the passage (input ablation), and skip the intervention entirely (no-hook control).

## What we found

- **Generic concepts (Paris/Rome, dog/cat, happy/sad).** All three models (GPT-2 124M, Qwen2-0.5B-Instruct, Llama-3.1-8B-Instruct) reach 100% accuracy on agreeing examples. In the conflict cells they follow the prompt: DFI = −1.0 for all three, and removing the direction never changes the report. High reporting accuracy alone is not evidence of introspection.
- **Unseen concepts collapse (Llama).** A held-out concept pair (France/Italy) is reported as the nearest trained pair (Paris/Rome).
- **Prompt-injection capstone.** With several different training phrasings per class, the report follows the edited direction (DFI = +1.0). The camera-ready controls (Llama-3.1-8B-Instruct, layer 19, 3 seeds, 150 samples per cell) add two caveats:
  - the result survives removing the label tokens' output-readout directions from the edit;
  - a **random** direction trained the same way does equally well, so the capstone shows that varied training text escapes the prompt shortcut, **not** that the J-lens direction carries safety-relevant content.

See [`llama3/PROTOCOL.md`](llama3/PROTOCOL.md) for definitions, scoring rules, and how to read each control.

> **Status of the committed results.** Some `*_results/` folders hold earlier runs with fewer samples per cell. The camera-ready reruns (exact counts, 3 seeds, bootstrap intervals, output-pathway control) are being added stage by stage; each run writes a `protocol.json` with its settings and code version, so check that file before comparing numbers across runs.

## Repository layout

```
paper/                       the submission PDF
gpt2/                        GPT-2 small (124M)             stages 1-7
qwen2/                       Qwen2-0.5B-Instruct            stages 1-7
llama3/                      Llama-3.1-8B-Instruct          stages 1-7, PROTOCOL.md, tests
01_verbalizable-access/      early standalone copy of the stage 1 direction-bank code (kept for history)
requirements.txt
```

Each model folder has the same seven stages and is run from inside the stage directory:

| Stage | Folder | What it does |
|---|---|---|
| 1 | `stage1/` | Builds the **direction bank**: J-lens, SAE, probe, random, and random-in-SAE-span directions, each tagged with a steering-efficacy score. Also `activation_scale.py` for the residual-stream norm per layer. |
| 2 | `stage2/` | **Native verbalizability**: clamps each direction, asks the model to report, scores the free text, and fits the covariate regression. `alpha_sweep.py` helps choose the clamp strength. |
| 3 | `stage3/` | **Decorrelation before training**: the conflict cells on the untrained model, plus controls (orthogonalized direction, logit-lens baseline, unembedding-projected direction). |
| 4 | `stage4/` | **LoRA training** on agreeing examples only. Records which pairs and layers are held out in `split.json`. |
| 5 | `stage5/` | **The decisive test**: unseen directions, layers and coefficients; decorrelated conflict cells; direction-ablation and input-ablation checks. |
| 6 | `stage6/` | Tables and statistics: metrics, markdown summary, and `aggregate_seeds.py` for seed-level aggregation. |
| 7 | `stage7/` | **Prompt-injection capstone** (injection vs. safe) with a fresh adapter trained on varied phrasings and tested on held-out ones. |

`llama3/` is the most complete version. It has the camera-ready controls: exact `k/n` counts in every results file, bootstrap intervals, the output-pathway (unembedding) control, random-direction twins, a dose-response sweep for Stage 7, `protocol.json` sidecars, and unit tests. `gpt2/` and `qwen2/` have the same unembedding-control code, but not yet the Stage 7 random-direction and sweep options.

## Setup

```bash
git clone https://github.com/amartyahatua/verbalization
cd verbalization
pip install -r requirements.txt
```

- **GPU:** the Llama-3.1-8B runs were done on a single A100 80GB. GPT-2 and Qwen2 are small enough for a modest GPU.
- **Llama access:** `meta-llama/Llama-3.1-8B-Instruct` is gated. Accept the license on the Hugging Face model page, confirm it under *Settings → Gated repos*, then run `huggingface-cli login` (or set `HF_TOKEN`).
- **J-lens package:** the stage code imports the `jlens` package from pip. Our own extraction code is `stage1/jspace_lens.py` (a best-effort reimplementation from the public description, not the original authors' code), so don't name a local file `jlens.py`.

## Quick start

Every stage has a `--dry_run` mode that uses a tiny model, needs no downloads and no GPU, and checks that the pipeline runs end to end.

```bash
cd llama3/stage1  && python build_direction_bank.py --dry_run
cd ../stage2      && python stage2_elicit.py --dry_run
cd ../stage3      && python stage3_run.py --dry_run
cd ../stage4      && python stage4_train.py --dry_run
cd ../stage5      && python stage5_run.py --dry_run
cd ../stage7      && python stage7_run.py --dry_run
```

Unit tests (from `llama3/`):

```bash
cd llama3 && pytest tests/
```

## Full run (Llama-3.1-8B-Instruct)

Run from `llama3/`. The step-by-step commands, with notes, are in [`llama3/running_command`](llama3/running_command); the camera-ready plan is in [`llama3/PROTOCOL.md`](llama3/PROTOCOL.md) §3.

```bash
# 0. choose the clamp strength per layer, and measure the residual-stream scale
cd stage2 && python alpha_sweep.py --layer 11 --alphas 4 8 12 16 24 32 48
cd ../stage1 && python activation_scale.py --layers 3 7 11 15 19 23 27 --n_contexts 50

# 1. direction bank
python build_direction_bank.py

# 2. native verbalizability, then decorrelation before training (with the output-pathway control)
cd ../stage2 && python stage2_elicit.py --bank_dir ../stage1/direction_bank
cd ../stage3 && python stage3_run.py --bank_dir ../stage1/direction_bank --unembed_control \
                  --activation_norms ../stage1/activation_norms.json --seed 0

# 3. train the adapter, then the decisive test
cd ../stage4 && python stage4_train.py --bank_dir ../stage1/direction_bank --seed 0
cd ../stage5 && python stage5_run.py --adapter_dir ../stage4/stage4_adapter --bank_dir ../stage1/direction_bank \
                  --activation_norms ../stage1/activation_norms.json --seed 0

# 4. report, and aggregate over seeds
cd ../stage6 && python stage6_report.py --help
python aggregate_seeds.py --kind stage5 <seed0>/stage5_results.json <seed1>/stage5_results.json <seed2>/stage5_results.json
```

### Stage 7 (capstone) options

```bash
cd llama3/stage7

# J-lens direction, output-pathway control, random-swap check, dose-response sweep
python stage7_run.py --bank_dir ../stage1/direction_bank --layer 19 --seed 0 \
    --n_trials 30 --unembed_control --unembed_mode train_and_eval \
    --random_swap --alpha_sweep --n_random_sweep 30 \
    --activation_norms ../stage1/activation_norms.json --out_dir stage7_train_jspace_s0

# The matched control: the same pipeline with a random unit direction
python stage7_run.py --bank_dir ../stage1/direction_bank --layer 19 --seed 0 \
    --n_trials 30 --unembed_control --unembed_mode train_and_eval \
    --direction_source random --alpha_sweep --n_random_sweep 30 --out_dir stage7_train_random_s0
```

| Flag | Meaning |
|---|---|
| `--unembed_control` | Project the label tokens' output-readout directions out of the edit. |
| `--unembed_mode train_and_eval` / `eval_only` | Train and test on the projected direction, or train on the raw direction and only test on the projected one. |
| `--direction_source random` | Use a seeded random unit direction for both training and testing. |
| `--random_swap` | After training on the J-lens direction, rerun the conflict test with a random direction, no retraining. |
| `--alpha_sweep`, `--n_random_sweep N` | Clamp at −1.5× to +1.5× of α and compare the trained direction's slope with N random directions. |
| `--n_trials` | Generations per phrasing. Held-out sets have 5 phrasings per class, so n per cell is 5 × `n_trials`. |

## What each run writes

- `*_results.json`: every rate comes with its integers (`counts: {k, n, n_total, n_degenerate, n_multi_label}`), and DFI comes with a bootstrap interval. Raw generations are stored, so numbers can be recounted without rerunning.
- `protocol.json`: model, layer, α (and α/‖h‖ if `--activation_norms` is given), seed, decoding settings, every prompt and training text, and the code version.

## Known limits

- The J-lens directions come from our own approximate reimplementation, not the original authors' code.
- The capstone uses one layer (19), one α (6.0) and one binary label pair, with three seeds per direction type.
- Because a random direction passes the same capstone test, the test shows that a report is **caused by** the edit, not that the edit is being **interpreted**.
- Llama reports are written after a leading `<|begin_of_text|>` token. Any analysis of next-token logits must read after it (see `llama3/PROTOCOL.md`).
- The ablation interventions may themselves shift the model's activations; the no-hook control is there to expose that.
