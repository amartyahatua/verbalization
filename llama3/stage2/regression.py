"""
Fit the Stage 2 mixed-effects regression:

    report_match ~ jacobian_magnitude + unembedding_alignment + workspace_participation,
    with direction identity as a random effect (random intercept per direction).

Uses per-trial binary match outcomes (not pre-aggregated per-direction
rates) as the response -- that's the statistically appropriate unit
of analysis when a random intercept per direction is meant to soak up
repeated-trial correlation within a direction.

If you've run blind human judging (scoring.export_for_blind_judging +
merge_judge_scores), pass --judged_csv/--answer_key to use those
scores in place of exact-match wherever a human score exists.
"""

import argparse
import json
from typing import Dict, Optional, Tuple

import pandas as pd
import statsmodels.formula.api as smf


def build_regression_dataframe(
    results_path: str,
    judged_scores: Optional[Dict[Tuple[str, int], int]] = None,
) -> pd.DataFrame:
    results = json.load(open(results_path))
    rows = []
    for r in results:
        for trial in r["trials"]:
            match = trial["exact_match"]
            key = (r["direction_key"], trial["trial"])
            if judged_scores is not None and key in judged_scores:
                match = judged_scores[key]  # human judgment overrides exact-match when available
            if match is None:
                continue  # no scorable label for this trial (no exact-match target, no human score yet)
            rows.append({
                "direction_key": r["direction_key"],
                "class": r["class"],
                "report_match": match,
                "jacobian_magnitude": r["jacobian_magnitude"],
                "unembedding_alignment": r["unembedding_alignment"],
                "workspace_participation": r["workspace_participation"],
            })
    return pd.DataFrame(rows)


def fit_mixed_effects(df: pd.DataFrame):
    """
    Linear mixed model (report_match treated as continuous 0/1, i.e. a
    "linear probability model" with random intercepts) -- simple and
    robust to fit. See fit_binomial_glmm() below for a stricter
    binomial alternative if this one's residuals look off.
    """
    model = smf.mixedlm(
        "report_match ~ jacobian_magnitude + unembedding_alignment + workspace_participation",
        data=df,
        groups=df["direction_key"],
    )
    return model.fit()


def fit_binomial_glmm(df: pd.DataFrame):
    """Stricter alternative: a proper binomial mixed model via variational Bayes."""
    import numpy as np
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    exog = df[["jacobian_magnitude", "unembedding_alignment", "workspace_participation"]].copy()
    exog.insert(0, "Intercept", 1.0)
    exog_vc = pd.get_dummies(df["direction_key"]).astype(float)  # one random-effect column per direction
    ident = np.zeros(exog_vc.shape[1], dtype=int)
    model = BinomialBayesMixedGLM(df["report_match"].values, exog.values, exog_vc.values, ident)
    return model.fit_vb()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="stage2_results/elicitation_results.json")
    parser.add_argument("--judged_csv", default=None)
    parser.add_argument("--answer_key", default=None)
    parser.add_argument("--binomial", action="store_true", help="Use the stricter binomial GLMM instead of the linear mixed model.")
    args = parser.parse_args()

    judged_scores = None
    if args.judged_csv and args.answer_key:
        from scoring import merge_judge_scores
        judged_scores = merge_judge_scores(args.judged_csv, args.answer_key)

    df = build_regression_dataframe(args.results, judged_scores)
    print(f"Regression dataset: {len(df)} scorable trials across {df['direction_key'].nunique()} directions")
    if df.empty:
        print("No scorable trials -- run blind judging and pass --judged_csv/--answer_key, "
              "or check that some directions have an inferable target_concept for exact-match.")
        return

    if args.binomial:
        result = fit_binomial_glmm(df)
    else:
        result = fit_mixed_effects(df)
    print(result.summary())


if __name__ == "__main__":
    main()
