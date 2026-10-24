"""
Scoring for Stage 2: automated exact-match scoring, plus utilities to
export elicited reports for blinded human judging and merge the
returned judgments back in.

Exact-match is a coarse, high-precision/low-recall automated signal
(catches only literal mentions of the target word) -- it's a cheap
check, not a replacement for human judging, which is why both are
called for in the plan. Both are tracked at trial-level granularity
(direction_key, trial_index), not aggregated per-direction, so a
human judge's per-report call doesn't silently overwrite a sibling
trial's score for the same direction.
"""

import csv
import json
import random
import re
from typing import List, Dict, Optional, Tuple


def is_degenerate_repetition(text: str, min_repeats: int = 3) -> bool:
    """
    True if `text` looks like a degenerate repetition loop rather than
    a genuine report -- caught via three related patterns, all of
    which trivially satisfy a naive whole-word target-concept match
    without being a coherent report:
      (a) the same whitespace-delimited word repeating min_repeats+
          times consecutively (e.g. "What What What What"),
      (b) one word dominating more than half the text even if not
          strictly consecutive,
      (c) a short substring repeating min_repeats+ times with NO
          separating whitespace at all (e.g. "catcatcatcatcat",
          "dogdogdogdogdogdog"). This is the pattern actually observed
          at high steering alpha (see alpha_sweep.py runs on layer 19/
          23/27 during the 2026-08 calibration pass): sampling gets
          stuck re-emitting the same subword token with no space
          between repeats, so the ENTIRE run parses as one "word"
          under a whitespace split -- (a) and (b) alone silently miss
          it, which is exactly how alpha=12's degenerate output could
          have inflated exact_match_rate before this check existed.
    """
    stripped = text.strip()
    words = [w.strip(".,!?\"'") .lower() for w in stripped.split()]

    if len(words) >= min_repeats:
        max_run, run = 1, 1
        for i in range(1, len(words)):
            if words[i] == words[i - 1]:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 1
        if max_run >= min_repeats:
            return True

        from collections import Counter
        most_common_count = Counter(words).most_common(1)[0][1]
        if most_common_count / len(words) > 0.5:
            return True

    # (c) glued/no-space repetition: a short unit (1-12 chars) repeated
    # min_repeats+ times back-to-back anywhere in the text. Case-
    # insensitive since the sampler sometimes drifts capitalization
    # mid-repeat (e.g. "DogdogDog"). Safe against false positives on
    # genuine prose: natural English essentially never repeats an
    # arbitrary short substring 3+ times with zero separating
    # whitespace, so this only fires on the actual collapse pattern.
    if re.search(r"(.{1," + str(12) + r"}?)\1{" + str(min_repeats - 1) + r",}", stripped, flags=re.IGNORECASE):
        return True

    return False


def mentions_multiple_labels(text: str, candidate_labels: List[str], min_distinct: int = 2) -> bool:
    """
    True if `text` mentions min_distinct or more DISTINCT labels from
    `candidate_labels` as whole words -- a shotgun-listing shortcut
    distinct from is_degenerate_repetition: instead of repeating one
    word, the model recites several different trained labels at once
    (e.g. "dog sad happy Rome Paris"), which trivially satisfies
    whole-word matching against whichever target happens to be
    checked, without genuinely committing to reporting any one of
    them specifically. Caught only when the caller knows the full set
    of labels in play (e.g. all target concepts trained/tested in a
    given run) -- exact_match_score alone can't detect this, since it
    only ever sees one target at a time.
    """
    seen = set()
    for label in candidate_labels:
        pattern = r"\b" + re.escape(label.strip()) + r"\b"
        if re.search(pattern, text, flags=re.IGNORECASE):
            seen.add(label.lower())
    return len(seen) >= min_distinct


def exact_match_score(
    report: str, target_concept: Optional[str], candidate_labels: Optional[List[str]] = None,
) -> Optional[int]:
    """
    1 if target_concept appears as a whole word in report (case-
    insensitive), 0 if not, OR if the report is a degenerate
    repetition loop (see is_degenerate_repetition), OR if
    candidate_labels is given and the report mentions multiple
    distinct labels from it (see mentions_multiple_labels) -- both
    are ways a report can trivially satisfy whole-word matching
    without being a genuine, specific report. None if there's no
    well-defined target concept for this direction (e.g. a generic
    SAE feature with no single associated word) -- those rely on
    human judging instead.

    candidate_labels is optional and backward compatible: omit it
    (as most existing call sites do) to skip the shotgun-listing
    check entirely -- only pass it when the caller actually knows the
    full set of labels a given run's model could be reciting from.

    NOTE on the glued-mention recovery below: plain \b-bounded
    matching turns out to have the OPPOSITE problem from what it looks
    like at first -- it already correctly scores 0 on pure repetition
    spam like "catcatcatcat" (no word boundary exists mid-run between
    identical glued copies, so \bcat\b never matches there at all).
    But that same \b strictness also silently misses genuine, on-
    target hits where the steering hook drops the token's leading-
    space marker and glues it onto the previous word with no space --
    e.g. "...currently thinking aboutFrance, the country" or "...the
    upcomingItaly vs Spain football match", both seen repeatedly in
    the alpha_sweep.py calibration output right in the good, near-
    sweet-spot alpha range (not the collapse range). Those are real
    verbalization hits that \btarget\b alone was silently scoring 0 --
    an undercount in exactly the results you'd want to trust most.
    """
    if not target_concept:
        return None
    if is_degenerate_repetition(report):
        return 0
    if candidate_labels and mentions_multiple_labels(report, candidate_labels):
        return 0
    target = target_concept.strip()
    pattern = r"\b" + re.escape(target) + r"\b"
    if re.search(pattern, report, flags=re.IGNORECASE):
        return 1
    # Recover the lowercase-into-Capitalized glue artifact for
    # proper-noun targets only (Paris, Rome, France, Italy in this
    # project's jspace token set). Requires the glue to occur at a
    # lowercase-to-uppercase transition immediately before the target
    # and a normal word boundary immediately after it -- specific
    # enough not to become a blanket substring match (it would NOT,
    # for example, rescue "cat" inside "category": no case transition
    # there). Deliberately NOT extended to lowercase targets (dog,
    # cat, happy, sad) -- there's no case signal to tell a genuine
    # glued mention apart from an ordinary word that happens to
    # contain the target as a substring (concatenate, indicate,
    # category, vacation, ...), so those stay strict/\b-only and any
    # undercount there has to be caught by human/blind judging instead.
    if target[:1].isupper():
        glued_pattern = r"(?<=[a-z])" + re.escape(target) + r"\b"
        if re.search(glued_pattern, report):
            return 1
    return 0


def score_reports(
    reports: List[str], target_concept: Optional[str], candidate_labels: Optional[List[str]] = None,
) -> Dict:
    """
    Exact counts behind a reported rate. Returns
        k            reports scored 1
        n            reports that were scorable (target_concept given)
        n_total      reports generated
        n_degenerate reports flagged as repetition loops (scored 0, kept in n)
        n_multi_label reports flagged as shotgun listings (scored 0, kept in n)
        rate         k / n  (None if n == 0)
    so every percentage in a table can be traced to an integer k out of an integer n.
    Scoring itself is exactly exact_match_score -- nothing here changes what counts as a match.
    """
    scores = [exact_match_score(r, target_concept, candidate_labels=candidate_labels) for r in reports]
    scored = [s for s in scores if s is not None]
    k = int(sum(scored))
    n = len(scored)
    return {
        "k": k, "n": n, "n_total": len(reports),
        "n_degenerate": sum(1 for r in reports if is_degenerate_repetition(r)),
        "n_multi_label": (sum(1 for r in reports if mentions_multiple_labels(r, candidate_labels))
                          if candidate_labels else 0),
        "rate": (k / n) if n else None,
    }


def infer_target_concept(direction_key: str) -> Optional[str]:
    """
    J-space directions are named "jspace::{layer}::{token}" -- the
    token IS the target concept, so exact-match is well-defined for
    them. Other classes don't have one known target word by
    construction, so this returns None for them.
    """
    parts = direction_key.split("::")
    if parts[0] == "jspace" and len(parts) >= 3:
        return parts[2]
    return None


def export_for_blind_judging(records: List[Dict], out_path: str, seed: int = 0) -> str:
    """
    records: [{"direction_key":..., "trial": int, "report": str}, ...]
    Writes a shuffled, ID-only CSV for a human judge to fill in a
    `judge_score` column (1 = report describes the intervened
    concept, 0 = it doesn't) with no direction_key visible. Returns
    the path to a separate answer-key JSON (not shown to the judge)
    used to merge scores back to (direction, trial) pairs afterward.
    """
    shuffled = records[:]
    random.Random(seed).shuffle(shuffled)

    answer_key = {}
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["blind_id", "report", "judge_score"])
        for i, r in enumerate(shuffled):
            blind_id = f"item_{i:04d}"
            writer.writerow([blind_id, r["report"], ""])
            answer_key[blind_id] = {"direction_key": r["direction_key"], "trial": r["trial"]}

    answer_key_path = out_path.replace(".csv", "_answer_key.json")
    with open(answer_key_path, "w") as f:
        json.dump(answer_key, f, indent=2)
    return answer_key_path


def merge_judge_scores(judged_csv_path: str, answer_key_path: str) -> Dict[Tuple[str, int], int]:
    """
    Reads a completed blind-judging CSV + its answer key. Returns
    {(direction_key, trial): judge_score}, skipping rows the judge
    left blank.
    """
    answer_key = json.load(open(answer_key_path))
    scores = {}
    with open(judged_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["judge_score"] not in ("", None):
                entry = answer_key[row["blind_id"]]
                scores[(entry["direction_key"], entry["trial"])] = int(row["judge_score"])
    return scores