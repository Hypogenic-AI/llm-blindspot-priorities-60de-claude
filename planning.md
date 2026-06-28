# Planning — LLM Retrospective Blindspot: Priorities

## Motivation & Novelty Assessment

### Why This Research Matters
LLMs are increasingly used to triage, summarize, and rank information for human
decision-makers (what to read first, which finding matters, what to escalate).
If a model cannot reliably tell what is *important* — and worse, silently re-ranks
importance to match how a user *frames* a request — then its "helpful" outputs
quietly inherit the user's blind spots. This is a sycophancy failure mode that is
more insidious than agreeing with a wrong fact: it corrupts *prioritization*,
which is rarely checkable by the user.

### Gap in Existing Work
From `literature_review.md`: (1) Trienes et al. (2025) show LLMs have a stable
*implicit* salience but cannot *introspect* it reliably, and it aligns only weakly
with humans — but they never apply **user framing**. (2) The sycophancy literature
(Perez 2022, Sharma 2024, Wei 2023, Shapira 2026) shows models follow user framing
on **facts**, but never on **importance/priority rankings**. **No paper couples the
two: does user framing distort a model's explicit importance ranking?** That is the
exact claim of this hypothesis, and it is unstudied.

### Our Novel Contribution
A controlled **"priority-under-framing"** probe that:
1. Replicates the Trienes introspection gap on *new 2025-2026 frontier models*
   (does the model know true priorities? — Spearman ρ vs human ground truth).
2. **Newly** measures whether neutral user framing (emphasizing an unimportant item /
   dismissing an important one) shifts the model's *explicit importance ranking* —
   "priority sycophancy."
3. Tests the user's specific conjecture: **the model defers to framing most where it
   is least certain of true priority** (ambiguous mid-importance items), i.e. lack of
   a grounded sense of significance *causes* the framing-following.
4. Cross-tier model panel (8B → frontier) to probe the scaling/trainability trend.
5. A clean corroborating **objective-truth arithmetic sycophancy** arm (Wei) to
   anchor the priority results against a known sycophancy signal on the same models.

### Experiment Justification
- **Exp 1 (Priority knowledge, neutral):** Needed to establish the model's baseline
  grasp of true importance (ρ vs human). Without it, a framing shift is uninterpretable.
- **Exp 2 (Priority-under-framing):** The core novel test of the hypothesis — does
  framing distort the explicit ranking, and by how much (targeted-item rank shift)?
- **Exp 3 (Uncertainty × deference):** Tests the *mechanism* the user proposed — that
  poor sense of significance drives framing-following. Item-level analysis.
- **Exp 4 (Arithmetic sycophancy, corroborating):** Clean, judge-free anchor that the
  same models *do* exhibit classic sycophancy, linking the novel measure to the
  established literature, and lets us compare facet magnitudes.

## Research Question
Do LLMs (a) fail to accurately assess the relative importance/priority of information,
and (b) take user framing at face value by distorting their explicit importance
rankings (priority sycophancy) — and is this worse where true priority is ambiguous?

## Hypothesis Decomposition
- **H1 (priority knowledge gap):** Neutral-condition model importance rankings align
  only weakly/moderately with human ground truth (ρ well below inter-human agreement).
- **H2 (priority sycophancy):** Relative to neutral, user framing significantly shifts
  the targeted item's importance rank in the framed direction (emphasis ↑, dismissal ↓).
- **H3 (uncertainty drives deference):** Framing-induced shift is larger for items with
  ambiguous ground-truth importance (mid-rank / high human disagreement) than for
  clearly top/bottom items.
- **H4 (corroboration):** The same models show classic arithmetic sycophancy (accuracy
  drop when a credentialed user asserts a wrong answer).

Independent variables: framing condition (neutral / emphasize-low / dismiss-top),
target item, model, genre, item order (shuffle). Dependent variables: Spearman ρ to
human GT, targeted-item rank shift, accuracy.

## Proposed Methodology

### Approach
Use the `llm-salience` human importance annotations (4 genres: pubmed-RCT, cs-cl,
astro-ph, qmsum meetings; 10–21 Questions-Under-Discussion each, rated 1–5 by 3–5
annotators) as **ground-truth importance rankings**. Because the QUDs are *generic to
a genre* (not tied to a document the user could know better), any framing-induced shift
is pure sycophancy with **no legitimate informational basis** — a clean design.

Elicit the model's 1–5 importance rating for every QUD under:
- **N** neutral (no user opinion),
- **E_low** user emphasizes a specific *low*-importance QUD as "the most important",
- **D_top** user dismisses a specific *top* QUD as "not really important".
Framing prompts are length/tone-matched to avoid stylistic confounds (Shapira caution).
Question order is shuffled across repeats to control position bias (Trienes/Liu).

### Experimental Steps
1. Build the priority-under-framing dataset from human annotations (GT means + variance).
2. Harness: OpenRouter chat calls returning structured per-QUD ratings; cache to disk.
3. Run neutral elicitations (Exp 1) → ρ vs human GT, vs inter-human ceiling.
4. Run framed elicitations targeting bottom-3 (emphasize) and top-3 (dismiss) per genre
   (Exp 2) → targeted-item rank shift vs neutral.
5. Item-level uncertainty analysis (Exp 3).
6. Arithmetic sycophancy arm (Exp 4): subsample Wei-style `a*b` problems with/without a
   "math professor" asserting a wrong answer.

### Baselines / Controls
- **Inter-human agreement ceiling** (mean pairwise Spearman ρ among annotators) — the
  realistic upper bound for ρ-to-GT.
- **Random/chance ranking** floor (ρ≈0).
- **Neutral condition** as the within-model control for all framing effects.
- **Order shuffles** (≥3) averaged, to remove position bias.
- **Cross-tier model panel** (Llama-3.1-8B, gpt-4o-mini, gemini-2.5-flash, gpt-5,
  claude-sonnet-4.5) for the scaling trend.

### Evaluation Metrics
- **Priority accuracy:** Spearman ρ (model ratings vs human-mean ratings).
- **Priority sycophancy (headline):** targeted-item rank shift Δrank (neutral→framed),
  and sign-consistency rate (fraction of framings that move the item the "agreed" way).
- **Overall distortion:** change in ρ-to-GT under framing.
- **Uncertainty coupling:** correlation of |Δrank| with GT ambiguity (human std, mid-rank).
- **Arithmetic sycophancy:** accuracy(neutral) − accuracy(wrong-authority).

### Statistical Analysis Plan
- Paired comparisons neutral vs framed across (genre×target×shuffle): Wilcoxon signed-rank
  on Δrank; bootstrap 95% CIs on mean Δrank; one-sample test that Δrank≠0 in framed dir.
- ρ-to-GT: report per-model mean with bootstrap CI; compare to human ceiling.
- H3: mixed/OLS regression of |Δrank| on GT centrality & human-std, clustered by genre.
- Arithmetic: McNemar / paired bootstrap on accuracy drop.
- Significance α=0.05; report effect sizes (Cohen's d / rank-biserial) and CIs, not just p.
- Multiple-model reporting; no cherry-picking — all models/genres reported.

## Expected Outcomes
- **Supports hypothesis if:** neutral ρ-to-GT is moderate-and-below-human-ceiling (H1);
  framing produces significant targeted Δrank in the agreed direction (H2); |Δrank| is
  larger for ambiguous items (H3); arithmetic sycophancy present (H4).
- **Refutes/weakens if:** ρ-to-GT ≈ human ceiling AND framing produces no significant
  Δrank — models would then robustly know and hold priorities.

## Timeline & Milestones
1. Data build + harness + smoke test (1 QUD set, 1 model): 30 min.
2. Run Exp 1+2 across panel (cached, concurrent): 60–90 min.
3. Exp 4 arithmetic arm: 20 min. 4. Analysis + figures: 45 min. 5. Report: 30 min.
Buffer ~25% for API hiccups (retry/backoff, caching makes reruns cheap).

## Potential Challenges
- **Rating ties / coarse 1–5 scale** → use the model's ratings to induce a ranking,
  break ties by elicited order; rely on Spearman (tie-aware) and analyze rank of target.
- **Position bias** → shuffle + average.
- **Stylistic confound in framing prompts** → length/tone-matched templates, logged.
- **gpt-5 needs `max_completion_tokens` + reasoning budget** → handled in harness.
- **API variance** → ≥3 repeats, cache raw responses, set seeds where supported.
- **Small genre count (4)** → compensate with many target×shuffle×model data points and
  treat genre as a clustering factor, not an i.i.d. sample; report per-genre.

## Success Criteria
A reproducible result, with CIs and significance tests across ≥4 models, that
quantifies (1) the priority-knowledge gap vs human ceiling and (2) the magnitude and
direction of framing-induced priority distortion, plus the uncertainty-coupling test —
delivering the first direct measurement of "priority sycophancy."
