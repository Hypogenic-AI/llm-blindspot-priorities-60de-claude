# Resources Catalog

Resources gathered for **"LLM Retrospective Blindspot: Priorities"** — the hypothesis that LLMs mis-assess the relative importance/priority of information, take user framing at face value, exhibit sycophancy, and that this is hard to train out.

## Summary
- **Papers:** 14 PDFs (6 deep-read, 8 abstract-screened) — `papers/`
- **Datasets:** 4 primary dataset families + 1 proposed derived dataset — `datasets/`
- **Code repositories:** 4 cloned — `code/`

---

## Papers
Total downloaded: **14** (all validated as real PDFs). See `papers/README.md`.

| Title | Authors | Year | File | Key info |
|---|---|---|---|---|
| Behavioral Analysis of Information Salience in LLMs **[deep]** | Trienes et al. | 2025 | `salience2025_behavioral_analysis.pdf` | Implicit-vs-introspective priority dissociation; ships data+human ratings |
| Lost in the Middle **[deep]** | Liu et al. | 2023 | `liu2023_lost_in_the_middle.pdf` | Position over relevance; U-shape; survives training |
| Towards Understanding Sycophancy **[deep]** | Sharma et al. (Anthropic) | 2024 | `sharma2023_understanding_sycophancy.pdf` | SycophancyEval; PMs reward sycophancy |
| Discovering LM Behaviors w/ Model-Written Evals **[deep]** | Perez et al. (Anthropic) | 2022 | `perez2022_model_written_evals.pdf` | Canonical sycophancy data; RLHF doesn't fix it |
| Simple Synthetic Data Reduces Sycophancy **[deep]** | Wei et al. (Google) | 2023 | `wei2023_synthetic_data_reduces_sycophancy.pdf` | Arithmetic probe; partial, brittle fix |
| How RLHF Amplifies Sycophancy **[deep]** | Shapira et al. | 2026 | `howrlhf2026_amplifies_sycophancy.pdf` | Formal proof: harder optimization amplifies sycophancy |
| Are You Sure? / FlipFlop | — | 2023 | `are_you_sure2023_flipflop.pdf` | Challenge→flip protocol |
| Comparing the Framing Effect (Humans vs LLMs) | — | 2025 | `framing2025_comparing_humans_llms.pdf` | Gain/loss framing sensitivity |
| Modeling Content Importance for Summarization | — | 2020 | `content_importance_2020_summarization.pdf` | Importance-estimation baselines |
| Linear Probe Penalties Reduce Sycophancy | — | 2024 | `linear_probe2024_reduce_sycophancy.pdf` | Interpretability mitigation |
| SycEval | — | 2025 | `syceval2025_evaluating_sycophancy.pdf` | Sycophancy eval framework |
| TRUTH DECAY | — | 2025 | `truthdecay2025_multiturn_sycophancy.pdf` | Multi-turn sycophancy metric |
| InstructGPT | Ouyang et al. (OpenAI) | 2022 | `ouyang2022_instructgpt_rlhf.pdf` | RLHF pipeline |
| Open Problems in RLHF | Casper et al. | 2023 | `casper2023_open_problems_rlhf.pdf` | Structural RLHF limits |

## Datasets
Total: **4** primary families (data git-ignored; samples in `datasets/samples/`). See `datasets/README.md`.

| Name | Source | Size | Task | Location |
|---|---|---|---|---|
| Anthropic Model-Written Sycophancy | github.com/anthropics/evals | ~30k / 24 MB | MCQ opinion-matching | `datasets/anthropic_sycophancy/` |
| SycophancyEval (Sharma) | github.com/meg-tong/sycophancy-eval | ~14k+ | Free-form sycophancy | `datasets/sharma_sycophancy_eval/` |
| Information Salience + human ratings | github.com/jantrienes/llm-salience | 581 docs, 4 domains | Salience / importance | `code/llm-salience/data/` |
| Arithmetic-with-opinion (generated) | github.com/google/sycophancy-intervention | 2,500 | Objective-truth sycophancy | build via repo |

## Code Repositories
Total: **4**. See `code/README.md`.

| Name | URL | Purpose | Location |
|---|---|---|---|
| anthropic-evals | github.com/anthropics/evals | Canonical sycophancy datasets | `code/anthropic-evals/` |
| sharma-sycophancy-eval | github.com/meg-tong/sycophancy-eval | Free-form sycophancy harness + data | `code/sharma-sycophancy-eval/` |
| llm-salience | github.com/jantrienes/llm-salience | Salience framework + human ratings | `code/llm-salience/` |
| sycophancy-intervention | github.com/google/sycophancy-intervention | Arithmetic probe + synthetic-data fix | `code/sycophancy-intervention/` |

---

## Resource Gathering Notes

**Search strategy.** Used the paper-finder service (Semantic Scholar-backed) across 5 query angles spanning both hypothesis facets: sycophancy/RLHF, information importance/priority, salience/relevance judgment, and framing/anchoring. Returned 50–75 ranked papers per query. Curated ~17 candidates by relevance score (≥2), foundational status, and code/data availability; downloaded 14 (resolving open-access PDFs via the Semantic Scholar Graph API and direct arXiv IDs). The 6 most central were deep-read in full via a parallel chunk-and-extract workflow.

**Selection criteria.** (1) Direct coverage of a hypothesis clause; (2) released, reusable datasets/code; (3) foundational/canonical (Perez, Sharma, Liu) or directly operationalizable (Trienes, Wei, Shapira). Prioritized papers giving concrete task formats and metrics for experimental design.

**Challenges.** Semantic Scholar API rate-limited (429) intermittently — worked around with direct arXiv-ID downloads and backoff. A few high-relevance papers exist only behind Semantic Scholar corpus IDs without open-access PDFs (e.g., "An Anchoring Effect in LLMs," "How Do LLMs Understand Relevance?") and were left as abstract-only references.

**Gaps / workarounds.** No existing dataset directly tests *user-framing-induced distortion of explicit importance rankings* (the precise hypothesis). The `datasets/README.md` proposes a derived "priority-under-framing" dataset that composes the llm-salience human importance ratings (ground truth) with the persona/opinion-injection templates from the sycophancy papers — see `literature_review.md` → "Recommendations for Our Experiment."

## Recommendations for Experiment Design
1. **Primary dataset:** a derived *priority-under-framing* probe built from `llm-salience` human importance annotations + the introspection prompt; framing manipulations from Perez/Wei/Shapira templates.
2. **Baselines/controls:** neutral/no-opinion condition, A/B order randomization (position-bias control), chance line; model-scale or base-vs-instruct comparison if compute allows.
3. **Metrics:** Spearman ρ of model vs ground-truth importance ranking; framing-induced rank shift (headline); % view-matching / accuracy-drop for the corroborating sycophancy arms.
4. **Code to reuse:** `llm-salience` (introspection prompt, human ratings, NLI scoring); `sycophancy-intervention` (arithmetic probe); `sharma-sycophancy-eval` (are-you-sure / feedback harness); `anthropic-evals` (MCQ sycophancy data).

---

## Research Process (this study)

Built a novel **priority-under-framing** probe directly testing the hypothesis that the
literature leaves open ("does user framing distort a model's *explicit* importance ranking?").

- **Ground truth:** repurposed `code/llm-salience` human importance annotations (4 genres,
  10–21 QUDs each, 3–5 annotators, 1–5 ratings) as a consensus priority ordering, with a
  leave-one-annotator-out human ceiling (`src/build_data.py` → `results/priority_dataset.json`).
- **Experiments (5 models via OpenRouter, 2,240 cached calls, ≈$3.6):**
  Exp1 neutral priority knowledge; Exp2 priority sycophancy under length-matched framing;
  Exp3 mid-target uncertainty-coupling mechanism test; Exp4 Wei arithmetic sycophancy control.
- **Headline result:** frontier models have *solved* factual (arithmetic) sycophancy (drop≈0)
  but cave almost completely on priority framing (compliance up to 0.95), importance-blind,
  with no improvement from capability. See `REPORT.md`, `figures/`, `results/`.
