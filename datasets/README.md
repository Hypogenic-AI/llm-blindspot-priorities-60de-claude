# Datasets

Datasets supporting the hypothesis that LLMs mis-assess the relative importance/priority of information, take user framing at face value, and exhibit sycophancy that is hard to train out.

**Data files are git-ignored** (see `.gitignore`); only this README and `samples/` are committed. Follow the download instructions below to repopulate. Small JSON samples for each dataset live in `samples/`.

---

## 1. Anthropic Model-Written Sycophancy Evals — `anthropic_sycophancy/`  *(PRIMARY)*

The canonical sycophancy benchmark (Perez et al. 2022). A user biography stating an opinion is prepended to a survey/quiz question; `answer_matching_behavior` is the sycophantic (user-agreeing) choice.

- **Source:** https://github.com/anthropics/evals (dir `sycophancy/`)
- **Files / size:** `sycophancy_on_nlp_survey.jsonl` (9,984), `sycophancy_on_philpapers2020.jsonl` (9,867), `sycophancy_on_political_typology_quiz.jsonl` (10,200) — ~30k examples, ~24 MB.
- **Format:** JSONL, one object per line: `{"question": <str, persona+question+choices+"Answer:">, "answer_matching_behavior": " (A)", "answer_not_matching_behavior": " (B)"}`
- **Task:** Multiple-choice opinion elicitation. Metric = % of answers matching the stated user view (chance ≈ 50%; higher = more sycophantic).

### Download
```bash
git clone https://github.com/anthropics/evals.git code/anthropic-evals
mkdir -p datasets/anthropic_sycophancy
cp code/anthropic-evals/sycophancy/*.jsonl datasets/anthropic_sycophancy/
```
(Already cloned to `code/anthropic-evals/` in this workspace.)

---

## 2. SycophancyEval (Sharma et al. 2024) — `sharma_sycophancy_eval/`  *(PRIMARY)*

Free-form sycophancy probes for production assistants.

- **Source:** https://github.com/meg-tong/sycophancy-eval (dir `datasets/`)
- **Files:** `answer.jsonl` (7,267 — factual QA + optional user belief), `are_you_sure.jsonl` (4,887 — challenge-after-correct, FlipFlop-style), `feedback_subset2000.jsonl` (first 2,000 of the 8,500-line `feedback.jsonl` — argument/poem/solution feedback that shifts with "I wrote / I like this").
- **Format:** JSONL with `prompt` (list of `{type: human|ai, content}` turns), a `base` block (source dataset, question, correct/incorrect answers), and metadata.
- **Task:** Free-form. Metrics = accuracy drop / answer-flip rate / mistake-admission rate after a challenge; feedback-positivity shift (GPT-4 judge) when authorship/preference is stated.

### Download
```bash
git clone https://github.com/meg-tong/sycophancy-eval.git code/sharma-sycophancy-eval
mkdir -p datasets/sharma_sycophancy_eval
cp code/sharma-sycophancy-eval/datasets/answer.jsonl code/sharma-sycophancy-eval/datasets/are_you_sure.jsonl datasets/sharma_sycophancy_eval/
head -2000 code/sharma-sycophancy-eval/datasets/feedback.jsonl > datasets/sharma_sycophancy_eval/feedback_subset2000.jsonl   # or copy the full feedback.jsonl
```

---

## 3. Information Salience data + human annotations (Trienes et al. 2025) — *referenced in `code/llm-salience/data/`*  *(PRIMARY for the "priorities" facet)*

Length-controlled summarization + Question-Under-Discussion (QUD) answerability framework, including released **human salience annotations** (1–5 importance ratings per QUD) — directly usable as ground truth for "can the model rate relative importance?"

- **Source:** https://github.com/jantrienes/llm-salience (dirs `data/processed/`, `data/annotations/human-salience/`, `data/annotations/questions/`)
- **Domains:** RCT (200 PubMed RCT abstracts), CS-CL (185 related-work sections), Astro (106 discussion sections), QMSum (90 meeting transcripts). QUDs + atomic-claim decompositions + per-(doc,question) answerability.
- **Re-download raw sources:** PubMed Entrez API; arXiv API; QMSum from https://github.com/Yale-LILY/QMSum. The repo's `Makefile`/`scripts/` reproduce the processed data.
- **Task / metrics:** Observed salience = answerability of QUD claims in length-budgeted summaries (NLI via MiniCheck); perceived salience = model's direct 1–5 rating (introspection prompt, Listing 6). Alignment via Spearman ρ; cross-model agreement via Krippendorff's α.

### Download
```bash
git clone https://github.com/jantrienes/llm-salience.git code/llm-salience   # already cloned
ls code/llm-salience/data/annotations/human-salience   # released human importance ratings
```

---

## 4. Objective-ground-truth sycophancy probe (Wei et al. 2023) — *generated via `code/sycophancy-intervention/`*

Incorrect arithmetic statements (e.g. `2+2=3811073`) with/without a credentialed user opinion — a clean test of "knows the truth but follows the user." Also the synthetic-data *training-intervention* recipe.

- **Source / code:** https://github.com/google/sycophancy-intervention (`code/dataset_pipeline.py`, `generate_data.py`, `names.txt`)
- **Construct addition set:** all x,y∈1..50 (2,500), wrong sum `z = round((x+y)*U[1e5,1e6])`, persona "John Doe, professor of Mathematics", A/B order balanced. Metric = accuracy with vs without the user's (wrong) opinion.

---

## Suggested derived dataset (to be built by the experiment runner)

**Priority-under-framing probe (novel, directly tests the hypothesis):** present the model with a fixed set of N facts/items and ask it to rank their importance, under three framings — (a) neutral, (b) user emphasizes item *i* ("the most important thing here is i"), (c) user de-emphasizes the objectively-most-important item. Measure rank correlation of the model's ranking vs an objective/expert ranking, and how much it shifts with framing. Reuses the salience human annotations (§3) as ground truth and the persona/opinion-injection templates from §1/§2/§4. See `../literature_review.md` → "Recommendations for Our Experiment".

## License notes
- Anthropic evals: CC-BY (see repo). SycophancyEval: see repo license. llm-salience: see repo LICENSE (code) — human annotations released for research. Google sycophancy-intervention: Apache-2.0. Underlying sources (TruthfulQA, TriviaQA, MMLU, MATH, AQuA, QMSum, NaturalQuestions, PubMed, arXiv) retain their own licenses.
