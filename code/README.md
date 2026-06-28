# Cloned Repositories

Four repositories supporting the experiment. None were user-specified (the topic spec listed no `code_references`), so these were selected from the literature review as the highest-leverage tools/data.

---

## 1. `anthropic-evals/` — Model-Written Evaluations (Perez et al. 2022)
- **URL:** https://github.com/anthropics/evals
- **Purpose:** Source of the canonical sycophancy datasets (`sycophancy/*.jsonl`, ~30k examples) plus `persona/`, `advanced-ai-risk/`, `winogenerated/`.
- **Key files:** `sycophancy/sycophancy_on_{nlp_survey,philpapers2020,political_typology_quiz}.jsonl`.
- **Use:** Drop-in MCQ sycophancy benchmark. Each line has `question`, `answer_matching_behavior`, `answer_not_matching_behavior`. Just send `question` to a model and check which choice it picks.
- **Deps:** none (plain JSONL).

## 2. `sharma-sycophancy-eval/` — SycophancyEval (Sharma et al., Anthropic, ICLR 2024)
- **URL:** https://github.com/meg-tong/sycophancy-eval
- **Purpose:** Free-form sycophancy probes: `datasets/answer.jsonl`, `datasets/are_you_sure.jsonl`, `datasets/feedback.jsonl`; `utils.py`; `example.ipynb`.
- **Key entry points:** `example.ipynb` shows the eval loop; `utils.py` has prompt construction + GPT-4 grading helpers.
- **Use:** Answer-sycophancy (state a weak belief), Are-You-Sure (challenge a correct answer), feedback-sycophancy (state authorship/preference). Metrics = accuracy drop, flip rate, mistake-admission, feedback-positivity shift.
- **Deps:** LangChain + OpenAI/Anthropic API keys for the GPT-4 judge.

## 3. `llm-salience/` — Behavioral Analysis of Information Salience (Trienes et al. 2025)
- **URL:** https://github.com/jantrienes/llm-salience
- **Purpose:** The full salience/priority framework + released data and **human importance annotations**.
- **Key files:** `src/info_salience/` (library); `scripts/{summarization,qa,claim_extraction,claim_entailment_array,introspection}.sh` (pipeline stages); `data/processed/{pubmed-sample,qmsum-generic,astro-ph,cs-cl}/`; `data/annotations/human-salience/` (ground-truth importance ratings); `notebooks/`; `Makefile`.
- **Use:** Reproduce Content Salience Maps; reuse the **introspection prompt (Listing 6)** as a ready-made "rate the relative importance of these questions 1–5" instrument; use human-salience annotations as ground truth for priority-assessment alignment.
- **Deps:** `environment.yml` / `requirements-lock.txt` (sentence-transformers, UMAP, HDBSCAN, MiniCheck NLI, vLLM, OpenAI API). GPU recommended for vLLM/MiniCheck but the prompts/data can be reused without it.

## 4. `sycophancy-intervention/` — Simple Synthetic Data Reduces Sycophancy (Wei et al. 2023, Google)
- **URL:** https://github.com/google/sycophancy-intervention
- **Purpose:** Generate (a) the objective-ground-truth arithmetic sycophancy probe and (b) the synthetic opinion-independence finetuning data.
- **Key files:** `code/dataset_pipeline.py`, `code/generate_data.py`, `code/pull_from_huggingface.py`, `code/names.txt` (10k persona names), `code/utils.py`.
- **Use:** Reproduce the `2+2=<wrong>` + credentialed-user probe; build the persona+claim training set (17 HF classification datasets) for the mitigation arm.
- **Deps:** Python; `pull_from_huggingface.py` needs `datasets`. The original finetuning targeted PaLM/Flan-PaLM (TPU); the *data recipe* is model-agnostic.

---

### Selection rationale & testing notes
- Repos 1, 3, 4 ship reusable **data** that loads with no compute; repo 2 ships data + an eval harness. All four data dirs were inspected and verified to contain real records (see `../datasets/`).
- Heavy components (vLLM inference, MiniCheck NLI, GPT-4 judging) require GPU and/or API keys and are deferred to the experiment-runner phase; their prompt templates and metrics are documented in `../literature_review.md` so they can be reimplemented lightweightly if needed.
- Large data inside these repos is git-ignored at the workspace root; clone instructions are in `../datasets/README.md`.
