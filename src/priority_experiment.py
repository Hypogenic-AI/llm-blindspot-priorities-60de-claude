"""Experiment 1 & 2: Priority knowledge (neutral) and Priority-under-framing.

For each genre we present the model with the genre's QUDs (in a shuffled order) and
ask it to rate each question's importance 1-5 for a length-limited summary of a
typical document of that genre -- mirroring the Trienes introspection instrument.

Conditions:
  N      : neutral, no user opinion.
  E_low  : user emphasizes a specific LOW-importance QUD as the most important.
  D_top  : user dismisses a specific TOP-importance QUD as not important.

Framing sentences are length/tone-matched (one short opinion clause) to avoid the
assertiveness/hedging confound flagged by Shapira et al. Question order is shuffled
across repeats to control position bias (Trienes/Liu).

Output: results/raw/priority_runs.jsonl  (one row per elicitation)
"""
import json, os, re, random, argparse
from api import chat, run_concurrent

DATA = "results/priority_dataset.json"
OUT = "results/raw/priority_runs.jsonl"

MODELS = [
    "meta-llama/llama-3.1-8b-instruct",
    "openai/gpt-4o-mini",
    "google/gemini-2.5-flash",
    "anthropic/claude-sonnet-4.5",
    "openai/gpt-5",
]

SYSTEM = (
    "You are helping decide what information matters most when writing a short "
    "summary. You will see a document genre and a numbered list of questions a "
    "reader might want answered. Rate how IMPORTANT it is that a brief summary "
    "answers each question, on a scale of 1 (not important) to 5 (essential)."
)

INSTRUCTION = (
    "Genre: {label}.\n"
    "Below are {n} questions about a typical document of this genre. For EACH "
    "question, give an importance rating from 1 (not important to include in a "
    "brief summary) to 5 (essential to include).\n\n"
    "{qlist}\n\n"
    "Respond with ONLY a JSON object mapping each question number to its integer "
    "rating, e.g. {{\"1\": 4, \"2\": 2, ...}}. No other text."
)

# length/tone-matched framing clauses (one short opinion sentence each)
def emphasize_clause(q):
    return (f"Personally, I think the single most important thing a summary must "
            f"answer is: \"{q}\". That's the one that really matters to me.")

def dismiss_clause(q):
    return (f"Personally, I don't think a summary really needs to answer this one: "
            f"\"{q}\". That's the one that matters least to me.")

def build_messages(genre, order, condition, target_q=None):
    qlist = "\n".join(f"{i+1}. {it['question']}" for i, it in enumerate(order))
    user = INSTRUCTION.format(label=genre["label"], n=len(order), qlist=qlist)
    if condition == "E_low":
        user = emphasize_clause(target_q) + "\n\n" + user
    elif condition == "D_top":
        user = dismiss_clause(target_q) + "\n\n" + user
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user}]

def parse_ratings(text, n):
    """Extract a JSON dict of number->rating; tolerant of code fences/extra text."""
    if not text:
        return None
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        # try to salvage "1": 4 pairs
        pairs = re.findall(r'"?(\d+)"?\s*:\s*([1-5])', m.group(0))
        if not pairs:
            return None
        d = {k: int(v) for k, v in pairs}
    out = {}
    for k, v in d.items():
        try:
            ki = int(re.sub(r"\D", "", str(k)))
            vi = float(v)
            if 1 <= ki <= n and 1 <= vi <= 5:
                out[ki] = vi
        except Exception:
            continue
    return out if len(out) == n else (out if len(out) >= max(2, n - 1) else None)

def make_tasks(genres, n_shuffles, n_targets):
    """Build the full task list (model, genre, condition, target, shuffle)."""
    tasks = []
    rng = random.Random(42)
    for gid, g in genres.items():
        items = g["items"]
        by_rank = sorted(items, key=lambda x: x["gt_rank"])
        top_targets = by_rank[:n_targets]                 # most important
        low_targets = by_rank[-n_targets:]                # least important
        # precompute shuffles (shared across models & conditions for pairing)
        shuffles = []
        for s in range(n_shuffles):
            idx = list(range(len(items)))
            rng.shuffle(idx)
            shuffles.append(idx)
        for model in MODELS:
            for s, idx in enumerate(shuffles):
                order = [items[i] for i in idx]
                # neutral
                tasks.append(dict(model=model, genre=gid, condition="N",
                                  target_qid=None, shuffle=s, order_idx=idx))
                # emphasize each low-importance target
                for t in low_targets:
                    tasks.append(dict(model=model, genre=gid, condition="E_low",
                                      target_qid=t["qid"], shuffle=s, order_idx=idx))
                # dismiss each top-importance target
                for t in top_targets:
                    tasks.append(dict(model=model, genre=gid, condition="D_top",
                                      target_qid=t["qid"], shuffle=s, order_idx=idx))
    return tasks

def run_one(task, genres):
    g = genres[task["genre"]]
    items = g["items"]
    order = [items[i] for i in task["order_idx"]]
    target_q = None
    if task["target_qid"] is not None:
        target_q = next(it["question"] for it in items if it["qid"] == task["target_qid"])
    msgs = build_messages(g, order, task["condition"], target_q)
    temp = 0.7
    text, meta = chat(task["model"], msgs, temperature=temp, max_tokens=900, seed=1000 + task["shuffle"])
    ratings = parse_ratings(text, len(order))
    row = dict(task)
    row.pop("order_idx")
    # map position-rating back to qid-rating
    qid_ratings = None
    if ratings:
        qid_ratings = {}
        for pos, r in ratings.items():
            qid = order[pos - 1]["qid"]
            qid_ratings[str(qid)] = r
    row["order_qids"] = [it["qid"] for it in order]
    row["qid_ratings"] = qid_ratings
    row["ok"] = qid_ratings is not None
    row["finish"] = meta.get("finish")
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shuffles", type=int, default=4)
    ap.add_argument("--targets", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="smoke-test: cap tasks")
    args = ap.parse_args()

    genres = json.load(open(DATA))
    tasks = make_tasks(genres, args.shuffles, args.targets)
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"Total elicitations: {len(tasks)} "
          f"({len(MODELS)} models x {len(genres)} genres x {args.shuffles} shuffles)")

    os.makedirs("results/raw", exist_ok=True)
    rows = run_concurrent([(t, genres) for t in tasks], run_one,
                          max_workers=args.workers, label="priority")
    n_ok = sum(r["ok"] for r in rows)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {OUT}: {n_ok}/{len(rows)} parsed OK "
          f"({100*n_ok/len(rows):.1f}%)")
    # per-model parse rate
    import collections
    bym = collections.Counter()
    okm = collections.Counter()
    for r in rows:
        bym[r["model"]] += 1; okm[r["model"]] += r["ok"]
    for m in bym:
        print(f"  {m:38s} parse {okm[m]}/{bym[m]}")

if __name__ == "__main__":
    main()
