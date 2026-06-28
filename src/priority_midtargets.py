"""Supplementary run for Exp 3: target MIDDLE-ranked QUDs (not just extremes).

The main run targets top-3 / bottom-3 items. To test 'models defer more on
ambiguous items' across the full ground-truth range, we additionally emphasize and
dismiss the middle-ranked items. Reuses the same harness, prompts, and cache.

Output: results/raw/priority_runs_mid.jsonl
"""
import json, os, random
from api import run_concurrent
from priority_experiment import MODELS, run_one

DATA = "results/priority_dataset.json"
OUT = "results/raw/priority_runs_mid.jsonl"
N_SHUFFLES = 4
N_MID = 3  # middle items per genre

def make_tasks(genres):
    tasks = []
    rng = random.Random(42)  # same seed -> same shuffles as main run (cache reuse)
    for gid, g in genres.items():
        items = g["items"]
        by_rank = sorted(items, key=lambda x: x["gt_rank"])
        mid = len(by_rank) // 2
        half = N_MID // 2
        mid_targets = by_rank[mid - half: mid - half + N_MID]
        shuffles = []
        for s in range(N_SHUFFLES):
            idx = list(range(len(items)))
            rng.shuffle(idx)
            shuffles.append(idx)
        for model in MODELS:
            for s, idx in enumerate(shuffles):
                for t in mid_targets:
                    # emphasize (push up) and dismiss (push down) each mid item
                    for cond in ("E_low", "D_top"):
                        tasks.append(dict(model=model, genre=gid, condition=cond,
                                          target_qid=t["qid"], shuffle=s, order_idx=idx,
                                          mid=True))
    return tasks

def main():
    genres = json.load(open(DATA))
    tasks = make_tasks(genres)
    print(f"Mid-target elicitations: {len(tasks)}")
    rows = run_concurrent([(t, genres) for t in tasks], run_one, max_workers=10, label="mid")
    n_ok = sum(r["ok"] for r in rows)
    os.makedirs("results/raw", exist_ok=True)
    with open(OUT, "w") as f:
        for r in rows:
            r["mid"] = True
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {OUT}: {n_ok}/{len(rows)} OK")

if __name__ == "__main__":
    main()
