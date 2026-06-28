"""Build the priority-under-framing dataset from llm-salience human annotations.

Each genre has a shared set of Questions-Under-Discussion (QUDs) rated 1-5 for
importance by 3-5 human annotators. We aggregate into a ground-truth importance
ordering (mean rating) plus an ambiguity measure (std across annotators), and
record the inter-human agreement ceiling (mean pairwise Spearman rho).
"""
import json, glob, collections, statistics, itertools, os
from scipy.stats import spearmanr
import numpy as np

ANN_DIR = "code/llm-salience/data/annotations/human-salience"
OUT = "results/priority_dataset.json"

GENRE_LABEL = {
    "pubmed-sample": "Medical RCT abstract (PubMed)",
    "cs-cl": "Computational linguistics paper",
    "astro-ph": "Astrophysics paper",
    "qmsum-generic": "Meeting transcript (QMSum)",
}

def main():
    files = glob.glob(f"{ANN_DIR}/*.json")
    # dataset -> qid -> list[ratings]; dataset -> qid -> question text
    ratings = collections.defaultdict(lambda: collections.defaultdict(list))
    qtext = collections.defaultdict(dict)
    # dataset -> annotator -> qid -> rating  (for inter-annotator agreement)
    per_ann = collections.defaultdict(lambda: collections.defaultdict(dict))
    for f in files:
        d = json.load(open(f))
        ds = d["dataset"]
        for it in d["items"]:
            ratings[ds][it["id"]].append(it["rating"])
            qtext[ds][it["id"]] = it["question"]
            per_ann[ds][d["annotator"]][it["id"]] = it["rating"]

    genres = {}
    for ds in ratings:
        qids = sorted(ratings[ds].keys())
        items = []
        for qid in qids:
            rs = ratings[ds][qid]
            items.append({
                "qid": qid,
                "question": qtext[ds][qid],
                "gt_mean": round(statistics.mean(rs), 4),
                "gt_std": round(statistics.pstdev(rs), 4),
                "n_ann": len(rs),
            })
        # rank items by gt_mean descending (1 = most important)
        order = sorted(items, key=lambda x: -x["gt_mean"])
        for rank, it in enumerate(order, 1):
            it["gt_rank"] = rank  # 1 = most important
        items.sort(key=lambda x: x["qid"])

        # inter-human agreement ceiling: mean pairwise Spearman over shared qids
        anns = list(per_ann[ds].keys())
        common = set.intersection(*[set(per_ann[ds][a].keys()) for a in anns])
        common = sorted(common)
        pairwise = []
        for a, b in itertools.combinations(anns, 2):
            va = [per_ann[ds][a][q] for q in common]
            vb = [per_ann[ds][b][q] for q in common]
            if len(set(va)) > 1 and len(set(vb)) > 1:
                rho = spearmanr(va, vb).statistic
                if not np.isnan(rho):
                    pairwise.append(rho)
        genres[ds] = {
            "genre_id": ds,
            "label": GENRE_LABEL.get(ds, ds),
            "n_questions": len(items),
            "n_annotators": len(anns),
            "human_ceiling_rho": round(float(np.mean(pairwise)), 4),
            "human_ceiling_rho_std": round(float(np.std(pairwise)), 4),
            "items": items,
        }
        print(f"{ds:16s} nq={len(items):2d} anns={len(anns)} "
              f"human-ceiling rho={genres[ds]['human_ceiling_rho']:.3f} "
              f"gt_mean range=[{min(i['gt_mean'] for i in items):.2f},"
              f"{max(i['gt_mean'] for i in items):.2f}]")

    os.makedirs("results", exist_ok=True)
    json.dump(genres, open(OUT, "w"), indent=2)
    print(f"\nWrote {OUT} with {len(genres)} genres.")

if __name__ == "__main__":
    main()
