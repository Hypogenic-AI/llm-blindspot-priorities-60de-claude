import requests, time, json, os, re

# Curated key papers: (short_name, identifier) where identifier is "CorpusId:xxx", "arxiv:xxx", or S2 paperId hash
PAPERS = {
    # --- Foundational sycophancy ---
    "sharma2023_understanding_sycophancy": "CorpusId:264405698",
    "perez2022_model_written_evals": "CorpusId:254854519",
    "wei2023_synthetic_data_reduces_sycophancy": "CorpusId:260704246",
    "ouyang2022_instructgpt_rlhf": "paper:d766bffc357127e0dc86dd69561d5aeb520d6f4c",
    "casper2023_open_problems_rlhf": "paper:6eb46737bf0ef916a7f906ec6a8da82a45ffb623",
    # --- Information salience / importance (priority facet) ---
    "salience2025_behavioral_analysis": "CorpusId:276482257",
    "liu2023_lost_in_the_middle": "paper:1733eb7792f7a43dd21f51f4d1017a1bffd217b5",
    "relevance2025_mechanistic": "CorpusId:277667643",
    "content_importance_2020_summarization": "CorpusId:226262337",
    # --- Framing / anchoring (take framing at face value) ---
    "anchoring2025_effect_in_llms": "paper:f03d6f215f163ec3e3e33158fde32a2945a48d85",
    "framing2025_comparing_humans_llms": "CorpusId:276576130",
    "fact_to_judgment2025_task_framing": "CorpusId:283054937",
    # --- Mitigation / training-out difficulty ---
    "howrlhf2026_amplifies_sycophancy": "CorpusId:285270178",
    "linear_probe2024_reduce_sycophancy": "CorpusId:274437329",
    "syceval2025_evaluating_sycophancy": "CorpusId:276287766",
    "are_you_sure2023_flipflop": "CorpusId:265213308",
    "truthdecay2025_multiturn_sycophancy": "CorpusId:277065947",
}

S2 = "https://api.semanticscholar.org/graph/v1/paper/"
HEADERS = {"User-Agent":"Mozilla/5.0 research"}
os.makedirs("papers", exist_ok=True)
meta = {}

def get_field(pid):
    url = S2 + pid + "?fields=title,year,authors,externalIds,openAccessPdf,abstract"
    for _ in range(4):
        r = requests.get(url, headers=HEADERS)
        if r.status_code==200: return r.json()
        if r.status_code==429: time.sleep(5); continue
        return None
    return None

def download(url, path):
    try:
        r = requests.get(url, headers=HEADERS, timeout=60, allow_redirects=True)
        if r.status_code==200 and r.content[:4]==b"%PDF":
            open(path,"wb").write(r.content); return True
        # arxiv pages sometimes need pdf suffix
    except Exception as e:
        print("  dl err", e)
    return False

for name, ident in PAPERS.items():
    pid = ident.replace("paper:","")
    d = get_field(pid)
    time.sleep(1.2)
    if not d:
        print(f"[META FAIL] {name} {ident}"); continue
    ext = d.get("externalIds") or {}
    arx = ext.get("ArXiv")
    oa = (d.get("openAccessPdf") or {}).get("url")
    path = f"papers/{name}.pdf"
    ok=False
    # Prefer arXiv
    if arx:
        ok = download(f"https://arxiv.org/pdf/{arx}.pdf", path)
    if not ok and oa:
        ok = download(oa, path)
    print(f"[{'OK ' if ok else 'NOPDF'}] {name} | arxiv={arx} | oa={'y' if oa else 'n'} | {d.get('title','')[:60]}")
    meta[name] = {"title":d.get("title"),"year":d.get("year"),
                  "authors":", ".join(a["name"] for a in (d.get("authors") or [])[:6]),
                  "arxiv":arx, "oa_pdf":oa, "downloaded":ok,
                  "abstract":d.get("abstract"), "corpusId":ext.get("CorpusId")}
json.dump(meta, open("papers/_metadata.json","w"), indent=2)
print("\nDownloaded:", sum(1 for m in meta.values() if m["downloaded"]), "/", len(PAPERS))
