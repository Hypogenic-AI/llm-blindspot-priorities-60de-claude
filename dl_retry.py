import requests, time, json, os
H={"User-Agent":"Mozilla/5.0 research"}
# Known arXiv IDs for the failed/important ones
DIRECT = {
  "sharma2023_understanding_sycophancy":"2310.13548",
  "perez2022_model_written_evals":"2212.09251",
  "wei2023_synthetic_data_reduces_sycophancy":"2308.03958",
  "ouyang2022_instructgpt_rlhf":"2203.02155",
  "liu2023_lost_in_the_middle":"2307.03172",
}
def dl(url,path):
    try:
        r=requests.get(url,headers=H,timeout=90,allow_redirects=True)
        if r.status_code==200 and r.content[:4]==b"%PDF":
            open(path,"wb").write(r.content); return True
    except Exception as e: print("  err",e)
    return False
for n,a in DIRECT.items():
    ok=dl(f"https://arxiv.org/pdf/{a}.pdf", f"papers/{n}.pdf")
    print(("OK   " if ok else "FAIL "), n, a)
    time.sleep(1)
# Retry S2 metadata for relevance mechanistic + anchoring with backoff
S2="https://api.semanticscholar.org/graph/v1/paper/"
for n,pid in {"relevance2025_mechanistic":"CorpusId:277667643",
              "anchoring2025_effect_in_llms":"f03d6f215f163ec3e3e33158fde32a2945a48d85"}.items():
    for attempt in range(5):
        r=requests.get(S2+pid+"?fields=title,externalIds,openAccessPdf",headers=H)
        if r.status_code==200:
            d=r.json(); arx=(d.get("externalIds") or {}).get("ArXiv"); oa=(d.get("openAccessPdf") or {}).get("url")
            ok=False
            if arx: ok=dl(f"https://arxiv.org/pdf/{arx}.pdf",f"papers/{n}.pdf")
            if not ok and oa: ok=dl(oa,f"papers/{n}.pdf")
            print(("OK   " if ok else "NOPDF"), n, "arxiv=",arx); break
        time.sleep(6)
    else: print("METAFAIL",n)
