import requests, time, json, os
H={"User-Agent":"Mozilla/5.0 research"}
remaining = {
  "are_you_sure2023_flipflop":"CorpusId:265213308",
  "truthdecay2025_multiturn_sycophancy":"CorpusId:277065947",
  "relevance2025_mechanistic":"CorpusId:277667643",
  "fact_to_judgment2025_task_framing":"CorpusId:283054937",
  "anchoring_bias2024_experimental":"CorpusId:a1fd12feb1b49219ad8d8ef231386ef26d7af60d",
}
ids=list(remaining.values())
# normalize: batch API takes CorpusId:xxx or hash
r=requests.post("https://api.semanticscholar.org/graph/v1/paper/batch",
    params={"fields":"title,year,externalIds,openAccessPdf"},
    json={"ids":[i for i in ids if not i.startswith('CorpusId:a')]+["a1fd12feb1b49219ad8d8ef231386ef26d7af60d"]},
    headers=H)
print("batch status", r.status_code)
def dl(url,path):
    try:
        x=requests.get(url,headers=H,timeout=90,allow_redirects=True)
        if x.status_code==200 and x.content[:4]==b"%PDF": open(path,"wb").write(x.content); return True
    except Exception as e: print("err",e)
    return False
names=list(remaining.keys())
if r.status_code==200:
    for n,d in zip(names, r.json()):
        if not d: print("NULL",n); continue
        arx=(d.get("externalIds") or {}).get("ArXiv"); oa=(d.get("openAccessPdf") or {}).get("url")
        ok=False
        if arx: ok=dl(f"https://arxiv.org/pdf/{arx}.pdf",f"papers/{n}.pdf")
        if not ok and oa: ok=dl(oa,f"papers/{n}.pdf")
        print(("OK   " if ok else "NOPDF"), n, "arxiv=",arx,"oa=",bool(oa))
else:
    print(r.text[:300])
