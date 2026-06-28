import subprocess, json, sys, re
q = sys.argv[1]
mode = sys.argv[2] if len(sys.argv)>2 else "fast"
out = subprocess.run([sys.executable, ".claude/skills/paper-finder/scripts/find_papers.py", q, "--mode", mode, "--format","json"], capture_output=True, text=True, timeout=300).stdout
# Extract the JSON object (from first { to matching, before "Results saved")
idx = out.find("{")
# cut at "Results saved" if present
cut = out.find("\nResults saved")
js = out[idx:cut] if cut!=-1 else out[idx:]
try:
    d = json.loads(js)
except Exception as e:
    print("PARSE_FAIL", e); print(out[:300]); sys.exit(1)
papers = d.get("papers", [])
print(f"QUERY: {q} | total={d.get('total')} | returned={len(papers)}")
for p in papers:
    print(f"  [{p.get('relevance')}] {p.get('year')} | {p.get('title')[:90]} | {p.get('url')}")
# save full
import os
os.makedirs("paper_search_results/full", exist_ok=True)
fn = "paper_search_results/full/" + re.sub(r'[^a-z0-9]+','_', q.lower())[:60] + ".json"
json.dump(d, open(fn,"w"), indent=2)
