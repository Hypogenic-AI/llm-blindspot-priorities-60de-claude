"""OpenRouter API client with on-disk caching, retry/backoff, and concurrency.

Caching is keyed on the full request (model, messages, params) so reruns are free
and reproducible. gpt-5 / o-series models require `max_completion_tokens` and a
reasoning-token budget, handled here.
"""
import os, json, hashlib, time, urllib.request, urllib.error, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

CACHE_DIR = "results/cache"
os.makedirs(CACHE_DIR, exist_ok=True)
_KEY = os.environ["OPENROUTER_KEY"]
_URL = "https://openrouter.ai/api/v1/chat/completions"
_lock = threading.Lock()

# Models that use the reasoning API contract (max_completion_tokens, fixed temp).
_REASONING = ("openai/gpt-5", "openai/o1", "openai/o3", "openai/o4")

def _cache_path(payload):
    h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
    return os.path.join(CACHE_DIR, h + ".json")

def chat(model, messages, temperature=0.7, max_tokens=900, seed=None, _retries=6):
    """Return (text, meta). Cached. Returns text='' on persistent failure."""
    payload = {"model": model, "messages": messages, "temperature": temperature}
    is_reasoning = any(model.startswith(p) for p in _REASONING)
    if is_reasoning:
        # reasoning models: temperature fixed at 1, need room for reasoning tokens
        payload["temperature"] = 1
        payload["max_completion_tokens"] = max(max_tokens, 4000)
    else:
        payload["max_tokens"] = max_tokens
    if seed is not None:
        payload["seed"] = seed

    cp = _cache_path(payload)
    if os.path.exists(cp):
        c = json.load(open(cp))
        return c["text"], c.get("meta", {})

    last_err = None
    for attempt in range(_retries):
        try:
            req = urllib.request.Request(
                _URL, data=json.dumps(payload).encode(),
                headers={"Authorization": f"Bearer {_KEY}",
                         "Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=180))
            msg = r["choices"][0]["message"]
            text = msg.get("content") or ""
            meta = {"model_resolved": r.get("model"),
                    "finish": r["choices"][0].get("finish_reason"),
                    "usage": r.get("usage", {})}
            with _lock:
                json.dump({"text": text, "meta": meta}, open(cp, "w"))
            return text, meta
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:200]
            last_err = f"HTTP {e.code}: {body}"
            # 429/5xx -> backoff; 4xx other -> short retry then give up
            time.sleep(min(2 ** attempt, 30))
        except Exception as e:
            last_err = str(e)[:200]
            time.sleep(min(2 ** attempt, 30))
    print(f"  [api] FAILED {model}: {last_err}")
    return "", {"error": last_err}

def run_concurrent(tasks, fn, max_workers=8, label=""):
    """tasks: list of arg-tuples; fn(*args)->result. Returns list aligned to tasks."""
    results = [None] * len(tasks)
    done = [0]
    def _wrap(i, args):
        return i, fn(*args)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_wrap, i, t) for i, t in enumerate(tasks)]
        for fut in as_completed(futs):
            i, res = fut.result()
            results[i] = res
            done[0] += 1
            if done[0] % 20 == 0 or done[0] == len(tasks):
                print(f"  [{label}] {done[0]}/{len(tasks)}")
    return results
