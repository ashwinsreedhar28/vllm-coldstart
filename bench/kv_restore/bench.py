"""Runs on the pod: send labelled requests to a local vLLM server, record TTFT, cached tokens and offload metrics.

    python3 bench.py --model Qwen/Qwen3-8B --label net-s1 --plan warm,A8,A8,A30,A30 --out /w/out/kv.jsonl --state /workspace/replies.json

Plan items: `warm` (short prompt), `<L><k>` (seed letter L, about k thousand prompt tokens),
`X:<L><k>` (that prompt, its stored reply, and one more user turn).
"""
import argparse, json, os, random, time, urllib.request

WORDS = ("alpha bravo cedar delta ember fjord grove harbor island jasper kettle lumen meadow nectar onyx prairie quartz "
         "river summit timber umbra valley willow xenon yarrow zephyr anchor beacon copper dune echo falcon garnet").split()


def prompt(tok, seed: str, k: int) -> str:
    rng = random.Random(f"kv-restore-{seed}")
    n = k * 1000
    text = f"Session {seed}. Log follows. " + " ".join(f"{i}:{rng.choice(WORDS)}-{rng.randint(0, 9999)}" for i in range(n))
    ids = tok.encode(text, add_special_tokens=False)[:n]
    return tok.decode(ids) + "\n\nReply with the single word OK."


def call(base, key, model, messages):
    body = {"model": model, "messages": messages, "max_tokens": 16, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t0 = time.time(); ttft = None; text = ""; usage = {}
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices") or []:
                piece = (c.get("delta") or {}).get("content") or (c.get("delta") or {}).get("reasoning_content") or ""
                if ttft is None:
                    ttft = time.time() - t0
                text += piece
    return {"ttft_s": round(ttft or -1, 3), "total_s": round(time.time() - t0, 3), "text": text,
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens")}


def metrics(base, key):
    req = urllib.request.Request(base + "/metrics", headers={"Authorization": f"Bearer {key}"})
    out = {}
    for line in urllib.request.urlopen(req, timeout=30).read().decode().splitlines():
        if line.startswith("#") or "_bucket" in line or "_created" in line:
            continue
        if any(s in line for s in ("offload", "prefix_cache", "kv_")):
            name, _, val = line.rpartition(" ")
            out[name[:160]] = float(val)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8001"); ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True); ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--state", required=True)
    a = ap.parse_args()
    key = os.environ.get("VLLM_API_KEY", "")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    state = json.load(open(a.state)) if os.path.exists(a.state) else {}
    for item in a.plan.split(","):
        ext = item.startswith("X:")
        name = item[2:] if ext else item
        if name == "warm":
            msgs = [{"role": "user", "content": "Reply with the single word OK."}]
        else:
            msgs = [{"role": "user", "content": prompt(tok, name[0], int(name[1:]))}]
            if ext:
                msgs += [{"role": "assistant", "content": state.get(f"{a.model}/{name}", "OK")},
                         {"role": "user", "content": "Now reply with the single word DONE."}]
        try:
            row = call(a.base, key, a.model, msgs)
        except Exception as e:  # keep going: a failed item is a result too
            row = {"error": repr(e)[:300]}
        if name != "warm" and not ext and "text" in row:
            state.setdefault(f"{a.model}/{name}", row["text"])
            row["same_text_as_first"] = row["text"] == state[f"{a.model}/{name}"]
        row.update(label=a.label, model=a.model, item=item, t=time.time())
        try:
            row["metrics"] = metrics(a.base, key)
        except Exception as e:
            row["metrics_error"] = repr(e)[:200]
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print({k: v for k, v in row.items() if k != "metrics"}, flush=True)
        json.dump(state, open(a.state, "w"))


if __name__ == "__main__":
    main()
