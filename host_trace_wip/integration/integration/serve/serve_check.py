# Fixed prompt set against a running `vllm serve`: token-id prompts (seeded), temperature 0, ignore_eos, and the output
# token ids (return_token_ids) per prompt. --conc 1 sends them one at a time (the same schedule in every arm: batch 1
# prefill then batch 1 decode); --conc K keeps K in flight (batching depends on step timing, so arms may differ).
#   python serve_check.py --port P --out X.json [--conc 1] [--n 24] [--max-tokens 64]
#   python serve_check.py --compare A.json B.json
import argparse
import asyncio
import json
import random
import sys

LENS = [1, 5, 16, 17, 33, 100, 129, 257, 500, 1000, 1025, 2048, 3000, 4097, 6000, 8000]


def compare(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    same, rows = 0, []
    for i, (x, y) in enumerate(zip(A["outputs"], B["outputs"])):
        if x == y:
            same += 1
            continue
        d = next((j for j, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
        rows.append({"prompt": i, "len": A["lens"][i], "first_diff_token": d})
    return {"a": a, "b": b, "equal": same, "n": min(len(A["outputs"]), len(B["outputs"])), "diffs": rows}


async def main(a):
    import aiohttp

    rng = random.Random(0)
    lens = [LENS[i % len(LENS)] for i in range(a.n)]
    prompts = [[rng.randrange(1000, 100000) for _ in range(n)] for n in lens]
    url = f"http://127.0.0.1:{a.port}/v1/completions"
    outs = [None] * a.n
    sem = asyncio.Semaphore(a.conc)

    async def one(s, i):
        body = {"model": a.model, "prompt": prompts[i], "max_tokens": a.max_tokens, "temperature": 0, "ignore_eos": True, "return_token_ids": True}
        async with sem, s.post(url, json=body) as r:
            res = await r.json()
            outs[i] = res["choices"][0]["token_ids"]

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600), connector=aiohttp.TCPConnector(limit=0)) as s:
        await asyncio.gather(*(one(s, i) for i in range(a.n)))
    json.dump({"conc": a.conc, "max_tokens": a.max_tokens, "lens": lens, "outputs": outs}, open(a.out, "w"))
    print("serve_check", a.out, "n", a.n, "conc", a.conc, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=18741)
    p.add_argument("--model", default="/data/eellison/models/Qwen3-8B")
    p.add_argument("--out")
    p.add_argument("--conc", type=int, default=1)
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--compare", nargs=2)
    a = p.parse_args()
    if a.compare:
        print(json.dumps(compare(*a.compare)))
        sys.exit(0)
    asyncio.run(main(a))
