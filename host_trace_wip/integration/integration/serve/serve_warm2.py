# serve_warm.py with no client connection limit (aiohttp caps a session at 100, which capped decode bs and stalled
# mixed nd 128/255 behind 8000-token decoders) and 2000-token decoders. Range warm-up of a running `vllm serve` over HTTP (phase 6 step 5): rangesweep.py's schedule as requests. Decode bs
# 1..256 (bs concurrent 16-token prompts, 3 tokens each), single and multi-request prefills, then mixed steps: nd
# streaming decoders plus one new prompt of each length. The server batches on its own, so the steps are close to,
# not exactly, rangesweep's; what matters is that every trace class is hit.
#   python bench/serve_warm.py --port 18731 [--dec-bs ...]
import argparse, asyncio, json, random, time

import aiohttp

p = argparse.ArgumentParser()
p.add_argument("--port", type=int, default=18731)
p.add_argument("--model", default="/data/eellison/models/Qwen3-8B")
p.add_argument("--dec-bs", type=int, nargs="*", default=list(range(1, 257)))
p.add_argument("--pre-lens", type=int, nargs="*", default=[1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 255, 256, 257,
                                                           511, 512, 513, 1000, 1023, 1024, 1025, 2047, 2048, 2049, 3000, 4095, 4096, 4097, 6000, 8191, 8192])
p.add_argument("--pre-multi", type=str, nargs="*", default=["2x8", "2x100", "3x500", "8x64", "16x300", "64x100", "128x64", "2x4096"])
p.add_argument("--mixed-nd", type=int, nargs="*", default=[1, 2, 3, 8, 16, 17, 32, 33, 64, 128, 255])
p.add_argument("--mixed-lens", type=int, nargs="*", default=[1, 2, 7, 16, 17, 40, 128, 500, 1000, 4000, 8192])
a = p.parse_args()
URL = f"http://127.0.0.1:{a.port}/v1/completions"
rng = random.Random(0)


def prompt(n):
    return [rng.randrange(1000, 100000) for _ in range(n)]


async def complete(s, n, max_tokens):
    body = {"model": a.model, "prompt": prompt(n), "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True}
    for attempt in range(3):  # sr3: one connection reset by peer at a 200+ request burst
        try:
            async with s.post(URL, json=body) as r:
                await r.read()
            return
        except (aiohttp.ClientError, asyncio.TimeoutError):  # serve scratch: any client error (a disconnect too) is retried, then skipped
            if attempt == 2:
                return


async def decoder(s, started, stop):
    body = {"model": a.model, "prompt": prompt(8), "max_tokens": 2000, "temperature": 0, "ignore_eos": True, "stream": True}
    async with s.post(URL, json=body) as r:
        async for _ in r.content:
            started.set()
            if stop.is_set():
                break


async def main():
    t = time.perf_counter()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600), connector=aiohttp.TCPConnector(limit=0)) as s:
        for bs in a.dec_bs:
            await asyncio.gather(*(complete(s, 16, 3) for _ in range(bs)))
        t_dec = time.perf_counter()
        for n in a.pre_lens:
            await complete(s, n, 1)
        for spec in a.pre_multi:
            k, n = map(int, spec.split("x"))
            await asyncio.gather(*(complete(s, n, 1) for _ in range(k)))
        t_pre = time.perf_counter()
        for nd in a.mixed_nd:
            stop, starts = asyncio.Event(), [asyncio.Event() for _ in range(nd)]
            tasks = [asyncio.create_task(decoder(s, e, stop)) for e in starts]
            await asyncio.gather(*(e.wait() for e in starts))
            for n in a.mixed_lens:
                await complete(s, n, 1)
            stop.set()
            for x in tasks:
                x.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        t_mix = time.perf_counter()
    print(json.dumps({"warm_s": {"decode": round(t_dec - t, 1), "prefill": round(t_pre - t_dec, 1), "mixed": round(t_mix - t_pre, 1)}}), flush=True)


asyncio.run(main())
