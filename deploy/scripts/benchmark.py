#!/usr/bin/env python3
"""Academic benchmark: 20 runs with statistics."""
import subprocess, json, time, statistics

print("=== WARMUP (2 runs discarded) ===")
for i in range(2):
    subprocess.run(["curl","-s","-m","30","http://localhost:9999/v1/chat/completions",
                    "-H","Content-Type: application/json",
                    "-d",'{"model":"qwen3","messages":[{"role":"user","content":"warmup"}],"max_tokens":10}'],
                   capture_output=True, text=True)

print()
print("=== 20 measurement runs ===")
runs = []
for i in range(20):
    start = time.time()
    r = subprocess.run(["curl","-s","-m","30","http://localhost:9999/v1/chat/completions",
                        "-H","Content-Type: application/json",
                        "-d",'{"model":"qwen3","messages":[{"role":"user","content":"Write 200 words about distributed computing"}],"max_tokens":250,"temperature":0}'],
                       capture_output=True, text=True)
    dur = time.time() - start
    try:
        d = json.loads(r.stdout)
        tokens = d["usage"]["completion_tokens"]
        prompt = d["usage"]["prompt_tokens"]
        tps = tokens / dur
        runs.append({"dur": dur, "tokens": tokens, "prompt": prompt, "tps": tps})
        print(f"Run {i+1:2d}: {dur:.2f}s | prompt={prompt} | gen={tokens} | {tps:.2f} tok/s")
    except Exception as e:
        print(f"Run {i+1}: ERROR {e}")

tps_vals = sorted([r["tps"] for r in runs])
dur_vals = sorted([r["dur"] for r in runs])

print()
print(f"=== STATISTICS (n={len(tps_vals)} after warmup) ===")
print(f"tok/s mean    : {statistics.mean(tps_vals):.3f}")
print(f"tok/s median  : {statistics.median(tps_vals):.3f}")
print(f"tok/s stdev   : {statistics.stdev(tps_vals):.3f}")
print(f"tok/s min     : {min(tps_vals):.3f}")
print(f"tok/s max     : {max(tps_vals):.3f}")
print(f"tok/s p50     : {tps_vals[len(tps_vals)//2]:.3f}")
print(f"tok/s p90     : {tps_vals[int(len(tps_vals)*0.9)]:.3f}")
print(f"tok/s p99     : {tps_vals[-1]:.3f}")
print(f"latency mean s: {statistics.mean(dur_vals):.3f}")
print(f"latency p50 s : {dur_vals[len(dur_vals)//2]:.3f}")
print(f"latency p99 s : {dur_vals[-1]:.3f}")
ci = 1.96 * statistics.stdev(tps_vals) / (len(tps_vals)**0.5)
print(f"95% CI ±      : {ci:.3f} tok/s")
