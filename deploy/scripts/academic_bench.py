#!/usr/bin/env python3
import subprocess, json, time, statistics
print("=== WARMUP (2 runs discarded) ===")
for i in range(2):
    subprocess.run(["curl","-s","-m","30","http://localhost:9999/v1/chat/completions",
                    "-H","Content-Type: application/json",
                    "-d",'{"model":"qwen3","messages":[{"role":"user","content":"warmup"}],"max_tokens":10}'],
                   capture_output=True, text=True)
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
        tps = tokens / dur
        runs.append({"dur": dur, "tokens": tokens, "tps": tps})
        print(f"Run {i+1:2d}: {dur:.2f}s | gen={tokens} | {tps:.2f} tok/s")
    except Exception as e:
        print(f"Run {i+1}: ERROR {e}")
tps_vals = sorted([r["tps"] for r in runs])
print(f"=== STATISTICS (n={len(tps_vals)}) ===")
print(f"mean    : {statistics.mean(tps_vals):.3f}")
print(f"median  : {statistics.median(tps_vals):.3f}")
print(f"stdev   : {statistics.stdev(tps_vals):.3f}")
ci = 1.96 * statistics.stdev(tps_vals) / (len(tps_vals)**0.5)
print(f"95% CI +/- : {ci:.3f}")
