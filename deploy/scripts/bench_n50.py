#!/usr/bin/env python3
import subprocess, json, time, statistics, math
print("=== WARMUP (3 runs discarded) ===")
for i in range(3):
    subprocess.run(["curl","-s","-m","60","http://localhost:9999/v1/chat/completions",
                    "-H","Content-Type: application/json",
                    "-d",'{"model":"q","messages":[{"role":"user","content":"warmup"}],"max_tokens":50}'],
                   capture_output=True, text=True)
print("=== 50 measurement runs ===")
runs = []
for i in range(50):
    start = time.time()
    r = subprocess.run(["curl","-s","-m","30","http://localhost:9999/v1/chat/completions",
                        "-H","Content-Type: application/json",
                        "-d",'{"model":"q","messages":[{"role":"user","content":"Write 200 words about distributed computing"}],"max_tokens":250,"temperature":0}'],
                       capture_output=True, text=True)
    dur = time.time() - start
    try:
        d = json.loads(r.stdout); tokens = d["usage"]["completion_tokens"]; tps = tokens/dur
        runs.append(tps); print(f"Run {i+1:2d}: {tps:.3f}")
    except: print(f"Run {i+1}: ERROR")
print(f"\nn={len(runs)} mean={statistics.mean(runs):.4f} median={statistics.median(runs):.4f} stdev={statistics.stdev(runs):.4f}")
ci=1.96*statistics.stdev(runs)/math.sqrt(len(runs))
print(f"95% CI +/- {ci:.4f}")
print(f"min={min(runs):.4f} max={max(runs):.4f}")
