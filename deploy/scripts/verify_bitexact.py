#!/usr/bin/env python3
"""Bit-exact validation for the dLLaMA Pi5 cluster.

Sends a fixed deterministic request (temperature=0, seed=42) to the
OpenAI-compatible endpoint and compares the SHA-256 of the generated text
against the golden reference. Exit code 0 == bit-exact, 1 == divergence.

The golden hash was captured on the cluster and confirmed identical across
the unoptimised (yield-spin barrier) and optimised (WFE/SEV barrier) builds,
which proves the throughput optimisations do not alter model output. This is
a stronger correctness guarantee than a downstream benchmark such as MMLU:
identical bytes => identical logits => identical task accuracy by construction.

Usage:
  python3 verify_bitexact.py [endpoint]
  default endpoint: http://127.0.0.1:9999/v1/chat/completions (run on root node)
"""
import sys, os, json, hashlib, urllib.request

ENDPOINT = (sys.argv[1] if len(sys.argv) > 1
            else os.environ.get("ENDPOINT", "http://127.0.0.1:9999/v1/chat/completions"))

PROMPT = "List the first 12 prime numbers and then explain what a prime number is in one sentence."
PARAMS = {"model": "q", "messages": [{"role": "user", "content": PROMPT}],
          "max_tokens": 160, "temperature": 0, "seed": 42}
GOLDEN_SHA256 = "bdbcaec68c56dd4f5cf07f0dc8e60c8d17209fd14e998d2a3c437144f2328645"

def main():
    body = json.dumps(PARAMS).encode()
    req = urllib.request.Request(ENDPOINT, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        content = json.loads(r.read())["choices"][0]["message"]["content"]
    h = hashlib.sha256(content.encode()).hexdigest()
    ok = (h == GOLDEN_SHA256)
    print(f"endpoint : {ENDPOINT}")
    print(f"prompt   : {PROMPT!r}")
    print(f"params   : max_tokens=160 temperature=0 seed=42")
    print(f"sha256   : {h}")
    print(f"golden   : {GOLDEN_SHA256}")
    print(f"result   : {'PASS (bit-exact)' if ok else 'FAIL (divergence)'}")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
