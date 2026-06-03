#!/usr/bin/env python3
"""ARC grounding step 0 — de-risk the CPU LLM runtime BEFORE building the DSL/proposer.
Loads a small GGUF via llama_cpp, times load + a short structured generation (an ARC-style 'name the
transformation' prompt). Gates feasibility: search-with-LLM-proposer needs usable tokens/sec on this CPU.
Run with /data/llm/.venv/bin/python."""
import time, sys
from llama_cpp import Llama

MODEL = "/data/pmra-runs/smoke-local/ggufs/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
f = lambda *a: print(*a, flush=True)

t0 = time.time()
llm = Llama(model_path=MODEL, n_ctx=2048, n_threads=4, verbose=False)
f(f"load: {time.time()-t0:.1f}s   model={MODEL.split('/')[-1]}")

prompt = (
    "You are solving an abstract grid puzzle. A grid is a list of rows of integers 0-9 (colors).\n"
    "Train example:\n"
    "INPUT:  [[0,0,0],[0,5,0],[0,0,0]]\n"
    "OUTPUT: [[0,0,0],[0,2,0],[0,0,0]]\n"
    "In one short sentence, what transformation maps INPUT to OUTPUT?"
)
msgs = [{"role": "user", "content": prompt}]
t1 = time.time()
out = llm.create_chat_completion(messages=msgs, max_tokens=80, temperature=0.7)
dt = time.time() - t1
txt = out["choices"][0]["message"]["content"]
ntok = out["usage"]["completion_tokens"]
f(f"gen: {dt:.1f}s for {ntok} tok = {ntok/dt:.1f} tok/s")
f(f"reply: {txt.strip()[:300]}")
f("OK" if "recolor" in txt.lower() or "color" in txt.lower() or "2" in txt else "ran (semantics tbd)")
