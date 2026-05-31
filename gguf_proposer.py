#!/usr/bin/env python3
"""
gguf_proposer.py — STAGED (v18b): a frozen small GGUF as an OUTER-loop proposer.

The inner-loop proposer (v18) is a KB-sized learned policy queried thousands of
times per task — it must be ~microseconds/call. A GGUF (~72 tok/s on this CPU) is
far too slow for that hot path. But called SPARINGLY — a handful of times per task,
to suggest a high-level decomposition / which top ops to try first — its latency
amortizes, and it brings prior knowledge a bigram can't.

This module is the plumbing for that arm: load the local SmolLM2-360M GGUF, hand it
a task's I/O examples + the op vocabulary, and parse a proposed op sequence to bias
the search. Whether its suggestions actually beat the learned policy on our
synthetic tasks is the v18b MEASUREMENT (next) — and the honest prior is that a
360M model knows nothing about this bespoke DSL, so it may add little here; the
real payoff is expected on the eventual ARC arm, where an LLM can read structure.

Run `python3 gguf_proposer.py` for a smoke test (loads the model, makes a proposal).
"""
import os
import re

MODEL = "/data/llm/models/smollm2-360m-q8.gguf"
_LLM = None


def _model():
    global _LLM
    if _LLM is None:
        from llama_cpp import Llama
        _LLM = Llama(model_path=MODEL, n_ctx=1024, verbose=False)
    return _LLM


def propose(examples, ops, k=5):
    """examples: list of (input, output) strings;  ops: allowed op names.
    Returns a list of op-name guesses (filtered to `ops`), to seed/bias search."""
    ex = "\n".join(f"  {i} -> {o}" for i, o in examples[:6])
    prompt = (
        "<|im_start|>user\n"
        f"Available operations: {', '.join(ops)}.\n"
        "Each maps an input to an output. Given these examples, list the operations "
        "(space-separated, in order) that most likely produced them. Output only the op names.\n"
        f"Examples:\n{ex}<|im_end|>\n<|im_start|>assistant\n"
    )
    out = _model()(prompt, max_tokens=40, temperature=0.4, stop=["<|im_end|>"])
    text = out["choices"][0]["text"]
    toks = re.findall(r"[A-Za-z_]\w*\*?", text)
    guess = [t for t in toks if t in set(ops)]
    return guess[:k], text.strip()


def _smoke():
    if not os.path.exists(MODEL):
        print(f"SMOKE: model missing at {MODEL} — staged but not runnable here.")
        return
    ops = ["inc", "dec", "dbl", "tpl", "sqr", "neg"]
    examples = [("1", "4"), ("2", "6"), ("3", "8"), ("0", "2")]   # x -> 2x+2  (dbl then... )
    import time
    t = time.time()
    guess, raw = propose(examples, ops)
    print(f"SMOKE ok ({time.time()-t:.1f}s). raw: {raw!r}")
    print(f"parsed proposal (filtered to vocab): {guess}")
    print("(integration works; v18b measures whether this guidance helps the search.)")


if __name__ == "__main__":
    _smoke()
