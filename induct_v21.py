#!/usr/bin/env python3
"""
v21: how much does the airfoil scaffold buy a small LLM? (and vice versa)

Alex's reframe: a 360M has TREMENDOUS breadth in its data — it knows "symmetry",
"tiling", "recolor", "crop" as concepts — it just can't EXECUTE a precise ARC
solution single-pass. The question isn't "is it a good recognizer", it's "does
airfoil's scaffold (search composes, verifier filters) let its latent breadth get
APPLIED?" The LLM need only be a noisy-but-broad proposer; search + verification
turn signal-amid-noise into a verified-correct answer.

Three numbers on real ARC:
  - LLM ALONE  : 360M asked to produce the output grid directly (expected ≈0).
  - airfoil ALONE: full geometric+recolor DSL, blind search (v20: 6.5%).
  - LLM + AIRFOIL: 360M proposes concept-CATEGORIES (the easiest way to apply its
                   breadth); airfoil composes only those ops + recolor, verifies.
0 -> X is how much the scaffold buys the LLM; X vs 6.5 is whether breadth ADDS.
Run with /data/llm/.venv/bin/python.  (Honest prior: a 360M may not PERCEIVE
grids-as-text well enough to apply the right breadth — if so, the bottleneck is
perception/grounding, not depth or breadth-in-principle. We'll measure it.)
"""
import glob
import itertools
import json
import re
from statistics import mean

import induct_v20 as A   # reuse ops, recolor engine, predict, grid

TASKS = sorted(glob.glob("/data/arc/data/training/*.json"))
MODEL = "/data/llm/models/smollm2-360m-q8.gguf"
_LLM = None

CAT_OPS = {
    "geometry": ["rot90", "rot180", "rot270", "flip_h", "flip_v", "transpose"],
    "tile":     ["tile_h", "tile_v", "mirror_h", "mirror_v"],
    "scale":    ["scale2", "scale3"],
    "shrink":   ["crop", "downscale2", "downscale3"],
}


def llm():
    global _LLM
    if _LLM is None:
        from llama_cpp import Llama
        _LLM = Llama(model_path=MODEL, n_ctx=2048, verbose=False)
    return _LLM


def render(g):
    return "\n".join("".join(str(v) for v in r) for r in g)


def gen(prompt, n):
    return llm()(prompt, max_tokens=n, temperature=0.3, stop=["<|im_end|>"])["choices"][0]["text"]


def show_pairs(train, k=2):
    out = []
    for gin, gout in train[:k]:
        out.append(f"in ({len(gin)}x{len(gin[0])}):\n{render(gin)}\nout ({len(gout)}x{len(gout[0])}):\n{render(gout)}")
    return "\n".join(out)


def llm_recognize(train):
    p = ("<|im_start|>user\nA grid puzzle. Categories: geometry (rotate/flip), "
         "tile (repeat/mirror), scale (enlarge), shrink (crop/shrink), recolor "
         "(change colors). Which categories transform input to output here? "
         f"List the category words.\n{show_pairs(train)}<|im_end|>\n<|im_start|>assistant\n")
    txt = gen(p, 30).lower()
    cats = [c for c in CAT_OPS if c in txt]
    ops = []
    for c in cats:
        ops += CAT_OPS[c]
    return list(dict.fromkeys(ops))   # de-dup, keep order; [] if it named nothing


def llm_solve_direct(train, test_in):
    p = ("<|im_start|>user\nSolve the grid puzzle (digits are colors). Give ONLY the "
         f"output grid for the last input.\n{show_pairs(train)}\nlast in:\n{render(test_in)}\n"
         "last out:<|im_end|>\n<|im_start|>assistant\n")
    txt = gen(p, 200)
    rows = [re.sub(r"[^0-9]", "", ln) for ln in txt.strip().splitlines()]
    rows = [r for r in rows if r]
    try:
        return tuple(tuple(int(c) for c in r) for r in rows) if rows else None
    except Exception:
        return None


def main():
    base = [(b,) for b in A.SMALL]  # not used directly; engine takes op-name lists
    # --- LLM + AIRFOIL on all tasks ---
    combo, named_something = 0, 0
    for f in TASKS:
        t = json.load(open(f))
        train = [(A.grid(p["input"]), A.grid(p["output"])) for p in t["train"]]
        ops = llm_recognize(train)
        if ops:
            named_something += 1
        sol = A.solve_one(train, ops) if ops else None
        if sol and all(A.predict(sol, A.grid(tt["input"])) == A.grid(tt["output"]) for tt in t["test"]):
            combo += 1
    n = len(TASKS)

    # --- LLM ALONE on a sample (direct grid output) ---
    sample = TASKS[::13][:30]
    alone = 0
    for f in sample:
        t = json.load(open(f))
        train = [(A.grid(p["input"]), A.grid(p["output"])) for p in t["train"]]
        pred = llm_solve_direct(train, A.grid(t["test"][0]["input"]))
        if pred == A.grid(t["test"][0]["output"]):
            alone += 1

    print("=" * 76)
    print("v21  how much does the airfoil scaffold buy a small (360M) LLM?")
    print("=" * 76)
    print(f"  LLM ALONE (direct grid output): {alone}/{len(sample)} = {alone/len(sample)*100:.1f}%  (sample)")
    print(f"  airfoil ALONE (full DSL, v20) : 6.5%")
    print(f"  LLM + AIRFOIL (combo)         : {combo}/{n} = {combo/n*100:.1f}%")
    print(f"    (the 360M named >=1 category on {named_something}/{n} tasks)")
    print("\n" + "=" * 76)
    print("RESULT")
    print("=" * 76)
    print(f"  airfoil buys the LLM: {alone/len(sample)*100:.0f}% (alone) -> {combo/n*100:.1f}% (scaffolded).")
    print(f"  does the LLM's breadth ADD over blind search (6.5%)? combo {combo/n*100:.1f}% vs 6.5%.")
    print("  [interpretation written to LOG after reading the numbers — honest, not pre-baked]")


if __name__ == "__main__":
    main()
