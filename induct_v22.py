#!/usr/bin/env python3
"""
v22: the airfoil STAIRCASE — does the scaffold's payoff scale with the model's
ability to PERCEIVE the grid? (Alex's framing: "every staircase needs a first
step"; "if we could make it easier for a 360M to perceive the grid and name the
right concept, IMAGINE what that does for a 60B.")

Two axes, on real ARC (400 training tasks):

  CAPABILITY (model size, the staircase):
    SmolLM2-135M  ->  SmolLM2-360M  ->  Qwen3-1.7B-PMRA  (Alex's own frankentensor quant)

  PERCEPTION (how we present the grid to the model):
    raw    = just the digit grid
    struct = digit grid + a cue header (sizes, size-ratio, color sets, symmetry)
             — a cheap "perception prosthetic": surface the exact features that
             map to concept categories, so the model need only NAME, not perceive.

Each model is measured three ways:
  - ALONE      : asked to emit the output grid directly (no scaffold)        -> floor
  - +AIRFOIL   : names concept CATEGORIES; airfoil composes those ops + recolor,
                 under an EQUAL-COMPUTE node budget (narrowing buys depth) -> scaffolded
  (and airfoil with NO llm = blind search over the full DSL = 6.5%, from v20).

The mechanism that lets breadth ADD over blind search: solve runs under a fixed
node budget. Blind full-DSL search spends it on depth-2 over 15 ops (240 progs);
a model that correctly narrows to ~6 ops spends the SAME budget reaching depth-3.
Breadth (which ops) -> affordable depth. That is the airfoil thesis.

Run with /data/llm/.venv/bin/python.
"""
import gc
import glob
import itertools
import json
import re
import time
from statistics import mean

import induct_v20 as A   # ops, applyops, infer_colormap, grid, predict

TASKS = sorted(glob.glob("/data/arc/data/training/*.json"))
MODELS = [
    # 135M dropped: measured net-negative (combo raw 1.5% < 6.5% blind floor; the bug
    # only slowed it, didn't depress the score) — a foregone-conclusion rung.
    ("360M",  "/data/Windows-files/Documents/frontier-lab/models/SmolLM2-360M-Instruct-q8_0.gguf", False),
    ("1.7B",  "/data/llm/models/qwen3-1.7b-pmra.gguf",            True),  # qwen3 -> /no_think
]
CAT_OPS = {
    "geometry": ["rot90", "rot180", "rot270", "flip_h", "flip_v", "transpose"],
    "tile":     ["tile_h", "tile_v", "mirror_h", "mirror_v"],
    "scale":    ["scale2", "scale3"],
    "shrink":   ["crop", "downscale2", "downscale3"],
}
ALL_OPS = [o for ops in CAT_OPS.values() for o in ops]
NODE_BUDGET = 256   # ~ full-DSL depth-2 (15+225); narrowing converts this to depth

_LLM = None
_QWEN = False


def load(path, is_qwen):
    global _LLM, _QWEN
    from llama_cpp import Llama
    _LLM = Llama(model_path=path, n_ctx=4096, n_threads=6, verbose=False)
    _QWEN = is_qwen


def unload():
    global _LLM
    _LLM = None
    gc.collect()


def gen(prompt, n):
    try:
        return _LLM(prompt, max_tokens=n, temperature=0.2, stop=["<|im_end|>"])["choices"][0]["text"]
    except ValueError:        # prompt longer than ctx -> treat as no perception
        return ""


def chat(user, n):
    tag = "/no_think\n" if _QWEN else ""
    p = f"<|im_start|>user\n{tag}{user}<|im_end|>\n<|im_start|>assistant\n"
    txt = gen(p, n)
    return re.sub(r"<think>.*?</think>", "", txt, flags=re.S)   # strip any qwen3 reasoning


# ---------- perception: two renderings ----------
def render(g):
    return "\n".join("".join(str(v) for v in r) for r in g)


def cues(gin, gout):
    ri, ci = len(gin), len(gin[0])
    ro, co = len(gout), len(gout[0])
    ratio = (ro * co) / (ri * ci)
    ci_set = sorted({v for r in gin for v in r})
    co_set = sorted({v for r in gout for v in r})
    sym = []
    if gin == A.flip_h(gin): sym.append("L-R")
    if gin == A.flip_v(gin): sym.append("U-D")
    return (f"in {ri}x{ci} colors{ci_set} sym[{','.join(sym) or '-'}]  ->  "
            f"out {ro}x{co} colors{co_set}  area x{ratio:.2g}")


def show(train, mode, k=2):
    out = []
    for gin, gout in train[:k]:
        head = cues(gin, gout) + "\n" if mode == "struct" else ""
        out.append(f"{head}IN:\n{render(gin)}\nOUT:\n{render(gout)}")
    return "\n".join(out)


# ---------- recognition (breadth) ----------
def llm_recognize(train, mode):
    u = ("A colored-grid puzzle (digits are colors). Categories that can transform "
         "input->output: geometry (rotate/flip), tile (repeat/mirror), scale "
         "(enlarge), shrink (crop/shrink), recolor (change colors only). Which "
         "categories are at work here? Reply with just the category words.\n"
         + show(train, mode))
    txt = chat(u, 40).lower()
    ops = []
    for c in CAT_OPS:
        if c in txt:
            ops += CAT_OPS[c]
    return list(dict.fromkeys(ops))


# ---------- depth from breadth: equal-compute budgeted search ----------
MAXCELLS = 2500   # ARC outputs are <= 30x30 = 900; any bigger intermediate can't
                  # be the answer -> prune (also stops growth-op size explosions).
MAXDEPTH = 6      # hard backstop so a tiny op-set can't run to pathological depth.


def apply_guarded(prog, g):
    try:
        for op in prog:
            g = A.OPS[op](g)
            if not g or not g[0] or len(g) * len(g[0]) > MAXCELLS:
                return None
        return g
    except Exception:          # ops can raise on ill-shaped grids (e.g. downscale)
        return None


def solve_budget(train, opset, budget=NODE_BUDGET):
    if not opset:
        return None
    ins = [i for i, _ in train]
    outs = [o for _, o in train]
    tried = 0
    for d in range(1, MAXDEPTH + 1):
        if tried >= budget:
            break
        for prog in itertools.product(opset, repeat=d):
            tried += 1
            if tried > budget:
                break
            trans = [apply_guarded(prog, i) for i in ins]
            if any(t is None for t in trans):
                continue
            if all(t == o for t, o in zip(trans, outs)):
                return (prog, None)
            if all(A.same_shape(t, o) for t, o in zip(trans, outs)):
                m = A.infer_colormap(list(zip(trans, outs)))
                if m and all(A.apply_map(t, m) == o for t, o in zip(trans, outs)):
                    return (prog, m)
        if len(opset) ** (d + 1) > budget:   # next depth unaffordable -> done
            break
    return None


def llm_solve_direct(train, test_in, mode):
    u = ("Solve the colored-grid puzzle (digits are colors). Give ONLY the output "
         "grid (rows of digits) for the last input.\n"
         + show(train, mode) + f"\nLAST IN:\n{render(test_in)}\nLAST OUT:")
    txt = chat(u, 220)
    rows = [re.sub(r"[^0-9]", "", ln) for ln in txt.strip().splitlines()]
    rows = [r for r in rows if r]
    try:
        return tuple(tuple(int(c) for c in r) for r in rows) if rows else None
    except Exception:
        return None


# ---------- arms ----------
def arm_combo(mode, limit=None):
    tasks = TASKS if limit is None else TASKS[:limit]
    solved = named = 0
    for f in tasks:
        t = json.load(open(f))
        train = [(A.grid(p["input"]), A.grid(p["output"])) for p in t["train"]]
        ops = llm_recognize(train, mode)
        named += bool(ops)
        sol = solve_budget(train, ops)
        if sol and all(A.predict(sol, A.grid(tt["input"])) == A.grid(tt["output"]) for tt in t["test"]):
            solved += 1
    n = len(tasks)
    return solved, n, named


def arm_alone(mode, sample):
    ok = 0
    for f in sample:
        t = json.load(open(f))
        train = [(A.grid(p["input"]), A.grid(p["output"])) for p in t["train"]]
        pred = llm_solve_direct(train, A.grid(t["test"][0]["input"]), mode)
        if pred == A.grid(t["test"][0]["output"]):
            ok += 1
    return ok, len(sample)


def main():
    sample = TASKS[::13][:30]
    print("=" * 80)
    print("v22  THE AIRFOIL STAIRCASE — does scaffold payoff scale with perception?")
    print("=" * 80)
    print(f"  {len(TASKS)} ARC tasks; node budget {NODE_BUDGET}; airfoil-alone (no LLM) = 6.5% (v20)\n")
    flush = lambda *a: print(*a, flush=True)

    for name, path, is_qwen in MODELS:
        t0 = time.time()
        flush(f"\n----- {name} -----")
        load(path, is_qwen)
        # capability staircase uses the perception prosthetic (struct);
        # perception ablation (raw vs struct) on the combo arm too.
        a_ok, a_n = arm_alone("struct", sample)
        flush(f"  {name} ALONE (direct, struct)   : {a_ok}/{a_n} = {a_ok/a_n*100:.1f}%  (sample)")
        for mode in ("raw", "struct"):
            s, n, named = arm_combo(mode, limit=None)
            flush(f"  {name} +AIRFOIL ({mode:<6})        : {s}/{n} = {s/n*100:.1f}%   "
                  f"(named>=1 on {named}/{n})")
        unload()
        flush(f"  ({name} done in {time.time()-t0:.0f}s)")

    flush("\n" + "=" * 80)
    flush("RESULT")
    flush("=" * 80)
    flush("  Read top-to-bottom: ALONE (no scaffold) vs +AIRFOIL (scaffolded), per size.")
    flush("  raw vs struct = does easing PERCEPTION lift recognition, and how does that")
    flush("  interact with model size? Compare every +AIRFOIL to the 6.5% blind-search floor.")
    flush("  [honest interpretation written to LOG after reading the numbers]")


if __name__ == "__main__":
    main()
