#!/usr/bin/env python3
"""ARC grounding step 1 — the grounded science test: does an LLM PROPOSER beat RANDOM/ENUMERATIVE proposal
under a fixed verify budget B, on the DSL-solvable arena? (= the toy's 'rich proposer beats coverage', on ARC.)
Each method emits an ordered list of <=B candidate programs; verify in order against the train pairs; a task
is SOLVED if any candidate reproduces ALL train outputs exactly (then we also check it generalizes to test).
Run with /data/llm/.venv/bin/python."""
import json, random, time, re
import numpy as np
import dsl
from itertools import product
from llama_cpp import Llama

f = lambda *a: print(*a, flush=True)
ARC = "/data/Windows-files/Documents/airfoil/incubation/arc"
MODEL = "/data/pmra-runs/smoke-local/ggufs/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
BUDGETS = (5, 15, 40)

OP_DOC = """identity: unchanged
reflect_h: mirror left-right
reflect_v: mirror top-bottom
rot90 / rot180 / rot270: rotate
transpose: flip along main diagonal
tile_h2 / tile_v2 / tile_2x2: duplicate the grid right / down / both
scale2: upscale each cell to 2x2
crop_content: crop to the bounding box of non-zero cells
gravity_down: drop non-zero cells to the bottom
largest_object: keep only the largest connected non-zero blob
fill_holes C: fill enclosed background regions with color C
recolor A B: replace every cell of color A with color B"""


def instantiate(pal):
    insts = []
    for name in dsl.OP_NAMES:
        na = dsl.OPS[name][1]
        if na == 0: insts.append((name, ()))
        elif na == 1:
            for c in pal:
                if c != 0: insts.append((name, (c,)))
        elif na == 2:
            for a in pal:
                for b in pal:
                    if a != b: insts.append((name, (a, b)))
    return insts


def candidate_space(train, max_len=2):
    insts = instantiate(dsl.palette(train))
    space = [[i] for i in insts]
    space += [[a, b] for a, b in product(insts, insts)]
    return space, insts


# ---------- proposers: each returns an ordered list of programs ----------
def propose_random(train, B, rng):
    space, _ = candidate_space(train)
    rng.shuffle(space); return space[:B]


def propose_enum(train, B):
    space, _ = candidate_space(train); return space[:B]                # length-1 first, then length-2


def render(g):
    return "\n".join("".join(str(int(x)) for x in row) for row in g)


def parse_progs(text, valid_ops):
    progs = []
    for line in text.splitlines():
        line = re.sub(r"[^a-z0-9_ ]", " ", line.strip().lower())
        toks = line.split()
        prog = []
        i = 0
        while i < len(toks):
            name = toks[i]
            if name not in valid_ops: i += 1; continue
            na = dsl.OPS[name][1]; args = []
            for k in range(na):
                if i + 1 + k < len(toks) and toks[i + 1 + k].isdigit(): args.append(int(toks[i + 1 + k]))
            if len(args) == na: prog.append((name, tuple(args))); i += 1 + na
            else: i += 1
        if prog: progs.append(prog)
    return progs


def propose_llm(llm, train, B):
    pal = dsl.palette(train)
    # cap examples shown to keep within context (most DSL-solvable tasks are small; large ones get fewer/none)
    shown = [(gi, go) for gi, go in train if gi.size <= 400 and go.size <= 400][:3]
    if not shown: shown = train[:1]
    ex = "\n\n".join(f"INPUT:\n{render(gi)}\nOUTPUT:\n{render(go)}" for gi, go in shown)
    prompt = (
        f"Grid puzzle. Colors present: {pal}. Available operations:\n{OP_DOC}\n\n"
        f"Training examples (each maps INPUT grid -> OUTPUT grid):\n{ex}\n\n"
        f"Propose {B} candidate programs that could transform INPUT into OUTPUT, best first. "
        f"One program per line, ops separated by spaces, args as integers. Example lines:\n"
        f"reflect_h\nrecolor 5 2\nfill_holes 4 crop_content\nOnly output the program lines.")
    try:
        out = llm.create_chat_completion(messages=[{"role": "user", "content": prompt}],
                                         max_tokens=200, temperature=0.6, seed=0)
    except ValueError:
        return []                                          # oversized prompt -> no LLM proposals for this task
    progs = parse_progs(out["choices"][0]["message"]["content"], set(dsl.OP_NAMES))
    # de-dup preserving order
    seen = set(); uniq = []
    for p in progs:
        key = tuple((n, a) for n, a in p)
        if key not in seen: seen.add(key); uniq.append(p)
    return uniq[:B]


def first_solver(progs, train):
    for p in progs:
        if dsl.solves(p, train): return p
    return None


def generalizes(prog, test):
    for gi, go in test:
        out = dsl.apply_prog(gi, prog)
        if out is None or out.shape != go.shape or not np.array_equal(out, go): return False
    return True


def main():
    arena = json.load(open(f"{ARC}/solvable.json"))
    tids = [a["tid"] for a in arena]
    tasks = {tid: (train, test) for tid, train, test in dsl.load_all(dsl.TRAIN_DIR, n=400) if tid in set(tids)}
    f(f"arena = {len(tasks)} DSL-solvable tasks; budgets={BUDGETS}")
    f("loading LLM proposer..."); t0 = time.time(); llm = Llama(model_path=MODEL, n_ctx=4096, n_threads=4, verbose=False)
    f(f"  loaded {MODEL.split('/')[-1]} in {time.time()-t0:.1f}s")

    rng = random.Random(0)
    # precompute LLM proposals once at the max budget, then truncate per budget
    res = {m: {B: {"train": 0, "test": 0} for B in BUDGETS} for m in ("random", "enum", "llm")}
    tg = time.time()
    for tid, (train, test) in tasks.items():
        llm_progs = propose_llm(llm, train, max(BUDGETS))
        for B in BUDGETS:
            cand = {"random": propose_random(train, B, rng), "enum": propose_enum(train, B), "llm": llm_progs[:B]}
            for m, progs in cand.items():
                sol = first_solver(progs, train)
                if sol:
                    res[m][B]["train"] += 1
                    if generalizes(sol, test): res[m][B]["test"] += 1
    f(f"  evaluated arena in {time.time()-tg:.0f}s\n")

    N = len(tasks)
    f(f"SOLVE RATE on {N}-task arena (train-consistent / test-generalizing), by proposer x budget:")
    f("    method     " + "   ".join(f"B={B:<2d}" for B in BUDGETS))
    for m in ("random", "enum", "llm"):
        cells = [f"{res[m][B]['train']:2d}/{res[m][B]['test']:2d}" for B in BUDGETS]
        f(f"    {m:<10s} " + "  ".join(f"{c:>7s}" for c in cells))
    f("\nREAD: if LLM solves MORE at small B than random (and rivals enum's fixed order), the LLM proposer's")
    f("BREADTH (naming relevant ops) buys sample-efficiency on ARC = the toy's rich-proposer result, grounded.")
    f("(enum = exhaustive fixed order: a strong non-learned baseline; LLM beating it at small B = real signal.)")


if __name__ == "__main__":
    main()
