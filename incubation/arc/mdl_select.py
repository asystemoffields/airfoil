#!/usr/bin/env python3
"""ARC grounding step 4 — does COMPRESSION predict GENERALIZATION? (the airfoil thesis, grounded.)
Step 3 found OVERFITTING: many programs are train-consistent but spurious (fail the held-out test pair).
When several programs fit all train pairs, which should we trust? The airfoil/MDL claim: the SHORTEST
(lowest description length) generalizes best. Test it directly: enumerate ALL train-consistent programs
(length<=3, capped) per task, then compare test-generalization under selection rules:
  shortest (MDL) | longest (anti-MDL) | first-found | random-among-consistent.
If shortest >> longest on test-generalization, compression->generalization holds on ARC, with NO learning,
NO LLM. Run with /data/llm/.venv/bin/python."""
import random, time
import numpy as np
import dsl
from proposer_eval import instantiate

f = lambda *a: print(*a, flush=True)
NTASK = 300
CAP = 4000                                                 # executions per task while collecting consistent programs


def prog_len(p):                                           # description length proxy: ops + integer args
    return sum(1 + len(a) for _, a in p)


def collect_consistent(train, cap, max_len=3):
    insts = instantiate(dsl.palette(train)); ins = [gi for gi, _ in train]; tgt = [go for _, go in train]
    found = []; seen = set(); nexec = 0
    frontier = [([], ins)]
    for L in range(1, max_len + 1):
        nxt = []
        for prog, outs in frontier:
            for inst in insts:
                if nexec >= cap: return found
                try: outs2 = [dsl.OPS[inst[0]][0](g, *inst[1]) for g in outs]
                except Exception: nexec += 1; continue
                nexec += 1
                ok = all(o.shape == t.shape and np.array_equal(o, t) for o, t in zip(outs2, tgt))
                if ok:
                    key = tuple((n, a) for n, a in prog + [inst])
                    if key not in seen: seen.add(key); found.append(prog + [inst])
                else:
                    nxt.append((prog + [inst], outs2))       # only extend NON-solving prefixes
        frontier = nxt
    return found


def generalizes(prog, test):
    for gi, go in test:
        o = dsl.apply_prog(gi, prog)
        if o is None or o.shape != go.shape or not np.array_equal(o, go): return False
    return True


def main():
    tasks = dsl.load_all(dsl.TRAIN_DIR, n=NTASK); rng = random.Random(0)
    f(f"collecting ALL train-consistent programs (length<=3, cap {CAP}/task) over {NTASK} tasks..."); t0 = time.time()
    rules = ["shortest", "longest", "first", "random"]
    hit = {r: 0 for r in rules}; arena = 0; ambig = 0; ambig_hit = {r: 0 for r in rules}
    for tid, train, test in tasks:
        cons = collect_consistent(train, CAP)
        if not cons: continue
        arena += 1
        gen = {tuple((n, a) for n, a in p): generalizes(p, test) for p in cons}
        pick = {
            "shortest": min(cons, key=prog_len),
            "longest": max(cons, key=prog_len),
            "first": cons[0],
            "random": cons[rng.randrange(len(cons))],
        }
        # ambiguous = some consistent program generalizes AND some does not (selection actually matters)
        gvals = list(gen.values()); is_ambig = any(gvals) and not all(gvals)
        if is_ambig: ambig += 1
        for r in rules:
            g = generalizes(pick[r], test)
            if g: hit[r] += 1
            if is_ambig and g: ambig_hit[r] += 1
    f(f"  done in {time.time()-t0:.0f}s\n")
    f(f"arena = {arena} tasks with >=1 train-consistent program; {ambig} are AMBIGUOUS")
    f("(ambiguous = both a generalizing AND a spurious consistent program exist -> the selection rule decides)\n")
    f(f"TEST-GENERALIZATION rate by selection rule:")
    f(f"    rule        all-arena ({arena})      ambiguous-only ({ambig})")
    for r in rules:
        a = f"{hit[r]:2d}/{arena} = {100*hit[r]/max(arena,1):3.0f}%"
        b = f"{ambig_hit[r]:2d}/{ambig} = {100*ambig_hit[r]/max(ambig,1):3.0f}%" if ambig else "n/a"
        f(f"    {r:<10s}  {a:>18s}   {b:>18s}")
    f("\nREAD: if SHORTEST (MDL) > LONGEST on test-generalization (esp. on the AMBIGUOUS tasks where it matters),")
    f("then COMPRESSION PREDICTS GENERALIZATION on ARC — the airfoil thesis, grounded, no learning/LLM. The")
    f("shortest program that fits the train pairs is the one to trust; description length IS the generalization prior.")


if __name__ == "__main__":
    main()
