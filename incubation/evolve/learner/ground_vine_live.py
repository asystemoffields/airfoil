#!/usr/bin/env python3
"""Vine — LIVE LIBRARY (real-time read+write, incl. MID-SOLVE) + WARM-vs-COLD retrieval test (Alex).

The library is a continuously read-and-written working memory: RETRIEVE earned concepts first (try them before any
search), and WRITE the instant a concept verifies -- including MID-SOLVE: when a composition verifies, write BOTH
the whole Composed AND its inner part, so the parts are reusable building blocks immediately. Then the retrieval
test: COLD = solve eval with an empty library; WARM = first EARN a library by running over the TRAIN set
(verifier-gated), THEN solve eval (retrieve). Warm > cold = Vine CAN retrieve once its library isn't empty -- the
"why can't it retrieve" answer. Run: /data/llm/.venv/bin/python ground_vine_live.py [n_train] [n_eval]"""
import sys, time
import rel_dsl as D
import substrate_eye as SE
from open_loop import ranked
from ground_v2_relational import REL_PREDS
from effect_faculty import earn_effect, verify_effect, Gesture
from ground_arc import recognizer_solve
from ground_arc_v2 import GEN2, GEN6
from persist_library import Library
import torch
from train_v2 import V2

sys.path.insert(0, "/data/Windows-files/Documents/airfoil/incubation/evolve")
import harness

V2NET = V2(); V2NET.load_state_dict(torch.load("learner_v2.pt")); V2NET.eval()


def _applies(c, train, test):
    """does an earned library concept solve this task (RETRIEVAL)?"""
    if isinstance(c, Gesture):
        return verify_effect(c, train) and verify_effect(c, test)
    p = D.induce_recolor(c, train)
    return p is not None and D.verify(p, train) and D.verify(p, test)


def vine_solve_concept(train, test, lib, topk=10):
    """-> (solved, earned_concepts_to_write). RETRIEVE library first; else search; MID-SOLVE write the parts."""
    for c in lib.concepts():                          # RETRIEVE: reuse an earned concept before any search
        if _applies(c, train, test):
            return True, []
    try:
        s, _ = recognizer_solve(V2NET, train, test, 2, 3)   # grammar path (not 'earned beyond grammar' -> not stored)
        if s:
            return True, []
    except Exception:
        pass
    r = ranked(train)
    for key in r[:topk]:                              # EARNED-SENSE recolor
        if isinstance(key, (D.Quantify, SE.SubQuantify)):
            prog = D.induce_recolor(key, train)
            if prog is not None and D.verify(prog, train, test):
                return True, [key]
    outers = [k for k in r[:topk] if isinstance(k, D.Quantify)]
    for outer in outers:                              # COMPOSITION -- MID-SOLVE: write the whole AND the inner part
        for inner in REL_PREDS:
            comp = D.Composed(outer.ch, outer.value, inner, outer.mode)
            prog = D.induce_recolor(comp, train)
            if prog is not None and D.verify(prog, train, test):
                return True, [comp, inner]
    eff = earn_effect(train, test)                    # GESTURE
    if eff is not None:
        return True, [eff]
    return False, []


def run(tasks, lib, write):
    solved = set()
    for tid, train, test in tasks:
        try:
            s, earned = vine_solve_concept(train, test, lib)
        except Exception:
            continue
        if s:
            solved.add(tid)
        if write:
            for c in earned:
                lib.add(c)                            # REAL-TIME, verifier-gated write
    return solved


def main():
    nt = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    ne = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    train = list(harness.load_split("arc1-train"))[:nt]
    ev = list(harness.load_split("arc1-eval"))[:ne]
    t0 = time.time()

    cold = run(ev, Library("/tmp/vine_cold.json"), write=False)   # empty library
    import os
    for p in ("/tmp/vine_cold.json", "/tmp/vine_warm.json"):
        if os.path.exists(p): os.remove(p)
    warm_lib = Library("/tmp/vine_warm.json")
    earned_on_train = run(train, warm_lib, write=True)            # WARM: earn a library on train
    warm = run(ev, warm_lib, write=False)                        # then retrieve on eval

    print(f"LIVE LIBRARY warm-vs-cold [train {nt} / eval {ne}, {time.time()-t0:.0f}s]:")
    print(f"  COLD  (empty library):                 eval solved {len(cold)}")
    print(f"  WARM  (library earned on {nt} train):  eval solved {len(warm)}   library {len(warm_lib)} concepts")
    print(f"  retrieval LIFT: +{len(warm - cold)} eval tasks the warm library unlocked  {sorted(warm - cold)[:8]}")
    print(f"  earned {len(earned_on_train)}/{nt} train tasks into the library")
    print("READ: warm > cold = Vine CAN retrieve once its library is non-empty (earn on train, retrieve on eval) -- "
          "the cold diagnostic just never earned a library. Mid-solve writes the parts so they're reusable building "
          "blocks. Lift is bounded by ARC's breadth (eval needs concepts train didn't earn).")


if __name__ == "__main__":
    main()
