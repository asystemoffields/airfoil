#!/usr/bin/env python3
"""Vine — the HEAD-TO-HEAD: the assembled EARNED-vocabulary solver on gen6_base's EXACT eval + gate (Alex's retest).

Steps 1-2-3: (1) grown faculties = the self-evolving substrate EYE (earned senses) + the motor HAND (move/copy);
(2) assembled = V2 routes the open vocabulary (regular + fixed relational + substrate senses) -> recolor singles
-> composition (recognizer-ranked OUTER x full relational INNER, the falsifier's fix) -> gesture; (3) RETEST on
ARC-1 eval(400) + the SAME beyond_gen2 / beyond_gen6 gate gen6_base used. The honest number: Vine-that-EARNS vs
gen6_base-that-was-HAND-AUTHORED (48/400, 14 beyond gen2), identical ground -- NOT the 48/400 hand-authored figure.
Run: /data/llm/.venv/bin/python ground_vine.py"""
import sys, time
import numpy as np
import torch
import torch
import grammar as G
import rel_dsl as D
import substrate_eye as SE
from open_loop import ranked                          # V2H router over the OPEN vocabulary (earned senses)
from ground_v2_relational import REL_PREDS            # fixed relational set = the composition INNER pool
from effect_faculty import earn_effect                # the motor hand (move/copy)
from ground_arc_v2 import GEN2, GEN6                   # gen2_base (retrieval) / gen6_base (families) solved sets
from ground_arc import recognizer_solve               # FULL grammar path: recolor/select/colormap x 4 decomps, routed
from train_v2 import V2

sys.path.insert(0, "/data/Windows-files/Documents/airfoil/incubation/evolve")
import harness

V2NET = V2(); V2NET.load_state_dict(torch.load("learner_v2.pt")); V2NET.eval()


def vine_solve(train, test, topk=10):
    # 1. FULL GRAMMAR path (the big recovery): recolor/select/colormap x all 4 decomps, recognizer-routed
    try:
        s, _ = recognizer_solve(V2NET, train, test, topk_eff=2, topk_feat=3)
        if s:
            return True
    except Exception:
        pass
    r = ranked(train)
    for key in r[:topk]:                              # 2. EARNED-SENSE recolor (relational / substrate, beyond grammar)
        if isinstance(key, (D.Quantify, SE.SubQuantify)):
            prog = D.induce_recolor(key, train)
            if prog is not None and D.verify(prog, train, test):
                return True
    outers = [k for k in r[:topk] if isinstance(k, D.Quantify)]
    for outer in outers:                              # 3. composition: ranked OUTER x FULL relational INNER
        for inner in REL_PREDS:
            prog = D.induce_recolor(D.Composed(outer.ch, outer.value, inner, outer.mode), train)
            if prog is not None and D.verify(prog, train, test):
                return True
    try:
        if earn_effect(train, test) is not None:       # 4. the motor hand
            return True
    except Exception:
        pass
    return False


def main():
    tasks = harness.load_split("arc1-eval")
    t0 = time.time(); solved = set()
    for i, (tid, train, test) in enumerate(tasks):
        try:
            if vine_solve(train, test):
                solved.add(tid)
        except Exception:
            continue
        if (i + 1) % 100 == 0:
            print(f"  ...{i+1}/400  solved {len(solved)}  [{time.time()-t0:.0f}s]", flush=True)
    print(f"\nVINE (earned vocabulary) on ARC-1 eval(400) [{time.time()-t0:.0f}s]:")
    print(f"  solved: {len(solved)}")
    print(f"  beyond gen2_base (retrieval): {len(solved - GEN2)}")
    print(f"  beyond gen6_base (families):  {len(solved - GEN6)}  {sorted(solved - GEN6)}")
    print(f"  -- reference: gen6_base (HAND-AUTHORED) = 48 solved, 14 beyond gen2_base, 0 regressions --")
    print(f"  Vine solves NOT in gen6_base: {sorted(solved - GEN6)}   gen6_base solves Vine MISSES: {len(GEN6 - solved)}")
    print("READ: this is the honest head-to-head -- Vine EARNS its vocabulary (recognizer + substrate senses + "
          "gestures + composition) vs gen6_base's hand-authored menu, same eval + gate. Vine-beyond-gen6 = tasks "
          "the hand-authored families miss; gen6-Vine-misses = the hand-authored coverage Vine's faculties don't "
          "yet reach (the grow-targets).")


if __name__ == "__main__":
    main()
