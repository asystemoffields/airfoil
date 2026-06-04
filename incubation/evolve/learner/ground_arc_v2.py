#!/usr/bin/env python3
"""Branch-B VALUE TEST: does the LEARNED proposer (widened + composed grammar) reach ARC tasks the families
miss, and does it do so with FEWER verify-calls than blind enumeration?

Pipeline (recognizer-driven composed solve): for each cheap pre-op, transform the inputs, run the v2 consistency
recognizer on the transformed demos -> top-K (effect, feature) -> induce_composed -> exact-verify -> generalize.
Compare: recognizer-top-K solves + induce-call budget VS blind enumeration of the same composed space, capped at
the SAME number of induce-calls. Report solved, beyond gen2_base (retrieval), beyond gen6_base (full families).
Run: /data/llm/.venv/bin/python ground_arc_v2.py"""
import sys, time, json
import numpy as np
import torch
import grammar as G
import grammar_comp as GC
from train_v2 import V2, task_VO, FEATS, EFFECTS

EVOLVE = "/data/Windows-files/Documents/airfoil/incubation/evolve"
sys.path.insert(0, EVOLVE)
import harness

GEN2 = set(json.load(open(EVOLVE + "/runs/gen2_base_solved.json"))["arc1_eval"]["solved"])
BEYOND14 = set("0a2355a6 21f83797 281123b4 3194b014 358ba94e 37d3e8b2 6a11f6da 84db8fc4 "
               "9a4bb226 aa18de87 cd3c21df d282b262 e0fb7511 e872b94a".split())
GEN6 = GEN2 | BEYOND14


def gen_comp(comp, test):
    for gi, go in test:
        out = GC.apply_composed(comp, gi)
        if out is None or out.shape != go.shape or not np.array_equal(out, go):
            return False
    return True


def recog_solve(net, train, test, ke=2, kf=3):
    """recognizer-driven: enumerate pre-ops, recognizer picks top-K (eff,feat) on transformed demos. -> (solved, n_induce)."""
    n_ind = 0
    for pre in GC.PRE_OPS:
        tt = [(GC._pre(pre, gi), go) for gi, go in train]
        if any(t[0] is None or t[0].size == 0 for t in tt):
            continue
        V, O, m, g = task_VO(tt)
        with torch.no_grad():
            le, lf = net(torch.from_numpy(V[None]), torch.from_numpy(O[None]),
                         torch.from_numpy(m[None]), torch.from_numpy(g[None]))
        effs = [EFFECTS[i] for i in le[0].argsort(descending=True).tolist()[:ke]]
        feats = [FEATS[i] for i in lf[0].argsort(descending=True).tolist()[:kf]]
        for eff in effs:
            if eff == "colormap":
                n_ind += 1
                rel = G.induce("colormap", None, None, tt)
                if rel is not None and gen_comp((pre, rel), test):
                    return True, n_ind
                continue
            for feat in feats:
                for dec in G.DECOMPS:
                    n_ind += 1
                    comp = GC.induce_composed(pre, eff, dec, feat, train)
                    if comp is not None and gen_comp(comp, test):
                        return True, n_ind
    return False, n_ind


def blind_solve(train, test, budget):
    """blind enumeration of the SAME composed space, capped at `budget` induce-calls. -> (solved, n_induce)."""
    n_ind = 0
    for pre in GC.PRE_OPS:
        for eff in EFFECTS:
            if eff == "colormap":
                n_ind += 1
                if n_ind > budget: return False, n_ind
                rel = G.induce("colormap", None, None, [(GC._pre(pre, gi), go) for gi, go in train])
                if rel is not None and gen_comp((pre, rel), test):
                    return True, n_ind
                continue
            for feat in FEATS:
                for dec in G.DECOMPS:
                    n_ind += 1
                    if n_ind > budget: return False, n_ind
                    comp = GC.induce_composed(pre, eff, dec, feat, train)
                    if comp is not None and gen_comp(comp, test):
                        return True, n_ind
    return False, n_ind


def main():
    net = V2(); net.load_state_dict(torch.load("learner_v2.pt")); net.eval()
    tasks = harness.load_split("arc1-eval")
    t0 = time.time()
    rec_solved = set(); rec_ind = []
    for tid, train, test in tasks:
        s, ni = recog_solve(net, train, test)
        rec_ind.append(ni)
        if s: rec_solved.add(tid)
    budget = int(np.percentile(rec_ind, 90)) + 1   # blind gets the recognizer's ~90th-pct call budget
    blind_solved = set()
    for tid, train, test in tasks:
        s, _ = blind_solve(train, test, budget)
        if s: blind_solved.add(tid)
    print(f"ARC-1 eval (400), composed+widened grammar [{time.time()-t0:.0f}s]")
    print(f"  RECOGNIZER-driven: solved {len(rec_solved)}  | mean induce-calls {np.mean(rec_ind):.0f} (budget cap {budget})")
    print(f"     beyond gen2_base (retrieval): {len(rec_solved-GEN2)}  {sorted(rec_solved-GEN2)}")
    print(f"     beyond gen6_base (families):  {len(rec_solved-GEN6)}  {sorted(rec_solved-GEN6)}")
    print(f"  BLIND enum @ same budget ({budget} calls): solved {len(blind_solved)}")
    print(f"     beyond gen6_base: {len(blind_solved-GEN6)}  {sorted(blind_solved-GEN6)}")
    print(f"  recognizer solves NOT reached by blind-under-budget: {sorted(rec_solved - blind_solved)}")
    print("READ: recognizer beyond_gen6 > 0 = the LEARNED proposer reaches tasks the families miss. recognizer "
          ">= blind @ same budget = the learner NAVIGATES the composed space more efficiently than enumeration.")


if __name__ == "__main__":
    main()
