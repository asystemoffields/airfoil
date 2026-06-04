#!/usr/bin/env python3
"""Branch-B SIM-TO-REAL test: does the v2 recognizer (trained ONLY on synthetic blobs) transfer to REAL ARC?

Step 1: enumerate our grammar over ARC-1 (train+eval) -> the tasks our grammar can EXPRESS (a relation that
        verifies on train AND generalizes to the held-out test). Those give ground-truth (effect, feature).
Step 2: run the v2 recognizer on those REAL tasks' demos -> does its top-K (effect, feature) match the truth,
        and does propose(top-K)->induce->exact-verify->generalize actually SOLVE them?
Compare the recognizer's solve-rate (top-K) to enumerate-all (the ceiling for this grammar).
Run: /data/llm/.venv/bin/python ground_arc.py"""
import sys, time
import numpy as np
import torch
import grammar as G
from train_v2 import V2, task_VO, FEATS, EFFECTS, FEAT_IX, EFF_IX

sys.path.insert(0, "/data/Windows-files/Documents/airfoil/incubation/evolve")
import harness


def generalizes(rel, test):
    for gi, go in test:
        out = G.apply_relation(rel, gi)
        if out is None or out.shape != go.shape or not np.array_equal(out, go):
            return False
    return True


def winning_relations(train, test):
    """all grammar relations that verify on train AND generalize to test. Returns list of (effect,feature,rel)."""
    wins = []
    rel = G.induce("colormap", None, None, train)
    if rel is not None and generalizes(rel, test):
        wins.append(("colormap", None, rel))
    for eff in ("recolor", "select"):
        for d in G.DECOMPS:
            for f in FEATS:
                rel = G.induce(eff, d, f, train)
                if rel is not None and generalizes(rel, test):
                    wins.append((eff, f, rel))
    return wins


def recognizer_solve(net, train, test, topk_eff=2, topk_feat=3):
    """recognizer proposes top-K (effect,feature); induce+verify+generalize each; return (solved, used_relation)."""
    V, O, m, g = task_VO(train)
    with torch.no_grad():
        le, lf = net(torch.from_numpy(V[None]), torch.from_numpy(O[None]),
                     torch.from_numpy(m[None]), torch.from_numpy(g[None]))
    eff_order = [EFFECTS[i] for i in le[0].argsort(descending=True).tolist()[:topk_eff]]
    feat_order = [FEATS[i] for i in lf[0].argsort(descending=True).tolist()[:topk_feat]]
    for eff in eff_order:
        if eff == "colormap":
            rel = G.induce("colormap", None, None, train)
            if rel is not None and generalizes(rel, test):
                return True, ("colormap", None)
            continue
        for f in feat_order:
            for d in G.DECOMPS:
                rel = G.induce(eff, d, f, train)
                if rel is not None and generalizes(rel, test):
                    return True, (eff, f)
    return False, None


def main():
    net = V2(); net.load_state_dict(torch.load("learner_v2.pt")); net.eval()
    t0 = time.time()
    solvable = []   # (split, tid, train, test, wins)
    for split in ("arc1-train", "arc1-eval"):
        for tid, train, test in harness.load_split(split):
            wins = winning_relations(train, test)
            if wins:
                solvable.append((split, tid, train, test, wins))
    f"grammar-solvable ARC tasks found in {time.time()-t0:.0f}s"
    by = {}
    for s, *_ in solvable:
        by[s] = by.get(s, 0) + 1
    print(f"grammar-solvable ARC-1 tasks: {by} (total {len(solvable)})  [{time.time()-t0:.0f}s]")

    # recognition + pipeline on the REAL solvable tasks
    rec_e1 = rec_f3 = solved_topk = solved_top1 = 0
    for split, tid, train, test, wins in solvable:
        truth_eff = set(e for e, _f, _r in wins); truth_feat = set(f for _e, f, _r in wins if f)
        V, O, m, g = task_VO(train)
        with torch.no_grad():
            le, lf = net(torch.from_numpy(V[None]), torch.from_numpy(O[None]),
                         torch.from_numpy(m[None]), torch.from_numpy(g[None]))
        pe1 = EFFECTS[int(le[0].argmax())]
        pf3 = [FEATS[i] for i in lf[0].argsort(descending=True).tolist()[:3]]
        rec_e1 += int(pe1 in truth_eff)
        rec_f3 += int(bool(truth_feat) and any(f in truth_feat for f in pf3) or not truth_feat)
        s_k, _ = recognizer_solve(net, train, test, topk_eff=2, topk_feat=3)
        s_1, _ = recognizer_solve(net, train, test, topk_eff=1, topk_feat=1)
        solved_topk += int(s_k); solved_top1 += int(s_1)
    N = max(1, len(solvable))
    print(f"\nSIM-TO-REAL on {len(solvable)} grammar-solvable REAL ARC tasks (recognizer trained on synthetic only):")
    print(f"  effect top-1 correct:   {rec_e1}/{N} = {rec_e1/N:.2f}")
    print(f"  feature top-3 correct:  {rec_f3}/{N} = {rec_f3/N:.2f}")
    print(f"  PIPELINE solved (recognizer top-1 eff / top-1 feat): {solved_top1}/{N} = {solved_top1/N:.2f}")
    print(f"  PIPELINE solved (recognizer top-2 eff / top-3 feat): {solved_topk}/{N} = {solved_topk/N:.2f}")
    print(f"  (enumerate-all ceiling for this grammar = {len(solvable)}/{len(solvable)} = 1.00 by construction)")
    print("READ: if recognizer top-K solve-rate is high, the synthetic-trained recognizer TRANSFERS to real ARC "
          "demos -> sim-to-real works. If low, real ARC demos differ from synthetic blobs -> need HF ARC-ish "
          "data / verified-ARC distillation to close the gap.")


if __name__ == "__main__":
    main()
