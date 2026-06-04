#!/usr/bin/env python3
"""Branch-B LAST BOX STRESS — does the recognizer's sim-to-real transfer hold on HARDER / DIFFERENT-distribution
real tasks? The sim-to-real test (ground_arc.py) showed feat-top3 24/24, pipeline 23/24 on grammar-solvable
ARC-1. Here we re-run the SAME pipeline on ARC-2 (genuinely harder, different distribution) + ARC-1 as the
in-distribution baseline. The recognizer was trained ONLY on synthetic blobs; if its feature-identification +
pipeline solve-rate hold on ARC-2's grammar-solvable tasks, the consistency inductive bias is distribution-
ROBUST (a strong scaling signature). If they drop, that's a transfer fragility to know before scaling.
Run: /data/llm/.venv/bin/python ground_arc_transfer.py"""
import sys, time
import numpy as np
import torch
import grammar as G
from train_v2 import V2, task_VO, FEATS, EFFECTS
from ground_arc import winning_relations, recognizer_solve

sys.path.insert(0, "/data/Windows-files/Documents/airfoil/incubation/evolve")
import harness


def eval_split(net, split, cap=None):
    t0 = time.time()
    solvable = []
    for tid, train, test in harness.load_split(split, n=cap):
        try:
            wins = winning_relations(train, test)
        except Exception:
            continue
        if wins:
            solvable.append((tid, train, test, wins))
    rec_e1 = rec_f3 = stopk = 0
    for tid, train, test, wins in solvable:
        truth_eff = set(e for e, _f, _r in wins); truth_feat = set(f for _e, f, _r in wins if f)
        V, O, m, g = task_VO(train)
        with torch.no_grad():
            le, lf = net(torch.from_numpy(V[None]), torch.from_numpy(O[None]),
                         torch.from_numpy(m[None]), torch.from_numpy(g[None]))
        pe1 = EFFECTS[int(le[0].argmax())]
        pf3 = [FEATS[i] for i in lf[0].argsort(descending=True).tolist()[:3]]
        rec_e1 += int(pe1 in truth_eff)
        rec_f3 += int((bool(truth_feat) and any(f in truth_feat for f in pf3)) or not truth_feat)
        s_k, _ = recognizer_solve(net, train, test, topk_eff=2, topk_feat=3)
        stopk += int(s_k)
    N = max(1, len(solvable))
    print(f"  {split:<11}: grammar-solvable {len(solvable):4d} | eff-top1 {rec_e1/N:.2f} | "
          f"feat-top3 {rec_f3/N:.2f} | PIPELINE {stopk/N:.2f}  [{time.time()-t0:.0f}s]")
    return split, len(solvable), rec_f3 / N, stopk / N


def main():
    net = V2(); net.load_state_dict(torch.load("learner_v2.pt")); net.eval()
    print("SIM-TO-REAL TRANSFER under harder/different distributions (recognizer trained on synthetic blobs only):")
    rows = []
    for split in ("arc1-train", "arc1-eval", "arc2-train", "arc2-eval"):
        rows.append(eval_split(net, split))
    a1 = [r for r in rows if r[0].startswith("arc1")]; a2 = [r for r in rows if r[0].startswith("arc2")]
    f1 = np.mean([r[2] for r in a1]); p1 = np.mean([r[3] for r in a1])
    f2 = np.mean([r[2] for r in a2]); p2 = np.mean([r[3] for r in a2])
    print(f"\nARC-1 (in-dist):  feat-top3 {f1:.2f}  pipeline {p1:.2f}")
    print(f"ARC-2 (harder):   feat-top3 {f2:.2f}  pipeline {p2:.2f}")
    print("READ: ARC-2 feat-top3 + pipeline ~= ARC-1 = the synthetic-trained recognizer's transfer is DISTRIBUTION-"
          "ROBUST (holds on harder, never-seen real tasks) = strong scaling signature, last box gate green. A big "
          "drop = transfer fragility to address with verified-ARC distillation at scale.")


if __name__ == "__main__":
    main()
