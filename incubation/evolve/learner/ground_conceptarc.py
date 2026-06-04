#!/usr/bin/env python3
"""Branch-B transfer stress — the MORE-DIVERSE axis: ConceptARC (160 human-designed, concept-organized tasks,
a different distribution from both ARC-1 and the synthetic curriculum). Same sim-to-real pipeline as
ground_arc_transfer.py. Run: /data/llm/.venv/bin/python ground_conceptarc.py"""
import json, time
import numpy as np
import torch
from train_v2 import V2, task_VO, FEATS, EFFECTS
from ground_arc import winning_relations, recognizer_solve

CH = "/data/conceptarc/arc-agi_concept-challenges.json"
SO = "/data/conceptarc/arc-agi_concept-solutions.json"


def load_conceptarc():
    ch = json.load(open(CH)); so = json.load(open(SO))
    for tid in ch:
        train = [(np.asarray(p["input"], int), np.asarray(p["output"], int)) for p in ch[tid]["train"]]
        test = [(np.asarray(t["input"], int), np.asarray(so[tid][i], int)) for i, t in enumerate(ch[tid]["test"])]
        yield tid, train, test


def main():
    net = V2(); net.load_state_dict(torch.load("learner_v2.pt")); net.eval()
    t0 = time.time(); solvable = []
    for tid, train, test in load_conceptarc():
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
    print(f"ConceptARC (160 human-designed, MORE-DIVERSE) [{time.time()-t0:.0f}s]:")
    print(f"  grammar-solvable {len(solvable)} | eff-top1 {rec_e1/N:.2f} | feat-top3 {rec_f3/N:.2f} | PIPELINE {stopk/N:.2f}")
    print("READ: feat-top3 + pipeline ~= ARC-1's = transfer is robust on a DIFFERENT (human concept) distribution.")


if __name__ == "__main__":
    main()
