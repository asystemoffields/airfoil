#!/usr/bin/env python3
"""Branch-B navigation 'push further' — the spurious-consistency tail vs #demos.

The scaling stress showed NAV find-cost grows with the feature space because, with few demos, random distractor
features are consistent BY CHANCE and outrank the true one. The lever: MORE demos -> a random feature must be
consistent across more pairs -> spurious-consistency rate falls -> the true feature's rank recovers. This
quantifies how far that lever pushes navigation efficiency, at a fixed large space (1514 features).
Run: /data/llm/.venv/bin/python ground_nav_demos.py"""
import numpy as np
import torch
import grammar as G
from train_v2 import V2, task_VO, FEAT_IX, T

net = V2(); net.load_state_dict(torch.load("learner_v2.pt")); net.eval()


def distractor_cols(rng, n):
    D = np.zeros((T, n), np.float32)
    for j in range(n):
        K = rng.choice([2, 3, 4, 5, 8]); D[:, j] = rng.randint(0, K, size=T).astype(np.float32)
    return D


def true_rank(V, O, m, g, ti, nd, rng):
    Vp = V if nd == 0 else np.concatenate([V, distractor_cols(rng, nd)], axis=1)
    with torch.no_grad():
        _, lf = net(torch.from_numpy(Vp[None]), torch.from_numpy(O[None]),
                    torch.from_numpy(m[None]), torch.from_numpy(g[None]))
    return lf[0].argsort(descending=True).tolist().index(ti)


def main():
    rng = np.random.RandomState(1)
    feats = ["size", "holes", "rank_size", "uniq_color", "rank_col", "n_same_color", "height"]
    Nd = 1500; M = 80
    print(f"NAVIGATION 'push further' — find-cost vs #demos, at {14+Nd} features")
    print(f"{'n_demos':>7} | {'NAV find-cost (mean rank)':>25} | {'top-1':>6} | {'top-5':>6}")
    for nde in [2, 3, 4, 6, 10]:
        ranks = []; t1 = t5 = tot = 0
        for _ in range(M):
            feat = feats[rng.randint(0, len(feats))]; dec = G.DECOMPS[rng.randint(0, len(G.DECOMPS))]
            out = G.sample_task(np.random.RandomState(rng.randint(0, 2**31 - 1)),
                                rtype=G.rtype_id("recolor", dec, feat), n_demos=nde)
            if out is None:
                continue
            demos, _ = out; V, O, m, g = task_VO(demos)
            r = true_rank(V, O, m, g, FEAT_IX[feat], Nd, np.random.RandomState(rng.randint(0, 2**31 - 1)))
            ranks.append(r); t1 += int(r == 0); t5 += int(r < 5); tot += 1
        print(f"{nde:>7} | {np.mean(ranks)+1:>25.2f} | {t1/max(1,tot):>6.2f} | {t5/max(1,tot):>6.2f}")
    print("READ: find-cost DROPS as #demos rises = more evidence buys back the spurious-consistency tail -> the "
          "lever that pushes navigation further. ARC has 3-5 demos; a scale-curriculum can use more.")


if __name__ == "__main__":
    main()
