#!/usr/bin/env python3
"""Branch-B NAVIGATION-EFFICIENCY STRESS TEST — decoupled from the expressiveness bottleneck.

The composed cash-out couldn't test navigation: only 5 tasks were expressible and all were found in 2-4 calls, so
the navigator was never under load. Here we put it under load directly. SYNTHETIC recolor tasks (every task HAS a
true-feature solution by construction -> no expressiveness bottleneck), then we INJECT N distractor features and
grow N from 0 to thousands. Because the v2 consistency head scores each feature INDEPENDENTLY (shared scorer over
pairwise stats), it ranks all 14+N features with no retraining. Find-cost for a verify-filtered solver = the rank
of the TRUE feature (it tries features in recognizer-rank order, inducing+verifying each). The question Alex asked:
how far does navigation efficiency push -- does true-feature find-cost stay ~FLAT as the search space explodes
(vs blind ~(14+N)/2), or does it grow (the spurious-consistency limit)?
Run: /data/llm/.venv/bin/python ground_nav_scaling.py"""
import time
import numpy as np
import torch
import grammar as G
from train_v2 import V2, task_VO, FEATS, FEAT_IX, T

net = V2(); net.load_state_dict(torch.load("learner_v2.pt")); net.eval()


def distractor_cols(rng, n):
    """N random per-object feature columns, cardinality varied to mimic real categorical features."""
    D = np.zeros((T, n), np.float32)
    for j in range(n):
        K = rng.choice([2, 3, 4, 5, 8])
        D[:, j] = rng.randint(0, K, size=T).astype(np.float32)
    return D


def true_rank(V_real, O, mask, gvec, true_idx, n_distract, rng):
    Vp = V_real if n_distract == 0 else np.concatenate([V_real, distractor_cols(rng, n_distract)], axis=1)
    with torch.no_grad():
        _, lf = net(torch.from_numpy(Vp[None]), torch.from_numpy(O[None]),
                    torch.from_numpy(mask[None]), torch.from_numpy(gvec[None]))
    return lf[0].argsort(descending=True).tolist().index(true_idx)


def main():
    rng = np.random.RandomState(0)
    true_feats = ["size", "holes", "rank_size", "uniq_color", "rank_col", "n_same_color", "height"]
    Ns = [0, 16, 50, 150, 500, 1500, 5000]
    M = 80
    t0 = time.time()
    print("NAVIGATION-EFFICIENCY STRESS: true-feature find-cost as the feature-search space explodes")
    print(f"{'total feats':>11} | {'NAV find-cost (mean rank, 1-based)':>33} | {'top-1':>6} | {'top-5':>6} | {'BLIND ~half-space':>17}")
    for Nd in Ns:
        ranks = []; t1 = t5 = tot = 0
        for _ in range(M):
            feat = true_feats[rng.randint(0, len(true_feats))]
            dec = G.DECOMPS[rng.randint(0, len(G.DECOMPS))]
            out = G.sample_task(np.random.RandomState(rng.randint(0, 2**31 - 1)),
                                rtype=G.rtype_id("recolor", dec, feat))
            if out is None:
                continue
            demos, _ = out
            V, O, m, g = task_VO(demos)
            r = true_rank(V, O, m, g, FEAT_IX[feat], Nd, np.random.RandomState(rng.randint(0, 2**31 - 1)))
            ranks.append(r); t1 += int(r == 0); t5 += int(r < 5); tot += 1
        total = 14 + Nd
        print(f"{total:>11} | {np.mean(ranks)+1:>33.2f} | {t1/max(1,tot):>6.2f} | {t5/max(1,tot):>6.2f} | {total/2:>17.0f}")
    print(f"\n[{time.time()-t0:.0f}s] READ: NAV find-cost ~FLAT while BLIND grows ~linearly = navigation efficiency "
          "SCALES (recognizer keeps the true feature top-ranked among arbitrarily many distractors); the speedup is "
          "blind/nav. find-cost GROWING with total feats = the spurious-consistency ceiling = how far it pushes.")


if __name__ == "__main__":
    main()
