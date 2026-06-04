#!/usr/bin/env python3
"""Branch-B navigation 'push further' — a DISTRACTOR-HARDENED recognizer.

The stress test found navigation's ceiling = the spurious-consistency tail: with few demos some random features
are consistent BY CHANCE and outrank the true one. ROOT CAUSE: v2's per-feature stats are pair-count-NORMALIZED
(means), so 'consistent on 2 pairs' == 'consistent on 50 pairs' to the scorer -> it can't tell robust from
spurious consistency. Two fixes, tested here: (1) add a SUPPORT-COUNT stat (log #same-value pairs) so the head
can downweight low-evidence agreement; (2) TRAIN with K_TRAIN distractor features injected so it learns to
suppress them. Then re-run the scaling + demos stress and compare to baseline v2.
Run: /data/llm/.venv/bin/python train_v2_hardened.py"""
import time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import grammar as G
from train_v2 import task_VO, FEATS, FEAT_IX, EFFECTS, EFF_IX, T

torch.manual_seed(0); torch.set_num_threads(6)
K_TRAIN = 40  # distractor features injected per task during training


class V2H(nn.Module):
    def __init__(self):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
        self.eff = nn.Sequential(nn.Linear(25, 64), nn.ReLU(), nn.Linear(64, len(EFFECTS)))

    def forward(self, V, O, mask, gvec):
        Tt = V.shape[1]
        Veq = (V.unsqueeze(2) == V.unsqueeze(1)).float()
        Oeq = (O.unsqueeze(2) == O.unsqueeze(1)).float().unsqueeze(-1)
        pm = (mask.unsqueeze(2) & mask.unsqueeze(1)).float() * (1 - torch.eye(Tt, device=V.device)).unsqueeze(0)
        stats = torch.stack([Veq, Oeq.expand_as(Veq), Veq * Oeq, Veq * (1 - Oeq), (1 - Veq) * Oeq], dim=-1)
        w = pm.unsqueeze(-1).unsqueeze(-1)
        agg = (stats * w).sum((1, 2)) / w.sum((1, 2)).clamp(min=1)         # (B,NF,5) means
        support = (Veq * pm.unsqueeze(-1)).sum((1, 2))                     # (B,NF) raw same-value pair count
        agg = torch.cat([agg, torch.log1p(support).unsqueeze(-1)], dim=-1)  # (B,NF,6) + evidence
        return self.eff(gvec), self.score(agg).squeeze(-1)


def distractors(rng, k):
    D = np.zeros((T, k), np.float32)
    for j in range(k):
        Kc = rng.choice([2, 3, 4, 5, 8]); D[:, j] = rng.randint(0, Kc, size=T).astype(np.float32)
    return D


def gen_batch_h(rng, combos, bs, kd=K_TRAIN, ndemos=4):
    Vs, Os, Ms, Gs, ye, yf = [], [], [], [], [], []
    while len(Vs) < bs:
        eff, feat = combos[rng.randint(0, len(combos))]
        rt = "colormap" if eff == "colormap" else G.rtype_id(eff, G.DECOMPS[rng.randint(0, len(G.DECOMPS))], feat)
        out = G.sample_task(np.random.RandomState(rng.randint(0, 2**31 - 1)), rtype=rt, n_demos=ndemos)
        if out is None: continue
        demos, _ = out; V, O, m, g = task_VO(demos)
        V = np.concatenate([V, distractors(rng, kd)], axis=1)
        Vs.append(V); Os.append(O); Ms.append(m); Gs.append(g)
        ye.append(EFF_IX[eff]); yf.append(FEAT_IX[feat] if eff != "colormap" else 0)
    return (torch.from_numpy(np.stack(Vs)), torch.from_numpy(np.stack(Os)),
            torch.from_numpy(np.stack(Ms)), torch.from_numpy(np.stack(Gs)), torch.tensor(ye), torch.tensor(yf))


def true_rank(net, V_real, O, m, g, ti, nd, rng):
    Vp = V_real if nd == 0 else np.concatenate([V_real, distractors(rng, nd)], axis=1)
    with torch.no_grad():
        _, lf = net(torch.from_numpy(Vp[None]), torch.from_numpy(O[None]),
                    torch.from_numpy(m[None]), torch.from_numpy(g[None]))
    return lf[0].argsort(descending=True).tolist().index(ti)


def stress(net, label):
    rng = np.random.RandomState(7)
    feats = ["size", "holes", "rank_size", "uniq_color", "rank_col", "n_same_color", "height"]
    print(f"\n{label}: scaling @4 demos")
    for Nd in [50, 514, 1500, 5000]:
        ranks = []; t5 = 0; tot = 0
        for _ in range(80):
            f = feats[rng.randint(0, len(feats))]; dec = G.DECOMPS[rng.randint(0, len(G.DECOMPS))]
            out = G.sample_task(np.random.RandomState(rng.randint(0, 2**31 - 1)), rtype=G.rtype_id("recolor", dec, f), n_demos=4)
            if out is None: continue
            d, _ = out; V, O, m, g = task_VO(d)
            r = true_rank(net, V, O, m, g, FEAT_IX[f], Nd, np.random.RandomState(rng.randint(0, 2**31 - 1)))
            ranks.append(r); t5 += int(r < 5); tot += 1
        print(f"  {14+Nd:>5} feats: find-cost {np.mean(ranks)+1:6.1f}  top-5 {t5/max(1,tot):.2f}  (blind {(14+Nd)//2})")


def main():
    rng = np.random.RandomState(0)
    feats_all = ["size", "color", "holes", "height", "width", "rank_size", "uniq_size", "uniq_color",
                 "rank_col", "rank_row", "n_same_color", "is_largest"]
    HELDOUT = [("recolor", "holes"), ("recolor", "rank_col"), ("select", "n_same_color"), ("select", "is_largest")]
    combos = [("colormap", "size")]
    for e in ("recolor", "select"):
        for f in feats_all:
            if (e, f) not in HELDOUT: combos.append((e, f))
    net = V2H(); opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    t0 = time.time()
    print(f"V2H (6-stat + support-count), training with K_TRAIN={K_TRAIN} distractors/task")
    for s in range(2500):
        V, O, m, g, ye, yf = gen_batch_h(rng, combos, 32)
        le, lf = net(V, O, m, g); msk = (ye != EFF_IX["colormap"])
        loss = F.cross_entropy(le, ye) + (F.cross_entropy(lf[msk], yf[msk]) if msk.any() else 0.0)
        opt.zero_grad(); loss.backward(); opt.step()
        if (s + 1) % 500 == 0:
            print(f"  step {s+1}/2500 loss {float(loss.detach()):.3f} ({time.time()-t0:.0f}s)", flush=True)
    net.eval()
    torch.save(net.state_dict(), "learner_v2h.pt")
    stress(net, "HARDENED V2H")
    print("\nBASELINE v2 (from ground_nav_scaling, 4 demos): 64f find 5.3/top5 .82 | 514f 12.6/.69 | 1514f 47.2/.71 | 5014f 157.9/.79")
    print(f"[{time.time()-t0:.0f}s] READ: hardened top-5 UP + find-cost DOWN vs baseline at large feats = the "
          "support-count + distractor-training pushes navigation's ceiling further (esp. at ARC's 4-demo regime).")


if __name__ == "__main__":
    main()
