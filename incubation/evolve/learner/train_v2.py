#!/usr/bin/env python3
"""Branch-B learner v2 — explicit PAIRWISE-CONSISTENCY feature head (the principled fix to v1's flat loss).

v0/v1 lesson: neither a pixel-CNN nor a generic relational transformer DISCOVERS "which feature's value
determines the outcome." v2 bakes that inductive bias in: for each feature j, over all object PAIRS, compute
the consistency statistics [same-value_j, same-outcome, and their cross-terms]; a SHARED (feature-index-
agnostic) scorer maps those stats -> a score_j; feature = argmax_j score_j. Because the scorer only sees
(value-agreement, outcome-agreement) statistics, it transfers to features it never trained on BY CONSTRUCTION.

THE TEST (same as v0/v1): held-out (effect x feature) combos. held-out feature-acc >> 0.10 = the architecture
can represent + transfer causal-feature relevance -> green light to scale to composed relations + ARC.
Run: /data/llm/.venv/bin/python train_v2.py"""
import sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import grammar as G

torch.manual_seed(0); torch.set_num_threads(6)
FEATS = G.FEATURE_NAMES; FEAT_IX = {f: i for i, f in enumerate(FEATS)}
EFFECTS = ["colormap", "recolor", "select"]; EFF_IX = {e: i for i, e in enumerate(EFFECTS)}
T = 32  # max object tokens per task
NF = len(FEATS)


def task_VO(demos):
    """V:(T,NF) raw feature values, O:(T,) outcome scalar, mask:(T,), gvec:(25,) global summary."""
    V = np.zeros((T, NF), np.float32); O = np.zeros(T, np.float32); mask = np.zeros(T, bool)
    n = 0
    for gi, go in demos:
        objs = G.objects(gi, 4, True)
        for o in objs:
            if n >= T: break
            V[n] = [float(G.FEATURES[f](o, objs)) for f in FEATS]
            if go.shape == gi.shape:
                cols = [int(go[r, c]) for r, c in o["cells"]]
                O[n] = max(set(cols), key=cols.count) if cols else 0
            else:
                r0, c0, h, w = o["r0"], o["c0"], o["h"], o["w"]
                crop = gi[r0:r0 + h, c0:c0 + w]
                O[n] = 1.0 if (go.shape == crop.shape and np.array_equal(go, crop)) else 0.0
            mask[n] = True; n += 1
    # global summary for the EFFECT head
    ih = np.zeros(10); oh = np.zeros(10); sshape = 0
    for gi, go in demos:
        for c in range(10):
            ih[c] += (gi == c).mean(); oh[c] += (go == c).mean()
        sshape += int(gi.shape == go.shape)
    ih /= len(demos); oh /= len(demos)
    gvec = np.concatenate([ih, oh, [sshape / len(demos), demos[0][0].shape == demos[0][1].shape,
                                    n / T, float(n), 0.0]]).astype(np.float32)
    return V, O, mask, gvec


class V2(nn.Module):
    def __init__(self):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
        self.eff = nn.Sequential(nn.Linear(25, 64), nn.ReLU(), nn.Linear(64, len(EFFECTS)))

    def forward(self, V, O, mask, gvec):
        B = V.shape[0]
        Veq = (V.unsqueeze(2) == V.unsqueeze(1)).float()                 # (B,T,T,NF)
        Oeq = (O.unsqueeze(2) == O.unsqueeze(1)).float().unsqueeze(-1)   # (B,T,T,1)
        pm = (mask.unsqueeze(2) & mask.unsqueeze(1)).float()
        pm = pm * (1 - torch.eye(T, device=V.device)).unsqueeze(0)       # exclude self-pairs
        stats = torch.stack([Veq, Oeq.expand_as(Veq), Veq * Oeq,
                             Veq * (1 - Oeq), (1 - Veq) * Oeq], dim=-1)   # (B,T,T,NF,5)
        w = pm.unsqueeze(-1).unsqueeze(-1)                               # (B,T,T,1,1)
        agg = (stats * w).sum((1, 2)) / w.sum((1, 2)).clamp(min=1)        # (B,NF,5)
        feat_logits = self.score(agg).squeeze(-1)                        # (B,NF)
        return self.eff(gvec), feat_logits

    def nparams(self):
        return sum(p.numel() for p in self.parameters())


def gen_batch(rng, combos, bs):
    Vs, Os, Ms, Gs, ye, yf = [], [], [], [], [], []
    while len(Vs) < bs:
        eff, feat = combos[rng.randint(0, len(combos))]
        rt = "colormap" if eff == "colormap" else G.rtype_id(eff, G.DECOMPS[rng.randint(0, len(G.DECOMPS))], feat)
        out = G.sample_task(np.random.RandomState(rng.randint(0, 2**31 - 1)), rtype=rt)
        if out is None: continue
        demos, _ = out
        V, O, m, g = task_VO(demos)
        Vs.append(V); Os.append(O); Ms.append(m); Gs.append(g)
        ye.append(EFF_IX[eff]); yf.append(FEAT_IX[feat] if eff != "colormap" else 0)
    return (torch.from_numpy(np.stack(Vs)), torch.from_numpy(np.stack(Os)),
            torch.from_numpy(np.stack(Ms)), torch.from_numpy(np.stack(Gs)),
            torch.tensor(ye), torch.tensor(yf))


def main():
    rng = np.random.RandomState(0)
    feats_all = ["size", "color", "holes", "height", "width", "rank_size", "uniq_size", "uniq_color",
                 "rank_col", "rank_row", "n_same_color", "is_largest"]   # incl. new relational/positional features
    # hold out a mix of OLD and NEW features entirely -> tests generalization over the WIDENED grammar
    HELDOUT = [("recolor", "holes"), ("recolor", "rank_col"), ("select", "n_same_color"), ("select", "is_largest")]
    train_combos = [("colormap", "size")]
    for e in ("recolor", "select"):
        for f in feats_all:
            if (e, f) not in HELDOUT: train_combos.append((e, f))
    net = V2()
    print(f"v2 consistency-head: {net.nparams()/1e3:.1f}K params | train {len(train_combos)} | held-out {HELDOUT}")
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    t0 = time.time(); STEPS, BS = 2000, 32
    for s in range(STEPS):
        V, O, m, g, ye, yf = gen_batch(rng, train_combos, BS)
        le, lf = net(V, O, m, g)
        msk = (ye != EFF_IX["colormap"])
        loss = F.cross_entropy(le, ye) + (F.cross_entropy(lf[msk], yf[msk]) if msk.any() else 0.0)
        opt.zero_grad(); loss.backward(); opt.step()
        if (s + 1) % 400 == 0:
            print(f"  step {s+1}/{STEPS} loss {float(loss.detach()):.3f} ({time.time()-t0:.0f}s)", flush=True)
    net.eval()
    def ev(combos, n=320):
        eok = fok = f3 = jok = tot = fcnt = 0
        with torch.no_grad():
            for _ in range(n // BS + 1):
                V, O, m, g, ye, yf = gen_batch(rng, combos, BS)
                le, lf = net(V, O, m, g); pe = le.argmax(1); pf = lf.argmax(1); mm = (ye != EFF_IX["colormap"])
                top3 = lf.topk(3, dim=1).indices
                eok += (pe == ye).sum().item(); tot += len(ye)
                fok += ((pf == yf) & mm).sum().item()
                f3 += (((top3 == yf.unsqueeze(1)).any(1)) & mm).sum().item()
                jok += ((pe == ye) & (pf == yf) & mm).sum().item()
                fcnt += mm.sum().item()
        c = max(1, fcnt)
        return eok / tot, fok / c, f3 / c, jok / tot
    te, tf, t3, tj = ev(train_combos); he, hf, h3, hj = ev(HELDOUT)
    print(f"\nfeatures: {len(FEATS)} ({FEATS})")
    print(f"TRAINED combos:  effect {te:.2f}  feat-top1 {tf:.2f}  feat-top3 {t3:.2f}  joint {tj:.2f}")
    print(f"HELD-OUT combos: effect {he:.2f}  feat-top1 {hf:.2f}  feat-top3 {h3:.2f}  joint {hj:.2f}   (random top1 {1/len(FEATS):.2f})")
    print("READ: held-out feat-top3 high = the consistency head TRANSFERS relevance to new (effect x feature) "
          "combos over the WIDENED grammar; feat-top3 is the metric the verify-filtered pipeline actually uses.")
    torch.save(net.state_dict(), "learner_v2.pt"); print(f"saved learner_v2.pt ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
