#!/usr/bin/env python3
"""Branch-B learner v1 — OBJECT-CENTRIC relational proposer (the fix v0 pointed to).

v0 lesson: a pixel-CNN learns the effect TYPE but can't extract the causal FEATURE (re-deriving perception
we already compute). v1 keeps perception as ARCHITECTURE: grammar.py deterministically segments objects and
computes every feature; we FEED those feature-tables + each object's OUTCOME to a small RELATIONAL TRANSFORMER
over the object-set (objects attend to each other -> can compare property->outcome across objects/demos), and
predict (effect, feature). Deep/relational, not wide (~few M params, trains on the 7GB box).

THE TEST (same as v0): hold out (effect x feature) combos entirely; held-out feature-acc >> 0.10 = the model
identifies a causal feature it never saw under that effect = compositional generalization over the grammar.
Run: /data/llm/.venv/bin/python train_v1.py"""
import os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import grammar as G

torch.manual_seed(0); torch.set_num_threads(6)
FEATS = G.FEATURE_NAMES; FEAT_IX = {f: i for i, f in enumerate(FEATS)}
EFFECTS = ["colormap", "recolor", "select"]; EFF_IX = {e: i for i, e in enumerate(EFFECTS)}
MAXTOK = 48                          # max object tokens per task (across all demos) + 1 global
_FSCALE = {"size": 25., "color": 9., "holes": 3., "height": 12., "width": 12.,
           "rank_size": 8., "rank_size_d": 8., "uniq_size": 1., "uniq_color": 1., "uniq_shape": 1.}


def _objtoken(o, objs, gi, go, demo_ix):
    """raw feature vector for one object (the CAUSE side) + its OUTCOME (the effect side)."""
    fv = [min(1.0, G.FEATURES[f](o, objs) / _FSCALE[f]) for f in FEATS]      # 10 normalized features
    incol = [0.0] * 10; incol[o["color"] if 0 <= o["color"] < 10 else 0] = 1.0  # input color one-hot
    pos = [o["r0"] / 30., o["c0"] / 30., o["h"] / 30., o["w"] / 30.]
    # outcome
    outcol = [0.0] * 10; flags = [0., 0., 0., 0.]  # [unchanged, removed, is_whole_output, out_unknown]
    if go.shape == gi.shape:
        cols = [int(go[r, c]) for r, c in o["cells"]]
        oc = max(set(cols), key=cols.count) if cols else 0
        outcol[oc if 0 <= oc < 10 else 0] = 1.0
        if oc == o["color"]: flags[0] = 1.0
        if oc == 0: flags[1] = 1.0
    else:
        r0, c0, h, w = o["r0"], o["c0"], o["h"], o["w"]
        crop = gi[r0:r0 + h, c0:c0 + w]
        if go.shape == crop.shape and np.array_equal(go, crop):
            flags[2] = 1.0
        flags[3] = 1.0
    demo = [0.0] * 4; demo[min(demo_ix, 3)] = 1.0
    return np.array(fv + incol + pos + outcol + flags + demo + [0.0], np.float32)  # last dim: is_global=0


def _globaltoken(demos):
    ih = np.zeros(10); oh = np.zeros(10); n = 0; sshape = 0
    for gi, go in demos:
        for c in range(10):
            ih[c] += (gi == c).mean(); oh[c] += (go == c).mean()
        n += 1; sshape += int(gi.shape == go.shape)
    ih /= max(n, 1); oh /= max(n, 1)
    base = list(ih) + list(oh)                                   # 20
    gi, go = demos[0]
    extra = [gi.shape[0] / 30., gi.shape[1] / 30., go.shape[0] / 30., go.shape[1] / 30., sshape / max(n, 1)]
    v = np.zeros(RAWDIM, np.float32)
    v[:20] = base; v[20:25] = extra; v[-1] = 1.0                 # is_global=1
    return v


RAWDIM = 10 + 10 + 4 + 10 + 4 + 4 + 1   # = 43


def task_tokens(demos):
    toks = [_globaltoken(demos)]
    for di, (gi, go) in enumerate(demos):
        for o in G.objects(gi, 4, True):
            toks.append(_objtoken(o, G.objects(gi, 4, True), gi, go, di))
            if len(toks) >= MAXTOK: break
        if len(toks) >= MAXTOK: break
    arr = np.zeros((MAXTOK, RAWDIM), np.float32); m = np.zeros(MAXTOK, bool)
    for i, t in enumerate(toks[:MAXTOK]):
        arr[i] = t; m[i] = True
    return arr, m


class RelProposer(nn.Module):
    def __init__(self, d=128, layers=4, heads=4):
        super().__init__()
        self.proj = nn.Linear(RAWDIM, d)
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, batch_first=True, dropout=0.0)
        self.tf = nn.TransformerEncoder(enc, layers)
        self.eff = nn.Linear(d, len(EFFECTS)); self.feat = nn.Linear(d, len(FEATS))

    def forward(self, x, mask):                    # x:(B,T,RAW) mask:(B,T) True=valid
        h = self.tf(self.proj(x), src_key_padding_mask=~mask)
        h = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)  # masked mean
        return self.eff(h), self.feat(h)

    def nparams(self):
        return sum(p.numel() for p in self.parameters())


def gen_batch(rng, combos, bs):
    xs, ms, ye, yf = [], [], [], []
    while len(xs) < bs:
        eff, feat = combos[rng.randint(0, len(combos))]
        rt = "colormap" if eff == "colormap" else G.rtype_id(eff, G.DECOMPS[rng.randint(0, len(G.DECOMPS))], feat)
        out = G.sample_task(np.random.RandomState(rng.randint(0, 2**31 - 1)), rtype=rt)
        if out is None: continue
        demos, _ = out
        a, m = task_tokens(demos)
        xs.append(a); ms.append(m); ye.append(EFF_IX[eff]); yf.append(FEAT_IX[feat] if eff != "colormap" else 0)
    return (torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ms)),
            torch.tensor(ye), torch.tensor(yf))


def main():
    rng = np.random.RandomState(0)
    feats_all = ["size", "color", "holes", "height", "width", "rank_size", "uniq_size", "uniq_color"]
    HELDOUT = [("recolor", "holes"), ("recolor", "uniq_color"), ("select", "size"), ("select", "uniq_size")]
    train_combos = [("colormap", "size")]
    for e in ("recolor", "select"):
        for f in feats_all:
            if (e, f) not in HELDOUT: train_combos.append((e, f))
    net = RelProposer()
    print(f"v1 RelProposer: {net.nparams()/1e6:.2f}M params | train combos {len(train_combos)} | held-out {HELDOUT}")
    opt = torch.optim.Adam(net.parameters(), lr=1.5e-3)
    t0 = time.time(); STEPS, BS = 2500, 32
    for s in range(STEPS):
        x, m, ye, yf = gen_batch(rng, train_combos, BS)
        le, lf = net(x, m)
        msk = (ye != EFF_IX["colormap"])
        loss = F.cross_entropy(le, ye) + (F.cross_entropy(lf[msk], yf[msk]) if msk.any() else 0.0)
        opt.zero_grad(); loss.backward(); opt.step()
        if (s + 1) % 500 == 0:
            print(f"  step {s+1}/{STEPS} loss {float(loss):.3f} ({time.time()-t0:.0f}s)", flush=True)
    net.eval()
    def ev(combos, n=320):
        eok = fok = jok = tot = fcnt = 0
        with torch.no_grad():
            for _ in range(n // BS + 1):
                x, m, ye, yf = gen_batch(rng, combos, BS)
                le, lf = net(x, m); pe = le.argmax(1); pf = lf.argmax(1); mm = (ye != EFF_IX["colormap"])
                eok += (pe == ye).sum().item(); tot += len(ye)
                fok += ((pf == yf) & mm).sum().item(); jok += ((pe == ye) & (pf == yf) & mm).sum().item()
                fcnt += mm.sum().item()
        return eok / tot, fok / max(1, fcnt), jok / tot
    te, tf, tj = ev(train_combos); he, hf, hj = ev(HELDOUT)
    print(f"\nTRAINED combos:  effect {te:.2f}  feature {tf:.2f}  joint {tj:.2f}")
    print(f"HELD-OUT combos: effect {he:.2f}  feature {hf:.2f}  joint {hj:.2f}   (random feature 0.10)")
    print("READ: held-out FEATURE >> 0.10 = object-centric model identifies the causal feature it never saw "
          "under that effect = compositional generalization over the grammar (Branch-B prerequisite MET).")
    torch.save(net.state_dict(), "learner_v1.pt"); print(f"saved learner_v1.pt ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
