#!/usr/bin/env python3
"""Branch-B learner v0 — encoder + FACTORED relation proposer, and the decisive CPU test:
COMPOSITIONAL generalization to HELD-OUT (effect x feature) combinations.

Model: a small CNN encodes each demo (input + output grids, one-hot + valid-mask, padded 30x30) -> demo
embedding; mean-pool over demos -> task embedding; two heads predict EFFECT {colormap,recolor,select} and
FEATURE {the 10 object features}. (decomp is tried at induce-time, cheap; the table/mode are induced+verified.)

THE TEST: hold out a set of (effect,feature) combos from TRAINING entirely. If the factored model predicts the
right effect+feature for a held-out combo from its demos -- having seen that effect with OTHER features and
that feature under OTHER effects -- the learner GENERALIZES over the grammar (the Branch-B prerequisite).
Run: /data/llm/.venv/bin/python train.py"""
import os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import grammar as G

torch.manual_seed(0)
DEV = "cpu"
torch.set_num_threads(6)
PAD = 30
EFFECTS = ["colormap", "recolor", "select"]
EFF_IX = {e: i for i, e in enumerate(EFFECTS)}
FEATS = G.FEATURE_NAMES                      # 10
FEAT_IX = {f: i for i, f in enumerate(FEATS)}


def grid_tensor(g):
    """grid -> (11, 30, 30): 10 color one-hot + 1 valid-mask, padded."""
    g = np.asarray(g, int)
    h, w = g.shape
    t = np.zeros((11, PAD, PAD), np.float32)
    hh, ww = min(h, PAD), min(w, PAD)
    for c in range(10):
        t[c, :hh, :ww] = (g[:hh, :ww] == c)
    t[10, :hh, :ww] = 1.0
    return t


def task_tensor(demos, max_d=4):
    """demos -> (max_d, 22, 30, 30): per demo = input(11) ++ output(11). Pad/truncate to max_d demos."""
    arr = np.zeros((max_d, 22, PAD, PAD), np.float32)
    for k, (gi, go) in enumerate(demos[:max_d]):
        arr[k, :11] = grid_tensor(gi)
        arr[k, 11:] = grid_tensor(go)
    return torch.from_numpy(arr)


class Encoder(nn.Module):
    def __init__(self, d=96):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(22, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 15
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 7
            nn.Conv2d(64, d, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.eff = nn.Linear(d, len(EFFECTS))
        self.feat = nn.Linear(d, len(FEATS))

    def forward(self, x):                         # x: (B, D, 22, 30, 30)
        B, D = x.shape[:2]
        z = self.conv(x.reshape(B * D, 22, PAD, PAD)).reshape(B, D, -1)   # (B,D,d)
        z = z.mean(1)                              # mean-pool over demos
        return self.eff(z), self.feat(z), z


def gen_batch(rng, combos, bs):
    """sample bs tasks from the given (effect,feature)-combo set; return tensors + labels."""
    xs, ye, yf = [], [], []
    while len(xs) < bs:
        eff, feat = combos[rng.randint(0, len(combos))]
        if eff == "colormap":
            rt = "colormap"
        else:
            d = G.DECOMPS[rng.randint(0, len(G.DECOMPS))]
            rt = G.rtype_id(eff, d, feat)
        out = G.sample_task(np.random.RandomState(rng.randint(0, 2**31 - 1)), rtype=rt)
        if out is None:
            continue
        demos, lab = out
        xs.append(task_tensor(demos))
        ye.append(EFF_IX[eff]); yf.append(FEAT_IX[feat] if eff != "colormap" else 0)
    return torch.stack(xs), torch.tensor(ye), torch.tensor(yf)


def main():
    rng = np.random.RandomState(0)
    # ---- combo split: hold out specific (effect,feature) pairs entirely from training ----
    recolor_feats = ["size", "color", "holes", "height", "width", "rank_size", "uniq_size", "uniq_color"]
    select_feats = ["size", "color", "holes", "height", "width", "rank_size", "uniq_size", "uniq_color"]
    HELDOUT = [("recolor", "holes"), ("recolor", "uniq_color"), ("select", "size"), ("select", "uniq_size")]
    train_combos = [("colormap", "size")]  # colormap (feature ignored)
    for f in recolor_feats:
        if ("recolor", f) not in HELDOUT: train_combos.append(("recolor", f))
    for f in select_feats:
        if ("select", f) not in HELDOUT: train_combos.append(("select", f))
    print(f"train combos: {len(train_combos)} | held-out combos: {HELDOUT}")

    net = Encoder().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    t0 = time.time()
    STEPS, BS = 1500, 32
    for step in range(STEPS):
        x, ye, yf = gen_batch(rng, train_combos, BS)
        le, lf, _ = net(x)
        # feature loss only on non-colormap tasks
        mask = (ye != EFF_IX["colormap"])
        loss = F.cross_entropy(le, ye)
        if mask.any():
            loss = loss + F.cross_entropy(lf[mask], yf[mask])
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 300 == 0:
            print(f"  step {step+1}/{STEPS} loss {loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)

    # ---- eval: trained combos vs HELD-OUT combos (compositional generalization) ----
    net.eval()
    def evaluate(combos, n=300):
        eok = fok = jok = tot = 0
        with torch.no_grad():
            for _ in range(n // BS + 1):
                x, ye, yf = gen_batch(rng, combos, BS)
                le, lf, _ = net(x)
                pe = le.argmax(1); pf = lf.argmax(1)
                m = (ye != EFF_IX["colormap"])
                eok += (pe == ye).sum().item()
                fok += ((pf == yf) & m).sum().item()
                jok += (((pe == ye) & (pf == yf)) & m).sum().item()
                tot += len(ye)
        return eok / tot, fok / max(1, m.sum().item() * (n // BS + 1)), jok / tot
    te, tf, tj = evaluate(train_combos)
    he, hf, hj = evaluate([c for c in HELDOUT])
    print(f"\nTRAINED combos:  effect-acc {te:.2f}  feature-acc {tf:.2f}  joint {tj:.2f}")
    print(f"HELD-OUT combos: effect-acc {he:.2f}  feature-acc {hf:.2f}  joint {hj:.2f}")
    print(f"(random baseline: effect {1/len(EFFECTS):.2f}, feature {1/len(FEATS):.2f})")
    print("READ: HELD-OUT feature-acc >> 0.10 means the model predicts a feature it NEVER saw under that "
          "effect, from demos = compositional generalization over the grammar (the Branch-B prerequisite).")
    torch.save(net.state_dict(), "learner_v0.pt")
    print(f"\nsaved learner_v0.pt ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
