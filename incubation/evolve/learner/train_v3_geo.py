#!/usr/bin/env python3
"""Branch-B scale-prep BOX-PREP 1 — V3-GEO, the STRUCTURAL twin of the V2 consistency recognizer.

V2 works because it computes deterministic pairwise CONSISTENCY stats and applies ONE shared, feature-index-
AGNOSTIC scorer -> generalizes to unseen features by construction. V3-GEO is the structural version: for each
candidate pre-op p in an enumerable bank, compute deterministic STRUCTURAL-consistency stats over the demos
(how well p(input) aligns with output: shape-ratio, nonzero-mask IoU, residual cell-match, object-count match,
all mean+std-pooled across demos = the cross-demo consistency signal), then ONE shared, transform-index-AGNOSTIC
scorer ranks the candidates. The op identity is NEVER an input to the scorer (exactly like V2's feature index),
so it transfers to HELD-OUT pre-ops without retraining. PASS BAR (V2-style unit test): held-out pre-op top-M
recall >> chance. Run: /data/llm/.venv/bin/python train_v3_geo.py"""
import sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import grammar as G
import grammar_comp as GC

sys.path.insert(0, "/data/Windows-files/Documents/airfoil/incubation/arc")
import dsl

torch.manual_seed(0); torch.set_num_threads(6)
BANK = ["identity", "reflect_h", "reflect_v", "rot90", "rot180", "rot270", "transpose", "crop_content",
        "gravity_down", "gravity_up", "gravity_left", "gravity_right", "shift_up", "shift_down",
        "shift_left", "shift_right", "sym_lr", "sym_ud", "tile_h2", "tile_v2", "tile_2x2", "scale2",
        "downscale2", "trim_border"]
BANK_IX = {n: i for i, n in enumerate(BANK)}
HELDOUT = ["rot270", "transpose", "gravity_left", "shift_right", "sym_ud", "scale2"]  # never trained, tested
TRAINBANK = [p for p in BANK if p not in HELDOUT]
SDIM = 16   # 8 structural stats x (mean, std); the 8th is the V2H-style EVIDENCE/support stat (raw alignment volume)


def _pre(name, g):
    if name == "identity":
        return np.asarray(g, int)
    try:
        out = dsl.OPS[name][0](np.asarray(g, int))
        return np.asarray(out, int) if out is not None else None
    except Exception:
        return None


def cand_stats(demos, p):
    """structural-consistency stats for candidate pre-op p over the demos -> 14-vec (7 stats x mean,std)."""
    rows = []
    for gi, go in demos:
        gi = np.asarray(gi, int); go = np.asarray(go, int)
        gp = _pre(p, gi)
        if gp is None or gp.size == 0:
            rows.append([0, 0, 0, 0, 0, 0, 0]); continue
        sm = 1.0 if gp.shape == go.shape else 0.0
        rh = min(go.shape[0] / gp.shape[0], 4) / 4.0
        rw = min(go.shape[1] / gp.shape[1], 4) / 4.0
        if gp.shape == go.shape:
            cell = float((gp == go).mean())
            inter = ((gp != 0) & (go != 0)).sum(); uni = ((gp != 0) | (go != 0)).sum()
            nz = float(inter) / max(1, int(uni))
        else:
            cell = 0.0; nz = 0.0
        no_gp = len(G.objects(gp, 4, True)); no_go = len(G.objects(go, 4, True))
        nom = 1.0 / (1 + abs(no_gp - no_go))
        support = np.log1p(int((gp == go).sum()) if gp.shape == go.shape else 0) / 6.0  # V2H-style EVIDENCE volume
        rows.append([sm, rh, rw, cell, nz, nom, min(no_gp, 20) / 20.0, support])
    a = np.asarray(rows, np.float32)
    return np.concatenate([a.mean(0), a.std(0)]).astype(np.float32)


def task_stats(demos):
    return np.stack([cand_stats(demos, p) for p in BANK]).astype(np.float32)   # (|BANK|, 14)


class V3GEO(nn.Module):
    def __init__(self):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(SDIM, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, S):                 # S: (B, |BANK|, SDIM)
        return self.score(S).squeeze(-1)  # (B, |BANK|)

    def nparams(self):
        return sum(p.numel() for p in self.parameters())


def make_set(rng, bank, n):
    """generate n composed tasks with pre_op drawn from `bank`; cache (stats, label)."""
    S = []; Y = []
    tries = 0
    while len(S) < n and tries < n * 40:
        tries += 1
        p = bank[rng.randint(0, len(bank))]
        out = GC.sample_composed(np.random.RandomState(rng.randint(0, 2**31 - 1)), pre_name=p)
        if out is None:
            continue
        demos, (pre_name, _brt) = out
        if len(demos) < 2:
            continue
        S.append(task_stats(demos)); Y.append(BANK_IX[pre_name])
    return torch.from_numpy(np.stack(S)), torch.tensor(Y)


def main():
    rng = np.random.RandomState(0)
    t0 = time.time()
    Str, Ytr = make_set(rng, TRAINBANK, 2000)
    Ste, Yte = make_set(rng, HELDOUT, 500)
    print(f"V3-GEO structural twin | bank {len(BANK)} pre-ops, held-out {HELDOUT} | "
          f"train {len(Ytr)} / held-out {len(Yte)} tasks  [{time.time()-t0:.0f}s]")
    net = V3GEO(); opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    print(f"  params {net.nparams()/1e3:.1f}K")
    BS = 64
    for ep in range(400):
        perm = torch.randperm(len(Ytr))
        for i in range(0, len(Ytr), BS):
            idx = perm[i:i + BS]
            logits = net(Str[idx])
            loss = F.cross_entropy(logits, Ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        def recall(S, Y, M):
            top = net(S).topk(M, dim=1).indices
            return (top == Y.unsqueeze(1)).any(1).float().mean().item()
        chance1 = 1 / len(BANK); chance3 = 3 / len(BANK)
        print(f"\nTRAINED pre-ops:  top-1 {recall(Str,Ytr,1):.2f}  top-3 {recall(Str,Ytr,3):.2f}")
        print(f"HELD-OUT pre-ops: top-1 {recall(Ste,Yte,1):.2f}  top-3 {recall(Ste,Yte,3):.2f}   "
              f"(chance top-1 {chance1:.2f}, top-3 {chance3:.2f})")
    torch.save(net.state_dict(), "learner_v3_geo.pt")
    print("READ: held-out top-3 >> chance = the structural-consistency scorer RANKS pre-ops it never trained on, "
          "by structural alignment alone = V3-GEO is V2's structural twin (transform-agnostic, generalizes by "
          f"construction). [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
