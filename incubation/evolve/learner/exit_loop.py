#!/usr/bin/env python3
"""Vine RL loop — the COMPOSE-POLICY head + STaR round + leave-one-FAMILY-out FALSIFIER (bricks 3-6, the GO/NO-GO).

The compose head learns the INNER-SELECTION V2 structurally can't surface (a property of the OTHER object). The
honest test: COMPOSED-ANCHOR families "recolor the object contained in the <ANCHOR>" where ANCHOR varies (largest /
tallest / unique-color). The OUTER is containment (V2 ranks it); the INNER is the anchor-defining concept, which
DIFFERS per family but shares the principle "the inner is the UNIQUELY-satisfied object." Distill the head on N-1
anchor families, score on the never-seen one: if held-out cost-to-solve FALLS over STaR rounds (and stays solved),
the policy LEARNED a generalizing inner-selection principle = reasoning-over-experience pays. If flat, the gradient
adds nothing beyond search/reuse (the honest NO-GO the synthesis flagged). Run: /data/llm/.venv/bin/python exit_loop.py"""
import sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import grammar as G
import rel_dsl as D
import substrate_eye as SE

sys.path.insert(0, "/data/Windows-files/Documents/airfoil/incubation/evolve")
torch.manual_seed(0)
rng = np.random.RandomState(0)

# the OUTER (containment) is fixed for these families; the head selects the INNER from candidates:
OUTER = (1, 1, "exists")                           # pair_signature ch=1 (b_contains_a), exists
INNERS = [
    ("is-largest",  D.Quantify(4, 1, "forall")),                                  # forall a_bigger
    ("is-tallest",  SE.SubQuantify(SE.SubChannel("h", ">", "h"), "forall")),
    ("uniq-color",  D.FeatKey("uniq_color")),
    ("is-smallest", SE.SubQuantify(SE.SubChannel("size", "<", "size"), "forall")),  # distractor (picks dots)
    ("has-adjacent", D.Quantify(2, 1, "exists")),                                 # distractor (many satisfy)
    ("is-widest",   SE.SubQuantify(SE.SubChannel("w", ">", "w"), "forall")),      # distractor
]
ANCHOR_FAMS = ["largest", "tallest", "uniq-color"]


def make_anchor_family(anchor, n):
    demos = []
    for _ in range(n):
        g = np.zeros((28, 28), int); boxes = []
        specs = [(8, 3), (5, 8), (4, 5)]; cols = [5, 5, 5]
        if anchor == "uniq-color":
            cols = [6, 5, 5]; rng.shuffle(cols)
        order = list(range(3)); rng.shuffle(order)
        for i in order:
            h, w = specs[i]
            for _t in range(60):
                r, c = rng.randint(0, 28 - h), rng.randint(0, 28 - w)
                if all(abs(r - br) > h + 2 or abs(c - bc) > w + 2 for br, bc, _, _ in boxes):
                    g[r:r+h, c:c+w] = cols[i]; g[r+1:r+h-1, c+1:c+w-1] = 0
                    g[r+h//2, c+w//2] = 4; boxes.append((r, c, h, w)); break
        out = g.copy(); objs = G.objects(g, 4, True)
        box_objs = [o for o in objs if o["size"] > 4]
        if anchor == "largest":   anc = max(box_objs, key=lambda o: o["size"])
        elif anchor == "tallest": anc = max(box_objs, key=lambda o: o["h"])
        else:                     anc = [o for o in box_objs if o["color"] == 6][0]
        for o in objs:
            col = 2 if D._contains(anc, o) else 3
            for (rr, cc) in o["cells"]: out[rr, cc] = col
        demos.append((g, out))
    return demos


def inner_feats(demos, inner):
    """per-inner features that GENERALIZE across anchors: is it UNIQUELY satisfied (an anchor) + its fraction."""
    gi = np.asarray(demos[0][0], int); objs = G.objects(gi, 4, True)
    sat = sum(int(bool(inner(o, objs))) for o in objs)
    return [1.0 if sat == 1 else 0.0, sat / max(1, len(objs))]


def solves(demos_tr, demos_te, inner):
    prog = D.induce_recolor(D.Composed(OUTER[0], OUTER[1], inner, OUTER[2]), demos_tr)
    return prog is not None and D.verify(prog, demos_te)


class ComposeHead(nn.Module):
    def __init__(self): super().__init__(); self.f = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, X): return self.f(X).squeeze(-1)      # (n_inners,)


def held_out_cost(head, fam, M=12):
    """rank inners by the head; cost = position of the first inner that SOLVES (induce-calls). + solve-rate."""
    costs = []; solved = 0
    for _ in range(M):
        tr, te = make_anchor_family(fam, 4), make_anchor_family(fam, 2)
        X = torch.tensor([inner_feats(tr, inn) for _n, inn in INNERS], dtype=torch.float32)
        with torch.no_grad():
            order = head(X).argsort(descending=True).tolist()
        n = 0; hit = False
        for idx in order:
            n += 1
            if solves(tr, te, INNERS[idx][1]): hit = True; break
        costs.append(n); solved += int(hit)
    return float(np.mean(costs)), solved / M


def main():
    t0 = time.time()
    head = ComposeHead(); opt = torch.optim.Adam(head.parameters(), lr=5e-3)
    HELD = "uniq-color"; distill = [f for f in ANCHOR_FAMS if f != HELD]
    print(f"COMPOSE-POLICY falsifier — distill on {distill}, HELD-OUT '{HELD}'")
    c0, s0 = held_out_cost(head, HELD)
    print(f"  round 0 (untrained head): held-out cost {c0:.2f}  solve {s0:.2f}")
    for rnd in range(1, 5):
        # STaR: on distill families, the VERIFIED inner is the reward=1 target; train head to rank it top
        Xs, ys = [], []
        for fam in distill:
            for _ in range(16):
                tr, te = make_anchor_family(fam, 4), make_anchor_family(fam, 2)
                feats = [inner_feats(tr, inn) for _n, inn in INNERS]
                winners = [i for i, (_n, inn) in enumerate(INNERS) if solves(tr, te, inn)]
                if not winners: continue
                Xs.append(torch.tensor(feats, dtype=torch.float32)); ys.append(winners[0])
        for _ in range(40):
            for X, y in zip(Xs, ys):
                loss = F.cross_entropy(head(X).unsqueeze(0), torch.tensor([y]))
                opt.zero_grad(); loss.backward(); opt.step()
        c, s = held_out_cost(head, HELD)
        print(f"  round {rnd}: held-out cost {c:.2f}  solve {s:.2f}  (distilled on {len(Xs)} verified composed solves)")
    print(f"\n[{time.time()-t0:.0f}s] READ: held-out cost FALLS over rounds at held solve-rate = the compose head "
          "LEARNED a generalizing inner-selection principle ('inner = the uniquely-satisfied anchor') from OTHER "
          "families and applied it to an anchor it never trained on = reasoning-over-experience pays. Flat = NO-GO "
          "(gradient adds nothing beyond search; the expressiveness track is the move).")


if __name__ == "__main__":
    main()
