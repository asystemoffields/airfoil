#!/usr/bin/env python3
"""Incubation step 12 — can the SEARCH be AMORTIZED into a fast learned proposer that still transfers to a
NOVEL chain? (the learned-not-heuristic + deployability question, in the controllable substrate.)

Value-guided search solves the held-out DEPTH-3 axis at ~90% (step 6) but COSTS test-time compute (step 11:
that compute IS the creative lever). Can we DISTILL it into a fast reactive net R(state,goal)->op — trained
ONLY on the search's solutions for the SUBSET axes {0,1,2,3} (depth-2) — and have R transfer ZERO-SHOT to the
structurally-different held-out depth-3 axis 4? Two outcomes, both decisive:
  - R TRANSFERS: creativity amortizes into weights -> replace search with a free reactive net at inference (huge
    for the 7GB target).
  - R FAILS (like the directly-trained reactive policy, step5/11 = 0 on held-out): the NOVEL requires SEARCH at
    test time; creativity does NOT compress into params -> reinforces size-for-time (search is irreducible).
Compare on the held-out axis: reactive-distilled-from-search (R) vs the search itself vs random oracle.
Run with /data/llm/.venv/bin/python."""
import torch
import torch.nn as nn

import multiaxis_struct as MS
import value_search as VS

torch.manual_seed(0)
apply_op, init_states, target_obs, reached = MS.apply_op, MS.init_states, MS.target_obs, MS.reached
NOP, D, H, HELD_OUT, SUBSET, T = MS.NOP, MS.D, MS.H, MS.HELD_OUT, MS.SUBSET, MS.T
Bs = (2, 4, 6, 8, 10)


class Recognizer(nn.Module):                               # fast reactive proposer R(E(s), goal) -> op
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(H + 10, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, NOP))

    def forward(self, e, t): return self.net(torch.cat([e, t], -1))


@torch.no_grad()
def collect(wl, V, episodes=60, n=256):
    """run the value-guided SEARCH on SUBSET axes; log (E(s), goal, chosen first-op) at each MPC step."""
    E, G, Y = [], [], []
    for ep in range(episodes):
        axis = SUBSET[ep % len(SUBSET)]                    # scalar axis per episode (plan_first_op expects scalar)
        s = init_states(n)
        for _ in range(T):
            t = target_obs(s, axis)
            op = VS.plan_first_op(wl, V, s, axis, W=10, L=5)   # the search's choice = supervision
            E.append(wl.E(s)); G.append(t); Y.append(op)
            s = apply_op(s, op)
    return torch.cat(E), torch.cat(G), torch.cat(Y)


def train_R(E, G, Y, steps=4000):
    R = Recognizer(); opt = torch.optim.Adam(R.parameters(), lr=1e-3); ce = nn.CrossEntropyLoss()
    N = E.shape[0]
    for _ in range(steps):
        idx = torch.randint(0, N, (256,))
        loss = ce(R(E[idx], G[idx]), Y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return R


@torch.no_grad()
def run_R(wl, R, axis, n=1500):
    s = init_states(n); ever = torch.zeros(n, dtype=torch.bool); out = {}
    for tt in range(max(Bs)):
        op = R(wl.E(s), target_obs(s, axis)).argmax(-1)
        s = apply_op(s, op); ever |= reached(s, axis)
        if (tt + 1) in Bs: out[tt + 1] = ever.float().mean().item()
    return [out[b] for b in Bs]


def main():
    f = lambda *a: print(*a, flush=True)
    hdr = "    budget B:                   " + "  ".join(f"{b:>4d}" for b in Bs)
    f("=" * 92)
    f(f"Incubation step 12 — AMORTIZE the search into a learned proposer R; transfer to held-out axis {HELD_OUT}?")
    f("=" * 92)
    f("pretrain latent + V_togo..."); wl = MS.pretrain_latent(); V = VS.train_value_togo(wl)
    f("collecting search demonstrations on SUBSET axes..."); E, G, Y = collect(wl, V)
    f(f"  {E.shape[0]} (state,goal,op) demos from search"); R = train_R(E, G, Y)
    f("  trained recognizer R\n")
    for axis, lab in [(HELD_OUT, f"HELD-OUT axis {HELD_OUT} = depth-3 (NOVEL) ZERO-SHOT"),
                      (1, "subset-ref axis 1 (depth-2, in distillation)")]:
        f(f"  --- {lab} ---"); f(hdr)
        f("    random oracle (floor)      " + "  ".join(f"{v*100:4.0f}" for v in VS.oracle(axis, Bs)))
        f("    R distilled-from-search    " + "  ".join(f"{v*100:4.0f}" for v in run_R(wl, R, axis)))
        f("    search (expert, W=10)      " + "  ".join(f"{v*100:4.0f}" for v in VS.run_search(wl, V, axis, Bs, W=10)))
    f("\n" + "=" * 92)
    f("READ: on the held-out depth-3 axis, if R (distilled, reactive, NO search) ~ the search -> creativity")
    f("AMORTIZES into weights (deploy free). If R ~ random/0 while search stays high -> the NOVEL requires")
    f("SEARCH at inference; creativity does NOT compress into params -> size-for-time is irreducible. R on the")
    f("subset-ref axis should be high either way (it distilled those) — that isolates transfer from competence.")


if __name__ == "__main__":
    main()
