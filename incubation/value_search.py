#!/usr/bin/env python3
"""
Incubation step 6 — VALUE-GUIDED SEARCH over the frozen world-model (beat the random floor, NO LLM).

Step 5 left the structurally-novel (depth-3) ceiling stuck at the RANDOM-imagination floor (~42%): the
selector (aiming) was structure-general but the proposer (incubation) was structure-bound. Fix the
imaginer WITHOUT an LLM by replacing random rollouts with SEARCH guided by a structure-general value.

KEY correction (why 1-ply Hybrid B failed): the old selector was a GOAL-CLASSIFIER (is THIS state at the
goal?) -> gives no credit to a setup move (post-P4 state hasn't reached -> scored low). A search needs a
VALUE-TO-GO heuristic: does this state LEAD to the target within a few steps? That credits P4 (post-P4 is
closer) and guides the search down the chain. We train V_togo by random-rollout reachability (= value of
the random policy, subset axes only -> structure-general); using SEARCH to improve over it = one step of
policy improvement, structure-agnostic by construction.

Compare on the held-out DEPTH-3 axis (+ refs): reactive (0) ; random oracle (~42 ceiling) ; greedy on
V_togo (W=1) ; beam search on V_togo (W=10). World + frozen latent reused from multiaxis_struct.
Run with /data/llm/.venv/bin/python.
"""
import torch
import torch.nn as nn

import multiaxis_struct as MS

torch.manual_seed(2)
apply_op, init_states, target_obs, torus5, reached, oh = (
    MS.apply_op, MS.init_states, MS.target_obs, MS.torus5, MS.reached, MS.oh)
NOP, D, H, HELD_OUT, SUBSET = MS.NOP, MS.D, MS.H, MS.HELD_OUT, MS.SUBSET


class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(H + 10, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, emb, t):                          # emb:(n,m,H), t:(n,10) -> (n,m)
        m = emb.shape[1]
        return self.net(torch.cat([emb, t.unsqueeze(1).expand(-1, m, -1)], -1)).squeeze(-1)


def train_value_togo(wl, k=6, steps=3500):
    """V_togo(s,t) ~ P(a random k-step rollout from s reaches the target axis). Higher for states ON the
    way to the chain (credits setup moves). Trained on SUBSET axes only -> structure-general test on axis 4."""
    V = ValueNet(); opt = torch.optim.Adam(V.parameters(), lr=2e-3); bce = nn.BCEWithLogitsLoss()
    for _ in range(steps):
        n = 256; s = init_states(n)
        axis = torch.tensor(SUBSET)[torch.randint(0, len(SUBSET), (n,))]; t = target_obs(s, axis)
        cur = s.clone(); ever = (torch.cos(cur[torch.arange(n), axis]) < 0.0)
        for _ in range(k):
            cur = apply_op(cur, torch.randint(0, NOP, (n,)))
            ever |= (torch.cos(cur[torch.arange(n), axis]) < 0.0)
        y = ever.float()
        loss = bce(V(wl.E(s).unsqueeze(1), t).squeeze(1), y)
        opt.zero_grad(); loss.backward(); opt.step()
    return V


@torch.no_grad()
def plan_first_op(wl, V, s, axis, W, L):
    """beam search over the frozen world-model guided by V_togo; returns the first op of the best plan."""
    n = s.shape[0]; ar = torch.arange(n); t = target_obs(s, axis)

    def expand(states):                                  # states:(n,m,D) -> children:(n, m*NOP, D), scores, first-carry-ready
        m = states.shape[1]
        kids = torch.stack([apply_op(states.reshape(n * m, D), torch.full((n * m,), j)) for j in range(NOP)], 1)
        kids = kids.view(n, m * NOP, D)                  # child order: [m0(op0..opN), m1(op0..opN), ...]
        emb = wl.E(kids.reshape(n * m * NOP, D)).view(n, m * NOP, H)
        reach = (torch.cos(kids[:, :, axis]) < 0.0).float()
        sc = V(emb, t) + 6.0 * reach                     # value-to-go + reach bonus
        return kids, sc

    # depth 1: expand the root by all ops
    kids, sc = expand(s.unsqueeze(1))                    # (n, NOP, D), (n, NOP)
    topv, topi = sc.topk(min(W, NOP), 1)
    beam = kids.gather(1, topi.unsqueeze(-1).expand(-1, -1, D))   # (n,W,D)
    first = topi.clone()                                 # op index == first op (root had 1 node)
    best_first = first[ar, topv.argmax(1)]               # first op of the best beam node so far
    # depths 2..L
    for _ in range(L - 1):
        Wc = beam.shape[1]
        kids, sc = expand(beam)                          # (n, Wc*NOP, D), (n, Wc*NOP)
        first_rep = first.repeat_interleave(NOP, 1)      # carry first op of each parent
        topv, topi = sc.topk(min(W, Wc * NOP), 1)
        beam = kids.gather(1, topi.unsqueeze(-1).expand(-1, -1, D))
        first = first_rep.gather(1, topi)
        best_first = first[ar, topv.argmax(1)]
    return best_first


@torch.no_grad()
def run_search(wl, V, axis, Bs, W, L=5, n=600):
    s = init_states(n); ever = torch.zeros(n, dtype=torch.bool); out = {}
    for t in range(max(Bs)):
        op = plan_first_op(wl, V, s, axis, W, L)
        s = apply_op(s, op); ever |= reached(s, axis)
        if (t + 1) in Bs: out[t + 1] = ever.float().mean().item()
    return [out[b] for b in Bs]


@torch.no_grad()
def oracle(axis, Bs, K=16, L=7, n=3000):
    s = init_states(n); ever = torch.zeros(n, dtype=torch.bool); out = {}
    for t in range(max(Bs)):
        S = s.repeat_interleave(K, 0); first = torch.randint(0, NOP, (n * K,)); cur = apply_op(S, first)
        for _ in range(L - 1):
            cur = apply_op(cur, torch.randint(0, NOP, (n * K,)))
        reach = (torch.cos(cur.view(n, K, D)[:, :, axis]) < 0.0)
        k = reach.float().argmax(1)
        s = apply_op(s, first.view(n, K)[torch.arange(n), k]); ever |= reached(s, axis)
        if (t + 1) in Bs: out[t + 1] = ever.float().mean().item()
    return [out[b] for b in Bs]


def main():
    f = lambda *a: print(*a, flush=True)
    Bs = (2, 4, 6, 8, 10)
    hdr = "    budget B:                  " + "  ".join(f"{b:>4d}" for b in Bs)
    f("=" * 94)
    f(f"Incubation step 6 — VALUE-GUIDED SEARCH over frozen world-model (held-out axis {HELD_OUT}=depth-3)")
    f("=" * 94)
    f("pretraining frozen latent..."); wl = MS.pretrain_latent()
    f("training V_togo (value-to-go, subset axes only)..."); V = train_value_togo(wl)
    for axis, lab in [(HELD_OUT, f"HELD-OUT axis {HELD_OUT} = depth-3 (STRUCTURALLY DIFFERENT) ZERO-SHOT"),
                      (1, "trained-ref axis 1 (depth-2)"), (0, "free axis 0")]:
        f(f"\n  --- {lab} ---"); f(hdr)
        f("    random oracle (~ceiling) " + "  ".join(f"{v*100:4.0f}" for v in oracle(axis, Bs)))
        f("    greedy  V_togo (W=1)     " + "  ".join(f"{v*100:4.0f}" for v in run_search(wl, V, axis, Bs, W=1)))
        f("    beam    V_togo (W=10)    " + "  ".join(f"{v*100:4.0f}" for v in run_search(wl, V, axis, Bs, W=10)))
    f("\n" + "=" * 94)
    f("READ: if beam/greedy on V_togo clears the random oracle on the DEPTH-3 axis, value-guided search")
    f("(frozen WM + structure-general value-to-go) lifts the structurally-novel ceiling above random —")
    f("NO LLM, reusing only the two pieces we trust. Value-to-go (not goal-classifier) is the fix that")
    f("credits the chain's setup move and lets the search find depth-3 chains it never trained on.")


if __name__ == "__main__":
    main()
