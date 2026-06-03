#!/usr/bin/env python3
"""Incubation step 16 — SCALE DEPTH: does the search+value stack hold as the held-out chain deepens?
World with goal axes of GROWING chain depth: 0 free; 1,2,3 depth-2 (P_i->C_i); 4 depth-3 (P4->T4->C4);
5 DEPTH-4 (P5->T5a->T5b->C5). Train the value ONLY on subset {0,1,2,3} (depth<=2); test ZERO-SHOT on axis 4
(depth-3) and axis 5 (depth-4). Question: does the beam+value ceiling HOLD or fall off a cliff with depth?
(robustness check on the triangulated thesis.) tanh couplings (ALPHA=2.2) keep reach well-defined at any depth.
Run with /data/llm/.venv/bin/python."""
import torch
import torch.nn as nn

torch.manual_seed(0)
NSCR = 6
D = 14 + NSCR                     # goals 0:6 ; reg1,2,3=6,7,8 ; reg4a,4b=9,10 ; reg5a,5b,5c=11,12,13 ; scratch 14:
ALPHA = 2.2
H = 56
T = 12
SUBSET = [0, 1, 2, 3]
NOP = 17 + NSCR                   # free 0-3 ; P1C1P2C2P3C3 4-9 ; P4T4C4 10-12 ; P5 T5a T5b C5 13-16 ; scratch 17+
NGOAL = 6


def reached(s, axis): return torch.cos(s[:, axis]) < 0.0


def apply_op(s, op):
    s = s.clone()
    e1 = torch.tensor([2.1, -2.1, 4.2, -4.2])
    for j in range(4):
        m = op == j
        if m.any(): s[m, 0] += e1[j]
    for i in (1, 2, 3):                                    # depth-2
        p = 4 + 2 * (i - 1); c = p + 1
        m = op == p
        if m.any(): s[m, 5 + i] += 1.5
        m = op == c
        if m.any(): s[m, i] += ALPHA * torch.tanh(s[m, 5 + i])
    m = op == 10                                           # depth-3 axis 4
    if m.any(): s[m, 9] += 1.5
    m = op == 11
    if m.any(): s[m, 10] += s[m, 9]
    m = op == 12
    if m.any(): s[m, 4] += ALPHA * torch.tanh(s[m, 10])
    m = op == 13                                           # depth-4 axis 5
    if m.any(): s[m, 11] += 1.5
    m = op == 14
    if m.any(): s[m, 12] += s[m, 11]
    m = op == 15
    if m.any(): s[m, 13] += s[m, 12]
    m = op == 16
    if m.any(): s[m, 5] += ALPHA * torch.tanh(s[m, 13])
    sbase = 17
    m = (op >= sbase) & (op < sbase + NSCR)
    if m.any(): s[m, 14 + (op[m] - sbase)] += 1.0
    return s


def torusg(s):
    cs = [torch.cos(s[:, a]) for a in range(NGOAL)]; sn = [torch.sin(s[:, a]) for a in range(NGOAL)]
    return torch.stack([v for pr in zip(cs, sn) for v in pr], -1)


def target_obs(s, axis):
    t = torusg(s).clone(); n = s.shape[0]; ar = torch.arange(n)
    ax = axis if torch.is_tensor(axis) else torch.full((n,), axis)
    t[ar, 2 * ax] = -1.0; t[ar, 2 * ax + 1] = 0.0
    return t


def init_states(n):
    s = torch.zeros(n, D)
    s[:, 0:NGOAL] = torch.randn(n, NGOAL) * 0.3
    s[:, 6:14] = torch.randn(n, 8) * 0.8
    s[:, 14:] = torch.randn(n, NSCR) * 0.5
    return s


def wide_states(n): return torch.randn(n, D) * 2.0
def oh(idx, k):
    v = torch.zeros(idx.shape[0], k); v[torch.arange(idx.shape[0]), idx] = 1.0; return v
def reach_ax(s, av): n = s.shape[0]; return torch.cos(s[torch.arange(n), av]) < 0.0


class WorldLatent(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(D, 112), nn.ReLU(), nn.Linear(112, H))
        self.fwd = nn.Sequential(nn.Linear(H + NOP, 112), nn.ReLU(), nn.Linear(112, D))
    def E(self, s): return self.enc(s)


def pretrain_latent(steps=3500):
    wl = WorldLatent(); opt = torch.optim.Adam(wl.parameters(), lr=2e-3)
    for _ in range(steps):
        n = 512; s = wide_states(n)
        for _ in range(torch.randint(0, 6, (1,)).item()):
            s = apply_op(s, torch.randint(0, NOP, (n,)))
        op = torch.randint(0, NOP, (n,)); s2 = apply_op(s, op)
        loss = ((wl.fwd(torch.cat([wl.E(s), oh(op, NOP)], -1)) - (s2 - s)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    for p in wl.parameters(): p.requires_grad_(False)
    return wl


class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(H + 2 * NGOAL, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, emb, t):
        m = emb.shape[1]
        return self.net(torch.cat([emb, t.unsqueeze(1).expand(-1, m, -1)], -1)).squeeze(-1)


def train_value_bellman(wl, gamma=0.92, steps=5000):
    V = ValueNet(); opt = torch.optim.Adam(V.parameters(), lr=2e-3); bce = nn.BCEWithLogitsLoss()
    for _ in range(steps):
        n = 256; s = init_states(n)
        axis = torch.tensor(SUBSET)[torch.randint(0, len(SUBSET), (n,))]; t = target_obs(s, axis)
        with torch.no_grad():
            now = reach_ax(s, axis); best = torch.zeros(n)
            for op in range(NOP):
                child = apply_op(s, torch.full((n,), op))
                best = torch.maximum(best, torch.maximum(reach_ax(child, axis).float(),
                                                         torch.sigmoid(V(wl.E(child).unsqueeze(1), t).squeeze(1))))
            target = torch.where(now, torch.ones(n), gamma * best)
        loss = bce(V(wl.E(s).unsqueeze(1), t).squeeze(1), target)
        opt.zero_grad(); loss.backward(); opt.step()
    return V


@torch.no_grad()
def plan_first_op(wl, V, s, axis, W, L):
    n = s.shape[0]; ar = torch.arange(n); t = target_obs(s, axis)
    def expand(states):
        m = states.shape[1]
        kids = torch.stack([apply_op(states.reshape(n * m, D), torch.full((n * m,), j)) for j in range(NOP)], 1).view(n, m * NOP, D)
        emb = wl.E(kids.reshape(n * m * NOP, D)).view(n, m * NOP, H)
        reach = (torch.cos(kids[:, :, axis]) < 0.0).float()
        return kids, V(emb, t) + 6.0 * reach
    kids, sc = expand(s.unsqueeze(1)); topv, topi = sc.topk(min(W, NOP), 1)
    beam = kids.gather(1, topi.unsqueeze(-1).expand(-1, -1, D)); first = topi.clone(); bf = first[ar, topv.argmax(1)]
    for _ in range(L - 1):
        Wc = beam.shape[1]; kids, sc = expand(beam); fr = first.repeat_interleave(NOP, 1)
        topv, topi = sc.topk(min(W, Wc * NOP), 1)
        beam = kids.gather(1, topi.unsqueeze(-1).expand(-1, -1, D)); first = fr.gather(1, topi); bf = first[ar, topv.argmax(1)]
    return bf


@torch.no_grad()
def run_search(wl, V, axis, Bs, W=12, L=6, n=800):
    s = init_states(n); ever = torch.zeros(n, dtype=torch.bool); out = {}
    for tt in range(max(Bs)):
        op = plan_first_op(wl, V, s, axis, W, L); s = apply_op(s, op); ever |= reached(s, axis)
        if (tt + 1) in Bs: out[tt + 1] = ever.float().mean().item()
    return [out[b] for b in Bs]


@torch.no_grad()
def oracle(axis, Bs, K=24, L=9, n=2000):
    s = init_states(n); ever = torch.zeros(n, dtype=torch.bool); out = {}
    for tt in range(max(Bs)):
        S = s.repeat_interleave(K, 0); first = torch.randint(0, NOP, (n * K,)); cur = apply_op(S, first)
        for _ in range(L - 1): cur = apply_op(cur, torch.randint(0, NOP, (n * K,)))
        reach = (torch.cos(cur.view(n, K, D)[:, :, axis]) < 0.0); k = reach.float().argmax(1)
        s = apply_op(s, first.view(n, K)[torch.arange(n), k]); ever |= reached(s, axis)
        if (tt + 1) in Bs: out[tt + 1] = ever.float().mean().item()
    return [out[b] for b in Bs]


def main():
    f = lambda *a: print(*a, flush=True)
    Bs = (2, 4, 6, 8, 10, 12)
    hdr = "    budget B:               " + "  ".join(f"{b:>4d}" for b in Bs)
    f("=" * 96); f("Incubation step 16 — SCALE DEPTH: search+value (trained on depth<=2) vs held-out depth-3 AND depth-4")
    f("=" * 96)
    f("pretrain latent + Bellman value (subset depth<=2 only)..."); wl = pretrain_latent(); V = train_value_bellman(wl)
    f("")
    for axis, lab in [(5, "HELD-OUT axis 5 = DEPTH-4 (P5->T5a->T5b->C5) ZERO-SHOT"),
                      (4, "held-out axis 4 = depth-3 ZERO-SHOT"),
                      (1, "subset-ref axis 1 = depth-2")]:
        f(f"  --- {lab} ---"); f(hdr)
        f("    random oracle (floor)  " + "  ".join(f"{v*100:4.0f}" for v in oracle(axis, Bs)))
        f("    search + V_bellman     " + "  ".join(f"{v*100:4.0f}" for v in run_search(wl, V, axis, Bs)))
    f("\n" + "=" * 96)
    f("READ: depth-2 (trained) high; the held-out depth-3 and depth-4 ceilings show whether the stack HOLDS as")
    f("the never-trained chain deepens, or falls off a cliff. A depth-4 chain is exponentially rarer under random")
    f("rollout (the value's bootstrap source) and needs more beam depth/width -> expect a lower but >floor ceiling.")


if __name__ == "__main__":
    main()
