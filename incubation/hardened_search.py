#!/usr/bin/env python3
"""
Incubation step 8 — HARDENED dynamics + a PANEL of better-than-random proposers (multi-seed).

Two open holes from step 7:
  (1) the realism gate was VACUOUS — the world was affine (+=const, +=ALPHA*reg) so the learned forward
      model hit Δs MSE ~0.000; there was no model error to compound, so "learned beam ≈ perfect beam"
      tested nothing.
  (2) the headline depth-3 number was seed-favorable (55 vs 90 across two seeds) — never quantified.

This file closes both. The world is HARDENED so the learned model has IRREDUCIBLE error:
  - process noise: every op adds N(0, NOISE) to the full state (model can only learn the MEAN -> MSE floor)
  - nonlinear couplings: C_i: s[i] += ALPHA*tanh(reg) ; T4: s9 += s8*(1+0.3*tanh(s8)) ; C4: s4 += ALPHA*tanh(s9)
    (an MLP must APPROXIMATE these -> residual error on the rare chain ops, esp. C4 — where drift compounds)
Reachability is preserved (ALPHA bumped to 2.2 so the chains still push cos<0).

Then we race a PANEL of structure-general "better than random" planners on the HELD-OUT depth-3 axis,
all PLANNING over the LEARNED model and ACTING in the true (noisy) world (MPC re-plan each step):
  0. random oracle                  — the floor (pick the op of the best random k-rollout)
  1. value-to-go beam   (k=6, W=10) — the step-6/7 champion
  2. value-to-go greedy (k=6, W=1)  — does beam WIDTH matter?
  3. value short-horizon(k=1, W=10) — does crediting the SETUP move matter? (k=1 ~ goal-classifier)
  4. value+novelty beam (k=6, W=10) — keep-dims-open: V_togo + latent-novelty bonus
  5. novelty+reach beam (no value)  — pure curiosity (goal-blind except the reach bonus)
  6. value-to-go beam over PERFECT model (k=6, W=10) — upper reference; gap to row 1 = the REALISM COST

Reported across SEEDS as mean[min..max] on the held-out axis. Run with /data/llm/.venv/bin/python.
"""
import torch
import torch.nn as nn

# ---------------- hardened world ----------------
NSCR = 8
D = 10 + NSCR
ALPHA = 2.2                         # bumped so tanh-saturated couplings still reach cos<0
NOISE = 0.12                        # process noise -> irreducible forward-model MSE floor
H = 48
T = 10
HELD_OUT = 4
SUBSET = [0, 1, 2, 3]
NOP = 13 + NSCR                     # 21
# index map: s[0:5]=goal angles ; s[5:8]=reg1,2,3 ; s[8]=reg4a s[9]=reg4b ; s[10:]=scratch


def reached(s, axis):
    return torch.cos(s[:, axis]) < 0.0


def apply_op(s, op, noise=True):
    s = s.clone()
    e1 = torch.tensor([2.1, -2.1, 4.2, -4.2])
    for j in range(4):                                   # 0..3 free axis 0
        m = op == j
        if m.any(): s[m, 0] += e1[j]
    for i in (1, 2, 3):                                  # depth-2 axes: P_i then C_i (NONLINEAR couple)
        p = 4 + 2 * (i - 1); c = p + 1
        m = op == p
        if m.any(): s[m, 4 + i] += 1.5
        m = op == c
        if m.any(): s[m, i] += ALPHA * torch.tanh(s[m, 4 + i])
    m = op == 10                                         # P4: arm reg4a
    if m.any(): s[m, 8] += 1.5
    m = op == 11                                         # T4: NONLINEAR state-dependent transfer reg4a->reg4b
    if m.any(): s[m, 9] += s[m, 8] * (1.0 + 0.3 * torch.tanh(s[m, 8]))
    m = op == 12                                         # C4: NONLINEAR couple reg4b into axis 4
    if m.any(): s[m, 4] += ALPHA * torch.tanh(s[m, 9])
    sbase = 13                                           # scratch movers
    m = (op >= sbase) & (op < sbase + NSCR)
    if m.any(): s[m, 10 + (op[m] - sbase)] += 1.0
    if noise:
        s = s + NOISE * torch.randn_like(s)
    return s


def torus5(s):
    cs = [torch.cos(s[:, a]) for a in range(5)]; sn = [torch.sin(s[:, a]) for a in range(5)]
    return torch.stack([v for pr in zip(cs, sn) for v in pr], -1)


def target_obs(s, axis):
    t = torus5(s).clone(); n = s.shape[0]; ar = torch.arange(n)
    ax = axis if torch.is_tensor(axis) else torch.full((n,), axis)
    t[ar, 2 * ax] = -1.0; t[ar, 2 * ax + 1] = 0.0
    return t


def init_states(n):
    s = torch.zeros(n, D)
    s[:, 0:5] = torch.randn(n, 5) * 0.3
    s[:, 5:10] = torch.randn(n, 5) * 0.8
    s[:, 10:] = torch.randn(n, NSCR) * 0.5
    return s


def wide_states(n):
    return torch.randn(n, D) * 2.0


def oh(idx, k):
    v = torch.zeros(idx.shape[0], k); v[torch.arange(idx.shape[0]), idx] = 1.0; return v


# ---------------- frozen latent + forward model ----------------
class WorldLatent(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(D, 96), nn.ReLU(), nn.Linear(96, H))
        self.fwd = nn.Sequential(nn.Linear(H + NOP, 96), nn.ReLU(), nn.Linear(96, D))

    def E(self, s): return self.enc(s)


def pretrain_latent(steps=3000):
    wl = WorldLatent(); opt = torch.optim.Adam(wl.parameters(), lr=2e-3)
    for _ in range(steps):
        n = 512; s = wide_states(n)
        for _ in range(torch.randint(0, 5, (1,)).item()):
            s = apply_op(s, torch.randint(0, NOP, (n,)))
        op = torch.randint(0, NOP, (n,)); s2 = apply_op(s, op)            # target carries process noise
        loss = ((wl.fwd(torch.cat([wl.E(s), oh(op, NOP)], -1)) - (s2 - s)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    for p in wl.parameters(): p.requires_grad_(False)
    return wl


@torch.no_grad()
def fwd_diagnostic(wl, n=6000):
    s = init_states(n)
    for _ in range(torch.randint(0, 4, (1,)).item()):
        s = apply_op(s, torch.randint(0, NOP, (n,)))
    rows = []
    for j in range(NOP):
        true = apply_op(s, torch.full((n,), j), noise=False) - s            # mean (noiseless) target
        pred = wl.fwd(torch.cat([wl.E(s), oh(torch.full((n,), j), NOP)], -1))
        rows.append(((pred - true) ** 2).mean().item())
    return sum(rows) / len(rows), rows[10], rows[11], rows[12]              # overall, P4, T4, C4


# ---------------- value-to-go ----------------
class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(H + 10, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, emb, t):                          # emb:(n,m,H), t:(n,10) -> (n,m)
        m = emb.shape[1]
        return self.net(torch.cat([emb, t.unsqueeze(1).expand(-1, m, -1)], -1)).squeeze(-1)


def train_value_togo(wl, k=6, steps=3000):
    """V_togo(s,t) ~ P(random k-step rollout from s reaches the target axis). Trained on SUBSET axes only
    -> structure-general. k controls how much SETUP credit it gives (k=1 ~ goal-classifier)."""
    V = ValueNet(); opt = torch.optim.Adam(V.parameters(), lr=2e-3); bce = nn.BCEWithLogitsLoss()
    for _ in range(steps):
        n = 256; s = init_states(n)
        axis = torch.tensor(SUBSET)[torch.randint(0, len(SUBSET), (n,))]; t = target_obs(s, axis)
        cur = s.clone(); ever = (torch.cos(cur[torch.arange(n), axis]) < 0.0)
        for _ in range(k):
            cur = apply_op(cur, torch.randint(0, NOP, (n,)))
            ever |= (torch.cos(cur[torch.arange(n), axis]) < 0.0)
        loss = bce(V(wl.E(s).unsqueeze(1), t).squeeze(1), ever.float())
        opt.zero_grad(); loss.backward(); opt.step()
    return V


# ---------------- planners (panel of scoring rules) ----------------
def make_perfect():
    return lambda states, op: apply_op(states, op, noise=False)            # true MEAN dynamics


def make_learned(wl):
    return lambda states, op: states + wl.fwd(torch.cat([wl.E(states), oh(op, NOP)], -1))


def sc_value(V, emb, t, kids, axis, mem):
    reach = (torch.cos(kids[:, :, axis]) < 0.0).float()
    return V(emb, t) + 6.0 * reach


def sc_value_novelty(V, emb, t, kids, axis, mem):
    reach = (torch.cos(kids[:, :, axis]) < 0.0).float()
    nov = (emb - mem.unsqueeze(1)).pow(2).mean(-1)
    return V(emb, t) + 6.0 * reach + 1.5 * nov


def sc_novelty(V, emb, t, kids, axis, mem):                                # goal-blind except reach
    reach = (torch.cos(kids[:, :, axis]) < 0.0).float()
    nov = (emb - mem.unsqueeze(1)).pow(2).mean(-1)
    return 6.0 * reach + 1.5 * nov


@torch.no_grad()
def plan_first_op(wl, V, s, axis, W, L, trans, score, mem):
    n = s.shape[0]; ar = torch.arange(n); t = target_obs(s, axis)

    def expand(states):
        m = states.shape[1]
        kids = torch.stack([trans(states.reshape(n * m, D), torch.full((n * m,), j)) for j in range(NOP)], 1)
        kids = kids.view(n, m * NOP, D)
        emb = wl.E(kids.reshape(n * m * NOP, D)).view(n, m * NOP, H)
        return kids, score(V, emb, t, kids, axis, mem)

    kids, sc = expand(s.unsqueeze(1))
    topv, topi = sc.topk(min(W, NOP), 1)
    beam = kids.gather(1, topi.unsqueeze(-1).expand(-1, -1, D)); first = topi.clone()
    best_first = first[ar, topv.argmax(1)]
    for _ in range(L - 1):
        Wc = beam.shape[1]
        kids, sc = expand(beam)
        first_rep = first.repeat_interleave(NOP, 1)
        topv, topi = sc.topk(min(W, Wc * NOP), 1)
        beam = kids.gather(1, topi.unsqueeze(-1).expand(-1, -1, D))
        first = first_rep.gather(1, topi)
        best_first = first[ar, topv.argmax(1)]
    return best_first


@torch.no_grad()
def run_search(wl, V, axis, Bs, W, trans, score, L=5, n=400):
    s = init_states(n); ever = torch.zeros(n, dtype=torch.bool); out = {}; mem = wl.E(s)
    for t in range(max(Bs)):
        op = plan_first_op(wl, V, s, axis, W, L, trans, score, mem)
        s = apply_op(s, op); mem = 0.9 * mem + 0.1 * wl.E(s); ever |= reached(s, axis)
        if (t + 1) in Bs: out[t + 1] = ever.float().mean().item()
    return [out[b] for b in Bs]


@torch.no_grad()
def oracle(axis, Bs, K=16, L=7, n=2000):
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


SEEDS = (1, 2, 3)
Bs = (4, 8, 12)


def main():
    f = lambda *a: print(*a, flush=True)
    hdr = "      budget B:               " + "  ".join(f"{b:>4d}" for b in Bs)
    f("=" * 96)
    f(f"Incubation step 8 — HARDENED dynamics + better-than-random PANEL (held-out axis {HELD_OUT}=depth-3)")
    f(f"   process noise={NOISE}, nonlinear couplings; seeds={SEEDS}")
    f("=" * 96)

    # rows accumulate held-out-axis results across seeds: name -> list of per-seed [B...] lists
    rows = {n: [] for n in ["random oracle", "value k6 beam W10", "value k6 greedy W1",
                            "value k1 beam W10", "value+novelty k6 W10", "novelty+reach W10",
                            "value k6 beam PERFECT"]}
    ref = {}  # champion on ref axes, last seed only (sanity)

    for sd in SEEDS:
        torch.manual_seed(sd)
        f(f"\n----- seed {sd} -----")
        wl = pretrain_latent()
        ov, p4, t4, c4 = fwd_diagnostic(wl)
        f(f"  fwd Δs MSE: overall {ov:.4f} | chain P4 {p4:.4f}  T4 {t4:.4f}  C4 {c4:.4f}")
        Vk6 = train_value_togo(wl, k=6); Vk1 = train_value_togo(wl, k=1)
        learned, perfect = make_learned(wl), make_perfect()
        ax = HELD_OUT
        res = {
            "random oracle":          oracle(ax, Bs),
            "value k6 beam W10":      run_search(wl, Vk6, ax, Bs, 10, learned, sc_value),
            "value k6 greedy W1":     run_search(wl, Vk6, ax, Bs, 1,  learned, sc_value),
            "value k1 beam W10":      run_search(wl, Vk1, ax, Bs, 10, learned, sc_value),
            "value+novelty k6 W10":   run_search(wl, Vk6, ax, Bs, 10, learned, sc_value_novelty),
            "novelty+reach W10":      run_search(wl, Vk6, ax, Bs, 10, learned, sc_novelty),
            "value k6 beam PERFECT":  run_search(wl, Vk6, ax, Bs, 10, perfect, sc_value),
        }
        f(hdr)
        for name, v in res.items():
            rows[name].append(v)
            f(f"      {name:<22s}" + "  ".join(f"{x*100:4.0f}" for x in v))
        if sd == SEEDS[-1]:                                               # sanity: champion on ref axes
            ref["axis1 depth-2"] = run_search(wl, Vk6, 1, Bs, 10, learned, sc_value)
            ref["axis0 free"]    = run_search(wl, Vk6, 0, Bs, 10, learned, sc_value)

    f("\n" + "=" * 96)
    f(f"SUMMARY — held-out depth-3 axis {HELD_OUT}, across {len(SEEDS)} seeds  (mean[min..max], % reached)")
    f(hdr)
    for name, runs in rows.items():
        cells = []
        for j in range(len(Bs)):
            vals = [r[j] * 100 for r in runs]
            cells.append(f"{sum(vals)/len(vals):3.0f}[{min(vals):3.0f}..{max(vals):3.0f}]")
        f(f"      {name:<22s}" + "  ".join(cells))
    f("\n  champion (value k6 beam) on reference axes, last seed:")
    f(hdr)
    for name, v in ref.items():
        f(f"      {name:<22s}" + "  ".join(f"{x*100:4.0f}" for x in v))
    f("\n" + "=" * 96)
    f("READ: (1) fwd MSE > 0 now (esp. C4) -> the realism gate is no longer vacuous; the gap between")
    f("'value k6 beam' (learned model) and 'value k6 beam PERFECT' is the REAL cost of an imperfect model.")
    f("(2) the mean[min..max] columns give the honest seed spread the step-7 single number hid.")
    f("(3) the panel shows WHICH better-than-random idea wins: beam>greedy (width), k6>k1 (setup credit),")
    f("and whether a novelty/keep-dims-open bonus helps or hurts on a structurally-novel chain.")


if __name__ == "__main__":
    main()
