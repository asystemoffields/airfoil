#!/usr/bin/env python3
"""
Incubation SCALING step 1b — NAIL AIMING (why goal-informed coverage matters once the latent is big).

Step 1 showed continuous VOLUME coverage de-toys the mechanism, but at a small latent the goal-AGNOSTIC
full-latent coverage won outright (no dilution) and a learned reconstruction READOUT for aiming was
fragile (entanglement faked coverage -> 0%). Here we (1) make the world big with MANY goal-irrelevant
movable directions so agnostic coverage DILUTES (can't afford the expensive locked-axis chain within
budget), and (2) aim ROBUSTLY the way stage (b) actually did — cover the TRUE GOAL OBSERVABLE (the goal
coordinate), not a learned readout. Claim: aimed coverage recovers the locked-direction repurposing
where agnostic dilutes. That restores the stage-(a/b) "aiming matters" result in continuous form.

World (R^28): s[0]=e1 (free goal angle), s[1]=e2 (LOCKED goal angle), s[2:4]=B (hidden register),
s[4:28]=scratch (24 dims, each independently movable, GOAL-IRRELEVANT). Goal-space is a torus.
Ops (NOP=30): 4 home e1-movers ; 24 scratch-movers (op 4+j pushes scratch dim j) ; 1 coupling
(s[1]+=alpha*s[2], the ONLY e2-mover) ; 1 repurposing P (s[2]+=1.5, the ONLY B-mover). Reaching an e2
goal needs P->coupling (mid-plan repurpose), state-dependent (skip P if B already active). Home never
touches e2 or B. The 24 scratch dims are cheap volume for an agnostic coverer to chase.

Three explorers (REINFORCE), trained on TYPICAL (e1) goals only:
  - directed(success)  : reach the e1 goal => fixates on e1 => fails e2.
  - agnostic(coverage) : volume (log-det Gram) of the FULL latent E(s) => chases the 24 cheap scratch
                         dims, neglects the expensive e2 chain within budget => DILUTES.
  - aimed(gcoverage)   : volume of the TRUE GOAL OBSERVABLE torus(s) (cos/sin of e1,e2) => ignores
                         scratch, is forced to open the locked e2 dim => discovers the chain.
Run with /data/llm/.venv/bin/python.
"""
import torch
import torch.nn as nn

torch.manual_seed(0)
NSCR = 24
D = 4 + NSCR            # 28
ALPHA = 1.5
H = 48                  # effect-latent dim
T_TRAIN = 12
NOP = 6 + NSCR          # 30


def reached(s, dim):
    return torch.cos(s[:, dim]) < 0.0


def apply_op(s, op):
    s = s.clone()
    e1 = torch.tensor([2.1, -2.1, 4.2, -4.2])
    for j in range(4):                                   # 0..3 home e1-movers
        m = op == j
        if m.any(): s[m, 0] += e1[j]
    m = (op >= 4) & (op < 4 + NSCR)                      # 4..27 scratch-movers (each a distinct dim)
    if m.any(): s[m, 4 + (op[m] - 4)] += 1.0
    m = op == (4 + NSCR)                                 # coupling: only e2-mover
    if m.any(): s[m, 1] += ALPHA * s[m, 2]
    m = op == (5 + NSCR)                                 # repurposing P: only B-mover
    if m.any(): s[m, 2] += 1.5
    return s


def torus(s):
    return torch.stack([torch.cos(s[:, 0]), torch.sin(s[:, 0]),
                        torch.cos(s[:, 1]), torch.sin(s[:, 1])], -1)


def init_states(n):
    s = torch.zeros(n, D)
    s[:, 0] = torch.randn(n) * 0.3
    s[:, 1] = torch.randn(n) * 0.3
    s[:, 2] = torch.randn(n) * 0.8                       # B: sometimes pre-activated, sometimes not
    s[:, 3] = torch.randn(n) * 0.5
    s[:, 4:] = torch.randn(n, NSCR) * 0.5
    return s


def wide_states(n):
    return torch.randn(n, D) * 2.0


def oh(idx, n):
    v = torch.zeros(idx.shape[0], n); v[torch.arange(idx.shape[0]), idx] = 1.0; return v


class WorldLatent(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(D, 96), nn.ReLU(), nn.Linear(96, H))
        self.fwd = nn.Sequential(nn.Linear(H + NOP, 96), nn.ReLU(), nn.Linear(96, D))

    def E(self, s): return self.enc(s)


def pretrain_latent(steps=6000):
    wl = WorldLatent(); opt = torch.optim.Adam(wl.parameters(), lr=2e-3)
    for _ in range(steps):
        n = 512; s = wide_states(n)
        for _ in range(torch.randint(0, 4, (1,)).item()):
            s = apply_op(s, torch.randint(0, NOP, (n,)))
        op = torch.randint(0, NOP, (n,)); s2 = apply_op(s, op)
        pred = wl.fwd(torch.cat([wl.E(s), oh(op, NOP)], -1))
        loss = ((pred - (s2 - s)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    for p in wl.parameters(): p.requires_grad_(False)
    return wl


class Explorer(nn.Module):
    def __init__(self):
        super().__init__()
        din = D + H + 4 + 1 + 2                          # state + mean-E + mean-torus + step + goal
        self.net = nn.Sequential(nn.Linear(din, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, NOP))

    def forward(self, s, meanE, meanTor, t, g):
        x = [s, meanE, meanTor, torch.full((s.shape[0], 1), t / T_TRAIN), g]
        return self.net(torch.cat(x, -1))


def volume_coverage(Zs):
    Z = torch.stack(Zs, 1)
    Z = Z - Z.mean(1, keepdim=True)
    k = Z.shape[-1]
    G = torch.einsum("ntk,ntj->nkj", Z, Z) / Z.shape[1] + 0.1 * torch.eye(k)
    return torch.linalg.slogdet(G)[1]


def train(wl, mode, steps=6000):
    ex = Explorer(); opt = torch.optim.Adam(ex.parameters(), lr=1.5e-3)
    g_typ = oh(torch.zeros(256, dtype=torch.long), 2)
    for _ in range(steps):
        n = 256; s = init_states(n)
        meanE = wl.E(s); meanTor = torus(s); cnt = 1.0
        logps = torch.zeros(n); ent = torch.zeros(n); solved = torch.zeros(n, dtype=torch.bool)
        Z_full = [wl.E(s)]; Z_goal = [torus(s)]
        for t in range(T_TRAIN):
            d = torch.distributions.Categorical(logits=ex(s, meanE, meanTor, t, g_typ))
            op = d.sample(); logps = logps + d.log_prob(op); ent = ent + d.entropy()
            s = apply_op(s, op)
            e = wl.E(s); Z_full.append(e); Z_goal.append(torus(s))
            meanE = (meanE * cnt + e) / (cnt + 1); meanTor = (meanTor * cnt + torus(s)) / (cnt + 1); cnt += 1
            solved |= reached(s, 0)
        reward = {
            "success":   solved.float(),
            "coverage":  volume_coverage(Z_full),         # goal-AGNOSTIC full-latent volume
            "gcoverage": volume_coverage(Z_goal),         # AIMED: volume of the true goal observable
        }[mode]
        adv = (reward - reward.mean()) / (reward.std() + 1e-6)
        loss = -(logps * adv.detach()).mean() - 0.02 * ent.mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return ex


@torch.no_grad()
def rollout(wl, ex, s, goal_dim, Bs):
    g = oh(torch.full((s.shape[0],), goal_dim), 2)
    meanE = wl.E(s); meanTor = torus(s); cnt = 1.0
    ever = torch.zeros(s.shape[0], dtype=torch.bool); out = {}
    for t in range(max(Bs)):
        op = ex(s, meanE, meanTor, t, g).argmax(-1)
        s = apply_op(s, op)
        e = wl.E(s); meanE = (meanE * cnt + e) / (cnt + 1); meanTor = (meanTor * cnt + torus(s)) / (cnt + 1); cnt += 1
        ever |= reached(s, goal_dim)
        if (t + 1) in Bs: out[t + 1] = ever.float().mean().item()
    return [out[b] for b in Bs]


def evaluate(wl, ex, goal_dim, Bs):
    return rollout(wl, ex, init_states(4000), goal_dim, Bs)


def evaluate_split(wl, ex, Bs, want_needsP):
    s = init_states(12000)
    mask = (s[:, 2] < 1.0) if want_needsP else (s[:, 2] >= 1.0)
    return rollout(wl, ex, s[mask][:4000], 1, Bs)


@torch.no_grad()
def diagnostics(Bs):
    n = 5000
    s = init_states(n); ever = torch.zeros(n, dtype=torch.bool)
    for t in range(max(Bs)):
        op = torch.where(s[:, 2] < 1.0, torch.full((n,), 5 + NSCR), torch.full((n,), 4 + NSCR))
        s = apply_op(s, op); ever |= reached(s, 1)
    oracle = ever.float().mean().item()
    s = init_states(n); ever = torch.zeros(n, dtype=torch.bool)
    for t in range(max(Bs)):
        s = apply_op(s, torch.randint(0, NOP, (n,))); ever |= reached(s, 1)
    return oracle, ever.float().mean().item()


def main():
    f = lambda *a: print(*a, flush=True)
    f("=" * 92)
    f(f"Incubation step 1b — NAIL AIMING: {NOP} ops ({NSCR} goal-irrelevant scratch-movers), latent H={H}")
    f("=" * 92)
    Bs = (4, 6, 8, 10, 12, 16)
    o, rnd = diagnostics(Bs)
    f(f"world-sanity: e2 ORACLE reaches {o*100:.0f}%  |  RANDOM-policy discovery {rnd*100:.1f}%")
    f("pretraining effect-latent (broad interventions)...")
    wl = pretrain_latent()
    directed = train(wl, "success")
    agn = train(wl, "coverage")
    aim = train(wl, "gcoverage")
    hdr = "    budget B:               " + "  ".join(f"{b:>4d}" for b in Bs)
    arms = [("directed(success)   ", directed), ("agnostic(full-latent)", agn), ("aimed(goal-observ.) ", aim)]
    for gd, label in [(0, "TYPICAL goals (free dir e1)"), (1, "REPURPOSING goals (LOCKED dir e2)")]:
        f(f"\n  --- {label} ---"); f(hdr)
        for name, ex in arms:
            f(f"    {name}" + "  ".join(f"{v*100:4.0f}" for v in evaluate(wl, ex, gd, Bs)))
    f("\n  --- REPURPOSING split by B activation (state-dependence) ---")
    for needsP, plab in [(True, "B low: must APPLY P"), (False, "B pre-activated: should SKIP P")]:
        f(f"    [{plab}]"); f(hdr)
        for name, ex in arms:
            f(f"    {name}" + "  ".join(f"{v*100:4.0f}" for v in evaluate_split(wl, ex, Bs, needsP)))
    f("\n" + "=" * 92)
    f("READ: with 24 cheap goal-irrelevant directions, AGNOSTIC volume coverage should DILUTE (chase")
    f("scratch, neglect the locked e2 chain within budget) while AIMED coverage (volume of the true")
    f("goal observable) ignores scratch and discovers the state-dependent repurposing chain. If so,")
    f("AIMING is what makes coverage scale — restoring stage-(a/b) in continuous, high-dim form.")


if __name__ == "__main__":
    main()
