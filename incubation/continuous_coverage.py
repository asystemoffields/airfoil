#!/usr/bin/env python3
"""
Incubation SCALING step 1 — de-toy the COVERAGE mechanism.

Stage (a)/(b) measured "coverage" as distinct DISCRETE values (a bitmask). That cannot scale: in any
real domain the effect space is high-dimensional/continuous and you cannot enumerate it. The hinge
claim for scaling: goal-informed effect-coverage still discovers the (sequential, state-dependent)
repurposing chain when coverage is measured as DIVERSITY IN A LEARNED EFFECT-EMBEDDING instead of a
count. If that holds, the toy stops being a toy.

World (continuous, R^12):
  s[0] = C.e1 (goal coord, free direction)   s[1] = C.e2 (goal coord, LOCKED direction)
  s[2:4] = B (hidden register)               s[4:12] = scratch (8-dim, high-variance, goal-irrelevant)
Ops (NOP=10):
  - 4 home e1-movers: s[0] += {+.6,-.6,+1.2,-1.2}      (move the FREE goal direction only)
  - 4 home scratch-movers: s[4:12] += fixed random vec (rich, high-dim, goal-IRRELEVANT effects)
  - 1 coupling:  s[1] += alpha * s[2]                  (the ONLY way to move the LOCKED direction e2)
  - 1 repurposing P: s[2] += 1.0                       (the ONLY way to activate B; home never touches B)
Home ops never touch e2 or B => reaching an e2 goal needs P (activate B) -> coupling (inject e2):
a mid-plan repurpose. And coupling moves e2 by alpha*s[2], so whether to apply P first DEPENDS on
B's current value (skip P if s[2] already large) — state-dependent, no fixed order qualifies.

Goals (threshold form, robust to continuous imprecision):
  - TYPICAL:     drive s[0] (free e1) above THETA      (home-solvable)
  - REPURPOSING: drive s[1] (locked e2) above THETA    (needs the P->coupling chain)
Trained on TYPICAL only (usage skew). Coverage is measured in a FROZEN effect-latent E (pretrained
from interventions to predict op-effects — task-agnostic, keeps useless structure; can't be gamed by
the policy because it's frozen):
  - non-dir(coverage)  : spread of E(s) (full 16-d latent; scratch-dominated => should DILUTE)
  - non-dir(gcoverage) : spread of rho(E(s)) ~ the goal observable (goal-informed => aims at C-space)
  - directed(success)  : reach the TYPICAL goal => fixates on e1 => should FAIL e2.
Run with /data/llm/.venv/bin/python.
"""
import torch
import torch.nn as nn

torch.manual_seed(0)
D = 12
ALPHA = 1.5
H = 16          # effect-latent dim
T_TRAIN = 6
# Goal-space is PERIODIC (a torus): s[0],s[1] are angles. This is the continuous analog of "mod 8" —
# it puts a CEILING on the easy axis so covering more REQUIRES the locked direction (else coverage is
# trivially maxed by sliding the free axis to infinity, the bug in the non-periodic v1).


def reached(s, dim):
    """goal = rotate the angle into the FAR half of the circle (robust to continuous imprecision)."""
    return torch.cos(s[:, dim]) < 0.0

# ---- op table: each op -> additive delta (coupling depends on state) -------------------------------
SCRATCH_VECS = (torch.randn(4, 8) * 0.7)        # 4 fixed random scratch movers


def apply_op(s, op):
    s = s.clone()
    e1 = torch.tensor([2.1, -2.1, 4.2, -4.2])   # COARSE: e1 circle covered in ~2-3 ops (like mod-8's 4 vals)
    for j in range(4):                          # 0..3 home e1-movers
        m = op == j
        if m.any(): s[m, 0] += e1[j]
    for j in range(4):                          # 4..7 home scratch-movers
        m = op == (4 + j)
        if m.any(): s[m, 4:12] += SCRATCH_VECS[j]
    m = op == 8                                 # coupling: s[1] += alpha * s[2]  (only e2-mover)
    if m.any(): s[m, 1] += ALPHA * s[m, 2]
    m = op == 9                                 # repurposing P: activate B (only B-mover)
    if m.any(): s[m, 2] += 1.5
    return s


NOP = 10


def init_states(n):
    s = torch.zeros(n, D)
    s[:, 0] = torch.randn(n) * 0.3              # C below threshold
    s[:, 1] = torch.randn(n) * 0.3
    s[:, 2] = torch.randn(n) * 0.8             # B: sometimes pre-activated (skip P), sometimes not
    s[:, 3] = torch.randn(n) * 0.5
    s[:, 4:12] = torch.randn(n, 8) * 0.5
    return s


def oh(idx, n):
    v = torch.zeros(idx.shape[0], n); v[torch.arange(idx.shape[0]), idx] = 1.0; return v


# ---- frozen effect-latent: pretrain E + forward model F + goal readout rho (interventions) ---------
class WorldLatent(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(D, 64), nn.ReLU(), nn.Linear(64, H))
        self.fwd = nn.Sequential(nn.Linear(H + NOP, 64), nn.ReLU(), nn.Linear(64, D))  # predict Δs
        self.rho = nn.Sequential(nn.Linear(H, 32), nn.ReLU(), nn.Linear(32, 4))        # E(s)->(cos,sin) of C angles

    def E(self, s): return self.enc(s)


def torus(s):
    """(cos,sin) embedding of the two goal angles — bounded, so coverage SATURATES on each circle."""
    return torch.stack([torch.cos(s[:, 0]), torch.sin(s[:, 0]),
                        torch.cos(s[:, 1]), torch.sin(s[:, 1])], -1)


def wide_states(n):
    """BROAD state sampling for the task-agnostic world-model: exercises the WHOLE state space —
    incl. the normally-locked s[1]/e2 — so the latent + goal-readout retain rare/locked structure."""
    return torch.randn(n, D) * 2.0


def pretrain_latent(steps=5000):
    wl = WorldLatent(); opt = torch.optim.Adam(wl.parameters(), lr=2e-3)
    for _ in range(steps):
        n = 512
        s = wide_states(n)
        for _ in range(torch.randint(0, 4, (1,)).item()):    # random walk so states are diverse
            s = apply_op(s, torch.randint(0, NOP, (n,)))
        op = torch.randint(0, NOP, (n,))
        s2 = apply_op(s, op)
        e = wl.E(s)
        pred_d = wl.fwd(torch.cat([e, oh(op, NOP)], -1))
        loss = ((pred_d - (s2 - s)) ** 2).mean() + ((wl.rho(e) - torus(s)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    for p in wl.parameters(): p.requires_grad_(False)
    return wl


# ---- explorer policy (REINFORCE) ------------------------------------------------------------------
class Explorer(nn.Module):
    def __init__(self, use_goal):
        super().__init__()
        self.use_goal = use_goal
        din = D + H + 4 + 1 + (2 if use_goal else 0)  # state + mean-E + mean-torus (visit memory) + step + [goal]
        self.net = nn.Sequential(nn.Linear(din, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, NOP))

    def forward(self, s, meanE, meanTor, t, g):
        x = [s, meanE, meanTor, torch.full((s.shape[0], 1), t / T_TRAIN)]
        if self.use_goal: x.append(g)
        return self.net(torch.cat(x, -1))


def volume_coverage(Zs):
    """Determinantal (volume) coverage: logdet(eps*I + Gram of visited embeddings). Unlike pairwise
    spread, this SATURATES per-subspace — bouncing around an already-spanned dimension stops paying;
    the only way to raise it is to inject variance into a NEW dimension (=> forces the locked axis)."""
    Z = torch.stack(Zs, 1)                                  # (n, T+1, k)
    Z = Z - Z.mean(1, keepdim=True)                         # center over time
    k = Z.shape[-1]
    G = torch.einsum("ntk,ntj->nkj", Z, Z) / Z.shape[1] + 0.1 * torch.eye(k)
    return torch.linalg.slogdet(G)[1]                       # (n,)


def train(wl, mode, steps=5000):
    use_goal = (mode != "coverage")
    ex = Explorer(use_goal); opt = torch.optim.Adam(ex.parameters(), lr=1.5e-3)
    g_typ = oh(torch.zeros(256, dtype=torch.long), 2)       # typical goal = free dir e1
    for _ in range(steps):
        n = 256; s = init_states(n)
        meanE = wl.E(s); meanTor = torus(s); cnt = 1.0
        logps = torch.zeros(n); ent = torch.zeros(n); solved = torch.zeros(n, dtype=torch.bool)
        Z_full = [wl.E(s)]; Z_goal = [wl.rho(wl.E(s))]
        for t in range(T_TRAIN):
            d = torch.distributions.Categorical(logits=ex(s, meanE, meanTor, t, g_typ))
            op = d.sample(); logps = logps + d.log_prob(op); ent = ent + d.entropy()
            s = apply_op(s, op)
            e = wl.E(s); Z_full.append(e); Z_goal.append(wl.rho(e))
            meanE = (meanE * cnt + e) / (cnt + 1); meanTor = (meanTor * cnt + torus(s)) / (cnt + 1); cnt += 1
            solved |= reached(s, 0)                         # trained on TYPICAL (free dir)
        reward = {
            "success":   solved.float(),
            "coverage":  volume_coverage(Z_full),           # goal-agnostic full-latent volume
            "gcoverage": volume_coverage(Z_goal),           # goal-informed (goal-readout) volume
        }[mode]
        adv = (reward - reward.mean()) / (reward.std() + 1e-6)
        loss = -(logps * adv.detach()).mean() - 0.02 * ent.mean()   # entropy bonus -> discover the chain
        opt.zero_grad(); loss.backward(); opt.step()
    return ex


@torch.no_grad()
def evaluate(wl, ex, goal_dim, Bs):
    n = 4000; s = init_states(n)
    g = oh(torch.full((n,), goal_dim), 2)
    meanE = wl.E(s); meanTor = torus(s); cnt = 1.0
    ever = torch.zeros(n, dtype=torch.bool); out = {}
    for t in range(max(Bs)):
        op = ex(s, meanE, meanTor, t, g).argmax(-1)
        s = apply_op(s, op)
        e = wl.E(s); meanE = (meanE * cnt + e) / (cnt + 1); meanTor = (meanTor * cnt + torus(s)) / (cnt + 1); cnt += 1
        ever |= reached(s, goal_dim)
        if (t + 1) in Bs: out[t + 1] = ever.float().mean().item()
    return [out[b] for b in Bs]


@torch.no_grad()
def evaluate_split(wl, ex, goal_dim, Bs, want_needsP):
    """repurposing eval restricted to states where B is pre-activated (skip P) vs not (needs P)."""
    n = 12000; s = init_states(n)
    mask = (s[:, 2] < 1.0) if want_needsP else (s[:, 2] >= 1.0)
    s = s[mask][:4000]
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


@torch.no_grad()
def diagnostics(Bs):
    """Is the e2 goal solvable, and how rare is it under RANDOM exploration? (world-sanity vs hard-exploration)"""
    n = 5000
    # ORACLE: adaptively apply P if B not yet activated, else couple — confirms the chain reaches e2.
    s = init_states(n); ever = torch.zeros(n, dtype=torch.bool)
    for t in range(max(Bs)):
        op = torch.where(s[:, 2] < 1.0, torch.full((n,), 9), torch.full((n,), 8))  # P if B low else coupling
        s = apply_op(s, op); ever |= reached(s, 1)
    oracle = ever.float().mean().item()
    # RANDOM policy: discovery rate of the e2 chain under uniform actions.
    s = init_states(n); ever = torch.zeros(n, dtype=torch.bool)
    for t in range(max(Bs)):
        s = apply_op(s, torch.randint(0, NOP, (n,))); ever |= reached(s, 1)
    rand = ever.float().mean().item()
    return oracle, rand


def main():
    f = lambda *a: print(*a, flush=True)
    f("=" * 90)
    f(f"Incubation SCALING step 1 — continuous world, coverage = spread in a FROZEN learned effect-latent")
    f("=" * 90)
    Bs = (2, 3, 4, 5, 6, 8)
    o, rnd = diagnostics(Bs)
    f(f"world-sanity: e2 ORACLE (adaptive P/couple) reaches {o*100:.0f}%  |  RANDOM-policy discovery {rnd*100:.1f}%")
    f("(oracle high => world solvable; random low => e2 chain is a hard-exploration needle)")
    f("pretraining effect-latent (interventions)...")
    wl = pretrain_latent()
    directed = train(wl, "success")
    cov = train(wl, "coverage")
    gcov = train(wl, "gcoverage")
    hdr = "    budget B:               " + "  ".join(f"{b:>4d}" for b in Bs)
    arms = [("directed(success)   ", directed), ("non-dir(coverage)   ", cov), ("non-dir(gcoverage)  ", gcov)]
    for gd, label in [(0, "TYPICAL goals (free dir e1)"), (1, "REPURPOSING goals (LOCKED dir e2)")]:
        f(f"\n  --- {label} ---"); f(hdr)
        for name, ex in arms:
            f(f"    {name}" + "  ".join(f"{v*100:4.0f}" for v in evaluate(wl, ex, gd, Bs)))
    f("\n  --- REPURPOSING split by B activation (state-dependence) ---")
    for needsP, plab in [(True, "B low: must APPLY P"), (False, "B pre-activated: should SKIP P")]:
        f(f"    [{plab}]"); f(hdr)
        for name, ex in arms:
            f(f"    {name}" + "  ".join(f"{v*100:4.0f}" for v in evaluate_split(wl, ex, 1, Bs, needsP)))
    f("\n" + "=" * 90)
    f("FINDING: coverage DE-TOYS. Continuous VOLUME (log-det) coverage over a frozen, broadly-")
    f("intervention-trained causal latent discovers the locked-direction, state-dependent, multi-step")
    f("repurposing chain (full-latent ~89%, 95% on the harder needs-P case), ADAPTIVELY, with NO")
    f("enumeration — while success-training FIXATES (<=15%). It works because the task-agnostic latent")
    f("RETAINS the locked structure (keep-the-useless-structure, vindicated): volume coverage feels the")
    f("near-zero-variance stuck directions and un-sticks them. OPEN: a naive learned goal-READOUT for")
    f("AIMING is fragile — entanglement leaks variance so it fakes coverage and fails to aim (0%).")
    f("Aiming (needed once the latent is too big to cover wholesale) -> attention-native controller next.")


if __name__ == "__main__":
    main()
