#!/usr/bin/env python3
"""
Incubation step 5 — STRUCTURALLY-DIFFERENT held-out affordance (the hard transfer test).

Trained locked axes 1,2,3 share ONE shape: P_i (arm register i) -> C_i (couple register i in) = a
DEPTH-2, single-register chain. The held-out axis 4 has a STRUCTURALLY DIFFERENT shape: a DEPTH-3,
two-register cascade  P4 (arm reg4a) -> T4 (TRANSFER reg4a->reg4b, an op type NO trained chain uses)
-> C4 (couple reg4b in). The model never sequenced a 3-op chain nor used a register-transfer. Question:
does the imagine+aim routine (coverage-proposer + axis-general target-match selector + progress-gated
fast/slow) fire on a plan whose SHAPE it never saw?
  - might transfer: the routine is defined over EFFECTS/TARGETS, not chain structure (coverage explores
    effects -> may discover the depth-3 route; selector is mechanism-agnostic).
  - might fail: a depth-3 chain is RARER under coverage exploration -> proposer may never generate it ->
    imagination never contains it. The ORACLE ceiling on axis 4 diagnoses which.
Train on axes {0,1,2,3}; evaluate ZERO-SHOT on axis 4 (depth-3) + ref axes. Run with /data/llm/.venv/bin/python.
"""
import torch
import torch.nn as nn

torch.manual_seed(1)
NSCR = 8
D = 10 + NSCR                      # 5 goal angles + reg1,2,3 + reg4a,reg4b + scratch = 18
ALPHA = 1.5
H = 48
T = 10                             # rollout horizon
HELD_OUT = 4
SUBSET = [0, 1, 2, 3]
# index map: s[0:5]=goal angles ; s[5:8]=reg1,reg2,reg3 ; s[8]=reg4a s[9]=reg4b ; s[10:]=scratch


def reached(s, axis):
    return torch.cos(s[:, axis]) < 0.0


def apply_op(s, op):
    s = s.clone()
    e1 = torch.tensor([2.1, -2.1, 4.2, -4.2])
    for j in range(4):                                   # 0..3 free axis 0
        m = op == j
        if m.any(): s[m, 0] += e1[j]
    for i in (1, 2, 3):                                  # depth-2 axes: P_i (reg s[4+i]) then C_i
        p = 4 + 2 * (i - 1); c = p + 1
        m = op == p
        if m.any(): s[m, 4 + i] += 1.5
        m = op == c
        if m.any(): s[m, i] += ALPHA * s[m, 4 + i]
    # depth-3 axis 4: ops 10,11,12 = P4 (reg4a), T4 (reg4b += reg4a), C4 (axis4 += alpha*reg4b)
    m = op == 10
    if m.any(): s[m, 8] += 1.5
    m = op == 11
    if m.any(): s[m, 9] += s[m, 8]
    m = op == 12
    if m.any(): s[m, 4] += ALPHA * s[m, 9]
    sbase = 13                                           # scratch movers
    m = (op >= sbase) & (op < sbase + NSCR)
    if m.any(): s[m, 10 + (op[m] - sbase)] += 1.0
    return s


NOP = 13 + NSCR                   # 21


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
    s[:, 5:10] = torch.randn(n, 5) * 0.8                 # registers (incl reg4a,reg4b) sometimes active
    s[:, 10:] = torch.randn(n, NSCR) * 0.5
    return s


def wide_states(n):
    return torch.randn(n, D) * 2.0


def oh(idx, k):
    v = torch.zeros(idx.shape[0], k); v[torch.arange(idx.shape[0]), idx] = 1.0; return v


def volume_coverage(Zs):
    Z = torch.stack(Zs, 1); Z = Z - Z.mean(1, keepdim=True); k = Z.shape[-1]
    G = torch.einsum("ntk,ntj->nkj", Z, Z) / Z.shape[1] + 0.1 * torch.eye(k)
    return torch.linalg.slogdet(G)[1]


# ---------- frozen latent ----------
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
        op = torch.randint(0, NOP, (n,)); s2 = apply_op(s, op)
        loss = ((wl.fwd(torch.cat([wl.E(s), oh(op, NOP)], -1)) - (s2 - s)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    for p in wl.parameters(): p.requires_grad_(False)
    return wl


# ---------- coverage-proposer (+memory) ----------
class Proposer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(H + 10 + 1, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, NOP))

    def forward(self, e, mem, t):
        return self.net(torch.cat([e, mem, torch.full((e.shape[0], 1), t / T)], -1))


def train_proposer(wl, steps=3000):
    p = Proposer(); opt = torch.optim.Adam(p.parameters(), lr=1.5e-3)
    for it in range(steps):
        n = 128; s = init_states(n); Z = [torus5(s)]; mem = torus5(s); cnt = 1.0
        logp = torch.zeros(n); ent = torch.zeros(n); beta = 0.05 if it < steps // 2 else 0.02
        for t in range(T):
            d = torch.distributions.Categorical(logits=p(wl.E(s), mem, t))
            op = d.sample(); logp = logp + d.log_prob(op); ent = ent + d.entropy()
            s = apply_op(s, op); tor = torus5(s); Z.append(tor); mem = (mem * cnt + tor) / (cnt + 1); cnt += 1
        r = volume_coverage(Z); adv = (r - r.mean()) / (r.std() + 1e-6)
        loss = -(logp * adv.detach()).mean() - beta * ent.mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return p


class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(H + 10, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, emb, t):
        m = emb.shape[1]
        return self.net(torch.cat([emb, t.unsqueeze(1).expand(-1, m, -1)], -1)).squeeze(-1)


def train_selector(wl, steps=2500):
    V = ValueNet(); opt = torch.optim.Adam(V.parameters(), lr=2e-3); bce = nn.BCEWithLogitsLoss()
    for _ in range(steps):
        n = 256; s = init_states(n)
        axis = torch.tensor(SUBSET)[torch.randint(0, len(SUBSET), (n,))]
        t = target_obs(s, axis); L = torch.randint(1, 7, (1,)).item(); term = s
        for _ in range(L):
            term = apply_op(term, torch.randint(0, NOP, (n,)))
        y = (torch.cos(term[torch.arange(n), axis]) < 0.0).float()
        loss = bce(V(wl.E(term).unsqueeze(1), t).squeeze(1), y)
        opt.zero_grad(); loss.backward(); opt.step()
    return V


# ---------- fast Arch-2 ----------
class GenPolicy(nn.Module):
    def __init__(self, dm=64, nhead=4, nlayers=2):
        super().__init__()
        self.state_emb = nn.Linear(H, dm); self.goal_emb = nn.Linear(10, dm); self.pos = nn.Embedding(T + 6, dm)
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(dm, nhead, dim_feedforward=128, batch_first=True), nlayers)
        self.head = nn.Linear(dm, NOP)

    def forward(self, seq, t):
        L = seq.shape[1]
        toks = self.state_emb(seq) + self.pos(torch.arange(1, L + 1)).unsqueeze(0)
        g = (self.goal_emb(t) + self.pos(torch.zeros(1, dtype=torch.long))).unsqueeze(1)
        return self.head(self.tf(torch.cat([g, toks], 1))[:, 0])


def train_fast(wl, steps=1500):
    pol = GenPolicy(); opt = torch.optim.Adam(pol.parameters(), lr=1e-3)
    for _ in range(steps):
        n = 256; s = init_states(n)
        ax = torch.tensor(SUBSET)[torch.randint(0, len(SUBSET), (n,))]; t = target_obs(s, ax)
        seq = [wl.E(s)]; logp = torch.zeros(n); ent = torch.zeros(n); ever = torch.zeros(n, dtype=torch.bool)
        for _ in range(T):
            d = torch.distributions.Categorical(logits=pol(torch.stack(seq, 1), t))
            op = d.sample(); logp = logp + d.log_prob(op); ent = ent + d.entropy()
            s = apply_op(s, op); seq.append(wl.E(s)); ever |= (torch.cos(s[torch.arange(n), ax]) < 0.0)
        r = ever.float(); adv = (r - r.mean()) / (r.std() + 1e-6)
        loss = -(logp * adv.detach()).mean() - 0.01 * ent.mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return pol


# ---------- imagination + hybrids ----------
def imagine_prop(wl, prop, s, K, L):
    n = s.shape[0]; S = s.repeat_interleave(K, 0); mem = torus5(S); cnt = 1.0; first = None; cur = S
    for l in range(L):
        op = torch.distributions.Categorical(logits=prop(wl.E(cur), mem, l)).sample()
        if l == 0: first = op
        cur = apply_op(cur, op); tor = torus5(cur); mem = (mem * cnt + tor) / (cnt + 1); cnt += 1
    return first.view(n, K), cur.view(n, K, D)


@torch.no_grad()
def run_A(wl, prop, V, axis, Bs, K=16, L=7, n=1500):
    s = init_states(n); ever = torch.zeros(n, dtype=torch.bool); out = {}
    for t in range(max(Bs)):
        first, term = imagine_prop(wl, prop, s, K, L)
        k = V(wl.E(term.reshape(n * K, D)).view(n, K, H), target_obs(s, axis)).argmax(1)
        s = apply_op(s, first[torch.arange(n), k]); ever |= reached(s, axis)
        if (t + 1) in Bs: out[t + 1] = ever.float().mean().item()
    return [out[b] for b in Bs]


@torch.no_grad()
def run_C(wl, fast, prop, V, axis, Bs, K=16, L=7, n=1500, stuck_max=2):
    s = init_states(n); ar = torch.arange(n); ever = torch.zeros(n, dtype=torch.bool); out = {}
    seq = [wl.E(s)]; mode_fast = torch.ones(n, dtype=torch.bool); stuck = torch.zeros(n); prev = torch.cos(s[:, axis])
    for t in range(max(Bs)):
        fop = fast(torch.stack(seq, 1), target_obs(s, axis)).argmax(-1)
        first, term = imagine_prop(wl, prop, s, K, L)
        dop = first[ar, V(wl.E(term.reshape(n * K, D)).view(n, K, H), target_obs(s, axis)).argmax(1)]
        op = torch.where(mode_fast, fop, dop); s = apply_op(s, op); seq.append(wl.E(s)); cur = torch.cos(s[:, axis])
        prog = cur < prev - 1e-3
        stuck = torch.where(mode_fast & ~prog, stuck + 1, torch.where(mode_fast, torch.zeros(n), stuck))
        mode_fast = mode_fast & (stuck < stuck_max); prev = cur; ever |= reached(s, axis)
        if (t + 1) in Bs: out[t + 1] = ever.float().mean().item()
    return [out[b] for b in Bs]


@torch.no_grad()
def run_arch2(wl, fast, axis, Bs, n=1500):
    s = init_states(n); ever = torch.zeros(n, dtype=torch.bool); out = {}; seq = [wl.E(s)]
    for t in range(max(Bs)):
        op = fast(torch.stack(seq, 1), target_obs(s, axis)).argmax(-1)
        s = apply_op(s, op); seq.append(wl.E(s)); ever |= reached(s, axis)
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
    f(f"Incubation step 5 — STRUCTURALLY-DIFFERENT held-out affordance (axis {HELD_OUT}=depth-3; train {SUBSET}=depth-2)")
    f("=" * 94)
    f("diagnostics — imagine-ORACLE ceiling (does random imagination even CONTAIN the chains?):"); f(hdr)
    f("    oracle axis 4 (DEPTH-3)   " + "  ".join(f"{v*100:4.0f}" for v in oracle(4, Bs)))
    f("    oracle axis 1 (depth-2)   " + "  ".join(f"{v*100:4.0f}" for v in oracle(1, Bs)))
    f("pretraining frozen latent..."); wl = pretrain_latent()
    f("training coverage-proposer, selector, fast Arch-2..."); prop = train_proposer(wl); V = train_selector(wl); fast = train_fast(wl)
    for axis, lab in [(HELD_OUT, f"HELD-OUT axis {HELD_OUT} = STRUCTURALLY DIFFERENT (depth-3) ZERO-SHOT"),
                      (1, "trained-ref axis 1 (depth-2)"), (0, "free axis 0")]:
        f(f"\n  --- {lab} ---"); f(hdr)
        f("    Arch2 alone (reactive)    " + "  ".join(f"{v*100:4.0f}" for v in run_arch2(wl, fast, axis, Bs)))
        f("    Hybrid A (imagine+aim)    " + "  ".join(f"{v*100:4.0f}" for v in run_A(wl, prop, V, axis, Bs)))
        f("    Hybrid C (progress-gated) " + "  ".join(f"{v*100:4.0f}" for v in run_C(wl, fast, prop, V, axis, Bs)))
    f("\n" + "=" * 94)
    f("READ: does the routine fire on a plan SHAPE it never saw (depth-3, with a register-transfer op no")
    f("trained chain uses)? Oracle axis-4 = whether imagination even contains it. If Hybrid A/C transfer")
    f("to axis 4 (>>reactive 0), the routine is STRUCTURE-GENERAL (effect/target-defined), not family-bound.")


if __name__ == "__main__":
    main()
