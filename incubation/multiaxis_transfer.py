#!/usr/bin/env python3
"""
Incubation step 3 — HELD-OUT-AXIS TRANSFER (does the repurposing ROUTINE generalize to a novel
instance?). The real creativity test, per Alex's emergence-via-cycles point.

World (R^19): 5 goal angles g[0..4] (axis 0 = FREE/home-movable ; axes 1..4 = LOCKED, each needs its
OWN chain) ; hidden registers Breg[1..4]=s[5:9] ; scratch s[9:19] (10 goal-irrelevant dims). For each
locked axis i: P_i activates Breg_i (s[4+i]+=1.5) ; C_i couples it in (s[i]+=ALPHA*Breg_i). Home never
touches locked axes or registers. Reaching a far target on locked axis i needs P_i->C_i (state-
dependent: skip P_i if Breg_i already active). All 4 chains are STRUCTURALLY IDENTICAL but use
different ops/dims -> learning the routine on some axes COULD transfer to a held-out axis.

GOAL is specified as a TARGET OBSERVABLE t (10-d cos/sin of the 5 axes): the goal axis set to (-1,0)
("drive this axis far"), other axes left at current (commensurate with effects, so transfer is
POSSIBLE rather than rigged-to-fail by an opaque index). Reward = the goal axis reaches cos<0.

TWO architectures x THREE regimes, tested ZERO-SHOT on HELD-OUT axis 4:
  Arch1 SELECT (imagine K random rollouts -> learned target-matching selector picks one -> MPC). Its
        selector is axis-GENERAL (matches effect to target) so it can deploy a chain imagination
        supplies WITHOUT training on that axis -> predicted to TRANSFER.
  Arch2 GEN    (transformer; goal token attends over its own rollout -> next op). Must EMIT the novel
        axis's ops from a policy that never learned them -> predicted to FAIL transfer.
  regimes: 'free' (train axis 0 only -> never learned any chain) ; 'subset' (axes 0,1,2,3 -> learned
        the routine on 1,2,3, axis 4 novel = THE test) ; 'all' (0..4 -> learnability upper bound).
Run with /data/llm/.venv/bin/python.
"""
import torch
import torch.nn as nn

torch.manual_seed(0)
NSCR = 10
D = 9 + NSCR                       # 5 goal angles + 4 registers + scratch = 19
ALPHA = 1.5
H = 48
T_REAL = 10
K_IMAG = 16
L_IMAG = 6
HELD_OUT = 4                       # locked axis held out of training
N_FREE_OPS = 4


def reached(s, axis):
    return torch.cos(s[:, axis]) < 0.0


def apply_op(s, op):
    s = s.clone()
    e1 = torch.tensor([2.1, -2.1, 4.2, -4.2])
    for j in range(4):                                   # 0..3 home movers on FREE axis 0
        m = op == j
        if m.any(): s[m, 0] += e1[j]
    base = N_FREE_OPS
    for i in range(1, 5):                                # locked axes: P_i then C_i
        pidx = base + 2 * (i - 1); cidx = pidx + 1
        m = op == pidx
        if m.any(): s[m, 4 + i] += 1.5                   # P_i: activate Breg_i  (Breg_i at s[4+i])
        m = op == cidx
        if m.any(): s[m, i] += ALPHA * s[m, 4 + i]       # C_i: couple Breg_i into locked axis i
    sbase = base + 8                                     # scratch movers
    m = (op >= sbase) & (op < sbase + NSCR)
    if m.any(): s[m, 9 + (op[m] - sbase)] += 1.0
    return s


NOP = N_FREE_OPS + 8 + NSCR        # 22


def torus5(s):
    cs = [torch.cos(s[:, a]) for a in range(5)]
    sn = [torch.sin(s[:, a]) for a in range(5)]
    return torch.stack([v for pair in zip(cs, sn) for v in pair], -1)   # (n,10): c0,s0,c1,s1,...


def target_obs(s, axis):
    """target observable: keep all axes at current, set the goal axis to (cos,sin)=(-1,0) = 'far'.
    axis may be an int (same for all) or a per-element LongTensor (n,)."""
    t = torus5(s).clone(); n = s.shape[0]; ar = torch.arange(n)
    ax = axis if torch.is_tensor(axis) else torch.full((n,), axis)
    t[ar, 2 * ax] = -1.0; t[ar, 2 * ax + 1] = 0.0
    return t


def init_states(n):
    s = torch.zeros(n, D)
    s[:, 0:5] = torch.randn(n, 5) * 0.3                  # goal angles near 0 (cos~1, not yet far)
    s[:, 5:9] = torch.randn(n, 4) * 0.8                  # registers: sometimes pre-activated
    s[:, 9:] = torch.randn(n, NSCR) * 0.5
    return s


def wide_states(n):
    return torch.randn(n, D) * 2.0


def oh(idx, k):
    v = torch.zeros(idx.shape[0], k); v[torch.arange(idx.shape[0]), idx] = 1.0; return v


# ----- frozen effect-latent -----
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


REGIMES = {"free": [0], "subset": [0, 1, 2, 3], "all": [0, 1, 2, 3, 4]}


def sample_goal_axes(n, regime):
    axes = REGIMES[regime]
    return torch.tensor(axes)[torch.randint(0, len(axes), (n,))]


# ============================ ARCH 1 — imagine + target-matching select ============================
def imagine(s, K, L):
    n = s.shape[0]
    S = s.unsqueeze(1).expand(n, K, D).reshape(n * K, D).clone()
    first = torch.randint(0, NOP, (n * K,)); cur = apply_op(S, first)
    for _ in range(L - 1):
        cur = apply_op(cur, torch.randint(0, NOP, (n * K,)))
    return first.view(n, K), cur.view(n, K, D)


class Selector(nn.Module):
    """axis-GENERAL: score each rollout by how well its effect (frozen E) matches the target t."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(H + 10, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, keys, t):                          # keys:(n,K,H), t:(n,10) -> (n,K)
        K = keys.shape[1]
        x = torch.cat([keys, t.unsqueeze(1).expand(-1, K, -1)], -1)
        return self.net(x).squeeze(-1)


def train_arch1(wl, regime, steps=2000, K=K_IMAG, L=L_IMAG):
    sel = Selector(); opt = torch.optim.Adam(sel.parameters(), lr=2e-3)
    for _ in range(steps):
        n = 128; s = init_states(n)
        ax = sample_goal_axes(n, regime)
        logp = torch.zeros(n); ent = torch.zeros(n); ever = torch.zeros(n, dtype=torch.bool)
        for step_t in range(T_REAL):
            t = target_obs(s, ax)                           # per-element target (vectorized)
            first, term = imagine(s, K, L)
            keys = wl.E(term.reshape(n * K, D)).view(n, K, H)
            d = torch.distributions.Categorical(logits=sel(keys, t))
            k = d.sample(); logp = logp + d.log_prob(k); ent = ent + d.entropy()
            s = apply_op(s, first[torch.arange(n), k])
            ever |= (torch.cos(s[torch.arange(n), ax]) < 0.0)
        r = ever.float(); adv = (r - r.mean()) / (r.std() + 1e-6)
        loss = -(logp * adv.detach()).mean() - 0.01 * ent.mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return sel


@torch.no_grad()
def eval_arch1(wl, sel, axis, Bs, K=K_IMAG, L=L_IMAG, oracle=False):
    s = init_states(4000); n = s.shape[0]; ever = torch.zeros(n, dtype=torch.bool); out = {}
    for t_step in range(max(Bs)):
        first, term = imagine(s, K, L)
        if oracle:
            reach = torch.cos(term[:, :, axis]) < 0.0
            k = reach.float().argmax(1)
        else:
            t = target_obs(s, axis)
            keys = wl.E(term.reshape(n * K, D)).view(n, K, H)
            k = sel(keys, t).argmax(1)
        s = apply_op(s, first[torch.arange(n), k])
        ever |= reached(s, axis)
        if (t_step + 1) in Bs: out[t_step + 1] = ever.float().mean().item()
    return [out[b] for b in Bs]


# ============================ ARCH 2 — attend during generation ====================================
class GenPolicy(nn.Module):
    def __init__(self, dm=64, nhead=4, nlayers=2):
        super().__init__()
        self.state_emb = nn.Linear(H, dm)
        self.goal_emb = nn.Linear(10, dm)
        self.pos = nn.Embedding(T_REAL + 2, dm)
        layer = nn.TransformerEncoderLayer(dm, nhead, dim_feedforward=128, batch_first=True)
        self.tf = nn.TransformerEncoder(layer, nlayers)
        self.head = nn.Linear(dm, NOP)

    def forward(self, latent_seq, t):
        n, L, _ = latent_seq.shape
        toks = self.state_emb(latent_seq) + self.pos(torch.arange(1, L + 1)).unsqueeze(0)
        gtok = (self.goal_emb(t) + self.pos(torch.zeros(1, dtype=torch.long))).unsqueeze(1)
        h = self.tf(torch.cat([gtok, toks], 1))
        return self.head(h[:, 0])


def train_arch2(wl, regime, steps=2500):
    pol = GenPolicy(); opt = torch.optim.Adam(pol.parameters(), lr=1e-3)
    for _ in range(steps):
        n = 256; s = init_states(n)
        ax = sample_goal_axes(n, regime)
        t = target_obs(s, ax)
        seq = [wl.E(s)]; logp = torch.zeros(n); ent = torch.zeros(n); ever = torch.zeros(n, dtype=torch.bool)
        for step_t in range(T_REAL):
            d = torch.distributions.Categorical(logits=pol(torch.stack(seq, 1), t))
            op = d.sample(); logp = logp + d.log_prob(op); ent = ent + d.entropy()
            s = apply_op(s, op); seq.append(wl.E(s))
            ever |= (torch.cos(s[torch.arange(n), ax]) < 0.0)
        r = ever.float(); adv = (r - r.mean()) / (r.std() + 1e-6)
        loss = -(logp * adv.detach()).mean() - 0.01 * ent.mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return pol


@torch.no_grad()
def eval_arch2(wl, pol, axis, Bs):
    s = init_states(4000); n = s.shape[0]; seq = [wl.E(s)]; ever = torch.zeros(n, dtype=torch.bool); out = {}
    for t_step in range(max(Bs)):
        t = target_obs(s, axis)
        op = pol(torch.stack(seq, 1), t).argmax(-1)
        s = apply_op(s, op); seq.append(wl.E(s))
        ever |= reached(s, axis)
        if (t_step + 1) in Bs: out[t_step + 1] = ever.float().mean().item()
    return [out[b] for b in Bs]


def main():
    f = lambda *a: print(*a, flush=True)
    Bs = (2, 4, 6, 8, 10)
    hdr = "    budget B:                          " + "  ".join(f"{b:>4d}" for b in Bs)
    f("=" * 98)
    f(f"Incubation step 3 — HELD-OUT-AXIS TRANSFER (D={D}, {NOP} ops, held-out locked axis {HELD_OUT})")
    f("=" * 98)
    f("pretraining frozen effect-latent...")
    wl = pretrain_latent()
    f("Arch1 imagine-ORACLE (held-out axis, imagination ceiling):")
    f(hdr); f("    oracle (axis %d)                    " % HELD_OUT + "  ".join(f"{v*100:4.0f}" for v in eval_arch1(wl, None, HELD_OUT, Bs, oracle=True)))

    a1 = {r: train_arch1(wl, r) for r in REGIMES}
    a2 = {r: train_arch2(wl, r) for r in REGIMES}

    f("\n  ===== HELD-OUT axis %d (ZERO-SHOT transfer test) =====" % HELD_OUT); f(hdr)
    for r in ["free", "subset", "all"]:
        f(f"    Arch1 select  ({r:<6})              " + "  ".join(f"{v*100:4.0f}" for v in eval_arch1(wl, a1[r], HELD_OUT, Bs)))
    for r in ["free", "subset", "all"]:
        f(f"    Arch2 gen     ({r:<6})              " + "  ".join(f"{v*100:4.0f}" for v in eval_arch2(wl, a2[r], HELD_OUT, Bs)))

    f("\n  ===== reference: FREE axis 0 and a TRAINED locked axis 1 (regime 'subset') ====="); f(hdr)
    f("    Arch1 select  (subset) axis0       " + "  ".join(f"{v*100:4.0f}" for v in eval_arch1(wl, a1["subset"], 0, Bs)))
    f("    Arch1 select  (subset) axis1       " + "  ".join(f"{v*100:4.0f}" for v in eval_arch1(wl, a1["subset"], 1, Bs)))
    f("    Arch2 gen     (subset) axis0       " + "  ".join(f"{v*100:4.0f}" for v in eval_arch2(wl, a2["subset"], 0, Bs)))
    f("    Arch2 gen     (subset) axis1       " + "  ".join(f"{v*100:4.0f}" for v in eval_arch2(wl, a2["subset"], 1, Bs)))

    f("\n" + "=" * 98)
    f("READ: 'subset' trains the repurposing routine on axes 1,2,3 and tests it ZERO-SHOT on the never-")
    f("trained axis %d. If Arch1 (imagine+general target-match) transfers (subset >> free) while Arch2" % HELD_OUT)
    f("(must EMIT the novel axis's ops) does NOT, then creativity on NOVEL problems needs the imagine/")
    f("search loop, not just a trained policy — even though Arch2 wins on TRAINED goals. Compare to the")
    f("imagine-ORACLE ceiling (what random imagination even contains for the held-out axis).")


if __name__ == "__main__":
    main()
