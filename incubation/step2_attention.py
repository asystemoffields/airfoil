#!/usr/bin/env python3
"""
Incubation SCALING step 2 — ATTENTION-NATIVE controller ("attend over self-generated rollouts").

Replaces the explicit coverage-search controller with a trainable attention module. Two architectures
(Alex's fork — test both head-to-head), two training regimes, on the step-1b high-dim world.

ARCH 1 — SELECT-AMONG-FINISHED (imagine -> attend -> pick, MPC):
  At each real step, imagine K complete random rollouts (length L) via the world-model; embed each
  through the FROZEN latent E; a learned GOAL-QUERY attends over the K rollout-effects and SELECTS one;
  execute its first op; re-imagine. Incubation (diverse imagined rollouts) and aiming (goal-attention
  selection) are SEPARATE stages. Aiming runs off the ENTANGLED latent (no given observable).

ARCH 2 — ATTEND-DURING-GENERATION (self-attending transformer policy):
  A transformer generates ONE trajectory; at each step a GOAL TOKEN attends over the model's own
  growing rollout (the latent states so far) to emit the next op. Incubation + aiming are FUSED.

REGIMES: train-ALL (both goal dims -> does the mechanism work?) ; train-TYPICAL (e1 only -> does
goal-query attention GENERALIZE to deploy the never-trained locked-axis repurposing chain?).

BASELINES: reactive feedforward policy (fixation control) ; imagination ORACLE for Arch 1 (pick the
rollout a verifier says reaches the goal — upper bound, separates "imagination had it" from "attention
found it"). World/frozen-latent reused from continuous_aiming.  Run with /data/llm/.venv/bin/python.
"""
import torch
import torch.nn as nn

import continuous_aiming as W

torch.manual_seed(0)
D, NOP, H = W.D, W.NOP, W.H
apply_op, init_states, oh = W.apply_op, W.init_states, W.oh
T_REAL = 8
K_IMAG = 16
L_IMAG = 5


def reached_dim(s, gd):
    """gd: (n,) goal-dim per element."""
    return torch.cos(s[torch.arange(s.shape[0]), gd]) < 0.0


def sample_goal_dims(n, train_all):
    return torch.randint(0, 2, (n,)) if train_all else torch.zeros(n, dtype=torch.long)


# ============================ ARCH 1 — select among finished rollouts ==============================
def imagine(s, K, L):
    """s:(n,D) -> first_ops:(n,K), terminal_states:(n,K,D). K random rollouts of length L (true WM)."""
    n = s.shape[0]
    S = s.unsqueeze(1).expand(n, K, D).reshape(n * K, D).clone()
    first = torch.randint(0, NOP, (n * K,))
    cur = apply_op(S, first)
    for _ in range(L - 1):
        cur = apply_op(cur, torch.randint(0, NOP, (n * K,)))
    return first.view(n, K), cur.view(n, K, D)


class Selector(nn.Module):
    """learned goal-query attention over rollout-effect embeddings (keys = frozen E of terminals)."""
    def __init__(self, qd=32):
        super().__init__()
        self.gq = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, qd))
        self.kk = nn.Linear(H, qd)
        self.qd = qd

    def forward(self, goal, keys):                       # goal:(n,2), keys:(n,K,H) -> scores:(n,K)
        q = self.gq(goal).unsqueeze(1)
        k = self.kk(keys)
        return (q * k).sum(-1) / (self.qd ** 0.5)


def train_arch1(wl, train_all, steps=2500, K=K_IMAG, L=L_IMAG):
    sel = Selector(); opt = torch.optim.Adam(sel.parameters(), lr=2e-3)
    for _ in range(steps):
        n = 128; s = init_states(n)
        gd = sample_goal_dims(n, train_all); goal = oh(gd, 2)
        logp = torch.zeros(n); ent = torch.zeros(n); ever = torch.zeros(n, dtype=torch.bool)
        for t in range(T_REAL):
            first, term = imagine(s, K, L)
            keys = wl.E(term.reshape(n * K, D)).view(n, K, H)
            d = torch.distributions.Categorical(logits=sel(goal, keys))
            k = d.sample(); logp = logp + d.log_prob(k); ent = ent + d.entropy()
            s = apply_op(s, first[torch.arange(n), k])
            ever |= reached_dim(s, gd)
        r = ever.float(); adv = (r - r.mean()) / (r.std() + 1e-6)
        loss = -(logp * adv.detach()).mean() - 0.01 * ent.mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return sel


@torch.no_grad()
def eval_arch1(wl, sel, goal_dim, Bs, s=None, K=K_IMAG, L=L_IMAG, oracle=False):
    if s is None: s = init_states(4000)
    n = s.shape[0]; gd = torch.full((n,), goal_dim); goal = oh(gd, 2)
    ever = torch.zeros(n, dtype=torch.bool); out = {}
    for t in range(max(Bs)):
        first, term = imagine(s, K, L)
        if oracle:
            reach = torch.cos(term[:, :, goal_dim]) < 0.0       # (n,K) which rollouts reach the goal
            k = reach.float().argmax(1)                          # first reaching rollout (else idx 0)
        else:
            keys = wl.E(term.reshape(n * K, D)).view(n, K, H)
            k = sel(goal, keys).argmax(1)
        s = apply_op(s, first[torch.arange(n), k])
        ever |= reached_dim(s, gd)
        if (t + 1) in Bs: out[t + 1] = ever.float().mean().item()
    return [out[b] for b in Bs]


# ============================ ARCH 2 — attend during generation ====================================
class GenPolicy(nn.Module):
    """transformer; a GOAL token attends over the self-generated latent trajectory -> next op."""
    def __init__(self, dm=64, nhead=4, nlayers=2):
        super().__init__()
        self.state_emb = nn.Linear(H, dm)
        self.goal_emb = nn.Linear(2, dm)
        self.pos = nn.Embedding(T_REAL + 2, dm)
        layer = nn.TransformerEncoderLayer(dm, nhead, dim_feedforward=128, batch_first=True)
        self.tf = nn.TransformerEncoder(layer, nlayers)
        self.head = nn.Linear(dm, NOP)

    def forward(self, latent_seq, goal):                 # latent_seq:(n,L,H), goal:(n,2) -> logits:(n,NOP)
        n, L, _ = latent_seq.shape
        toks = self.state_emb(latent_seq) + self.pos(torch.arange(1, L + 1)).unsqueeze(0)
        gtok = (self.goal_emb(goal) + self.pos(torch.zeros(1, dtype=torch.long))).unsqueeze(1)
        h = self.tf(torch.cat([gtok, toks], 1))
        return self.head(h[:, 0])                         # goal token output (attended over the rollout)


def train_arch2(wl, train_all, steps=2500):
    pol = GenPolicy(); opt = torch.optim.Adam(pol.parameters(), lr=1e-3)
    for _ in range(steps):
        n = 256; s = init_states(n)
        gd = sample_goal_dims(n, train_all); goal = oh(gd, 2)
        seq = [wl.E(s)]; logp = torch.zeros(n); ent = torch.zeros(n); ever = torch.zeros(n, dtype=torch.bool)
        for t in range(T_REAL):
            d = torch.distributions.Categorical(logits=pol(torch.stack(seq, 1), goal))
            op = d.sample(); logp = logp + d.log_prob(op); ent = ent + d.entropy()
            s = apply_op(s, op); seq.append(wl.E(s))
            ever |= reached_dim(s, gd)
        r = ever.float(); adv = (r - r.mean()) / (r.std() + 1e-6)
        loss = -(logp * adv.detach()).mean() - 0.01 * ent.mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return pol


@torch.no_grad()
def eval_arch2(wl, pol, goal_dim, Bs, s=None):
    if s is None: s = init_states(4000)
    n = s.shape[0]; gd = torch.full((n,), goal_dim); goal = oh(gd, 2)
    seq = [wl.E(s)]; ever = torch.zeros(n, dtype=torch.bool); out = {}
    for t in range(max(Bs)):
        op = pol(torch.stack(seq, 1), goal).argmax(-1)
        s = apply_op(s, op); seq.append(wl.E(s))
        ever |= reached_dim(s, gd)
        if (t + 1) in Bs: out[t + 1] = ever.float().mean().item()
    return [out[b] for b in Bs]


# ============================ reactive baseline (fixation control) =================================
class Reactive(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D + 2 + 1, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, NOP))

    def forward(self, s, goal, t):
        return self.net(torch.cat([s, goal, torch.full((s.shape[0], 1), t / T_REAL)], -1))


def train_reactive(train_all, steps=2500):
    pol = Reactive(); opt = torch.optim.Adam(pol.parameters(), lr=1.5e-3)
    for _ in range(steps):
        n = 256; s = init_states(n)
        gd = sample_goal_dims(n, train_all); goal = oh(gd, 2)
        logp = torch.zeros(n); ent = torch.zeros(n); ever = torch.zeros(n, dtype=torch.bool)
        for t in range(T_REAL):
            d = torch.distributions.Categorical(logits=pol(s, goal, t))
            op = d.sample(); logp = logp + d.log_prob(op); ent = ent + d.entropy()
            s = apply_op(s, op); ever |= reached_dim(s, gd)
        r = ever.float(); adv = (r - r.mean()) / (r.std() + 1e-6)
        loss = -(logp * adv.detach()).mean() - 0.01 * ent.mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return pol


@torch.no_grad()
def eval_reactive(pol, goal_dim, Bs, s=None):
    if s is None: s = init_states(4000)
    n = s.shape[0]; gd = torch.full((n,), goal_dim); goal = oh(gd, 2)
    ever = torch.zeros(n, dtype=torch.bool); out = {}
    for t in range(max(Bs)):
        op = pol(s, goal, t).argmax(-1)
        s = apply_op(s, op); ever |= reached_dim(s, gd)
        if (t + 1) in Bs: out[t + 1] = ever.float().mean().item()
    return [out[b] for b in Bs]


def main():
    f = lambda *a: print(*a, flush=True)
    Bs = (2, 3, 4, 5, 6, 8)
    hdr = "    budget B:                      " + "  ".join(f"{b:>4d}" for b in Bs)
    f("=" * 96)
    f(f"Incubation step 2 — ATTENTION-NATIVE controller (2 archs x 2 regimes), high-dim world (D={D}, {NOP} ops)")
    f("=" * 96)
    f("pretraining frozen effect-latent (broad interventions)...")
    wl = W.pretrain_latent()
    f("training controllers...")
    react = train_reactive(True)
    a1_all = train_arch1(wl, True); a1_typ = train_arch1(wl, False)
    a2_all = train_arch2(wl, True); a2_typ = train_arch2(wl, False)

    arms = [
        ("reactive (train-all)        ", lambda gd: eval_reactive(react, gd, Bs)),
        ("Arch1 imagine-ORACLE        ", lambda gd: eval_arch1(wl, None, gd, Bs, oracle=True)),
        ("Arch1 select (train-all)    ", lambda gd: eval_arch1(wl, a1_all, gd, Bs)),
        ("Arch1 select (train-typical)", lambda gd: eval_arch1(wl, a1_typ, gd, Bs)),
        ("Arch2 gen    (train-all)    ", lambda gd: eval_arch2(wl, a2_all, gd, Bs)),
        ("Arch2 gen    (train-typical)", lambda gd: eval_arch2(wl, a2_typ, gd, Bs)),
    ]
    for gd, label in [(0, "TYPICAL goals (free dir e1)"), (1, "REPURPOSING goals (LOCKED dir e2)")]:
        f(f"\n  --- {label} ---"); f(hdr)
        for name, fn in arms:
            f(f"    {name}" + "  ".join(f"{v*100:4.0f}" for v in fn(gd)))
    f("\n" + "=" * 96)
    f("READ: reactive should FIXATE on e2. Arch1/2 (train-all) test whether attention-over-rollouts")
    f("DEPLOYS the repurposing chain at all (vs the imagine-ORACLE ceiling). (train-typical) tests")
    f("whether goal-query attention GENERALIZES to a NEVER-TRAINED creative goal. Select-vs-generate")
    f("= the two ways to make 'attend over self-generated rollouts' native.")


if __name__ == "__main__":
    main()
