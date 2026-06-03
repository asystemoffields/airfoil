#!/usr/bin/env python3
"""
Incubation step 4b — push the two named levers from step 4:
  LEVER 1 (raise the ~65% ceiling): a BETTER coverage-proposer. The cap was what the near-random
  proposer GENERATES (selector already aimed as well as the oracle). Give the coverage-proposer a
  COVERAGE-MEMORY input (running mean of the goal observable) so it steers toward UNCOVERED axes and
  reliably emits EVERY axis's chain (incl. the held-out one) -> more imagined rollouts contain axis-4's
  arc -> higher ceiling. Still goal-AGNOSTIC (transfer-safe).
  LEVER 2 (best-of-both for Hybrid C): replace the myopic V-threshold gate with a PROGRESS gate. Trust
  the fast policy by default; switch to deliberation only when it is STUCK (target coordinate stops
  improving). Known goals -> fast makes progress -> stays fast (~100); novel axis -> immediately stuck
  -> deliberates (~ceiling). World + frozen latent + fast Arch-2 reused from multiaxis_transfer.
Run with /data/llm/.venv/bin/python.
"""
import torch
import torch.nn as nn

import multiaxis_transfer as M

torch.manual_seed(1)
apply_op, init_states, target_obs, torus5, reached = (
    M.apply_op, M.init_states, M.target_obs, M.torus5, M.reached)
NOP, D, H, HELD_OUT, T = M.NOP, M.D, M.H, M.HELD_OUT, M.T_REAL
SUBSET = [0, 1, 2, 3]


def volume_coverage(Zs):
    Z = torch.stack(Zs, 1); Z = Z - Z.mean(1, keepdim=True); k = Z.shape[-1]
    G = torch.einsum("ntk,ntj->nkj", Z, Z) / Z.shape[1] + 0.1 * torch.eye(k)
    return torch.linalg.slogdet(G)[1]


# ---------- LEVER 1: coverage-proposer WITH coverage memory ----------
class Proposer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(H + 10 + 1, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, NOP))

    def forward(self, e, mem, t):                 # mem = running-mean torus (what's been covered)
        return self.net(torch.cat([e, mem, torch.full((e.shape[0], 1), t / T)], -1))


def train_proposer(wl, steps=3000):
    p = Proposer(); opt = torch.optim.Adam(p.parameters(), lr=1.5e-3)
    for it in range(steps):
        n = 128; s = init_states(n); Z = [torus5(s)]; mem = torus5(s); cnt = 1.0
        logp = torch.zeros(n); ent = torch.zeros(n)
        beta = 0.04 if it < steps // 2 else 0.02
        for t in range(T):
            d = torch.distributions.Categorical(logits=p(wl.E(s), mem, t))
            op = d.sample(); logp = logp + d.log_prob(op); ent = ent + d.entropy()
            s = apply_op(s, op); tor = torus5(s); Z.append(tor)
            mem = (mem * cnt + tor) / (cnt + 1); cnt += 1
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


def train_selector(wl, steps=3000):
    V = ValueNet(); opt = torch.optim.Adam(V.parameters(), lr=2e-3); bce = nn.BCEWithLogitsLoss()
    for _ in range(steps):
        n = 256; s = init_states(n)
        axis = torch.tensor(SUBSET)[torch.randint(0, len(SUBSET), (n,))]
        t = target_obs(s, axis); L = torch.randint(1, 6, (1,)).item(); term = s
        for _ in range(L):
            term = apply_op(term, torch.randint(0, NOP, (n,)))
        y = (torch.cos(term[torch.arange(n), axis]) < 0.0).float()
        loss = bce(V(wl.E(term).unsqueeze(1), t).squeeze(1), y)
        opt.zero_grad(); loss.backward(); opt.step()
    return V


def imagine_prop(wl, prop, s, K, L):
    n = s.shape[0]; S = s.repeat_interleave(K, 0)
    mem = torus5(S); cnt = 1.0; first = None; cur = S
    for l in range(L):
        op = torch.distributions.Categorical(logits=prop(wl.E(cur), mem, l)).sample()
        if l == 0: first = op
        cur = apply_op(cur, op); tor = torus5(cur); mem = (mem * cnt + tor) / (cnt + 1); cnt += 1
    return first.view(n, K), cur.view(n, K, D)


@torch.no_grad()
def run_A(wl, prop, V, axis, Bs, K=16, L=6, n=2000):
    s = init_states(n); ever = torch.zeros(n, dtype=torch.bool); out = {}
    for t in range(max(Bs)):
        first, term = imagine_prop(wl, prop, s, K, L)
        k = V(wl.E(term.reshape(n * K, D)).view(n, K, H), target_obs(s, axis)).argmax(1)
        s = apply_op(s, first[torch.arange(n), k]); ever |= reached(s, axis)
        if (t + 1) in Bs: out[t + 1] = ever.float().mean().item()
    return [out[b] for b in Bs]


@torch.no_grad()
def run_C(wl, fast, prop, V, axis, Bs, K=16, L=6, n=2000, stuck_max=2):
    """LEVER 2: progress-gated. Fast by default; an element switches to deliberate (A) once it has gone
    stuck_max steps in fast-mode without improving the target coordinate."""
    s = init_states(n); ar = torch.arange(n); ever = torch.zeros(n, dtype=torch.bool); out = {}
    seq = [wl.E(s)]; mode_fast = torch.ones(n, dtype=torch.bool); stuck = torch.zeros(n); prev = torch.cos(s[:, axis])
    for t in range(max(Bs)):
        fop = fast(torch.stack(seq, 1), target_obs(s, axis)).argmax(-1)
        first, term = imagine_prop(wl, prop, s, K, L)
        dop = first[ar, V(wl.E(term.reshape(n * K, D)).view(n, K, H), target_obs(s, axis)).argmax(1)]
        op = torch.where(mode_fast, fop, dop)
        s = apply_op(s, op); seq.append(wl.E(s)); cur = torch.cos(s[:, axis])
        progressed = cur < prev - 1e-3
        stuck = torch.where(mode_fast & ~progressed, stuck + 1, torch.where(mode_fast, torch.zeros(n), stuck))
        mode_fast = mode_fast & (stuck < stuck_max)
        prev = cur; ever |= reached(s, axis)
        if (t + 1) in Bs: out[t + 1] = ever.float().mean().item()
    return [out[b] for b in Bs]


@torch.no_grad()
def run_arch2(wl, fast, axis, Bs, n=2000):
    s = init_states(n); ever = torch.zeros(n, dtype=torch.bool); out = {}; seq = [wl.E(s)]
    for t in range(max(Bs)):
        op = fast(torch.stack(seq, 1), target_obs(s, axis)).argmax(-1)
        s = apply_op(s, op); seq.append(wl.E(s)); ever |= reached(s, axis)
        if (t + 1) in Bs: out[t + 1] = ever.float().mean().item()
    return [out[b] for b in Bs]


def main():
    f = lambda *a: print(*a, flush=True)
    Bs = (2, 4, 6, 8, 10)
    hdr = "    budget B:                  " + "  ".join(f"{b:>4d}" for b in Bs)
    f("=" * 92)
    f(f"Incubation step 4b — better proposer + progress-gated C (held-out axis {HELD_OUT}, train {SUBSET})")
    f("=" * 92)
    f("pretraining frozen latent..."); wl = M.pretrain_latent(steps=5000)
    f("training: coverage-proposer(+memory), selector V, fast Arch-2...")
    prop = train_proposer(wl); V = train_selector(wl); fast = M.train_arch2(wl, "subset", steps=1800)

    for axis, lab in [(HELD_OUT, f"HELD-OUT axis {HELD_OUT} (ZERO-SHOT transfer)"),
                      (1, "trained-ref locked axis 1"), (0, "free axis 0")]:
        f(f"\n  --- {lab} ---"); f(hdr)
        f("    Arch2 alone (no hybrid)   " + "  ".join(f"{v*100:4.0f}" for v in run_arch2(wl, fast, axis, Bs)))
        f("    Hybrid A (better proposer)" + "  ".join(f"{v*100:4.0f}" for v in run_A(wl, prop, V, axis, Bs)))
        f("    Hybrid C (progress-gated) " + "  ".join(f"{v*100:4.0f}" for v in run_C(wl, fast, prop, V, axis, Bs)))
    f("\n" + "=" * 92)
    f("READ: Lever 1 works if Hybrid A on the HELD-OUT axis rises well above 65 (the old ceiling).")
    f("Lever 2 works if Hybrid C ~ Arch2 on trained/free axes (≈100) AND ~ Hybrid A on the held-out axis")
    f("= best-of-both: reactive-fast on the known, creative on the novel.")


if __name__ == "__main__":
    main()
