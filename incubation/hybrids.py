#!/usr/bin/env python3
"""
Incubation step 4 — THREE HYBRID controllers, compared on HELD-OUT-AXIS transfer (fast-but-legit).

Decomposition: incubation (propose candidate rollouts) -> aiming (select/commit). We train the LEARNED
pieces ONCE and let the three hybrids be different SEARCH STRATEGIES over them (this isolates the
strategy and is far cheaper than three full pipelines):
  shared: frozen latent E ; a COVERAGE-proposer (goal-AGNOSTIC volume coverage -> learns EVERY axis's
          chain incl. the held-out one, because coverage is axis-blind = the transfer enabler) ; a
          selector/value V(effect, target) trained ONLY on subset axes {0,1,2,3} (axis-general aiming,
          axis 4 never trained) ; a fast Arch-2 policy.
  Hybrid A (deep imagine + select): proposer samples K rollouts (len L); V scores terminals; MPC first op.
  Hybrid B (shallow exhaustive lookahead): enumerate all ops 1-ply; V scores next-states; take best.
  Hybrid C (two-timescale): fast Arch-2 by default; deliberate (A) only when V says the fast move is weak.
Transfer test: train on subset, evaluate ZERO-SHOT on HELD-OUT locked axis 4 (+ axis1 trained-ref,
axis0 free). World reused from multiaxis_transfer.  Run with /data/llm/.venv/bin/python.
"""
import torch
import torch.nn as nn

import multiaxis_transfer as M

torch.manual_seed(1)
apply_op, init_states, target_obs, torus5, reached, oh = (
    M.apply_op, M.init_states, M.target_obs, M.torus5, M.reached, M.oh)
NOP, D, H, HELD_OUT, T = M.NOP, M.D, M.H, M.HELD_OUT, M.T_REAL
SUBSET = [0, 1, 2, 3]


def volume_coverage(Zs):
    Z = torch.stack(Zs, 1); Z = Z - Z.mean(1, keepdim=True); k = Z.shape[-1]
    G = torch.einsum("ntk,ntj->nkj", Z, Z) / Z.shape[1] + 0.1 * torch.eye(k)
    return torch.linalg.slogdet(G)[1]


# ---------- shared learned pieces ----------
class Proposer(nn.Module):
    """goal-agnostic op policy; trained for VOLUME coverage of the goal observable -> proposes the
    chains for ALL axes (incl. held-out), because coverage is axis-blind."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(H + 1, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, NOP))

    def forward(self, e, t):
        return self.net(torch.cat([e, torch.full((e.shape[0], 1), t / T)], -1))


def train_proposer(wl, steps=1500):
    p = Proposer(); opt = torch.optim.Adam(p.parameters(), lr=1.5e-3)
    for _ in range(steps):
        n = 128; s = init_states(n); Z = [torus5(s)]; logp = torch.zeros(n); ent = torch.zeros(n)
        for t in range(T):
            d = torch.distributions.Categorical(logits=p(wl.E(s), t))
            op = d.sample(); logp = logp + d.log_prob(op); ent = ent + d.entropy()
            s = apply_op(s, op); Z.append(torus5(s))
        r = volume_coverage(Z); adv = (r - r.mean()) / (r.std() + 1e-6)
        loss = -(logp * adv.detach()).mean() - 0.02 * ent.mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return p


class ValueNet(nn.Module):
    """V(effect-embedding, target) -> P(target-axis reached far). Axis-general (reads the axis from t)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(H + 10, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, emb, t):                      # emb:(n,m,H), t:(n,10) -> (n,m)
        m = emb.shape[1]
        return self.net(torch.cat([emb, t.unsqueeze(1).expand(-1, m, -1)], -1)).squeeze(-1)


def train_selector(wl, steps=2500):
    V = ValueNet(); opt = torch.optim.Adam(V.parameters(), lr=2e-3); bce = nn.BCEWithLogitsLoss()
    for _ in range(steps):
        n = 256; s = init_states(n)
        axis = torch.tensor(SUBSET)[torch.randint(0, len(SUBSET), (n,))]      # subset axes only
        t = target_obs(s, axis)
        L = torch.randint(1, 6, (1,)).item(); term = s
        for _ in range(L):
            term = apply_op(term, torch.randint(0, NOP, (n,)))
        y = (torch.cos(term[torch.arange(n), axis]) < 0.0).float()   # per-element axis (avoid s[:,tensor])
        pred = V(wl.E(term).unsqueeze(1), t).squeeze(1)
        loss = bce(pred, y)
        opt.zero_grad(); loss.backward(); opt.step()
    return V


# ---------- imagination helpers ----------
def imagine_prop(wl, prop, s, K, L):
    n = s.shape[0]; S = s.repeat_interleave(K, 0); first = None; cur = S
    for l in range(L):
        op = torch.distributions.Categorical(logits=prop(wl.E(cur), l)).sample()
        if l == 0: first = op
        cur = apply_op(cur, op)
    return first.view(n, K), cur.view(n, K, D)


def next_states_all(s):
    """(n,D) -> (n,NOP,D): the next state for each possible op (1-ply exhaustive lookahead)."""
    n = s.shape[0]
    return torch.stack([apply_op(s, torch.full((n,), j)) for j in range(NOP)], 1)


# ---------- the three hybrids (eval-time strategies) ----------
@torch.no_grad()
def run_A(wl, prop, V, axis, Bs, K=12, L=4, n=2500):
    s = init_states(n); ever = torch.zeros(n, dtype=torch.bool); out = {}
    for t in range(max(Bs)):
        first, term = imagine_prop(wl, prop, s, K, L)
        k = V(wl.E(term.reshape(n * K, D)).view(n, K, H), target_obs(s, axis)).argmax(1)
        s = apply_op(s, first[torch.arange(n), k]); ever |= reached(s, axis)
        if (t + 1) in Bs: out[t + 1] = ever.float().mean().item()
    return [out[b] for b in Bs]


@torch.no_grad()
def run_B(wl, V, axis, Bs, n=2500):
    s = init_states(n); ever = torch.zeros(n, dtype=torch.bool); out = {}
    for t in range(max(Bs)):
        nxt = next_states_all(s)
        op = V(wl.E(nxt.reshape(n * NOP, D)).view(n, NOP, H), target_obs(s, axis)).argmax(1)
        s = apply_op(s, op); ever |= reached(s, axis)
        if (t + 1) in Bs: out[t + 1] = ever.float().mean().item()
    return [out[b] for b in Bs]


@torch.no_grad()
def run_C(wl, fast, prop, V, axis, Bs, thr=0.0, K=12, L=4, n=2500):
    """fast Arch-2 by default; if V deems the fast move weak (logit<thr), deliberate via A's imagine+select."""
    s = init_states(n); ever = torch.zeros(n, dtype=torch.bool); out = {}; seq = [wl.E(s)]
    for t in range(max(Bs)):
        fop = fast(torch.stack(seq, 1), target_obs(s, axis)).argmax(-1)            # fast proposal
        fres = apply_op(s, fop)
        fval = V(wl.E(fres).unsqueeze(1), target_obs(s, axis)).squeeze(1)          # confidence in fast move
        first, term = imagine_prop(wl, prop, s, K, L)                              # deliberate option
        k = V(wl.E(term.reshape(n * K, D)).view(n, K, H), target_obs(s, axis)).argmax(1)
        dop = first[torch.arange(n), k]
        op = torch.where(fval >= thr, fop, dop)                                    # arbitrate per element
        s = apply_op(s, op); seq.append(wl.E(s)); ever |= reached(s, axis)
        if (t + 1) in Bs: out[t + 1] = ever.float().mean().item()
    return [out[b] for b in Bs]


@torch.no_grad()
def run_arch2(wl, fast, axis, Bs, n=2500):
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
    f(f"Incubation step 4 — THREE HYBRIDS on HELD-OUT-AXIS transfer (axis {HELD_OUT}); train on subset {SUBSET}")
    f("=" * 92)
    f("pretraining frozen latent..."); wl = M.pretrain_latent()
    f("training shared pieces: coverage-proposer, selector V, fast Arch-2 (subset)...")
    prop = train_proposer(wl)
    V = train_selector(wl)
    fast = M.train_arch2(wl, "subset")

    for axis, lab in [(HELD_OUT, f"HELD-OUT axis {HELD_OUT} (ZERO-SHOT transfer)"),
                      (1, "trained-ref locked axis 1"), (0, "free axis 0")]:
        f(f"\n  --- {lab} ---"); f(hdr)
        f("    Arch2 alone (no hybrid)   " + "  ".join(f"{v*100:4.0f}" for v in run_arch2(wl, fast, axis, Bs)))
        f("    Hybrid A (deep imagine)   " + "  ".join(f"{v*100:4.0f}" for v in run_A(wl, prop, V, axis, Bs)))
        f("    Hybrid B (1-ply lookahead)" + "  ".join(f"{v*100:4.0f}" for v in run_B(wl, V, axis, Bs)))
        f("    Hybrid C (fast+deliberate)" + "  ".join(f"{v*100:4.0f}" for v in run_C(wl, fast, prop, V, axis, Bs)))
    f("\n" + "=" * 92)
    f("READ: on the HELD-OUT axis, Arch2-alone should FAIL (can't emit a never-trained chain). Hybrids")
    f("A/B/C should TRANSFER if the coverage-proposer supplies axis-4 chains and the axis-general V aims")
    f("at them. B = cheapest (exhaustive 1-ply). The winner = the native, trained form of propose->verify.")


if __name__ == "__main__":
    main()
