#!/usr/bin/env python3
"""
Incubation step 9 — PARTIAL OBSERVABILITY: the realism gate that bites CONSISTENTLY (+ does ensemble
pessimism rescue it?).

Step 8 showed the learned-model realism cost is real but STOCHASTIC: a fully-observed MLP nails the mean
dynamics (fwd MSE ~0), so the learned-vs-perfect beam gap only opens when deep search happens to exploit
the tiny error (seed-3). To make model error CONSISTENT and PRINCIPLED — and to match WHY a real
LLM-as-world-model is imperfect (it does not observe the full environment state) — we give the learned
model a PARTIAL OBSERVATION:

    obs(s) MASKS the internal registers s[5..9] (reg1,2,3,reg4a,reg4b) to 0, keeping goal angles s[0:5]
    and scratch s[10:].

This targets the failure exactly: the model CAN predict SETUP ops (P_i / P4: s[reg]+=1.5, a constant Δ,
no dependence on the hidden register) but CANNOT predict COUPLING ops (C_i: s[goal]+=ALPHA*tanh(reg) ;
T4/C4: depend on the hidden reg). Blind to the register, the model learns the POPULATION-MEAN coupling
effect -> it believes a coupling fires whether or not the setup armed the register -> it plans the payoff
WITHOUT the setup and gets fooled in the true world. That is the honest realism ceiling of planning over
an imperfect/partial model on a structurally-novel chain.

Then we test the standard MBRL fix — ENSEMBLE PESSIMISM (penalize beam branches where an ensemble of
forward models DISAGREES). Hypothesis: it will NOT rescue this gap, because the error here is shared BIAS
from missing input (all members are blind to the same register -> they AGREE on the wrong mean), not
epistemic VARIANCE. If so, the lesson is sharp: a systematic observation gap needs a model that actually
KNOWS more (-> the LLM's world knowledge), not an uncertainty penalty.

Panel on the held-out DEPTH-3 axis, planning over the model / acting in the true (hardened) world (MPC),
seeds {1,2}:
  random oracle ; perfect/full beam (true sim + full-obs value, UPPER REF) ;
  learned partial-obs beam (single) ; ensemble-mean partial-obs beam ; ensemble + pessimism.
Run with /data/llm/.venv/bin/python.
"""
import torch
import torch.nn as nn

import hardened_search as HS                              # reuse the HARDENED world (noise + nonlinear)
apply_op, init_states, target_obs, reached, oh, torus5, wide_states = (
    HS.apply_op, HS.init_states, HS.target_obs, HS.reached, HS.oh, HS.torus5, HS.wide_states)
D, NOP, H, HELD_OUT, SUBSET = HS.D, HS.NOP, HS.H, HS.HELD_OUT, HS.SUBSET

MASK_DIMS = [5, 6, 7, 8, 9]                               # hidden internal registers (reg1,2,3,reg4a,reg4b)
M_ENS = 3
SEEDS = (1, 2)
Bs = (4, 8, 12)


def obs(s):
    o = s.clone(); o[:, MASK_DIMS] = 0.0; return o        # the learned model sees this; true dynamics use full s


# ---- partial-obs latent + forward model (enc reads obs(s); fwd still predicts FULL Δs) ----
class POLatent(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(D, 96), nn.ReLU(), nn.Linear(96, H))
        self.fwd = nn.Sequential(nn.Linear(H + NOP, 96), nn.ReLU(), nn.Linear(96, D))

    def E(self, s): return self.enc(obs(s))               # <-- observation is masked HERE


def pretrain_po(steps=2500):
    wl = POLatent(); opt = torch.optim.Adam(wl.parameters(), lr=2e-3)
    for _ in range(steps):
        n = 512; s = wide_states(n)
        for _ in range(torch.randint(0, 5, (1,)).item()):
            s = apply_op(s, torch.randint(0, NOP, (n,)))
        op = torch.randint(0, NOP, (n,)); s2 = apply_op(s, op)
        loss = ((wl.fwd(torch.cat([wl.E(s), oh(op, NOP)], -1)) - (s2 - s)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    for p in wl.parameters(): p.requires_grad_(False)
    return wl


@torch.no_grad()
def fwd_diag(wl, n=6000):
    s = init_states(n)
    for _ in range(torch.randint(0, 4, (1,)).item()):
        s = apply_op(s, torch.randint(0, NOP, (n,)))
    rows = []
    for j in range(NOP):
        true = apply_op(s, torch.full((n,), j), noise=False) - s
        pred = wl.fwd(torch.cat([wl.E(s), oh(torch.full((n,), j), NOP)], -1))
        rows.append(((pred - true) ** 2).mean().item())
    return sum(rows) / len(rows), rows[10], rows[11], rows[12]    # overall, P4, T4, C4


def train_value_po(wl, k=6, steps=3000):
    V = HS.ValueNet(); opt = torch.optim.Adam(V.parameters(), lr=2e-3); bce = nn.BCEWithLogitsLoss()
    for _ in range(steps):
        n = 256; s = init_states(n)
        axis = torch.tensor(SUBSET)[torch.randint(0, len(SUBSET), (n,))]; t = target_obs(s, axis)
        cur = s.clone(); ever = (torch.cos(cur[torch.arange(n), axis]) < 0.0)
        for _ in range(k):
            cur = apply_op(cur, torch.randint(0, NOP, (n,)))
            ever |= (torch.cos(cur[torch.arange(n), axis]) < 0.0)
        loss = bce(V(wl.E(s).unsqueeze(1), t).squeeze(1), ever.float())   # value reads PARTIAL obs too
        opt.zero_grad(); loss.backward(); opt.step()
    return V


# ---- planners ----
def trans_true(states, op):                               # perfect simulator (full obs)
    return apply_op(states, op, noise=False)


def make_trans_po(wl):
    return lambda states, op: states + wl.fwd(torch.cat([wl.E(states), oh(op, NOP)], -1))


@torch.no_grad()
def beam_single(wl, V, s, axis, W, L, trans):
    n = s.shape[0]; ar = torch.arange(n); t = target_obs(s, axis)

    def expand(states):
        m = states.shape[1]
        kids = torch.stack([trans(states.reshape(n * m, D), torch.full((n * m,), j)) for j in range(NOP)], 1).view(n, m * NOP, D)
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
def beam_ensemble(ens, V, s, axis, W, L, pessimism):
    """plan over an ENSEMBLE of partial-obs models. score = mean_m[V+reach] - pessimism*std_m[V+reach].
    Uses ensemble member 0's encoder for V/keys (shared); transitions vary across members -> disagreement."""
    n = s.shape[0]; ar = torch.arange(n); t = target_obs(s, axis); E0 = ens[0]

    def expand(states):
        m = states.shape[1]; flat = states.reshape(n * m, D)
        per = []
        for wl in ens:
            kids = torch.stack([wl.fwd_step(flat, j) for j in range(NOP)], 1).view(n, m * NOP, D)
            emb = E0.E(kids.reshape(n * m * NOP, D)).view(n, m * NOP, H)
            reach = (torch.cos(kids[:, :, axis]) < 0.0).float()
            per.append((kids, V(emb, t) + 6.0 * reach))
        scores = torch.stack([p[1] for p in per], 0)      # (M, n, MNOP)
        kids_mean = torch.stack([p[0] for p in per], 0).mean(0)
        sc = scores.mean(0) - pessimism * scores.std(0)
        return kids_mean, sc

    kids, sc = expand(s.unsqueeze(1)); topv, topi = sc.topk(min(W, NOP), 1)
    beam = kids.gather(1, topi.unsqueeze(-1).expand(-1, -1, D)); first = topi.clone(); bf = first[ar, topv.argmax(1)]
    for _ in range(L - 1):
        Wc = beam.shape[1]; kids, sc = expand(beam); fr = first.repeat_interleave(NOP, 1)
        topv, topi = sc.topk(min(W, Wc * NOP), 1)
        beam = kids.gather(1, topi.unsqueeze(-1).expand(-1, -1, D)); first = fr.gather(1, topi); bf = first[ar, topv.argmax(1)]
    return bf


@torch.no_grad()
def run(planner, axis, n=400):
    s = init_states(n); ever = torch.zeros(n, dtype=torch.bool); out = {}
    for tt in range(max(Bs)):
        op = planner(s, axis); s = apply_op(s, op); ever |= reached(s, axis)
        if (tt + 1) in Bs: out[tt + 1] = ever.float().mean().item()
    return [out[b] for b in Bs]


def main():
    f = lambda *a: print(*a, flush=True)
    hdr = "      budget B:                  " + "  ".join(f"{b:>4d}" for b in Bs)
    f("=" * 96)
    f(f"Incubation step 9 — PARTIAL OBSERVABILITY realism gate (mask registers {MASK_DIMS}); held-out axis {HELD_OUT}=depth-3")
    f(f"   seeds={SEEDS}, ensemble M={M_ENS}")
    f("=" * 96)

    rows = {n: [] for n in ["random oracle", "perfect/full beam", "learned PO beam (single)",
                            "ensemble-mean PO beam", "ensemble+pessimism PO"]}
    for sd in SEEDS:
        torch.manual_seed(sd); f(f"\n----- seed {sd} -----")
        wl_full = HS.pretrain_latent(); V_full = HS.train_value_togo(wl_full, k=6)
        ovf, _, _, c4f = fwd_diag(wl_full)
        ens = [pretrain_po() for _ in range(M_ENS)]
        for wl in ens: wl.fwd_step = (lambda WL: (lambda flat, j: WL.fwd(torch.cat([WL.E(flat), oh(torch.full((flat.shape[0],), j), NOP)], -1)) + flat))(wl)
        ovp, _, _, c4p = fwd_diag(ens[0])
        V_po = train_value_po(ens[0], k=6)
        f(f"  fwd Δs MSE  FULL: overall {ovf:.4f} C4 {c4f:.4f}  |  PARTIAL-OBS: overall {ovp:.4f} C4 {c4p:.4f}")

        tr_po = make_trans_po(ens[0])
        res = {
            "random oracle":            HS.oracle(HELD_OUT, Bs),
            "perfect/full beam":        run(lambda s, ax: beam_single(wl_full, V_full, s, ax, 10, 5, trans_true), HELD_OUT),
            "learned PO beam (single)": run(lambda s, ax: beam_single(ens[0], V_po, s, ax, 10, 5, tr_po), HELD_OUT),
            "ensemble-mean PO beam":    run(lambda s, ax: beam_ensemble(ens, V_po, s, ax, 10, 5, 0.0), HELD_OUT),
            "ensemble+pessimism PO":    run(lambda s, ax: beam_ensemble(ens, V_po, s, ax, 10, 5, 1.0), HELD_OUT),
        }
        f(hdr)
        for name, v in res.items():
            rows[name].append(v); f(f"      {name:<26s}" + "  ".join(f"{x*100:4.0f}" for x in v))

    f("\n" + "=" * 96)
    f(f"SUMMARY — held-out depth-3 axis {HELD_OUT}, mean[min..max] over {len(SEEDS)} seeds (% reached)")
    f(hdr)
    for name, runs in rows.items():
        cells = [f"{sum(r[j]*100 for r in runs)/len(runs):3.0f}[{min(r[j]*100 for r in runs):3.0f}..{max(r[j]*100 for r in runs):3.0f}]" for j in range(len(Bs))]
        f(f"      {name:<26s}" + "  ".join(cells))
    f("\n" + "=" * 96)
    f("READ: (1) does PARTIAL-OBS fwd MSE (esp. C4) now exceed full-obs MSE -> model error is CONSISTENT")
    f("(structural, from masking the registers the couplings depend on), not a seed-3 fluke. (2) does the")
    f("learned PO beam fall CONSISTENTLY below perfect/full beam -> the honest realism ceiling of planning")
    f("over a partial model on a novel chain. (3) does ENSEMBLE pessimism close the gap? Prediction: little/")
    f("none — the error is shared BIAS from missing input (members agree on the wrong mean), not epistemic")
    f("variance. If so: a systematic observation gap needs a model that KNOWS more (the LLM), not a penalty.")


if __name__ == "__main__":
    main()
