#!/usr/bin/env python3
"""
Incubation step 10 — the REALISM FRONTIER: planning success vs MODEL FIDELITY (graded observability),
and where ensemble pessimism finally helps.

Step 9 was BINARY: full register masking -> the partial-obs planner collapses to 0% (worse than random),
and ensemble pessimism can't help (shared bias, not variance). That is one endpoint. This traces the whole
curve with a graded OBSERVATION-QUALITY knob rho in [0,1] on the registers:

    obs_rho(s)[reg] = rho * s[reg] + (1-rho) * OBS_NOISE * randn

  rho=1 -> registers seen exactly (model can be near-perfect, recovering the step-8 learned~perfect result)
  rho=0 -> registers are pure noise (no signal -> ~the step-9 mask -> collapse)
  0<rho<1 -> PARTIAL but not blind: the model has a NOISY view of the register. Critically, the noise is
             resampled per member -> ensemble members now DISAGREE where the register matters -> there is
             genuine EPISTEMIC VARIANCE for a pessimism penalty to act on (unlike rho=0's pure bias).

Question 1: where on the fidelity axis does plan-over-model OVERTAKE the random floor? (the realism threshold)
Question 2: in the partial regime (rho>0), does ENSEMBLE PESSIMISM lift the curve (now that disagreement
            carries signal), even though it was useless at rho=0?
Held-out DEPTH-3 axis; plan over the model, act in the true (hardened) world (MPC). Reuses step-8 world +
step-9 planners. Run with /data/llm/.venv/bin/python.
"""
import torch
import torch.nn as nn

import hardened_search as HS
import partial_obs_gate as PO
apply_op, init_states, reached = HS.apply_op, HS.init_states, HS.reached
D, NOP, H, HELD_OUT, SUBSET = HS.D, HS.NOP, HS.H, HS.HELD_OUT, HS.SUBSET
oh, wide_states = HS.oh, HS.wide_states

MASK_DIMS = PO.MASK_DIMS
OBS_NOISE = 1.5
M_ENS = 3
RHOS = (0.0, 0.33, 0.66, 1.0)
Bs = (4, 8, 12)
SEED = 1


def obs_rho(s, rho):
    o = s.clone()
    o[:, MASK_DIMS] = rho * s[:, MASK_DIMS] + (1.0 - rho) * OBS_NOISE * torch.randn(s.shape[0], len(MASK_DIMS))
    return o


class POLatentRho(nn.Module):
    def __init__(self, rho):
        super().__init__()
        self.rho = rho
        self.enc = nn.Sequential(nn.Linear(D, 96), nn.ReLU(), nn.Linear(96, H))
        self.fwd = nn.Sequential(nn.Linear(H + NOP, 96), nn.ReLU(), nn.Linear(96, D))

    def E(self, s): return self.enc(obs_rho(s, self.rho))


def pretrain_rho(rho, steps=2500):
    wl = POLatentRho(rho); opt = torch.optim.Adam(wl.parameters(), lr=2e-3)
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
def run(planner, axis, n=500):
    s = init_states(n); ever = torch.zeros(n, dtype=torch.bool); out = {}
    for tt in range(max(Bs)):
        op = planner(s, axis); s = apply_op(s, op); ever |= reached(s, axis)
        if (tt + 1) in Bs: out[tt + 1] = ever.float().mean().item()
    return [out[b] for b in Bs]


def main():
    f = lambda *a: print(*a, flush=True)
    hdr = "      budget B:                       " + "  ".join(f"{b:>4d}" for b in Bs)
    f("=" * 98)
    f(f"Incubation step 10 — REALISM FRONTIER: planning success vs model fidelity (rho); held-out axis {HELD_OUT}")
    f(f"   OBS_NOISE={OBS_NOISE}, ensemble M={M_ENS}, seed={SEED}, rho grid={RHOS}")
    f("=" * 98)
    torch.manual_seed(SEED)

    # fixed references
    wl_full = HS.pretrain_latent(); V_full = HS.train_value_togo(wl_full, k=6)
    floor = HS.oracle(HELD_OUT, Bs)
    perfect = run(lambda s, ax: PO.beam_single(wl_full, V_full, s, ax, 10, 5, PO.trans_true), HELD_OUT)
    f(hdr)
    f("      random oracle (floor)         " + "  ".join(f"{x*100:4.0f}" for x in floor))
    f("      perfect/full beam (true sim)  " + "  ".join(f"{x*100:4.0f}" for x in perfect))

    f("\n  --- graded model fidelity (rho 0=blind .. 1=full) ---")
    for rho in RHOS:
        ens = [pretrain_rho(rho) for _ in range(M_ENS)]
        for wl in ens:
            wl.fwd_step = (lambda WL: (lambda flat, j: WL.fwd(torch.cat([WL.E(flat), oh(torch.full((flat.shape[0],), j), NOP)], -1)) + flat))(wl)
        ov, _, _, c4 = PO.fwd_diag(ens[0])
        V_po = PO.train_value_po(ens[0], k=6)
        tr = PO.make_trans_po(ens[0])
        single = run(lambda s, ax: PO.beam_single(ens[0], V_po, s, ax, 10, 5, tr), HELD_OUT)
        ensm = run(lambda s, ax: PO.beam_ensemble(ens, V_po, s, ax, 10, 5, 0.0), HELD_OUT)
        pess = run(lambda s, ax: PO.beam_ensemble(ens, V_po, s, ax, 10, 5, 1.0), HELD_OUT)
        f(f"\n  rho={rho:.2f}  (fwd C4 MSE {c4:.3f}, overall {ov:.3f})"); f(hdr)
        f("      single PO beam               " + "  ".join(f"{x*100:4.0f}" for x in single))
        f("      ensemble-mean PO beam        " + "  ".join(f"{x*100:4.0f}" for x in ensm))
        f("      ensemble+pessimism PO        " + "  ".join(f"{x*100:4.0f}" for x in pess))

    f("\n" + "=" * 98)
    f("READ: (1) the REALISM THRESHOLD — the rho at which single PO beam overtakes the random floor: below")
    f("it, a partial model is WORSE than no model; above it, plan-over-model pays off. Maps fidelity->payoff.")
    f("(2) PESSIMISM in the partial regime — if ensemble+pessimism > single at mid-rho (but not at rho=0),")
    f("then once model error carries epistemic VARIANCE (noisy view, members disagree) an uncertainty penalty")
    f("helps; at rho=0 (pure shared BIAS) it cannot. (3) fwd C4 MSE is the fidelity proxy on the chain op.")


if __name__ == "__main__":
    main()
