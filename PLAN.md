# Compression-as-generalization — research plan

**Thesis (Alex's):** a system whose objective is to find the shortest reusable
description of its data will *generalize*, because the pressure to be short
forces it to discover abstractions that recur. "Within the particular is
contained the universal." The learner is *domesticated*: it only guides the
search for short programs, it doesn't roam the open space of all algorithms.

**The measurement that would prove it (the one Alex loved):**
> Does solving corner A make corner B cheaper to describe?
Learn abstractions from a TRAIN set; measure the description length of solutions
to a held-out TEST set. If structurally-*related* held-out tasks compress while
an unrelated *control* does not, the gain is genuine structural transfer, not a
free lunch.

## Roadmap (autonomous)
- [x] **v0** toy unary-int DSL, BPE-style library learning, frequency-code bits.
      Weak signal + messy control (frequency code specialises → control inflates).
- [ ] **v1** clean metric = *expression length in learned symbols* (control stays
      flat by construction); depth-3 held-out compositions; two controls
      (disjoint-ops + same-ops-scrambled). Target: related ≥1.5× shorter, controls flat.
- [x] **v2** two-part MDL in bits (library cost + uniform codebook data cost).
      Train MDL 209→153 (1.37×, library pays for itself); related 1.70×; control
      RISES (bigger codebook taxes unrelated work — honest no-free-lunch).
- [x] **v3** search-cost transfer: learned library solves held-out RELATED tasks
      in 21× fewer search nodes (25,761→1,207); CONTROL ~3× slower (branching tax,
      no shortcut). Abstractions = search accelerators where structure matches.
- [ ] **v4** depth generalisation: train on depth ≤k, test on depth >k. Real
      compositional generalisation, not interpolation.
- [ ] **v5** richer DSL (lists: map/fold/filter, parameterised ops) → a proper
      DreamCoder-style "list functions" testbed.
- [ ] **v6** the domesticated learner: a learned search guide (features → which
      symbol to try) vs uniform search. (A small GGUF could play proposer here.)
- [ ] **v7** graduate toward an abstraction benchmark (mini-ARC subset).

## Constraints
- Free / CPU-only. No paid APIs. No `sudo` (needs Alex's password) → no apt installs.
- GGUFs ARE pullable (network confirmed); SmolLM2-360M / PMRA-1.7B available.
  But running them locally is blocked: `cmake` missing → can't build llama.cpp,
  and the pip llama-cpp-python wheel is AVX-512 (crashes this Zen3). So the
  symbolic core is the focus; LLM-as-proposer is deferred to when/if v6 needs it.
- Everything logged in LOG.md; anything downloaded/installed tracked in INVENTORY.md.
