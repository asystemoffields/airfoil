# Incubation line — journal

**Question:** can we architecturally instill, in a *small* model, the thing big models get from
scale — finding non-obvious connections / "thinking bigger"? Functionally this collapses to
**cause-and-effect / affordance understanding**: *what (non-obvious) thing is capable of solving
the problem?* (caveman: a towel breaks a window safer; math: OpenAI disproving the Erdős unit-
distance conjecture via class field towers — a number-theory tool repurposed for geometry).

**Methodology guard (no woo):** biology is the muse, computation is the judge. Every borrowed
idea must restate as mechanism + benefit + a falsifiable *differential* prediction, with no brain
words (Marr: borrow computational/algorithmic levels, never implementational).

## The frame we converged on
- Creativity = narratology = explanatory coverage = **causal/affordance understanding**. "More
  correct the more it connects" = MDL/compression (one cause → many effects = short code), under
  a consistency/verify guard (else apophenia).
- Affordance = forward, goal-directed causal reasoning ("what produces effect E"). Needs
  **interventions** (Pearl rung 2; can't get from passive observation) — and interventions are
  the *sample-efficient* way to give a small model big-model affordance coverage.
- "Non-obvious" = **functional repurposing**: deploy a known operator-effect *outside its training
  usage-distribution*, against the usage prior, verified by reaching the goal.

## Results
- **RULESWITCH (step0_task / step0_hard):** retired. Long-range-retrieval task; fixation only
  appeared out-of-distribution (entangled with length extrapolation) — muddy, and the wrong
  capability (mechanical retrieval).
- **affordance_v1 — prerequisite CONFIRMED (clean):** abstract operator world (3 regs mod 8, 9 ops;
  repurposing ops have a latent odd effect on an off-home register). Dual head on a shared trunk:
  world-model (effects, from interventions) vs policy (goal→op, usage-skewed). Result — the TRIPLE:
  - KNOWS (world-model): **100%** (incl. each repurposing op's latent-odd register)
  - USES (reactive policy) on repurposing goals: **0%** reached, picks a typical/lure op 100% — textbook functional fixedness, in an MLP (not a transformer quirk)
  - PLANNER (simulate via the model's OWN world-model, pick op predicted to reach target): **100%**
  Read: the affordance was KNOWN and DEPLOYABLE; only the deployment MECHANISM (simulation) was
  missing. The planner = the airfoil propose→verify scaffold, over affordances. Planner 100% is an
  expected upper bound (perfect world-model + 9 ops); the *value* is the dissociation.

## Next — the real architectural test
Make the simulation **internal and trained** (an incubation channel) rather than explicit external
search; ask whether **non-directed > directed** at deploying the affordance, measured as
accuracy-vs-compute *frontiers* (3 outcomes: capability win / efficiency win / null). Staged:
(a) single-op repurposing [done as prerequisite], (b) compositional depth (where size-for-time bites).
