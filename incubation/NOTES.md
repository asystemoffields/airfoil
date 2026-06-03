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

## Architecture decision: Option A "but not like before"
A = explicit simulated world-model + internal controller, but the KEY break from prior MBRL
(Dyna→World Models→Dreamer[reconstruction]→MuZero[value-equivalence]→I2A→Diffuser): value-
equivalent/reward-centric world-models DISCARD currently-useless structure, which is exactly what
repurposing needs. So our world-model is TASK-AGNOSTIC, trained on INTERVENTIONS for FULL causal
effects (KEEP the useless structure), queried by a LEARNED internal controller doing GOAL/OUTCOME-
matching (not reward search), non-directed. Closest ancestor for the controller = I2A.

## Result: incubation_controller.py — the crossover (proof-of-principle)
Controller simulates ops through the frozen 100%-accurate world-model under a CAPPED budget B;
directedness = the search ORDER. directed = the policy's learned usage-prior preference;
non-directed = random or coverage (outcome-diversity) exploration. reached% vs B:
- TYPICAL goals: directed 100% @B=1 (obvious is right) ; exploration slower.
- REPURPOSING goals: directed 0% until B=5, 100% only @B=9 (prior ranks the repurposing op last
  → wastes budget on typical ops that can't work — fixation WITH a perfect simulator);
  non-dir(coverage) 88% @B=3, 100% @B=5.
Three findings: (1) simulator NECESSARY-BUT-NOT-SUFFICIENT (directed+perfect-sim still 0% cheap);
(2) CROSSOVER = the thesis (directed wins on obvious, loses on creative; non-direction is the
mirror — and it COSTS on easy goals, honest); (3) non-trivial nugget COVERAGE≫RANDOM (88 vs 34
@B=3) — explore by OUTCOME-DIVERSITY, not randomly ("imagine the pieces that do something
different"). Caveats: non-directed orders are HEURISTIC not yet learned; directed-side crossover
partly by-construction (prior built to down-rank repurposing op — the point). Proof-of-principle.

## Next
(1) LEARN the exploration policy end-to-end (trained incubation channel, not heuristic order);
(2) stage (b) COMPOSITIONAL depth (repurposing op needed mid-composition → budget/size-for-time
bites hardest); (3) eventually the attention-native version (attend over self-generated rollouts).
