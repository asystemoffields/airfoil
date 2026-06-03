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

## Result: learned_controller.py — the dissociation, now LEARNED (no heuristics)
Two explorers trained by REINFORCE, differing ONLY in reward (goal-success vs coverage);
verifier (goal-check) is the only non-learned piece. reached% vs budget B:
- TYPICAL:     directed(success) 58->100 by B=3 ; non-dir(coverage) 21->90.
- REPURPOSING: directed(success) 0% at EVERY B (learned, target-blind, total fixation) ;
               non-dir(coverage) 33->68%.
=> non-directedness is EMERGENT from the objective (coverage), not a coded order; success-
trained explorer fixates, coverage-trained explorer deploys the affordance. Crossover holds
(non-direction costs on typical: 90 vs 100). Caveats: non-dir plateaus 68% (goal-AGNOSTIC,
greedy coverage imperfect); directed 0%-flat is extreme (target-blind + greedy). Next:
goal-INFORMED-but-coverage-trained explorer (focus coverage without the usage prior) to lift
68->higher; then compositional/state-dependent world (where learned >> any fixed order);
then attention-native (attend over self-generated rollouts).

## Refinement: goal-INFORMED coverage = clean 100% (the concept sharpened)
Added a 3rd learned explorer: reward = coverage of distinct effects ON THE GOAL REGISTER
(goal-informed, but coverage-trained NOT success-trained). reached% vs B:
- TYPICAL:     directed 100%@B3 ; op-coverage(agnostic) ->89 ; r-coverage 100%@B3.
- REPURPOSING: directed 0% everywhere ; op-coverage ->67 ; r-coverage 33->67->100%@B5.
=> goal-informed coverage hits 100% on BOTH (incl. repurposing) vs directed's flat 0%.
CONCEPTUAL SHARPENING: "non-directed" is NOT goal-blind — it means exploration driven by
EFFECT-COVERAGE, not by success/the prior. The winner is goal-AWARE but coverage-DRIVEN:
use the goal to pick WHICH dimension to explore effects on, then cover that effect-space
instead of beelining the prior-preferred op. Because the repurposing op is the only source
of the odd register-r effect, "cover the effects you can cause on what you care about"
FORCES its discovery; success-on-skewed-goals never does. Incubation, precisely: direct
attention at the goal, but search the space of EFFECTS, not the space of known solutions.
Stage (a) is now clean + learned. Next: compositional/state-dependent world (coverage stops
being trivial — learning earns its keep), then attention-native.

## Result: learned_scaled.py — learned exploration is NECESSARY at scale (30 ops, budget 8)
When coverage isn't cheap (can't try all 30 ops in 8 sims), the three-way separation is clean:
- REPURPOSING goals: directed(success) 0% (fixates) ; op-coverage 33% FLAT (blind exploration
  fails — can't find the 1-in-30 op in budget) ; r-coverage 0->66->100% (learned goal-focused).
- TYPICAL goals: directed 100%@B4 ; op-coverage ->79 ; r-coverage 100%@B4.
=> The deploying ingredient is LEARNED exploration of the GOAL-RELEVANT effect-space — not
success-maximization (fixates) and not blind coverage (doesn't scale). r-coverage is genuinely
adaptive (conditions on what it's already covered) — can't be a fixed order. Stage-(a) single-op
repurposing is now: learned, necessary, and scales. NEXT (the real frontier): SEQUENTIAL /
state-dependent compositional world (non-commutative ops, plan evolves state) where adaptivity
matters within a plan and the size-for-time payoff is largest; then attention-native.

## Result: learned_sequential.py — stage (b): SEQUENTIAL, STATE-DEPENDENT repurposing (learned >> any fixed order)
World: 3 regs mod 8, 15 ops, NO P_c. Reaching an opposite-parity target on c is impossible by home
adds (parity-preserving) and impossible directly (no P_c) — you must TRANSFER from an odd source,
which needs a parity op first => mid-plan repurpose. And `c+=b` flips c only when b is ODD => apply
P_b iff b is even => the optimal plan is STATE-DEPENDENT (a fixed order cannot meet it). reached% vs B:
- TYPICAL:     directed(success) 50->100@B4 ; coverage(agnostic) 26 FLAT ; gcoverage 25->100@B8.
- REPURPOSING: directed 0% EVERYWHERE (total fixation) ; coverage(agnostic) 0% FLAT ; gcoverage 24->100@B8.
- REPURPOSING split by initial b-parity (the state-dependence test):
    b EVEN (must APPLY P_b): gcoverage 25/37/50/62/74/100   (directed & agnostic 0 flat)
    b ODD  (must SKIP  P_b): gcoverage 24/36/49/62/75/100   (directed & agnostic 0 flat)
  => gcoverage is ADAPTIVE: near-identical curves for both parities — it applies P_b when needed and
  skips it when not. No fixed order can do both; "always P_b" fails b-odd, "never P_b" fails b-even.
THREE findings: (1) directed success-training FIXATES even harder when the affordance is sequential
(0% flat, not just slow). (2) goal-AGNOSTIC novelty FAILS entirely (26/0 flat) — pure exploration
doesn't aim; "non-directed" must mean goal-AIMED effect-coverage, not goal-blind (sharpens stage a).
(3) goal-informed effect-coverage DISCOVERS the multi-step P_b->transfer->tune chain and deploys it
ADAPTIVELY to state — learned, and necessary (no fixed order qualifies). Honest caveats: gcoverage
"solves" by SWEEPING c's reachable values (incubation-by-coverage) so the verifier picks the hit — it
pays a coverage COST on easy goals (100@B8 vs directed 100@B4), the crossover holds; world still
abstract/low-dim (coverage = distinct discrete values, enumerable — the next scaling problem).
Stage (b) DONE: sequential + state-dependent repurposing is learned and necessary.
NEXT (scaling): coverage must move from distinct-discrete-values to DIVERSITY IN A LEARNED EFFECT-
EMBEDDING (so it works when the effect space is huge/continuous); then attention-native controller
(attend over self-generated rollouts); then the size-for-time frontier (small+incubation == big+
reactive on accuracy-vs-compute = the headline); then ground onto a real CPU LLM (PMRA).

## Result: continuous_coverage.py — SCALING step 1: de-toy the COVERAGE mechanism (continuous, no enumeration)
Goal: show goal-informed effect-coverage survives when the effect space is continuous/high-dim so it
CANNOT be enumerated (the discrete bitmask is dead). World: R^12 — s[0]=e1 (free goal angle), s[1]=e2
(LOCKED goal angle), s[2:4]=B (hidden register), s[4:12]=scratch (8-d, high-variance, goal-irrelevant).
Goal-space PERIODIC (torus) = continuous analog of "mod 8" — ceiling on the easy axis. Home ops move e1
(coarse) + scratch but NEVER e2 or B; coupling (s[1]+=alpha*s[2]) is the ONLY e2-mover; P (s[2]+=1.5)
the ONLY B-mover. Reaching an e2 goal => P (activate B) -> coupling (inject e2): mid-plan repurpose,
state-dependent (skip P if B already active). Coverage measured in a FROZEN intervention-trained causal
latent E (can't be gamed — frozen). reached% vs budget (trained on TYPICAL/e1 only):
DIAGNOSTIC: e2 ORACLE 100% (world solvable), RANDOM-policy discovery 24.5% (chain is findable, not a
needle) — so any 0% is the OBJECTIVE's fault, not the world's.
Iteration that mattered (honest debugging, all logged):
- non-periodic e1 (unbounded) => coverage trivially maxed by sliding e1 => e2 never pressured. FIX: torus.
- episodic min-distance spread => coarse e1 jumps stay "novel" forever, never saturates => still no e2
  pressure. FIX: VOLUME (log-det of Gram) coverage — saturates per-subspace, so raising it REQUIRES
  variance in a NEW dimension (the locked axis). This is the principled scalable coverage objective.
RESULT (volume coverage, broad-intervention pretrain):
- TYPICAL (e1): directed 79->100 ; full-latent-coverage 100 ; readout-coverage 100.
- REPURPOSING (e2): directed(success) FIXATES (0->15%) ; FULL-LATENT volume coverage 0->89%
  (B-low/needs-P 0->95%, B-pre-activated 0->30%) ; readout("gcoverage") 0% (failed).
TWO FINDINGS: (1) The coverage mechanism SCALES off the bitmask: continuous volume coverage over a
frozen causal latent discovers state-dependent multi-step repurposing, adaptively, no enumeration —
and it works BECAUSE the task-agnostic latent KEEPS the locked structure (volume feels the stuck dims).
That is the keep-everything causal world-model thesis, vindicated in continuous form. (2) NEGATIVE/OPEN:
a naive learned reconstruction READOUT for goal-AIMED coverage is fragile — spurious entanglement leaks
variance into the readout so e1 motion fakes e2 coverage => never aims => 0%. Note the reversal from
stage a/b (there AIMED won, agnostic diluted): at THIS latent size agnostic doesn't dilute, so it wins;
AIMING only becomes necessary once the latent is too big to cover wholesale within budget (real scale).
=> Step 1 hinge CONFIRMED with nuance: coverage de-toys (volume-over-causal-latent). Robust AIMING is
the open sub-problem, deferred to the attention-native controller (goal QUERY attends over self-
generated rollouts to aim coverage at the goal-relevant directions) — which is also step 2 / the finale.
