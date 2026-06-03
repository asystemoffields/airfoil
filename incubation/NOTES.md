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

## Result: continuous_aiming.py — step 1b: AIMING NAILED (goal-informed coverage matters at scale)
Resolves the step-1 open problem. Big world (R^28, 30 ops incl. 24 goal-IRRELEVANT scratch-movers,
latent H=48) so goal-AGNOSTIC volume coverage has many cheap dims to chase. e2 ORACLE 100%, random
discovery 16.3%. Trained on TYPICAL (e1) only. reached% vs budget:
- TYPICAL (e1):     directed 50->100 ; agnostic(full-latent vol) 80 flat ; aimed(goal-observ vol) 97->100.
- REPURPOSING (e2): directed 0 FLAT (fixates) ; agnostic 0 FLAT (DILUTES — chases 24 scratch dims, never
  opens the expensive locked e2 chain in budget) ; AIMED 98->100%.
- e2 split: aimed = 98->100 on B-low (must APPLY P) AND 100 on B-pre-activated (must SKIP P) => ADAPTIVE,
  no fixed order; directed & agnostic 0 on both.
FINDING: AIMING is what makes coverage scale. When the latent has many goal-irrelevant movable
directions, agnostic volume coverage dilutes to 0 on the locked axis; coverage of the TRUE GOAL
OBSERVABLE (volume over the goal coordinate) ignores scratch and discovers the state-dependent multi-
step repurposing chain at ~100%. This is exactly stage-(b) gcoverage (cover the goal register's values),
now continuous + high-dim. The step-1 fragility was specifically a LEARNED reconstruction readout
standing in for the observable (entanglement faked coverage); the observable is given by the task, so
covering it is legitimate and robust. OPEN (carried to step 2): when the goal-relevant readout is NOT a
given coordinate (e.g., a real LLM), aiming must come from a robust learned mechanism — the natural one
is the attention-native controller's GOAL QUERY attending over self-generated rollouts (= the finale).
STEP 1 COMPLETE: coverage de-toys (continuous volume over a frozen causal latent) AND aims (cover the
goal observable) — learned, necessary, adaptive, no enumeration, high-dim. Next: step 2 attention-native.

## Result: step2_attention.py — ATTENTION-NATIVE controller (2 archs x 2 regimes). REFRAMES the evidence bar.
Two architectures for "attend over self-generated rollouts": Arch1 SELECT-AMONG-FINISHED (imagine K=16
random rollouts len 5 via WM, frozen-latent keys, learned goal-query attention selects one, MPC exec
first op) ; Arch2 ATTEND-DURING-GENERATION (transformer; goal token attends over its own growing latent
trajectory -> next op). Regimes: train-ALL goals vs train-TYPICAL(e1)-only. Baselines: reactive FF;
Arch1 imagine-ORACLE (pick the verifier-winning rollout = imagination ceiling). reached% vs budget:
- TYPICAL(e1): reactive 100 ; Arch1-oracle 40->88 ; Arch1-select(all) 46->91, (typ) 47->92 ; Arch2 100/100.
- REPURPOSING(e2): reactive(train-all) 63->100 ; Arch1-oracle 9->40 ; Arch1-select(all) 8->47, (typ) 1->4 ;
  Arch2-gen(all) 11->72->100 ; Arch2-gen(typ) 0 FLAT.
THREE FINDINGS:
(1) **TRAIN-ALL IS NOT A CREATIVITY TEST.** Reactive FF, trained on e2, solves e2 at 100% — a goal-
   conditioned policy with the goal signal just learns the chain reactively. So train-all shows only
   LEARNABILITY for every arch; the creativity claim REQUIRES the generalization (held-out) regime.
   (The stage-a/b "reactive fixates" result was specifically under USAGE-SKEW, not when trained on the goal.)
(2) **ARCH FORK RESOLVED: attend-during-generation (Arch2) >> select-among-finished (Arch1).** Arch1 is
   bottlenecked by RANDOM imagination — the oracle ceiling is only 40% on e2 because random length-5
   rollouts rarely CONTAIN the rare chain (K=16 not enough); learned select (47%) even beats naive
   terminal-oracle (it picks by first-op value) but is capped by what imagination generates. Splitting
   incubation(random)+aiming(select) FAILS when the creative chain is rare under random proposal -> a
   learned/goal-aware PROPOSER would be needed. Arch2 has no such bottleneck (it GENERATES the chain).
(3) **ZERO-SHOT generalization (train-typical) FAILS for both (Arch1 4%, Arch2 0%).** BUT per Alex's
   emergence-via-cycles point [[incubation-emergence-via-cycles]], train-on-never-saw-it-at-all is the
   HARSHEST regime, not the verdict — the general imagine+aim routine may only emerge after cycles on a
   VARIETY of goals, then transfer to HELD-OUT ones.
NEXT (the real creativity test): enrich the goal space with MULTIPLE locked axes (each needing its own
repurposing chain), train Arch2 (winner) to deploy repurposing on a SUBSET of locked axes, then test
transfer to a HELD-OUT locked axis never trained — "learned the general repurposing ROUTINE, applies to
a novel instance" = emergent creativity. Then size-for-time; then ground onto a real CPU LLM.

## Result: hybrids.py — step 4: HELD-OUT-AXIS TRANSFER, three hybrids. THE indirect-fire result.
Shared learned pieces trained ONCE (frozen E ; coverage-proposer [goal-agnostic volume coverage ->
learns every axis's chain incl. held-out] ; selector V(effect,target) trained on SUBSET axes {0,1,2,3}
only ; fast Arch-2). Three hybrids = search strategies over them. Trained on subset, ZERO-SHOT on
HELD-OUT locked axis 4 (never a training goal). reached% vs budget:
- HELD-OUT axis 4 (the test): Arch2-alone 0 FLAT ; Hybrid A (deep imagine+select) 10->65 ; Hybrid B
  (1-ply lookahead) 1->5 ; Hybrid C (fast+deliberate) 10->66.
- trained-ref axis 1: Arch2-alone 72->100 ; A 5->51 ; B 6->46 ; C 4->52.
- free axis 0: Arch2-alone 100 ; A 21->86 ; B 56->79 ; C 20->85.
FOUR FINDINGS (big):
(1) **TRANSFER TO A NEVER-TRAINED PROBLEM: SUCCESS via imagination, IMPOSSIBLE without it.** Arch2-alone
   (reactive/generative) = 0 on the held-out axis — it cannot EMIT a chain it never learned. Hybrid A/C
   (imagine + axis-general select) = 65/66% on an axis they were NEVER trained on. Creativity-as-transfer
   REQUIRES the imagine/search loop. (Alex's artillery image: direct fire can't hit defilade; indirect
   fire — compute the arc via the world-model — lands on a ridge it never shelled. The fire-control
   computer = the world-model+selector; it generalizes to new coordinates.)
(2) **The crossover holds, now in TRANSFER form.** Hybrids WIN on the novel axis (65 vs 0) but LOSE on
   trained/free axes (51/86 vs 100). Incubation COSTS on the obvious (line-of-sight), is ESSENTIAL on the
   novel (defilade). Same crossover as stage a/b, now across train/test.
(3) **DEPTH matters: 1-ply lookahead (B) FAILS transfer (5%)** — the chain's first move (P_4 activates the
   register) yields NO immediate goal-progress, so a myopic value can't credit it. Only deep rollouts
   (A/C) contain+credit the full chain. "Attend over self-generated rollouts" needs enough DEPTH.
(4) **A ~ C ~ the imagination CEILING (~65%, cf. oracle 59%).** The selector aims as well as the verifier-
   oracle; the ceiling is set by what the (near-random) proposer GENERATES. To push past 65 -> a better
   coverage/learned PROPOSER (more reliably generates the chain). And Hybrid C did NOT realize best-of-both
   (52 on trained, should be ~100): its V-threshold arbitration over-deliberated -> tune the confidence
   gate so C = fast on known, deliberate on novel. Both are clear, named next levers.
NET: first demonstration in this program that the mechanism deploys a NON-OBVIOUS affordance on a problem
it was NEVER trained on, and that this REQUIRES imagination (reactive=0). Honest caveats: 65% not 100%
(proposer-limited); toy world; costs on trained goals. NEXT: (a) better proposer -> raise the ceiling;
(b) tune C's arbitration -> best-of-both; then size-for-time; then ground on a real CPU LLM.

## Result: hybrids2.py — step 4b: BOTH LEVERS WORK -> a best-of-both creative controller.
LEVER 1 (better coverage-proposer, +coverage-memory so it steers to UNCOVERED axes): the held-out-axis
ceiling rose 65 -> 93% (Hybrid A: 3/19/55/78/93 over B=2..10). It WAS proposer-limited; better
incubation -> higher ceiling. LEVER 2 (progress-gated C: fast by default, deliberate only when the
target coordinate stops improving): C is now BEST-OF-BOTH. reached% vs budget:
- HELD-OUT axis 4 (never trained): Arch2-alone 0 FLAT ; Hybrid A 3->93 ; Hybrid C 0->88.
- trained-ref axis 1:              Arch2-alone 68->100 ; Hybrid A 1->30 (pays deliberation cost) ;
                                    Hybrid C 67->95 (~matches reactive).
- free axis 0:                     Arch2-alone 100 ; Hybrid A 81->100 ; Hybrid C 100 (=reactive).
=> Hybrid C = a SINGLE trained controller that is reactive-FAST on known goals (95-100%) AND deploys a
NEVER-TRAINED repurposing chain on a held-out problem (88%), where a pure reactive/generative policy
gets 0. Fast on the obvious (line-of-sight), creative on the hidden (indirect fire) — chosen
automatically by a progress gate ("deliberate only when the direct approach stalls"). This is the
native, trained form of Airfoil's propose->verify: frozen causal world-model + coverage-proposer
(incubation) + axis-general selector (aiming) + fast policy, arbitrated by progress.
HONEST CAVEATS: 93/88 not 100 (residual proposer ceiling + small switch cost); 19-D toy; transfer is to
a STRUCTURALLY IDENTICAL axis (same chain pattern, new dims) = "apply the learned routine to a new
instance of the same kind" — real within-family creativity, not yet a structurally-different affordance.
Attribution: A's 65->93 is proposer-memory + rollout-len L 4->6 together (isolating arm optional).
NEXT: (a) structurally-DIFFERENT held-out affordance (harder transfer) ; (b) push ceiling 93->~100 ;
then size-for-time frontier ; then ground on a real CPU LLM.

## Result: multiaxis_struct.py — step 5: STRUCTURALLY-DIFFERENT held-out affordance. Localizes the gap.
Trained locked axes 1,2,3 = DEPTH-2 (P_i->C_i). Held-out axis 4 = STRUCTURALLY DIFFERENT: DEPTH-3
two-register cascade P4->T4(register-transfer, an op type no trained chain uses)->C4. Train on {0,1,2,3},
ZERO-SHOT on axis 4. Imagine-ORACLE (random) ceiling: axis4(depth-3) 8->42% @B10 ; axis1(depth-2) 8->55.
reached% vs budget:
- HELD-OUT axis 4 (depth-3): Arch2-alone 0 FLAT ; Hybrid A 2->36 ; Hybrid C 0->37.
- trained-ref axis 1 (depth-2): Arch2-alone 57->100 ; A 26->98 ; C 58->99.
- free axis 0: Arch2-alone 100 ; A 44->100 ; C 100.
THE FINDING (a clean decomposition): the routine DOES fire on a plan SHAPE it never saw (37% vs reactive
0) -> NOT family-bound. BUT it is capped at the RANDOM-oracle ceiling (~42%), unlike the within-family
depth-2 case where the coverage-proposer BEAT random (93 vs 59). So:
  * AIMING (selector) is STRUCTURE-GENERAL — it picks out depth-3 chains it never trained on (hence 37,
    not 0; it aims over whatever depth-3 rollouts exist).
  * INCUBATION (coverage-proposer) is STRUCTURE-BOUND — its learned richness only covers the trained
    families; for the NOVEL structure it falls back to ~random quality, so the ceiling drops 93->~37.
=> The bottleneck for structurally-novel creativity is the PROPOSER/imagination RICHNESS, not the aiming.
This is exactly why Alex's point matters: "in a trained model the imagination can be very rich." Our tiny
coverage-MLP proposer is family-bound; a RICH, BROAD imaginer (= a real LLM's own generation across
structures) is what would lift the structurally-novel ceiling above the random floor. STRONG motivation
for grounding on a real model: the LLM IS the structure-general rich proposer this toy lacks.
Hybrid C stays best-of-both (novel 37 ~ A, trained 99 ~ reactive 100, free 100). NEXT: (a) a STRUCTURE-
GENERAL / richer proposer (or curriculum over structures) to lift novel-structure transfer above random;
(b) the obvious one: ground the imaginer in a real CPU LLM (rich proposer) + this selector/gate; (c)
size-for-time frontier.

## Result: value_search.py — step 6: VALUE-GUIDED SEARCH lifts structurally-novel transfer 42%->90% (NO LLM).
Fixes step 5's bottleneck (structure-bound incubation) by replacing random imagination with SEARCH over
the frozen world-model guided by a structure-general VALUE-TO-GO. Key correction: the old selector was a
GOAL-CLASSIFIER (no credit for setup moves -> 1-ply failed); V_togo(s,t) = P(random k-rollout from s
reaches target), trained on SUBSET axes only, credits the chain's setup move. Search over the (perfect)
world-model + reach-bonus + V_togo heuristic = one step of policy improvement, structure-agnostic.
reached% vs budget on the held-out DEPTH-3 (structurally different) axis:
- random oracle (floor): 8->45 ; greedy V_togo (W=1): 17->59 (already beats random) ; beam V_togo (W=10):
  22->53->75->87->90.  trained axis1(depth-2): greedy 80->99, beam 33->100. free axis0: greedy 100, beam 76->100.
=> VALUE-GUIDED SEARCH (frozen WM + structure-general V_togo) deploys a DEPTH-3 chain it NEVER trained on
at 90% — DOUBLING the random floor (45) — using ONLY the two pieces we trust, NO LLM. This is exactly
Alex's call ("we can do better than random without an LLM, for sure"). The structure-general imaginer =
SEARCH (searches the model; doesn't depend on a learned proposer's family-bound richness); the structure-
general aimer = V_togo. Together: a small controller that is reactive-fast on the known AND deploys novel,
structurally-different affordances at ~90% by spending test-time SEARCH compute (= the size-for-time
trade in action: beam width/depth buys novel-transfer accuracy).
CAVEATS: search uses the PERFECT world-model (apply_op) as simulator (isolates the search/value question;
learned-imperfect-model search is the next check — error compounding). 19-D toy. Greedy(W=1) is more
budget-efficient than beam at tiny B on EASY goals (myopic-best first op); beam dominates at higher B and
on the hard novel axis. NEXT: (a) search over the LEARNED (imperfect) world-model wl.fwd, not true apply_op
-> realism check; (b) size-for-time frontier explicitly (accuracy vs beam compute, small-net vs big-reactive);
(c) then ground on a real CPU LLM. STATE: structure-general creative deployment achieved in the toy.

## Result: value_search_learned.py — step 7: learned-model search — realism gate PASSES but VACUOUSLY + a variance catch.
Plan over the LEARNED forward model (s'=s+wl.fwd(E(s),op)), act in the true world (apply_op), MPC re-plan.
forward-model Δs MSE: overall 0.000 | chain ops P4 0.000 T4 0.000 C4 0.000  <-- the learned model is
NEAR-PERFECT, because the toy dynamics are trivially learnable (additive constants + linear couplings).
reached% vs budget:
- HELD-OUT axis 4 (depth-3): beam PERFECT 21/41/48/53/55 ; beam LEARNED 28/46/51/53/54 ; greedy LEARNED 21->24.
- trained axis1: beam PERFECT 63->98 ; beam LEARNED 64->99 ; greedy LEARNED 79->98.
- free axis0: beam PERFECT 77->100 ; beam LEARNED 76->99 ; greedy LEARNED 100.
TWO HONEST FINDINGS:
(1) **Learned-model beam ~ perfect-model beam everywhere (54 vs 55 on depth-3)** => value-guided search
   survives the learned model — BUT VACUOUSLY: fwd MSE is ~0, so there is essentially NO model error to
   compound. The toy's affine/linear dynamics are too easy; this does NOT yet test error-compounding. A
   real realism gate needs HARDER-to-learn dynamics (nonlinear/stochastic/higher-dim) where wl.fwd has
   genuine error.
(2) **Variance catch:** perfect-model beam on the depth-3 axis is 55 here (seed 3) vs 90 in step 6
   (seed 2) — same code path. So step-6's 90 was partly seed-favorable; the ROBUST claim is "value-guided
   search beats random(~45)+reactive(0) on the structurally-novel axis, but the margin is SEED-VARIABLE
   (~55-90)." Greedy is the unreliable one (24 here vs 59 step6); beam is needed. Report the range, not 90.
NET: search-over-learned-model = search-over-perfect-model ONLY because the model is ~perfect here. NEXT:
(a) HARDEN dynamics (nonlinear coupling / observation noise / partial obs) so fwd MSE > 0, THEN re-run the
realism gate — the real test; (b) multi-seed to quantify the structurally-novel transfer variance honestly;
(c) size-for-time frontier; (d) ground on a real CPU LLM (where the model is genuinely imperfect).
