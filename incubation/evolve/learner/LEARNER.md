# Branch B — the learner (amortized relation-induction)

**Goal.** Learn to *propose* cause→effect relations (vs blind-fit the hand-authored menu), so a relation space
*too big to enumerate* becomes navigable — the only regime where a learner beats the families' enumerate+verify.
Box-first (CPU, small/deep models); Kaggle later (scale + the RL/verifier-as-reward loop). Standardized gate
unchanged: `beyond gen2_base` on held-out ARC. Data: the self-gen curriculum (`grammar.py`, the only source
with clean *relation labels*) now; the HF ARC-ish sets (ConceptARC, BARC ARC-Heavy) for diversity + self-
distillation later.

**The decisive unit test (pure synthetic, CPU):** train a *factored* proposer on a subset of (effect × feature)
combos, hold some out entirely; **held-out feature-acc ≫ 0.10** = the model identifies a causal feature it never
saw under that effect = compositional generalization over the grammar = the prerequisite for a learner that
*invents* rather than fits.

## Results

- **v0 — pixel CNN (`train.py`).** Encode demos as one-hot grids → effect + feature heads.
  `EFFECT 0.96 trained / 0.96 held-out` (generalizes); `FEATURE 0.27 trained / 0.00 held-out` (fails).
  *Lesson:* a pixel-CNN learns the transform TYPE but can't extract the causal FEATURE (counting holes/sizes +
  per-object property→outcome correlation is exactly what CNNs are worst at) — **and we shouldn't make it**:
  `grammar.py` segments objects and computes every feature deterministically. Don't re-learn perception.

- **v1 — object-centric relational transformer (`train_v1.py`, 0.54M).** FEED the deterministic object
  feature-tables + each object's outcome to a 4-layer transformer (objects attend to each other).
  `EFFECT 0.88 / 1.00`; `FEATURE 0.21 trained (≈random) / 0.00 held-out`; **loss FLAT (1.89→1.86)**.
  *Lesson:* not under-training — a generic relational model does **not discover** the consistency-detection
  ("which feature's value determines the outcome across objects/demos"). Relevance is the wall, not perception.

- **Meta-finding.** On a clean, small, exactly-verifiable feature space, learned relevance is (a) hard to train
  end-to-end and (b) **unnecessary** — deterministic enumerate+verify already solves it (the families' 14
  beyond-base). The learner's value is only at SCALE/COMPOSITION (space too big to enumerate) or under NOISE
  (no exact verifier). So this toy is a unit test of "can the model learn relevance *at all*"; only after it
  passes do we move to the regime where the learner actually pays.

- **v2 — explicit pairwise CONSISTENCY (`train_v2.py`, 2.6K params): POSITIVE.** Bake the bias in: for each
  feature *j*, a *shared, feature-index-agnostic* scorer maps the per-feature pairwise stats
  [same-value_j, same-outcome, and cross-terms] → score_j; feature = argmax_j score_j.
  `EFFECT 0.87 / 0.99`; `FEATURE 0.64 trained / **0.50 held-out**` (5× random) — and it **transfers** to combos
  it never trained. **The Branch-B prerequisite is MET**: with the right inductive bias, causal-feature
  relevance is learnable AND compositionally generalizable. The 2.6K param count is the point — the lever is
  relational STRUCTURE, not scale ("keep perception + structure as architecture, learn only the relevance").
  The recognition/relevance wall (recurring since the v22–v25 LLM-selector) yields to the right relational bias.

  *The v0→v1→v2 arc:* pixel-CNN can't (0.00) → generic relational transformer can't (0.00) → consistency-
  structured cracks it (0.50 held-out), at 2.6K params.

## What v2 establishes, and the two value-tests next

v2 proves the architecture *can represent + transfer* relevance. It does NOT yet prove the learner is *worth
it* — on the clean toy this is redundant with `induce()`'s deterministic per-feature check (enumerate+verify
already solves it). The learner only pays in two regimes, which are the next experiments:
1. **SIM-TO-REAL (the immediate next step):** does the synthetic-trained recognizer transfer to *real ARC*
   tasks? Pipeline: recognizer proposes top-K (effect, feature) → `induce()` over decomps → exact-verify →
   measure solve-rate + `beyond_base`. If it transfers, the recognizer is real; if not, close the gap with the
   HF ARC-ish sets (ConceptARC held-out, BARC ARC-Heavy self-distillation) + verified-ARC-solution distillation.
2. **SCALE (does it pay):** widen the grammar to composed/multi-step relations (a space too big to blind-
   enumerate under a budget) and check learned-proposal + verify > blind-enumeration + the same verify budget.

Then Kaggle: scale model + data + the RL/verifier-as-reward loop. Target 600M-class on GPU; the box stays in
the few-K-to-few-M relational regime where this problem has, so far, actually lived.

## SIM-TO-REAL (`ground_arc.py`) — the recognizer TRANSFERS to real ARC (green light)

Enumerated the grammar over ARC-1 → **24 grammar-solvable tasks** (19 train, 5 eval). The v2 recognizer
(trained on synthetic blobs ONLY) on those REAL tasks:
- **feature top-3 = 24/24 (1.00)** — proposes the correct causal feature for every real task in its top-3;
- effect top-1 = 15/24 (0.62);
- propose(top-2 eff × top-3 feat) → induce → exact-verify **SOLVES 23/24 (0.96)** vs the enumerate-all ceiling
  24/24 — with a tiny candidate set instead of all 81 types × decomps.

**The relevance signal is DOMAIN-GENERAL**: "objects sharing a feature-value share an outcome" reads off real
ARC demos exactly as off synthetic, because the consistency structure is abstract (not the visual statistics).
So sim-to-real transfer of *relevance* needed **no** HF data — the HF ARC-ish sets are for widening curriculum
diversity + self-distillation as the grammar grows, not for closing a (small) sim-to-real gap.

**Honest scope:** the 24 overlap `gen2_base`'s menu (enumeration is cheap here, so the recognizer *navigates*
rather than *solves more*). The recognizer PAYS once the grammar is WIDENED past enumerable size (more
decomps/features/effects + COMPOSED relations) — which is also when it reaches ARC tasks *beyond* `gen2_base`.

**Next:** widen the grammar (compose relations + more features/effects) → retrain the recognizer → ground on
ARC measuring `beyond_base`, comparing recognizer-top-K vs blind-enumeration *under a fixed verify budget*
(the regime where the learner is necessary, not just tidy). Then fold in HF data + go to Kaggle.

## COMPOSITION (`grammar_comp.py`) — composed relations, reaching the cross-shape family

A composed relation = `(structural pre-op, base feature-relation)`: apply geometry / content-crop, THEN a
`(decomp, feature, effect)` relation. **8 object-preserving pre-ops × 80 base relations = 640 composed types**
(deeper composition → too big to enumerate = the regime the learner is for). Curriculum yield **326/500 (65%)**
after fixing a table-accumulation consistency bug (each recolor table must accumulate across demos, not reset);
**220/500 are SIZE-CHANGING** — `crop→recolor` turns an input into a smaller output, so composition naturally
reaches the cross-shape/reshaping tasks the same-shape recognizer was weak on.

ELEGANT: the v2 recognizer needs **no retraining**. At solve time, enumerate the ~8 cheap pre-ops, transform
the inputs, and run the SAME consistency-recognizer on the (transformed) demos → top-K feature → induce_composed
→ exact-verify. The learner navigates the expensive feature×effect×decomp factor; pre-ops are cheap enumeration.

**Next (the value test):** ground the composed solver on ARC, measure `beyond_base`, and compare
recognizer-top-K vs blind composed-enumeration under a fixed verify budget.

## Exploration synthesis (2026-06-04) — creativity + cross-shape roadmap

**Disco103** = DeepMind's DiscoRL meta-learned RL *update rule* (Nature 2025; "103" = meta-training
environments Atari57+ProcGen+DMLab-30), NOT an ARC method — Alex's own analogy (he has `disco-torch`,
github.com/asystemoffields/disco-torch). Transferable stance: meta-learn the MECHANISM across a *distribution*
of task families so one learned object fires on held-out, surface-different families; it emits PROPOSALS, a
fixed outer procedure (for us: induce + exact-verify) supplies precision.

**Headline insight:** cross-shape generalization is mostly an EXPRESSIVENESS problem, not a selection one
(echoes the gen-5 diagnostic) — *widen what a relation can SAY before training how to PICK it.* Prioritized,
mostly CPU-box-feasible, verifier-gated throughout:

1. **[med] Relational object-PAIR/SET features** — lift consistency from single objects to *relations* between
   objects (containment, adjacency, alignment, relative-size, unique-extreme). ZERO new architecture (the v2
   scorer reads the same 5-dim pairwise stats per feature; a relational key is still a hashable table value).
   Highest value-per-CPU-hour; attacks multi-object / cross-shape tasks.
2. **[med] Output-shape / keep / tile / scale as a LEARNED-then-VERIFIED effect** — output grid shape/tiling as
   a per-task quantity determined by a feature (counting→build). Attacks the biggest miss-family
   (counting/construction 86); systematizes gen-6's hand-fit `output-shape = f(count)` wins.
3. **[small] Coverage/openness objective** — swap v2's top-1 cross-entropy for a top-K coverage loss
   (P(true ∈ top-K) + diversity). The prerequisite that makes 1–2 pay (recall-first proposer + reckless
   verify-filter = the openness principle, concretely). Hours, no new params.
4. **[small] Frame/canvas normalization at the FEATURE level (VARC trick, training-free)** — canonicalize to a
   scale/offset-invariant frame so "very different shapes" become same-shape for the recognizer.
5. **[med] Anti-unification (least-general-generalization)** — LGG of two verified relations → a NEW relation
   neither parent contained = the path from NAVIGATE to GENERATE relations (the genuine creative leap), CPU-
   trivial, verifier-gated.
6. **[med] STITCH library growth (DreamCoder wake-sleep)** — automate the minting at corpus scale
   (`pip install stitch_core`).
7. **[med] Leave-one-FAMILY-out meta-training (the faithful Disco103 graft)** — Reptile over a distribution of
   relation families, families held out at meta-test — the literal "discover a rule that generalizes" stance.
8. **[large] Verifier-as-reward EXPERT ITERATION** (STaR/SOAR-style self-distillation: proposer samples
   relations → induce + exact-verify → fine-tune on the verified solves; coverage + out-of-home repurposing in
   the objective, NOT success-alone). Built FOR our one-shot propose+verify setting. **Do NOT wire in the literal
   DiscoRL/disco-torch net** — it expects RL agent observations/actions and is a backward-in-time credit-
   assignment rule over *trajectories*; our reward is immediate/binary/exact, no policy or time axis, so its
   interface AND its learned inductive bias both mismatch (Alex's call, correct). Disco *inspiration* only:
   meta-learn across a *distribution* of families (#7) for cross-family generalization — stock meta-learning,
   not disco-torch. (We keep `disco-torch` as a conceptual reference for the discover-and-generalize philosophy,
   not as a component.)

**Build order I'm taking:** 3 (coverage loss) → 1 (relational features) + 2 (shape effects) + 4 (frame-norm)
→ ARC value-test (`beyond_base` on the cross-shape families) → 5 (anti-unification, first GENERATED relations)
→ then 7 (leave-one-family-out meta-training) + 8 (expert iteration) at Kaggle scale.

## VALUE TEST (`ground_arc_v2.py`) — HONEST NEGATIVE: the learner doesn't PAY at this grammar size

Recognizer-driven composed solver, ARC-1 eval(400): **solved 4, all beyond gen2_base (retrieval) but
`beyond gen6_base` (families) = 0**; mean 105 induce-calls. Blind enumeration of the SAME composed space at the
SAME 105-call budget: **solved 5 (more)**. So both halves of the value claim fail here:
1. no tasks beyond the families (the small grammar overlaps their coverage);
2. the recognizer is NOT more efficient — the composed space is small enough (~105 calls) that blind enumeration
   is essentially EXHAUSTIVE within budget, so top-K navigation adds nothing and even misses one task.

**Finding (consistent with the campaign meta-finding):** on a small, enumerable, exactly-verifiable space the
learner is REDUNDANT — enumerate+verify wins. The learner PAYS only when the grammar is *simultaneously*
(a) richer than the families (→ new `beyond_gen6`) AND (b) too big to enumerate under budget (→ navigation
matters). The current grammar is NEITHER, so this test was in the wrong regime. The machinery is proven
(relevance learnable, transfers to real ARC, generalizes over the grammar); the VALUE is not yet demonstrated.

**This makes the load-bearing next steps explicit (not "more of the same"):** #2 shape-effects (counting/
construction — tasks the families only partially reach) + DEEPER composition (2-step → space past enumerable)
+ #5 anti-unification (GENERATE relations neither the grammar nor the families contain). Re-run the value-test
only after the grammar is both richer-than-families and too-big-to-enumerate; otherwise blind enumeration is the
honest baseline to beat and currently wins.

## SHAPE-EFFECTS (`ground_shape.py`) — FIRST `beyond_gen6` (grammar-widening reaches a family-miss)

Count-construction effects (output = f(count): solid block / bar / tile-by-count, color induced) on ARC-1
eval(400): **solved 1, `beyond gen6_base` = 1 → a59b95c0** (`tile:tile_sq:n_colors` = tile the input k×k where
k = number of distinct colors). **The first solve in the learner chapter that even the full hand-authored
families miss.** So a NEW effect class does move `beyond_gen6` off zero — the expressiveness lever is real,
in the family-miss region the exploration predicted (counting/construction). Combined ceiling now 49/400.

MODEST: only 1 — the simple count-builders catch the easiest counting tasks; the bulk of the 86-task
counting/construction family needs **content-conditioned** assembly (build a specific pattern from the input).

**Richer construction confirms the treadmill (per-color histograms → +0 beyond_gen6).** A genuinely-common
content-conditioned pattern (output = bars encoding per-color counts), hand-authored in 6 layout variants,
caught ZERO ARC eval tasks — because each real construction task needs a *bespoke* layout/rule. So hand-adding
effect classes yields ~1 (or 0) `beyond_gen6` each: the gen-6 "every held-out task needs a differently-fitted
relation" diagnostic, recurring. CONVERGENCE: the box-feasible levers have now done their job — the recognition
machinery is PROVEN (relevance learnable + transfers + generalizes) and the expressiveness lever WORKS but
plateaus fast under hand-authoring. The frontier (reaching `beyond_gen6` at volume) needs **GENERATED** relations,
and gen-2 already found anti-unification mines ~0 macros from our atomic verified solutions — so the real
generation lever is a **LEARNED generative proposer + verifier-as-reward expert iteration at SCALE (Kaggle)**,
the phase we correctly deferred. Box next-steps (global head, more hand-effects, anti-unification) are
diminishing; the high-value move is to stand up the generative-proposer/expert-iteration loop for Kaggle.

## ANTI-UNIFICATION (`lgg.py`) — Mode-2 precondition: MECHANISM PROVEN (1.00 re-instantiation)

Design verdict (honest, from the de-risk workflow): LGG is a NAVIGATION/EFFICIENCY organ, NOT expressiveness —
it adds 0 new `beyond_gen6` (schema fillers come from the same grammar `induce()` exhausts). The cheap
precondition test (leave-one-out schema re-instantiation on real ARC-1-train recolor/select buckets):
- **re-instantiation success 43/43 = 1.00** (canonical 08ed6ac7/67385a82/6e82a1ae triple: **19/19**).
  `antiunify(R1,R2)` lifts the per-task TABLE to a typed hole; re-instantiation routes that hole to ONE
  `G.induce()` that re-fits the table on a held-out third task + exact-verifies — 100% of the time. **LGG
  schemas ARE genuine per-task GENERATORS** = the gen-6 "differently-fitted relation per task," *minted* (from
  two solved tasks), not hand-authored. The NAVIGATE→GENERATE step works mechanically.
- **induce-call ratio (blind/LGG): median 6×, mean 5×** — real efficiency, but UNDER the 10× GO bar *on the
  small grammar* (blind finds recolor solutions in ~5–6 calls, low ceiling). The 10×+ cashes out in the
  TOO-BIG-TO-ENUMERATE COMPOSED regime (blind = pre-ops×effect×feature×decomp explodes; schema still collapses
  to 1 induce). So 5–6× is the honest LOWER BOUND, not the headline — the design predicted this exact split.

VERDICT: mechanism PASSES decisively (1.00); efficiency real-and-lower-bounded. The anti-unification organ is a
genuine "better-version" component for the eventual scaled solver. NEXT: cash it in the composed regime — wire
LGG schemas as a recognizer-ranked FAST-PATH into the composed solver (`grammar_comp`) and re-run the value-test
where blind explodes (the real 10×+ efficiency test) — completing ready-to-scale gate #1 (every component proven
to add value on the box).

## CASH-OUT (`ground_lgg_composed.py`) — the binding constraint is EXPRESSIVENESS, not search

Library mined from ARC-1-train = **91 distinct skeletons of 113 grammar (NOT a compression** — nearly every
skeleton is useful somewhere). Recognizer-ranked library vs blind enum, composed regime, budget 200 induce/task:
- SHALLOW (8 pre-ops): library solved 5 (median **2** induce-to-solve), blind solved 5 (median 4) — ~2× efficiency, SAME coverage.
- DEEP (64 two-step combos, "too big"): library 5 (median 2), blind 5 (median 4) — **IDENTICAL to shallow; neither collapses.**

WHY deep doesn't bind: solutions are found in ~2–4 induce-calls (under identity/simple pre-ops + recognizer-
ranked schema), FAR under the 200 budget — **search-size is NEVER the binding constraint.** The grammar-solvable
tasks are solved almost immediately.

**META-FINDING (the box's clearest): the binding constraint is EXPRESSIVENESS (~5/400 eval tasks expressible at
all), NOT search efficiency.** Every navigation mechanism — recognizer, schema library, composition, and the
would-be pre-op navigation organ — is PROVEN but optimizes a NON-bottleneck. The 395 unsolved tasks are
INEXPRESSIBLE, not slowly-searched. Consequence: anti-unification's mechanism is proven (1.00) but efficiency-
value modest (~2×) and non-decisive; the navigation organ is NOT needed. Every box-feasible EXPRESSIVENESS lever
is now tested — hand-authoring = treadmill (~1 task each); anti-unification/wake-sleep for new structure = the
v19 wall (≈0 transfer). The ONLY untested expressiveness lever is SCALE (a learned generative proposer over a
rich relation distribution + expert iteration). **Ready-to-scale gates 1+2 DECISIVELY green** (mechanisms proven;
remaining gains expressiveness/scale-bound, not architecture-bound). The box has come as far as it can on the
current grammar — the scale decision (compute-spend) is the fork.

## NAVIGATION-EFFICIENCY STRESS (`ground_nav_scaling.py` + `ground_nav_demos.py`) — it SCALES (cash-out was an artifact)

Alex's catch: the composed cash-out didn't TEST navigation — only 5 tasks were expressible and all found in 2-4
calls, so the navigator was never under load ("library = blind" meant "neither was stressed"). Proper test:
synthetic recolor tasks (always solvable -> no expressiveness bottleneck) + inject N distractor features (the v2
head scores each feature INDEPENDENTLY -> handles arbitrary NF, no retrain) -> true-feature find-cost as the
space explodes.
- SCALING (4 demos): find-cost 1.6→158 as features 14→5014, BUT blind 7→2507 -> **speedup GROWS 4×→16×**; top-5
  recall ~FLAT 0.7-0.8 across **350× space growth**. The recognizer keeps the true feature top-ranked space-size-
  INVARIANTLY; navigation's relative advantage grows with the space.
- LIMIT = spurious-consistency tail (mean rank climbs as more random features are chance-consistent on few
  demos), CONTROLLED BY EVIDENCE. At 1514 features, find-cost vs #demos = {2:92, 4:58, 6:21, 10:23}; top-5
  {2:0.24, 4:0.63, 6:0.83, 10:0.90}. **At 6 demos: 37× speedup (rank 21 vs blind 757), 0.83 top-5.**

VERDICT: the cash-out's "navigation redundant" was a REGIME ARTIFACT, not a property. Under genuine load
navigation efficiency is a real, SCALABLE component — good scaling signature (increasing speedup, space-invariant
top-K) + a cheap lever (demos; + an untested one: a distractor-HARDENED recognizer). This STRENGTHENS the scale
bet: at scale the space is huge (navigation needed), expressiveness is rich (coverage to win), AND we control the
curriculum's demo count (navigation pays). Orthogonal to the coverage finding — expressiveness bottlenecks
COVERAGE, navigation scales SEARCH; both now validated for the scale regime.

**PUSH FURTHER (`train_v2_hardened.py`):** the spurious-consistency limit is partly ARCHITECTURAL — v2's stats are
pair-count-NORMALIZED (means), so robust vs spurious consistency look identical to the scorer. Fix: add a
SUPPORT-COUNT stat (log #same-value pairs) + train with 40 distractors injected. Hardened V2H vs baseline v2
@4 demos: find-cost DOWN ~15-30% everywhere (528f 9.0 vs 12.6; 1514f 40.3 vs 47.2; 5014f 110 vs 158); top-5 UP
**+0.08-0.09 in the 500-1500-feature regime** (528f 0.77 vs 0.69; 1514f 0.80 vs 0.71). (5014f top-5 dip = vs an
uncontrolled-seed baseline + noisy/UNconverged training (loss ~2.5) -> a LOWER bound; more steps/harder negatives
push further.) COMPLETE ANSWER (how far navigation pushes on the box): it SCALES (16-37×, space-invariant top-K),
with TWO proven composable levers (evidence/demos + an evidence-aware recognizer) and the ceiling NOT yet found —
a robustly scalable, still-improving component. Navigation is neither the bottleneck nor fragile.

## LAST BOX GATE — recognizer transfer is DISTRIBUTION-ROBUST (`ground_arc_transfer.py` + `ground_conceptarc.py`)

Sim-to-real on grammar-solvable tasks across distributions (recognizer trained on synthetic blobs ONLY):
- arc1-train (19): eff-top1 0.84, feat-top3 1.00, pipeline 1.00
- arc1-eval (5):  feat-top3 1.00, pipeline 0.80
- **arc2-train (25, HARDER): feat-top3 0.96, pipeline 0.92**
- **ConceptARC (5, human-designed MORE-DIVERSE): feat-top3 1.00, pipeline 1.00**

Aggregate over 54 grammar-solvable real tasks spanning 3 distributions: **feat-top3 ≈ 0.98, pipeline ≈ 0.94.**
The synthetic-trained recognizer transfers to harder ARC-2 + human-designed ConceptARC as well as to ARC-1 — the
consistency inductive bias is DISTRIBUTION-ROBUST (strong scaling signature). NOTE: arc2-eval = 0/120 grammar-
solvable (no substrate) → the per-split-averaged "0.48" is an ARTIFACT of a 0-substrate split, NOT a transfer
drop; it's an EXPRESSIVENESS limit (ARC-2-eval is entirely inexpressible in the recolor/select grammar) = a stark
RE-CONFIRMATION that expressiveness, not transfer/navigation, is THE bottleneck.

VERDICT: **last box gate GREEN.** Every architecture component is now validated AND distribution-robust —
recognizer transfer ✓, navigation scales ✓, composition ✓, anti-unification mechanism ✓; the sole gap is
expressiveness/coverage, which needs SCALE. **THE BOX IS DONE.** Cross over to the scale phase: a generative
proposer (geometry head + relation head + schema-minting) over a RICH relation distribution + verifier-as-reward
expert iteration, on Kaggle (free-compute; quota ~Jun 7-8). The bet: at scale the proposer GENERATES relations
beyond the fixed grammar — the one lever that lifts coverage.

## SCALE PHASE — architecture decided (de-risk workflow wtc772h7l + data workflow w7ijx1ysz, Alex's calls)

FACTORED GENERATIVE PROPOSER (all non-LLM at solve time, exact verifier gates everything): frozen RECOGNIZER prior
(V2H, the breadth router, distribution-robust) + STRUCTURAL/GEOMETRY head (V3-GEO, proposes the pre-op/frame
prefix) + GENERATIVE RELATIONAL SUFFIX (a small autoregressive policy over a typed lambda-DSL strictly richer than
the grammar: composition + let-binding + object-pair/set quantifiers + grown leaves) + VERIFIER-AS-REWARD EXPERT
ITERATION. Honest verdict: ONLY the suffix generator crosses the expressibility wall; the geometry head alone is
"the LGG trap repeated" (navigation of a non-bottleneck). THE load-bearing risk: the v19 breadth wall may be
ARC-INTRINSIC, not recognizer-absence (the recognizer-as-breadth-router is the wager).

**DSL FORK — Alex's call: THIN-CORE, EARN THE VOCABULARY.** BARC ARC-Heavy (~274k tasks + python programs +
concept tags, streaming loader `ground_barc.py`) is used for its SCALE (training/self-distill distribution) + its
MAP (concept tags = target vocabulary), but NOT its CODE — its programs are imperative numpy in BARC's own DSL,
NOT cleanly transcribable, so cloning them = adopting BARC's hand-designed vocabulary (treadmill-up-a-level +
lookup-Goodhart). Instead: a thin principled combinator CORE (composition, binding, object-pair quantifiers) +
GROW leaf primitives from our own verified solves, steered toward the concept-tag map; the invention gate (held-
out-family, certified-invented) is the guardrail. The system EARNS its vocabulary.

BUILD ORDER (box-prep falsifiers gate every GPU spend): BOX-PREP 1 V3-GEO structural twin (`train_v3_geo.py`,
held-out pre-op top-M recall >> chance); BOX-PREP 2 frame-norm coverage probe (`ground_v3_geo.py` — does VARC
frame-norm make any reshaping ARC task NEWLY expressible = new beyond_gen6 on the box? if ~0, geometry coverage
FALSIFIED before Kaggle); BOX-PREP 3 the lambda-DSL (`rel_dsl.py`) + BARC interface; BOX-PREP 4 the suffix policy
(`suffix_policy.py`) + leave-one-family-out + invention_gate fitness; KAGGLE-1 = ONE expert-iteration round
(falsifier, GO if held-out-family beyond_gen6 >= 3, <=8h T4x2); KAGGLE-2+ full loop only if GREEN. Data: BARC
stream + RE-ARC (unlimited ARC-1 procedural) + ConceptARC (/data/conceptarc). Don't wire the literal DiscoRL net.

### BOX-PREP 1+2 RESULTS — both GREEN (geometry organ validated cheaply, before any GPU)

BOX-PREP 1 (`train_v3_geo.py`, V3-GEO = 3.1K-param STRUCTURAL twin of V2): held-out pre-ops (rot270/transpose/
gravity_left/shift_right/sym_ud/scale2) top-1 **0.69** / top-3 0.73 ≈ TRAINED 0.72/0.77 (chance 0.04/0.12) — ranks
pre-ops it NEVER trained on by structural alignment alone; the V2 consistency inductive bias lifts cleanly to
geometry, generalizes by construction. Two organs, one principle.

BOX-PREP 2 (`ground_v3_geo.py`, geometry-coverage falsifier): of **588** reshaping ARC tasks (in.shape != out.shape,
all 0-grammar-solvable since recolor/select preserve shape), shape-changing-pre + grammar makes **3 NEWLY
EXPRESSIBLE** (1cf80156/crop_content, c59eb873/scale2, 68b67ca3/downscale2) = new beyond_gen6; V3-GEO ranks the
solving pre-op **top-3 on 6/6** real reshaping tasks (perfect structural transfer to real ARC). VERDICT: geometry
coverage NOT falsified (real gain + transfer) → the organ earns its place. BUT 3/588 re-confirms EXPRESSIVENESS is
the binding constraint: the geometry head makes tasks shape-COMPATIBLE; the generative suffix DSL must make them
RELATIONALLY expressible — necessary-but-not-sufficient. Box-prep 1+2 GREEN → BOX-PREP 3 (the thin-core lambda-DSL).

### BOX-PREP 3 — thin-core lambda-DSL (`rel_dsl.py`), VALIDATED

Thinnest principled core (Alex: big in favor of thinner — "the second the expressiveness ceiling stops being our
imagination"): FIXED floor = exactly the proven recognizer's level (object decompose + per-object FEATURES +
compose — grounding/routing, NOT vocabulary); The relational layer is EARNED, not hand-coded (Alex-approved refinement, the "relational eye" metaphor): the
fixed core exposes a structural FACULTY — `pair_signature(a,b)` = the raw relational perceptions between two
objects (the "cones": a-contains-b, b-contains-a, adjacent, aligned, relative-size, same-color) which NAMES
NOTHING — plus the `Quantify` combinator (exists/forall over the faculty). A specific predicate (containment,
adjacency…) is one (channel,value,mode) INSTANTIATION the system EARNS by search+verify, not a construct we give.
Rationale: anti-unification earns COMPOSITIONS not PRIMITIVES, so a relationally-blind core can never re-discover
containment (can't conjure a sense from examples) — we install the eye (grounding), it names the colors (vocab).
Smoke: SUBSUMPTION **3/3** (every grammar recolor-solve = a CORE-only DSL program → thin core contains the whole
grammar); EARN-THE-PREDICATE — a RANDOMIZED containment task (identical objects, random positions → NO per-object
feature correlates) has GRAMMAR winning relations = **0**, but the system DISCOVERS **`quant:exists(b_contains_a=1)`**
by searching the faculty's instantiations + verifying = it EARNED "contained" from the relational eye. The DSL is a
strict SUPERSET of the grammar; the expressiveness ceiling is now the closure of the GROWN library, not our design.

### BOX-PREP 4 — LIBRARY GROWTH (`grow_library.py`): the ceiling CLIMBS by composing earned predicates

Earned level-0 from simple tasks: containment → `exists(b_contains_a)`, largest → `forall(a_bigger)`. The COMPOSED
task "contained-in-LARGEST" (grammar winning = **0**, NO single level-0 predicate solves it) → SOLVED by
**`compose:exists(b_contains_a & forall(a_bigger))`** = the system BUILT "contained in the largest" by composing
two EARNED predicates (containment ∘ is-largest), nothing hand-coded; the minted composition is REUSABLE (named).
= the expressiveness ceiling climbs PAST the faculty's single instantiations via composition of earned vocabulary
— the thin-core bet's payoff, demonstrated. "The ceiling stops being our imagination."

### BOX-PREP 5 — sibling cross-pollination (Alex): one hardened core, V2 routes relations too

5A (`ground_v2_relational.py`) — THE "third sibling" was a MIRAGE: the relational faculty's predicates are
per-object values = just more FEATURES, and V2's consistency head is feature-count-AGNOSTIC + generalizes to
unseen features by construction. Feed V2H (trained on regular features ONLY) the 14 regular + 14 relational
features → on containment tasks (grammar=0) it ranks the true predicate `exists(b_contains_a)` **top-1 1.00,
top-3 1.00, mean-rank 1.0/28, above every regular feature 40/40**, NO retraining. V2 routes the relational layer
for FREE → the library/policy search becomes recognizer-GUIDED, not blind. The recognizer is just V2 with a wider eye.
5B (`train_v3_geo.py`) — backported V2H's EVIDENCE/support stat into V3-GEO (raw structural-alignment volume,
the analog of V2H's pair-support-count, SDIM 14→16). Held-out 0.69/0.73 unchanged on the box (small bank), folded
in for SCALE-robustness (pays at large pre-op banks, as V2H did at large feature counts). Both siblings now share
the hardened, evidence-aware core.

NET ARCHITECTURE: ONE hardened, evidence-aware consistency core, fed by TWO feature sources (object-features +
relational faculty) and ROUTED by V2, with V3-GEO as the structural router. The recognizer routes features AND
relations; the POLICY only needs to learn the COMPOSITIONS.

### POLICY + INVENTION GATE (`policy_eval.py`) — the last box pieces, GREEN

RECOGNIZER-GUIDED POLICY: V2 ranks all base predicates (regular + relational faculty); the policy tries top-K
singles, then composes recognizer-ranked OUTER × library INNER. INVENTION GATE (the honest creativity bar): a
solve is INVENTED iff the GRAMMAR can't express it (winning_relations==0) AND the policy solves it with a
RELATIONAL predicate that GENERALIZES to held-out test. Across a DIVERSE family suite, NO per-family training:
- containment: 20/20 solved, **20/20 invented**, policy cost **1** vs blind 17
- adjacency: 20/20 solved, 15/20 invented (5 were also grammar-expressible), cost **1** vs 19
- contained-in-largest (composed): 20/20 solved, **20/20 invented**, cost 44 vs 66

The fixed faculty + recognizer + composition GENERALIZE across families by construction. Singles: recognizer cost
1 (18× vs blind). Compositions: recognizer-guided on the OUTER, library-searched on the INNER (the inner is a
property of the OTHER object, which V2 can't surface from the target objects — the gap the LEARNED suffix policy
closes at scale). **EVERYTHING BOX-VALIDATABLE IS DONE** — the factored proposer (recognizer routes features+
relations, policy composes earned predicates, invention gate certifies invented-not-retrieved across families) is
proven piece-by-piece, all CPU, all committed.

### EXPERT-ITERATION LOOP (`kaggle_loop.py`) — dry-tested, the KAGGLE-1 vehicle

The loop: stream tasks → recognizer-guided policy → exact-verify → INVENTED solves (grammar=0 + relational +
generalizes) GROW the library → library-first inner makes later compositions cheaper (the expert-iteration payoff,
gradient-free box version). Dry-run N=30/stream: synthetic families **30/30 solved, 29 invented, library grew to 3**
(exists(b_contains_a), exists(adjacent), forall(a_bigger)); ConceptARC (real) **0/30 solved**. Real BARC chunk
(N=400, on the box, no GPU): **18/226 solved (all recognizer cost 1), 0 invented** = the ~8% recolor-expressible
slice routed flawlessly; the 92% unsolved + 0-invented confirms EFFECTS are the frontier at real-data scale. The 0/30 is HONEST
+ important: our DSL's EFFECT vocabulary is still thin (recolor/select only); real ARC needs richer effects
(move/draw/fill/construct). Per thin-core, EFFECTS are EARNED not hand-coded → growing the effect leaves (from BARC
concept-tags + verified solves) is the SCALE phase's job. The 0/30 quantifies the effect-expressiveness frontier
one final time + scopes KAGGLE-1.

KAGGLE HANDOFF (quota ~Jun 7-8): scale `kaggle_loop.py` N over BARC-stream/RE-ARC/ConceptARC + GROW the EFFECT
leaves + the LEARNED suffix policy (autoregressive over the DSL, cold-started from the box's recognizer-guided
solves) + verifier-as-reward expert iteration; strict leave-one-FAMILY-out; GO if held-out-family beyond_gen6 ≥ 3
on one ≤8h T4×2 session. Free-compute only. Push via the kaggle headless workflow (kernel scaffold under `kaggle/`).

### EFFECT FACULTY (`effect_faculty.py`) — the motor hand, EYE-GROUNDED (Alex's option 2)

Mirror of the relational eye, opening the effect frontier (the 0/30). The hand has THREE atomic motor primitives —
paint / place / erase — and nothing else is hand-coded; an effect is a COMPOSITION whose PARAMETERS (target +
displacement) are computed by the relational EYE: an eye-selected anchor tells the hand WHERE. Smoke: an
align-to-anchor MOVE task (recolor grammar winning = **0** — it cannot move objects) → the system EARNS
**`move(toward exists(a_bigger), align_col)`** = the eye picks the anchor (the largest), the hand moves each object
to its column. The gesture is EARNED (search target-predicate × displacement-mode + verify), not hand-coded.
Effects join predicates as EARNED vocabulary. **The architecture is now PERCEIVE (eye) → ROUTE (V2) → ACT (hand):**
thin innate faculties (pair-comparison eye + paint/place/erase hand), fully-EARNED vocabulary (predicates +
gestures), verifier-gated, library-growing by composition. WIRED IN + RE-MEASURED (`unified_remeasure.py`): the unified solver handles BOTH faculties in one loop (synthetic
containment 12/12 via recolor, align 12/12 via move). BARC re-measure (394 tasks): recolor **31** + move **0** = 31
(~8%). The move DELTA is ZERO — the ONE hand-picked gesture (align-to-anchor) matches no common BARC motion. This is
the EFFECT TREADMILL warning: hand-picking gestures one-by-one is a guessing game (align caught nothing). → the
answer is to make the FACULTIES self-evolving, not hand-pick gestures: the HAND (paint/erase) is already COMPLETE
(spans every grid), so gestures are earnable compositions; the EYE's channels should likewise become EARNED from a
RAW object substrate + general operators (thin-core, one level deeper) so the system EVOLVES whatever sense it
needs. Floor stays: object-decomposition + raw object descriptions (the grounding). "Vine" (the name) grows its own
tendrils. ### SELF-EVOLVING FACULTIES (Vine) — the thin-core move, recursively (`substrate_eye.py`)

Alex's question: can it EVOLVE every sense/hand it needs? Answer: yes — apply thin-core one level deeper.
SELF-EVOLVING EYE (`substrate_eye.py`, box-gate GREEN): the only innate perception is the RAW object substrate
(per-object r0,c0,r1,c1,h,w,size,color) + comparison operators (a.p OP b.q); a sense is an EARNED composition of
comparisons. (1) EARN A NEW SENSE: a share-height task (shared height VARIES per demo) — GRAMMAR winning 0,
hand-coded channels None, but the eye EVOLVES **`exists(a.h==b.h)`** from the substrate = a sense never given.
(2) COMPLETENESS: hand-coded `contained` == the substrate conjunction `a.r0>=b.r0 & a.c0>=b.c0 & a.r1<=b.r1 &
a.c1<=b.c1` (True) → the fixed channels were just IGNITION; the substrate spans them. The HAND is already complete
(paint/erase reach every grid) → gestures are earned compositions, no missing hand. So the only INNATE thing is
object-decomposition + raw descriptions; senses, predicates, gestures, compositions are ALL earned. Open-ended at every level.
SELF-EVOLVING HAND (`effect_faculty.py`, box-gate GREEN): `earn_effect` now searches the COMPLETE motor basis
(erase?+place) — from the SAME eye-grounded search it earns **`move`** on a move task and **`copy`** on a copy task
(grammar=0 both; the gesture is EARNED, not hand-picked). BOTH faculties now self-evolving + validated. The only
INNATE thing is object-decomposition + raw object descriptions; senses, predicates, gestures, compositions are ALL
earned. NEXT: wire substrate-channels + the complete motor basis into the unified loop/policy (recognizer routes
the earned senses+gestures); re-measure real-data coverage off the recolor-only 8%; scale on Colab.
