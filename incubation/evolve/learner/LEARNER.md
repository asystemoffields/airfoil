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
