# Vine: Verifier-Grounded Open-Ended Creativity from a Thin Core

*A research narrative — from "airfoil" to "Vine." The honest intellectual story, dead ends included.*

---

## 0. What this document is

This is the spine of the project as it actually unfolded: the motivations, the falsifiable bars, the
experiments that worked, and — more importantly — the experiments that *failed in instructive ways and steered
everything that followed*. It is written to be the backbone of a paper, so it favors specifics, numbers, and
milestone commits over slogans. Where a result is proven, it says proven. Where the project is making a bet, it
says bet. The single most load-bearing claim — that the system **earns** its concepts rather than memorizing
them — is also the one most easily faked, so it is guarded throughout.

The system is named **Vine** (it "grows its own tendrils"). Its fossil codename was **airfoil**, after Alex's
guiding metaphor: heavier-than-air flight was physically possible in the neolithic — the bottleneck was never
materials, it was *configuration*. The whole research stance follows from that: assume the substrate (a little
search + a verifier + a few tiny trained nets) is already sufficient in principle, and treat the entire problem
as finding the *arrangement*. Biology is the existence proof (a 20 W brain is "the bird"); overpowered search is
"the rocket" that proves the destination is reachable, to be distilled into the cheap "wing." Configuration is
scale-invariant, which is the non-mystical reason a discovery made on a 7 GB laptop is expected to transfer
upward unchanged.

---

## 1. Motivation and the creativity bar

The origin thesis was **compression *is* generalization**: optimizing the shortest *reusable* description forces
the discovery of recurring structure (Solomonoff/MDL). An early arc of tiny CPU experiments (the "v0–v19"
chapter) established this cleanly on synthetic domains — library reuse paying its own bits, 2.28× transfer,
depth-generalization, ~21× faster solving from a learned library, a verifier "wind tunnel" showing diversity (not
volume) rescues a systematically-weak verifier. Then v19 took the depth-engine to real ARC and measured the
boundary: a geometric+recolor DSL reached ~6.8% and the cross-task loop transferred ≈0. The conceptual result was
sharp and became the project's hinge: **airfoil was one half of a complete tool — the DEPTH engine** (compositional
reuse of a *given* vocabulary). ARC's difficulty is **BREADTH** — recognizing *which* concepts a novel task needs.
A long detour (v20–v27) tried to borrow breadth from a small LLM as a recognition/selection front-end. It failed
decisively and usefully: across v22–v25 the LLM never beat blind search; v24/v25 localized *why* — the model HAS
the breadth but cannot **rank task-relevance** (a 1.7B over-includes to ≈the whole DSL; a 360M under-includes).
**Relevance-discrimination, not knowledge, is the gap.** That finding, and Alex's resulting steer — "if we can make
something better than an LLM, pivot to that and toss the LLM" — set the terms for everything after: **no LLM at
solve time.**

In parallel, a synthetic "incubation" line refined what *creativity* even means here, and produced the second
hinge. Its operational, no-woo definition (governing the whole project):

> **Creativity = an unrestricted grasp of cause-and-effect** — inducing the invariant causal mechanism that
> generates a task's observations and deploying its cause→effect relations *freely*, including outside their
> typical role (repurposing) — **plus** the real-time *invention* of novel cause→effect heuristics, not retrieval
> from a fixed menu. The two reduce to one target: *invent, fast, at inference, a novel invariant cause→effect
> mechanism, from accumulated experience, unrestricted by any stored set.*

This definition is paired with a **bio anchor**: the strongest predictor of exceptional human creativity is
*openness to experience*. Operationalized, that says the lever is the *experience-acquisition* side, not a
smarter solve-time filter — train for **coverage not precision** (propose broadly; let an exact verifier supply
precision), keep the library **unpruned** (low latent inhibition — old material is raw stock for the next
repurposing), and **credit repurposing as a disposition**.

The incubation line then triangulated, three independent ways, that **creativity does not live in weights**:
scaling a reactive policy 45× in parameters reaches **0%** on a structurally-novel held-out affordance (s11);
distilling a successful search into a reactive net falls **below random** (s12); training cycles *specialize the
net away* from novel transfer rather than growing it (s15). What *does* transfer is **inference-time search over a
task-agnostic world-model, guided by a structure-general value function** — reaching **90–92%** on a depth-3
chain it never trained on (s6, s11), with the artillery image as its intuition: direct fire cannot hit a target
in defilade; indirect fire computes the arc through the world-model and lands on a ridge it never shelled. Two
further laws fell out and constrain the endgame: model **exploitation**, not inaccuracy, is the binding failure
of planning over a learned model (a confidently-wrong model is worse than no model, s9); and **"Alex's Law" —
any uncertainty-averse pessimism penalty always hurts a creativity-seeker**, because novelty *is* model-
uncertainty, so the fix is verify-by-acting, not a penalty (s10/s10b). These are the design priors Vine inherits.

**The bar, made falsifiable (the gate).** A solve counts as creative only if the *system* grasps/invents an
invariant cause→effect mechanism that **generalizes to held-out tasks** and that **a strong retrieval baseline
cannot produce**. Coverage that generalizes is good *material* but is not creativity. The honest metric:
**invented-and-generalizing, beyond the best retrieval solver, on a held-out eval that was never used for
selection.** This bar is non-negotiable because ARC solve-rate alone is Goodhart-prone — primitive-adding +
search + overfit can climb it with zero creativity. *"It needs to actually be creative, not a thing that gestures
at creativity."*

---

## 2. Approach: DIY-AlphaEvolve and the gen-0→6 campaign

**Method.** *DIY-AlphaEvolve*: agents are the *design-time* variation operator, evolving a whole non-LLM solver
generation by generation; the deployed artifact is **search + small trained nets, no LLM at solve time.** A fixed
fitness harness (2-attempt ARC rule, exact verify, held-out arc1-eval never used for selection) and a **standardized
ablation gate** — count only solves the *best retrieval solver* misses — keep the creativity claim honest.

**The campaign (ARC-AGI-1; dev = arc1-train 400, held-out = arc1-eval 400).** The arc of the headline number,
and the much more important arc of *certified-beyond-retrieval*:

| gen | held-out solved | beyond strong retrieval | lever |
|---|---|---|---|
| seed (gen-0) | 2 (0.5%) | — | best-first DSL search (perfect world-model = interpreter) |
| gen-1 | 28 (7%) | 0 | parametric concept-fitting |
| gen-2 | 37 (9.25%) | 0 | consolidated retrieval + linking/transfer → **gen2_base** (the standard ablation) |
| gen-3 | ≤34 | 0 vs gen2_base | generative mechanism-inventor (composition) |
| gen-4 | 38 combined | 4 | **first** beyond-retrieval (per-task *fitted* relations) |
| gen-5 | 45 combined (11.25%) | 11 | systematic relation-induction |
| **gen-6** | **48 single solver (12%)** | **14** | relation-induction sweep + consolidated merge |

Two findings from this campaign were decisive and shaped the entire learner chapter.

**(a) The standardized-ablation truth (gen-3).** Graded on *its own* curve, gen-3's causal-decomposition inventor
banked "9 certified-invented" solves. Re-graded against gen2_base as the shared strong ablation: its 11 solves
were a strict *subset* of base, and the union of all three inventors added **0 beyond retrieval.** Every "invented"
solve was already reachable by the rich parametric menu, reached by a different route. The standardized ablation
was non-negotiable precisely because, graded on its own curve, gen-3 would have banked a fake "9." This is the
discipline that makes the rest of the project trustworthy.

**(b) Verb-composition is a dead end; per-task fitted RELATIONS are where ARC's creativity lives (gen-5
diagnostic).** Before building a learned proposer over compositions, the gating question was asked: is
composition's non-transfer a *selection* problem (a prior could fix) or an *expressiveness* one (it can't)? The
answer, high-confidence: **expressiveness.** Of the 366 held-out eval tasks gen2_base misses, the number with even
*one* train-consistent program at any depth (exhaustive d1–2, 3×-budget d4 spot-check) in the relational-verb
alphabet was **0**. The verbs don't *engage* these tasks — there is nothing to compose, so nothing for a prior to
select. This *corrected* the earlier "composition overfits" read: on *train*, where compositions fit they
generalize ~100% (tight geometry equivalence classes); composition just almost never *fits* the held-out
families. **Every** real beyond-base win came from per-task *fitted* cause→effect relations (recolor-by-hole-
count, select-extreme-over-uniqueness, …), never from verb composition.

The consolidated `cand/gen6_base.py` — verified as a *single* non-LLM solver — reached **48/400 held-out (12%),
14 certified beyond-strong-retrieval, 0 regressions**, from a 0.5% seed. But gen-6 also confirmed the ceiling:
counting/construction added one genuinely new expressive class (output-shape = *f*(count)); line/draw and
object-movement re-derived existing wins and added **0** new. Each held-out task in the hard families needs a
*differently-fitted* relation, which an agent authoring a *fixed menu* cannot provide. The campaign therefore
rested at a clearly-stated fork: **the genuine next leap is the system generating its own relations per task** —
a learned generator / RL-expert-iteration phase. That fork is the learner chapter.

---

## 3. The recognition wall, and the consistency-head crack

To *generate* relations the system first needs to *recognize* which cause→effect feature a task is about — the
breadth/relevance organ the LLM detour failed to deliver. The learner chapter (Branch-B) built a clean unit test
on a **self-generated relation grammar** (`grammar.py`) — the only data source with clean *relation labels*. A
relation is a typed `(decomposition, feature, effect)` rule; the grammar both *renders* curriculum tasks forward
and *induces+verifies* relations inverse. The decisive unit test: train a proposer on a subset of
(effect × feature) combos, **hold some out entirely**; held-out feature-accuracy ≫ chance = the model identified
a causal feature it never saw under that effect = compositional generalization over the grammar = the prerequisite
for a learner that *invents* rather than fits.

The wall, and how it was cracked, is the chapter's pivot:

- **v0 — pixel CNN.** EFFECT 0.96/0.96 (generalizes); **FEATURE 0.27 train / 0.00 held-out.** A pixel-CNN learns
  the transform *type* but cannot extract the causal *feature* (counting holes, per-object property→outcome
  correlation is exactly what CNNs are worst at) — *and we shouldn't make it*: the grammar already segments
  objects and computes every feature deterministically. Don't re-learn perception.
- **v1 — object-centric relational transformer (0.54M).** Fed the deterministic feature tables to a 4-layer
  transformer where objects attend to each other. EFFECT 0.88/1.00; **FEATURE ≈random / 0.00 held-out; loss
  FLAT** (1.89→1.86). Not under-training — a *generic* relational model does not *discover* the consistency
  detection. **Relevance is the wall, not perception.**
- **v2 — explicit pairwise CONSISTENCY (2.6K params): POSITIVE.** Bake the inductive bias in. For each feature
  *j*, over all object PAIRS, compute the consistency statistics — `[same-value_j, same-outcome, and their
  cross-terms]` — and a *shared, feature-index-agnostic* scorer maps those stats → a score; feature = argmax.
  Because the scorer sees only *(value-agreement, outcome-agreement)* statistics, it transfers to features it
  never trained on **by construction.** Result: EFFECT 0.87/0.99; **FEATURE 0.64 train / 0.50 held-out** (5×
  random), transferring to combos it never trained.

The arc is the headline: pixel-CNN can't (0.00) → generic relational transformer can't (0.00) →
**consistency-structured cracks it (0.50 held-out), at 2.6K params.** The 2.6K is the point — *the lever is
relational STRUCTURE, not scale*. The recognition/relevance wall that had recurred since the v22–v25 LLM-selector
yields not to a bigger model but to the *right relational bias*. Keep perception + structure as architecture,
learn only the relevance.

**Sim-to-real transfer, and that it needed no extra data.** The v2 recognizer, trained on synthetic blobs
*only*, was run on real ARC tasks the grammar can solve (24 such on ARC-1). It proposed the correct causal
feature in its top-3 for **24/24 (1.00)**; the propose→induce→exact-verify pipeline solved **23/24 (0.96)** with a
tiny candidate set instead of all 81 types × decomps. The relevance signal is **domain-general**: "objects sharing
a feature-value share an outcome" reads off real ARC demos exactly as off synthetic, because the consistency
structure is *abstract*, not the visual statistics. Later the last box gate confirmed this is
**distribution-robust** across three distributions (synthetic-trained, no fine-tuning): arc1-train feat-top3 1.00 /
pipeline 1.00; arc1-eval 1.00/0.80; **harder arc2-train 0.96/0.92; human-designed ConceptARC 1.00/1.00** —
aggregate over 54 grammar-solvable real tasks ≈ 0.98 / 0.94. The consistency inductive bias is a strong scaling
signature.

---

## 4. The honest convergence: expressiveness is the bottleneck

With the recognizer proven, the chapter ran a sequence of *value-tests* — and the most important results were the
negatives, which together produced the chapter's clearest finding.

- **The value-test negative.** A recognizer-driven composed solver on ARC-1 eval solved 4 (all beyond gen2_base
  retrieval) but **0 beyond gen6_base** (the families); blind enumeration of the *same* composed space at the
  *same* 105-call budget solved 5 — *more*. On a small, enumerable, exactly-verifiable space the learner is
  **redundant**: blind enumerate+verify wins. The learner pays only when the grammar is *simultaneously* richer
  than the families (→ new coverage) **and** too big to enumerate under budget (→ navigation matters). The current
  grammar was neither — the test was in the wrong regime, and saying so honestly is what kept the project pointed
  at the real lever.
- **The treadmill.** Hand-authoring a new effect class yields ~1 (or 0) new beyond-gen6 each. The first
  `beyond_gen6` solve in the whole chapter came from a *new* shape-effect (`tile-by-color-count`, task a59b95c0) —
  proving the expressiveness lever is real in exactly the family-miss region the analysis predicted. But the very
  next, richer construction effect (per-color-count histograms, 6 layout variants) caught **0** — because each real
  construction task needs a *bespoke* layout. The gen-6 diagnostic, recurring: hand-adding effects is a treadmill.
- **Anti-unification: mechanism proven, expressiveness not.** Least-general-generalization of two verified
  relations lifts the per-task TABLE to a typed hole, a generative *schema*; re-instantiation routes that hole to
  one `induce()` that re-fits + exact-verifies. On real ARC-1-train recolor/select buckets, leave-one-out schema
  **re-instantiation succeeded 43/43 (1.00)** — LGG schemas *are* genuine per-task generators, *minted* from two
  solved tasks, not hand-authored. The NAVIGATE→GENERATE step works mechanically. But the efficiency cash-out was
  only ~2–6× (under the 10× bar), and the deeper cash-out exposed *why*: solutions are found in ~2–4 induce-calls,
  far under budget — **search-size is never the binding constraint.** Anti-unification earns *compositions*, not
  *primitives* (a relationally-blind core can never re-discover containment from examples) — proven as mechanism,
  not as expressiveness.
- **Navigation scales — but it isn't the bottleneck.** A separate stress test caught that the cash-out never
  *loaded* the navigator (only 5 expressible tasks, all found instantly). Under genuine load (synthetic recolor +
  injected distractor features) navigation efficiency is real and *scalable*: speedup **grows 4×→16×** as the
  space grows 350×, top-K recall stays ~flat (space-invariant), and with more evidence (demos) + an evidence-aware
  recognizer it reaches **37× speedup at 6 demos**. So navigation is neither fragile nor the bottleneck — it is a
  robustly-scalable component validated *for the scale regime*.

**The convergence.** Every navigation/transfer mechanism is proven — recognizer transfer ✓, navigation scales ✓,
composition ✓, anti-unification mechanism ✓ — and every one optimizes a *non-bottleneck*. The clearest single
datum: ARC-2-eval is **0/120 grammar-expressible** (no substrate at all), a stark re-confirmation that the
unsolved tasks are **inexpressible, not slowly-searched.** **The bottleneck is EXPRESSIVENESS/COVERAGE, not
search or transfer.** The only untested expressiveness lever is SCALE — a learned generative proposer over a rich
relation distribution. *The box is done.*

---

## 5. The thin-core architecture: faculties + an earned vocabulary

The scale decision came with a fork over *how* to get expressiveness. The available shortcut was BARC ARC-Heavy
(~274k tasks, with python programs and concept tags). The decision — **thin-core, EARN the vocabulary** — used
BARC for its **scale** (a training/self-distillation distribution) and its **map** (the concept tags = a target
vocabulary), but pointedly **not its code**. BARC's programs are imperative numpy in BARC's own hand-designed DSL;
cloning them = adopting someone else's vocabulary = the treadmill moved up a level, and a lookup-Goodhart that
would quietly violate the earns-not-memorizes bar. Instead: a thin principled combinator *core* + **grow** the
vocabulary from the system's own verified solves, with the invention gate as the guardrail.

This is the project's central architectural idea, and its governing metaphor is the **relational eye**.

**The relational eye (`rel_dsl.py`).** The fixed core is exactly the proven recognizer's level — object
*decomposition* + per-object *features* + *compose*. That is *grounding/routing, not vocabulary.* The relational
layer above it is **earned, not hand-coded.** The core exposes a structural *faculty*: `pair_signature(a,b)` — the
raw relational perceptions between two objects (a-contains-b, b-contains-a, adjacent, aligned, relative-size,
same-color) — which **names nothing**. A *specific* predicate (containment, adjacency, …) is *one
(channel, value, mode) instantiation* the system EARNS by search + verify, never a construct we give. The
rationale is precise: anti-unification earns *compositions*, not *primitives*, so a relationally-blind core could
never re-discover containment — it cannot conjure a sense from examples. So **we install the eye (the grounding);
it names the colors (the vocabulary).** The validation:

- **Subsumption 3/3** — every grammar recolor-solve transcribes to a *core-only* DSL program; the thin core
  *contains the whole proven grammar.*
- **Earn-the-predicate** — a *randomized* containment task (identical objects, random positions → **no per-object
  feature correlates**, grammar winning relations = **0**) is SOLVED because the system *discovers*
  `quant:exists(b_contains_a=1)` by searching the faculty's instantiations + verifying. It earned "contained" from
  the relational eye.

The DSL is a strict *superset* of the grammar; the expressiveness ceiling is now **the closure of the grown
library, not our design** — *"the second the expressiveness ceiling stops being our imagination."*

**Library growth (`grow_library.py`).** The real test of that claim is whether the ceiling *climbs* by composing
*earned* predicates. The composed task "recolor the object contained in the LARGEST container" — grammar = 0, and
*no single level-0 predicate solves it* — is solved by
`compose:exists(b_contains_a & forall(a_bigger))`: the system **built** "contained-in-the-largest" by composing
two earned predicates (containment ∘ is-largest), nothing hand-coded, and the minted composition is a *reusable
named* library entry. The ceiling climbs *past* the faculty's single instantiations. The thin-core bet's payoff,
demonstrated.

**The recognizer routes relations too — "the third sibling was a mirage."** A natural worry was that relations
needed a separate recognizer. They don't. A relational predicate's per-object value is *just another feature*, and
V2's consistency head is feature-count-agnostic and generalizes to unseen features by construction. Fed the 14
regular + 14 relational features (trained on regular features *only*), V2 ranks the true predicate
`exists(b_contains_a)` **top-1 1.00** on containment tasks, above every regular feature 40/40, **no retraining.**
The recognizer is just V2 with a wider eye. (The hardened V2H also backports a *support-count* evidence stat — the
fix for the spurious-consistency tail under many distractors.) **Net: one hardened, evidence-aware consistency
core, fed by two feature sources (object-features + relational faculty), routed by V2, with a structural twin
V3-GEO (3.1K params, held-out pre-op top-1 0.69 ≈ trained — "two organs, one principle") proposing the geometric
prefix. The recognizer routes features AND relations; the policy only needs to learn the COMPOSITIONS.**

**The motor hand (`effect_faculty.py`).** Symmetric to the eye, this opens the effect frontier. The hand has
**three atomic motor primitives — paint / place / erase — and nothing else is hand-coded.** An effect is a
composition whose *parameters* (which target, which displacement) are computed by the relational eye: an
eye-selected anchor tells the hand *where* to act. The smoke: an align-to-anchor MOVE task (recolor grammar = 0,
because recolor *cannot move objects*) is solved by the system EARNING `move(toward exists(a_bigger), align_col)`
— the eye picks the anchor (the largest), the hand moves each object to its column; and from the same eye-grounded
search it earns `copy` (place *without* erase) on a copy task. Gestures join predicates as **earned vocabulary.**

The architecture is now **PERCEIVE (the pair-comparison eye) → ROUTE (V2) → ACT (the paint/place/erase hand)**:
thin innate faculties, fully-earned vocabulary (predicates *and* gestures), verifier-gated, library-growing by
composition.

---

## 6. Self-evolving faculties: open-ended at every level

The first hand-picked gesture exposed the same treadmill one level up. Wired into the unified loop and re-measured
on real data, the BARC re-measure was recolor **31** + move **0**: the *one* hand-picked gesture (align-to-anchor)
matched no common BARC motion. Hand-picking gestures one-by-one is a guessing game. The answer is not to guess
better — it is to make the **faculties themselves self-evolving**, applying the thin-core move *one level deeper.*

**The self-evolving eye (`substrate_eye.py`).** The eye's channels (containment, adjacency, …) were themselves
hand-coded. Make them earned too: the *only* innate perception is the **raw object substrate** — each object's
`(r0, c0, r1, c1, h, w, size, color)` — plus general **comparison operators** between two objects (`a.p OP b.q`).
A "sense" is then an earned composition of comparisons, found by search + verify, exactly like a predicate. Two
gates: **(1)** a share-height task where the *shared height varies per demo* (so no per-object feature
generalizes, grammar = 0, hand-coded channels None) is solved because the eye **evolves `exists(a.h == b.h)`** — a
sense it was never given; **(2)** the hand-coded `contained` channel *equals* the substrate conjunction
`a.r0>=b.r0 & a.c0>=b.c0 & a.r1<=b.r1 & a.c1<=b.c1` (True) — proving the fixed channels were merely *ignition*;
the substrate **spans** them.

**The self-evolving hand (`effect_faculty.py`).** The paint/erase basis already reaches every grid, so the hand
is *complete* — gestures are earnable compositions, no missing primitive. From the same eye-grounded search over
the complete basis (erase? + place), the system earns `move` on a move task and `copy` on a copy task.

So the **only innate thing is object-decomposition + raw object descriptions.** Senses, predicates, gestures,
compositions are *all earned*. **Open-ended at every level** — this is what the name *Vine* means.

**The open-ended loop, and that it pays on real data (`open_loop.py`).** V2 is fed the full open sense vocabulary
— regular features + fixed relational channels + 80 substrate channels — and ranks them all (feature-agnostic,
proven to scale). On synthetic probes all four faculties route correctly (containment / share-height-via-substrate /
align-via-move / copy-via-copy, 10/10 each). On a BARC re-measure (295 tasks): recolor/compose **27 — of which 10
via a substrate sense the fixed vocabulary couldn't say** (new expressiveness the self-evolving eye adds on real
data); gestures **0** (the displacement vocabulary, still align-only, needs broadening — measure-then-grow). The
self-evolving *eye* pays; the *hand's* gesture vocabulary is the next coverage frontier. This re-measure is also
*cold* — frozen recognizer, no persistent library, every task from scratch — so it **understates** the persistent-
experience ceiling.

---

## 7. Experience and the RL endgame

Alex's flag — *does it accumulate across tasks, or start from scratch?* — names the last gap and the endgame.

**Layer 1 — the persistent library (`persist_library.py`, DONE).** Every earned concept (feature / relational /
substrate sense, composition, motor gesture) serializes to disk and back; a growing, deduped library survives
across runs and sessions (demo: containment + a substrate-sense + move + copy accumulated across a mixed stream,
persisted, reloaded clean). Experience now *persists* instead of resetting.

**Layer 2a — compounding by reuse, and an honest negative (`expert_iter.py`).** Route the accumulated library
*first* (cheap reuse), fall back to open search, grow the library from new solves. The contained-in-largest
compounding demo *did not fire*: `open_solve`'s composition-inner is restricted to top-K, which drops `is-largest`,
so the composed task returned None and nothing accumulated. The mechanism (reuse earned concepts) is sound, but
**pure retrieval is brittle** — a fixed heuristic misses the right composition. This is exactly the motivating
negative: **Vine needs to *reason* about its experiences, not merely retrieve them.** Retrieval ≠ reasoning.

**Layer 2 — the RL / expert-iteration loop (the frontier).** The endgame is a **learned exploring policy that
reasons over accumulated experience**, with the **exact verifier + the invention gate** as the reward. The honest
design constraints are inherited from the synthetic line and from gen-3:

- **Reward-weighted self-distillation (STaR-style), not temporal RL.** The reward is immediate, binary, exact —
  there is no policy-over-time or credit-assignment-over-trajectories. So the literal DiscoRL/disco-torch net is
  *not* wired in: it is a backward-in-time credit-assignment rule over RL trajectories — both its interface and
  its learned inductive bias mismatch. Disco-as-inspiration only (meta-learn across a *distribution* of families
  for cross-family generalization).
- **Never success-alone.** Naive verified-success-only RL *specializes away* creativity (s15: cycles specialized
  the reactive net off the novel transfer). The reward must be **success + coverage/novelty + out-of-home
  repurposing** (the openness terms), with repurposing credited as a disposition (tag each relation's curriculum
  home).
- **Box-feasible.** V2 is ~2.6K params, so the full experience loop (persist → accumulate → fine-tune on verified
  solves) runs on CPU; scale (Kaggle/Colab, free-compute only) is for *volume*, not for a different mechanism.
- **The yank-the-verifier endgame.** The AlphaGo analogy: use the verifier as the self-play reward to RL a
  policy+value that *internalizes its judgment* (taste), progressively lowering search-reliance. But — the honest
  constraint — **never fully yank**: open-ended creativity is OOD *by design* (unlike Go's fixed game), so a
  frontier always needs search + verify (keep the progress-gated fast/slow controller from the synthetic line:
  reactive-fast on the known, search-slow on the novel). The prize is *taste for domains where verification is
  costly or absent.*

The Kaggle-1 handoff is turnkey and falsifier-gated: scale the loop over BARC-stream / RE-ARC / ConceptARC, grow
the EFFECT leaves, cold-start the learned suffix policy from the box's recognizer-guided solves, **strict
leave-one-FAMILY-out**, **GO iff held-out-family beyond_gen6 ≥ 3** on one ≤8 h T4×2 session. The real-data scope is
quantified one last time: a BARC chunk (394 tasks, on the box, no GPU) solves the ~8% recolor-expressible slice
flawlessly (all recognizer cost 1) and **0 invented** — confirming **EFFECTS are the frontier** at real-data scale.
Growing the effect leaves (from BARC concept-tags + verified solves) is precisely the scale phase's job.

---

## 8. Significance, limits, and the falsifiable bet

**The north star.** Vine is a general architecture for **verifier-grounded open-ended creativity**: drop in a
*perception substrate* + an *action basis* + a *verifier*, and it **earns its own concepts** — predicates,
senses, gestures, and their compositions — open-ended at every level, with experience accumulating in a persistent
library and a policy that learns to reason over it. ARC is the first instance, not the point; the same shape
applies wherever there is a substrate, an action basis, and a cheap check: code, math, scientific hypotheses,
engineering design, robotics.

**What is proven (on a 7 GB CPU box).**
- Recognition/relevance is learnable *and compositionally generalizable* with the right inductive bias, at **2.6K
  params** — the lever is relational *structure*, not scale.
- That recognizer transfers sim-to-real with no fine-tuning, and is **distribution-robust** across ARC-1, harder
  ARC-2, and human-designed ConceptARC (feat-top3 ≈ 0.98, pipeline ≈ 0.94 over 54 tasks).
- Navigation efficiency scales (16–37×, space-invariant top-K); anti-unification re-instantiates per-task **1.00**;
  geometry transfers 6/6 to real reshaping tasks. Every navigation/transfer component is validated.
- The thin core **subsumes** the whole hand-built grammar (3/3) and **earns** vocabulary beyond it — containment,
  share-height, move, copy, and *composed* predicates like contained-in-largest — each with grammar = 0 and an
  exact-verify generalization check, certified across diverse families with **no per-family training** (singles at
  recognizer cost 1, ~18× cheaper than blind).
- Creativity is **inference-time search**, triangulated three ways against weights (params s11, distillation s12,
  cycles s15); the deployable form is a progress-gated fast/slow controller.

**What is a bet (and must be guarded).**
- **ARC-2 is unproven.** ARC-2-eval is currently 0/120 grammar-expressible; the whole scale phase is the wager
  that a learned generative proposer + earned effect-leaves crosses the *expressiveness* wall there. ARC-2 is the
  *falsifiable first proof* — designed to crush ARC-1 winners — and "smoking" it non-LLM is a real moonshot, not
  an instant SOTA.
- **The breadth wall may be ARC-intrinsic.** The load-bearing risk is that the v19/v22-25 breadth wall is a
  property of ARC, not of the absence of a recognizer. The recognizer-as-breadth-router is the wager; if it holds,
  the suffix generator crosses the expressibility wall; if not, that is the finding.
- **EARNS-not-MEMORIZES is the property to defend with our lives.** The single most load-bearing claim is also the
  easiest to fake. Drinking BARC's code would have been a lookup-Goodhart; the invention gate (held-out-family,
  grammar = 0, generalizes) is the guardrail, and it must be guarded *under scale*, where the pressure to memorize
  is strongest. A solve that the gate cannot certify as invented does not count, however much it raises the
  solve-rate.

**Why it matters if it works.** Vine would vindicate a **small-models / earned-not-scaled** paradigm: a tiny
*earning* core out-reasoning giant *memorizing* models on genuine novelty — configuration over resources, the
airfoil thesis, instantiated. And it would be a uniquely *honest* artifact: tiny, CPU-runnable, with **every
earned concept and the verifier auditable**, so the creativity claim is *checkable by strangers, not taken on
trust.* If it succeeds entirely, it ships open-source on HF (@Asystemoffields).

---

## Appendix: milestone commits

- `d26cc89 … 17694a1` — the v0–v19 compression-as-generalization arc; depth-engine proven, real-ARC boundary
  measured (depth vs breadth).
- `d711c1a … 4ed9a67` — the breadth detour (v20–v27): LLM recognition/selection never beats blind search;
  relevance-discrimination localized as the gap → "toss the LLM."
- `4a73692 … 75db896` — the synthetic incubation line: the crossover, learned dissociation, the realism frontier,
  Alex's Law, size-for-time, value-guided search (creativity = inference-time search, triangulated).
- `df5016e` — campaign gen-0→3: coverage 0.5%→9%, **invention-beyond-retrieval = 0** (the standardized-ablation
  truth).
- `40e154e` gen-4 first beyond-retrieval (4); `4308eaa` gen-5 diagnostic (**verb-composition dead, fitted
  relations live**); `b40adb8` gen-5 (11); `68026da` gen-6 consolidated **48/400, 14 beyond-retrieval**.
- `f3fcbb5 … 9e6eeaa` — learner v0/v1 honest negatives → **v2 consistency head cracks relevance (2.6K params,
  0.50 held-out)**.
- `5846449` sim-to-real (top-3 1.00, 23/24); `6216079` composition; `6ec1959` value-test negative; `ab9d95e`
  first beyond_gen6 (a59b95c0); `4e0b544` the treadmill; `f7cf3fa` anti-unification 1.00; `3dcb7f7` **the
  expressiveness convergence**; `83c1b28`/`8c21792` navigation scales; `29bdf02` distribution-robust — **box done.**
- `8049ea2` V3-GEO + geometry coverage; `042fe34` **the relational eye** (earn the predicate); `f49c713` **library
  growth** (compose earned predicates); `3e0e465` "third sibling was a mirage"; `0bceb06` policy + invention gate;
  `c8c1c86` expert-iteration loop + Kaggle handoff.
- `a7fda0a` **the motor hand**; `814571b` move-delta-0 treadmill warning; `e4c0801` **self-evolving faculties**
  (Vine); `a1a0c92` open-ended loop (substrate eye pays, 10/27); `a19efbd` persistent library; `7326ab8`
  compounding negative → *reason, don't retrieve* (the RL endgame).
