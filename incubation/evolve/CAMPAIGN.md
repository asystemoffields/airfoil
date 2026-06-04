# Creativity Campaign — a functionally-creative non-LLM ARC solver

**Objective.** Make the non-LLM creative engine *actually* creative (not a gesture) and move ARC-AGI
(ARC-AGI-1 first, then ARC-AGI-2). Method = **DIY-AlphaEvolve**: agents are the *design-time* variation
operator evolving a whole non-LLM solver; the deployed artifact = search + small trained nets, **no LLM at
solve time**.

**Operational, falsifiable definition of creativity (the gate).** A solve counts as creative only if the
*system* grasps/invents an **invariant cause→effect mechanism** that **generalizes** to held-out tasks and
that a **strong retrieval baseline cannot produce**. Coverage that generalizes is good *material* but is not
creativity. Honest metric: **invented-and-generalizing beyond the best retrieval solver, on held-out eval.**

## Results (ARC-AGI-1; dev = arc1-train, held-out = arc1-eval, 400 each)

- **gen-0 (seed)** — best-first grid-distance search over the 32-op DSL. **24/400 train, 2/400 eval (0.5%).**
  Failure taxonomy: ~93–99% "bucket A" = the rule isn't *expressible* in the DSL (breadth/expressiveness
  ceiling, not search; confirmed by exhaustive enumeration on small-space unsolved tasks).

- **gen-1 (6 evolved operators)** — per-task **parametric concept-fitting** (color_perm, symmetric_tiling,
  fractal, panel_logic, object_recolor, …). Roughly doubled dev and **generalizes**: best (param-struct)
  **28/400 eval (7%, 14× seed)**. Honest attribution: gains are generalizing *competence*; the creative
  mechanisms (in-session experience reuse, novel multi-concept linking) contributed **~0**.

- **gen-2 (consolidate + force creativity)** — merged gen-1 winners → **gen2_base 45/200 dev, 34/400 eval**;
  mutators pushed novel-linking + transferable-experience on the hard relational families with creativity
  metrics in the gate. Coverage rose to **37/400 eval (9.25%)**, but certified creative mechanisms stayed
  **marginal** (novel-links 1–2/candidate, experience-transfer **0** everywhere). The retrieval/coverage
  paradigm is near its creative ceiling.

- **gen-3 (generative mechanism-INVENTOR — paradigm branch)** — built reusable infra:
  `mechanism_curriculum.py` (alphabet = 33 ops in 6 relation-kinds, sentence-grammar = 5 combinators, a tiny
  2-head MLP prior trained on a self-generated curriculum) and `invention_gate.py`
  (`INVENTED = solved − solve_ablated`; positive/negative controls validated). Three inventors:
  compositional-synthesis, analogical-repurposing, **causal-decomposition** (decompose the output into
  per-cell / per-object / per-region parts, induce local cause→effect rules from cross-pair invariance,
  compose, exact-verify). Graded against **each inventor's own** ablation, causal-decomposition looked
  strong: **9 certified-invented, generalizing solves on held-out eval**.

## THE KEY FINDING — the standardized-ablation truth

"Invented" is only meaningful against a **standardized, strong** ablation. Re-graded against
**gen2_base (the best retrieval solver)** as the shared ablation, on held-out eval(400):

```
gen2_base solves:                 34
causal-decomp solves:             11   (a strict SUBSET of base → beyond base = 0)
compositional beyond base:         0
analogical   beyond base:          0
union(base, ALL 3 inventors):     34   → invention adds 0 beyond best retrieval
```

**Invention-beyond-retrieval = 0.** Every "invented" solve was already reachable by gen2_base's rich
parametric menu; the inventors reached a *subset* of the same tasks by a different (compositional) route.
So across gen-1/2/3 we climbed **coverage 0.5% → ~9%** on held-out, but **certified creativity beyond strong
retrieval is still 0.** This is the honest state — a clarifying negative, and the reason the standardized
ablation was non-negotiable: graded on its own curve, gen-3 would have banked a fake "9."

## gen-4 reframe — the only honest target

Solve a held-out task that **gen2_base cannot** (invented-beyond-retrieval > 0). That means attacking the
~366/400 held-out tasks the parametric menu can't express — the multi-step relational families — with genuine
mechanism *invention*, not a richer menu:
- **standardized gate = gen2_base** (count only solves it misses), gen2_base kept as the retrieval backstop
  (attempt 1), invention as attempt 2;
- **deep OPEN compositional invention** — multi-step mechanisms no template contains;
- **active causal discovery** — *intervention* (counterfactual probes to disambiguate among train-consistent
  mechanisms → attacks overfit / ARC-2's adversarial design), *invariance* (the mechanism stable across train
  pairs), *non-directed/coverage* exploration (find repurposings greedy search misses);
- **openness** — coverage-trained, unpruned, repurposing-credited.

If gen-4 still adds 0 beyond retrieval, that's the finding (the open-compositional frontier likely needs a
learned proposer and/or the RL/expert-iteration phase). Verifier merciless; proposer broad.

## gen-4 — FIRST invention beyond retrieval (modest, real, verified)

Standardized gate: each candidate's `solve_ablated` == `gen2_base.solve` (the strong retrieval ablation);
headline = `eval_beyond_base`. Mapped gen2_base's 318 train / 366 eval misses into families (biggest:
counting/construction 86, line/ray-draw-connect 77, object-to-marker-copy 57, relational-recolor 46). Levers:

| lever | train_solved | train>base | eval_solved | **eval>base** |
|---|---|---|---|---|
| relational-depth        | 89 | 7 | 36 | **2** (0a2355a6, 37d3e8b2) |
| non-directed-coverage   | 89 | 7 | 36 | **2** (21f83797, d282b262) |
| active-causal-discovery | 87 | 5 | 34 | 0 |

**First nonzero beyond-retrieval in the campaign**, independently re-verified: 4 DISTINCT held-out tasks
beyond `gen2_base` → combined ceiling **34 + 4 = 38/400 eval (9.5%)** with 4 certified-creative solves.

**The sharpened honest finding: fitting generalizes, composition overfits.** The 4 held-out wins came from
richer relational *fitting* (e.g. recolor-by-hole-count/border/aspect — a relation base's size-only menu
can't express). The open multi-step *composition* (value-guided / coverage beam) found beyond-base solves on
TRAIN (5–7 per lever) but **~0 transferred** to held-out; active-causal-discovery's intervention/invariance
cut train overfit (5 beyond-base) yet none generalized. So hand-built compositional search *invents on train
but doesn't transfer* — the obstacle is the **generalization** of composition, not its discovery.

## gen-5 — make composition GENERALIZE (the experience / openness lever)

Hand-built search has given what it has (a trickle of relational fitters beyond retrieval). The
open-composition path holds the high ceiling but doesn't transfer. Per the project thesis — and the bio anchor
(openness to experience) — the lever is a **learned proposer trained on the curriculum** so compositional
invention becomes *findable and transferable*: trained for COVERAGE not precision (the verifier supplies
precision), library unpruned, repurposing credited. gen-5 tests this **cheaply and decisively first** — does a
small CPU-trained proposer make even one composition family generalize beyond base? — before any heavy
training; scale to Kaggle only if it earns the spend.

## gen-5 DIAGNOSTIC — composition's non-transfer is EXPRESSIVENESS, not selection (decisive)

Before building the learned proposer, ran the gating diagnostic: SELECTION (a prior could fix) vs
EXPRESSIVENESS (it can't)? **Answer: EXPRESSIVENESS, high confidence.** Of the 366 held-out eval tasks
gen2_base misses, the number with even ONE train-consistent program at ANY depth — exhaustive at depth 1–2,
3× budget (120k execs) depth-4 spot-check — in gen4_01's relational alphabet = **0**. The atomic verbs don't
*engage* these tasks; there is nothing to compose, so nothing for a prior to select. A learned proposer over
compositions is aimed at a bottleneck that does not exist here.

**This CORRECTS the gen-4 read.** On *train*, composition is the OPPOSITE of overfit — 3 of gen4_01's 4
composed solves generalize, and where train-consistent compositions exist they generalize **100%** (tight
geometry equivalence classes like `transpose→ray_diag→transpose`). Composition isn't unreliable; it just
almost never *fits* the held-out families. **Every** real eval-beyond-base win came from per-task **fitted
cause→effect relations** (region_recolor by hole-count/border/aspect, …), never from verb composition. So on
real ARC's frontier the "compose a fixed alphabet + search + verify" depth-engine **doesn't engage** — ARC's
mechanisms are rich per-task RELATIONS, not compositions of these verbs.

**Pivoted gen-5 (the genuine, evidence-backed lever): LEARN THE RELATIONAL INDUCERS, don't compose verbs.**
A curriculum-trained model that, from a task's train pairs, INDUCES a cause→effect relation *beyond* the
hand-coded families — structure (decompose → induce → verify) kept as architecture, the relation CONTENT
learned/generated. This is "make the inducers learn without robbing them," now the evidence-backed frontier.
Standardized gate unchanged (beyond gen2_base, held-out). If a small CPU model can't make generated relations
generalize, that is the decisive case for the RL/expert-iteration phase (which can invent primitives) — a
scale decision (Kaggle/gradient training) to weigh with Alex on his return.

## gen-5 — relation-induction WORKS: 11 certified beyond-retrieval (combined), 45/400 held-out

The diagnostic-corrected lever — systematic rich per-task RELATION induction (decompose → rich feature vector
per part → induce feature→effect invariant across all train pairs → exact-verify), standardized gate (beyond
gen2_base) — pays off across all three granularities:

| inducer | eval_solved | **eval_beyond_base** | notable new held-out wins |
|---|---|---|---|
| object-relational  | 39 | **5** | 358ba94e, 9a4bb226, cd3c21df (select-extreme over a uniqueness feature) |
| cell-region        | 38 | **4** | aa18de87, e0fb7511 |
| two-part-relational| 37 | **3** | 281123b4, 6a11f6da |

**Combined, independently verified: 11 DISTINCT beyond-base held-out tasks** (gen-4's 4 ∪ gen-5's) →
**45/400 eval = 11.25%**, all 11 certified beyond strong retrieval (seed was 2/400 = 0.5%). Relation-induction
is the campaign's real engine: it adds *genuine creativity* (beyond-retrieval), not coverage, and every
granularity contributes. The new `select_extreme`-over-uniqueness effect is a clean example — a cause→effect
relation gen2_base's size-only menu can't express, and it generalizes.

**Honest framing.** The wins still come from *hand-authored relation families* (richer ones — hole-count,
uniqueness/extremeness, connect-with-fitted-fill — fitted per task). The direction has clear headroom: the two
biggest miss-families, **counting/construction (86)** and **line/draw-connect (77)** — roughly half the misses
— are still largely untouched. So enriching it is genuine beyond-retrieval progress, not coverage-padding. But
the leap to the *system itself generating* novel relation families (rather than agents authoring them) remains
the learned-generator / RL phase — the scale decision for Alex. gen-6 sweeps the two biggest families; then the
loop rests at a consolidated milestone with that fork clearly stated.

### Campaign arc at a glance (ARC-AGI-1 held-out eval, 400)

| gen | held-out solved | beyond strong retrieval | lever |
|---|---|---|---|
| seed | 2 (0.5%) | — | DSL search |
| gen-1 | 28 (7%) | 0 | parametric concept-fitting |
| gen-2 | 37 (9.25%) | 0 | consolidated retrieval + linking/transfer |
| gen-3 | (≤34) | 0 vs gen2_base | generative mechanism-inventor (composition) |
| gen-4 | 38 combined | 4 | first beyond-retrieval (fitted relations) |
| **gen-5** | **45 combined (11.25%)** | **11** | **systematic relation-induction** |
