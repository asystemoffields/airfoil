# VINE — self-handoff (resume here)

*Written 2026-06-04 at a clean checkpoint (commit `8f71dfd`), pre-compaction. Read this first, then `LEARNER.md`
(full journal) + the files named below. `VINE_NARRATIVE.md` is the paper backbone.*

## What Vine is
A **non-LLM ARC-AGI architecture**: **PERCEIVE → ROUTE → ACT** with *thin innate faculties + a fully-EARNED
vocabulary*, exact verifier as reward. Renamed from "airfoil." Repo: `/data/Windows-files/Documents/airfoil`,
code in `incubation/evolve/learner/`.
- PERCEIVE: the relational EYE (`substrate_eye.py`: pair_signature faculty + earned senses) + the **CELL substrate**
  (`cell_eye.py`, NEW); object-decomposition + per-object features (`grammar.py`).
- ROUTE: V2H, a 2.6K consistency recognizer (`train_v2_hardened.py`) — feature-index-agnostic, routes any earned
  sense/gesture for free, distribution-robust sim→real.
- ACT: the motor HAND (`effect_faculty.py`: paint/place/erase complete; gestures EARNED, eye-grounded).
- Vocabulary EARNED by search + the exact verifier; INVENTION GATE (`policy_eval.py`) = grammar-can't-express AND
  generalizes-to-held-out. Persistent live library (`persist_library.py`, `ground_vine_live.py`: retrieve-first +
  mid-solve part-write).

## The decisive finding (settled — do NOT re-derive)
Every downstream organ — routing, retrieval, live library, mid-solve, RL expert-iteration, navigation — is **built,
correct, and IDLE**. The single lever is **EXPRESSIVENESS**: on real ARC Vine solves ~4/400 because its earned
faculties express almost nothing real. (gen6_base's 48/400 is HAND-AUTHORED + ~34 is retrieval; don't conflate.)

## WHERE WE ARE — the expressiveness lever is finally MOVING
The elegant fix (design workflow `w2rlbnowm`): the rich effects were a PERCEPTION gap one level down — the eye saw
objects, not cells. **`cell_eye.py` adds the CELL substrate** (a literal copy of the object eye over cell-coords
under innate index-maps {mirror_h/v, transpose, translate}). The **DUALITY**: a forall-over-cells predicate
localizes its own violations → detection earns completion (violation-set = the edit, map = the source). **GO GATE
PASSED 2/2**: symmetry-repair 12/12 + fill-holes 12/12, both invention-certified (grammar=0, move/copy can't),
earned from ONE substrate with no per-effect code. **BUT this GO is on SHAPED SYNTHETIC, not real ARC.**

## IMMEDIATE NEXT (do this first)
The honest, still-UNPROVEN bet: **does the residual-directed search IGNITE on REAL ARC** (partial symmetry, messy
holes, the right cell-predicate buried among hundreds)? Build order:
1. Wire **V2H routing on cell-predicates** (feed them as features; confirm top-K recall is non-chance — the early
   falsifier for enumeration blowup).
2. **Size-changing handling**: residual = output XOR input only exists same-shape; for tile/scale induce against the
   **eye-computed output shape**.
3. **Assemble `cell_eye` into `ground_vine_live.py`** (the live-library solver) and run the **real-ARC head-to-head**:
   ARC-1 eval(400) with the SAME `beyond_gen6` gate as gen6_base (the paper number) + BARC stream — does symmetry/
   fill/draw move coverage off the recolor-only **4**? (Each new effect should add >0 `beyond_gen6`.)
4. ONLY IF real coverage rises → **Kaggle/Colab scale** (free-tier; `kaggle/colab_run.ipynb`) for the earn-loop
   VOLUME phase; the parked RL/expert-iteration loop (`reward.py`, `held_out_family.py`, `exit_loop.py` — design in
   `LEARNER.md` "RL FALSIFIER") is finally in its genuinely-too-big-to-enumerate regime there.

## Honest open risks
Ignition on real ARC (UNPROVEN — box used shaped synthetic, the make-or-break); the innate index-map set is a small
design choice (treadmill pushed down a level, not eliminated); duality is FALSE for bespoke-layout construction;
residual breaks on size-changing (see step 2).

## Working context (Alex)
- **No pausing** — keep driving to the next result, commit/journal each step; surface only genuine decisions
  (compute-spend) or blockers. Compacting is fine (not a pause).
- **Analogical reasoning** is the maturation vision: "this rhymes with THINGY which needed STUFF → try THINGY-modded
  + STUFF-modded" = recognizer judges "looks-like" + anti-unification schema-mod (param-mod built; structural-mod = next).
- North star: Vine = general **verifier-grounded open-ended creativity** (code/math/science/design/robotics); ARC-2
  the falsifiable first proof; load-bearing property = **EARNS-not-memorizes** (guard the invention gate). **If it
  succeeds entirely → open-source on HF (@asystemoffields).**
- Constraints: **free-compute only** (Kaggle/Colab); **non-LLM at solve time**; **don't wire the literal DiscoRL net**
  (temporal RL rule, mismatched); the RL loop is short-horizon STaR/reward-weighted self-distillation, NOT temporal.
- Retest discipline: Vine vs gen6_base on the SAME eval+gate (not the 48/400 hand-authored figure).

## Map of the key files (in `incubation/evolve/learner/`)
`cell_eye.py` (NEW, the live frontier) · `substrate_eye.py` (self-evolving object eye) · `effect_faculty.py` (hand) ·
`train_v2_hardened.py` (V2H router, `learner_v2h.pt`) · `rel_dsl.py` (relations/composition/verify) ·
`open_loop.py` / `ground_vine.py` / `ground_vine_live.py` (assembled solver + head-to-head) · `policy_eval.py`
(invention gate) · `persist_library.py` (live library) · `lgg.py` (anti-unification) · `LEARNER.md` (FULL journal) ·
`VINE_NARRATIVE.md` (paper backbone) · `kaggle/` (Colab+Kaggle scale lanes).
