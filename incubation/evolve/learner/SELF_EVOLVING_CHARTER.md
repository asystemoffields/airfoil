# Vine Self-Evolving Charter — the anti-treadmill law

*This protects the concept of the cell-substrate fix: effects are EARNED from a generative substrate, never hand-
coded per family. It is binding on all future work (Claude's and Alex's). Enforced by `test_generativity.py`.*

## THE PRINCIPLE (binding)
Vine **earns** its vocabulary. Every sense and every effect must be EARNED from a thin innate substrate by **ONE
search + the exact verifier** — never hand-coded per family. Hand-authoring an effect/sense menu is **THE TREADMILL**:
the proven dead end (~1 task per hand-coded effect; the hand-coded `align` gesture caught 0/295 BARC; the hand-coded
`earn_symmetry/fill/periodic` earned 0/400 real eval — yet the SELF-EVOLVING loop over the same substrate ignited).

## THE ONLY EXTENSION POINT: the generative basis
The substrate is a small set of innate **GENERATORS** — the object/cell primitives `{r,c,color,is_bg,...}`, the index-
map generators `{isometries, translations}`, the comparison operators `{==,<,>,...}`, the quantifiers `{exists,forall,
count}`. The earn loop searches the **CLOSURE** of the generators (their compositions) + applies the duality.
- **ALLOWED — add a GENERATOR** (rare, principled): a primitive the substrate genuinely lacks. A new generator
  expands the searchable closure, so *new effects are earned for free*. (e.g. a new index-map family; a new operator.)
- **FORBIDDEN — add a per-family EARNER** (`earn_symmetry`, `earn_fill`, `earn_periodic`, …): that is the treadmill.
  If you are writing an effect-SPECIFIC function, STOP — express the missing capability as a GENERATOR and let the
  single loop earn it.

## THE SMELL TEST (use it every time you reach to add capability)
> "Am I adding a GENERATOR (a basis primitive whose closure grows) or an EARNER (one effect's recipe)?"

Generators grow the closure; earners are the treadmill. **When in doubt, it's an earner — go find the generator.**
A tell: if your new code names a specific effect ("symmetry", "fill"), it is almost certainly an earner. Generators
are effect-agnostic (a map, an operator) and the effect's NAME only appears as a *discovered* label at solve time.

## THE TEST (compliance — runnable, `test_generativity.py`)
- **POSITIVE (generativity):** the *existing* earn loop must earn HELD-OUT effects — compositions of generators it
  was NOT built for (e.g. glide-reflection = mirror ∘ translate) — with **ZERO new code**. If it cannot, EITHER the
  basis is missing a GENERATOR (add ONE, principled) OR you are about to hand-code (forbidden). It must never be
  "add an earner."
- **NEGATIVE (treadmill detector):** the cell-effect module exposes ONE earn entry point; its map-set is produced by
  `closure(generators)`, NOT a hand-listed per-family menu; the count of effect-specific earner functions stays ~0.

## WHY THIS MATTERS
Vine's entire bet is **earns-not-memorizes** / verifier-grounded open-ended creativity. The treadmill is
memorization-by-developer — it launders hand-authored answers as "capability" and silently kills the thesis.
Protecting the generative substrate IS protecting the thesis. A number that went up because we hand-coded the family
is worth less than zero, because it hides the regression. See [[airfoil-project]] (north star) and `LEARNER.md`
("SELF-EVOLVING CORRECTION").
