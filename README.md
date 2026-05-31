# Airfoil

> The airfoil was never a question of materials — wood and cloth were lying around
> in the neolithic. What was missing was the *shape*: the configuration that turns
> already-sufficient substrate into lift by exploiting the medium. Airfoil (the
> project) is a search for that shape in learning systems — the arrangement that
> turns a little structure into a lot of generalization, on hardware you already have.

A small, honest line of experiments testing one idea:

**Compression is generalization.** A system whose objective is the *shortest
reusable description* of its data should generalize — not because generalization
was designed in, but because the only way to be short across many examples is to
find and reuse the structure that recurs. (The practical shadow of Solomonoff /
MDL: the shortest program that fits the data is the optimal predictor of the rest.)

These are deliberately tiny, CPU-only, sub-second experiments with **honest
controls** — the goal is to isolate the *mechanism*, not to chase a benchmark.

## What's here (v0 → v12)

Three arcs, built so each experiment answers the previous one's objection:

- **Compression → generalization (v1–v7).** Abstractions learned from shallow
  training shorten *novel, deeper* compositions (2.28×) and not unrelated work;
  they pay for their own bits; they make *solving* 21× faster; they generalize to
  unseen depths; **Occam ≡ generalization** (compressing *past* reuse is just
  memorization); it's domain-agnostic; and it lifts from linear pipelines to
  higher-order **trees**.
- **The verifier (v8–v10).** The whole loop rests on verification. A 10%-wrong
  verifier *alone* collapses solving to 4%; redundancy rescues it exponentially —
  but only if errors are *independent*, so **diversity, not volume**, is the lever;
  and in a reuse system a verifier error *poisons a reusable skill* and compounds
  with depth, which a diverse-consensus gate contains.
- **Out of the sandbox (v11–v12).** Antiunification reuses a *pattern* (a schema
  with a hole) where exact reuse is blind; and v12 **measures** generalization for
  real — a library *discovered* (not planted) from training solves **100%** of
  novel held-out tasks (base: 35%), with an honest boundary where structure isn't
  shared.

Full writeup: **[RESULTS.md](RESULTS.md)**. Chronological notes: **[LOG.md](LOG.md)**.
Roadmap: **[PLAN.md](PLAN.md)**.

## Reproduce

```bash
python3 induct.py        # v0
python3 induct_v1.py     # … through induct_v12.py
```

Pure Python standard library. Each script prints its own table and an honest
verdict (including its own caveats).

## Status

Mechanism demonstrations, not benchmark results — and they say so. The honest
frontier from here: a real *external* benchmark (ARC), the full iterative
wake-sleep loop, parameterized abstraction wired into the loop, and a learned
proposer. See PLAN.md.
