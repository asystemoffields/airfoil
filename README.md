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

## What's here (v0 → v27, then the incubation engine)

### The induction line (v0 → v27)

Built so each experiment answers the previous one's objection:

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
- **Out of the sandbox, and the loop (v11–v18).** Antiunification reuses a *pattern*
  (a schema with a hole) where exact reuse is blind; v12 **measures** generalization
  for real (a *discovered* library solves 100% of novel held-out tasks, base 35%);
  the wake-sleep loop **bootstraps** competence on a frozen base (v13) but only if
  the sleep **suppresses** junk (v14) and there's deeper structure to climb (v15);
  and deeper abstraction is **necessary, not just cheaper**, the deeper a task is
  relative to the search budget (v17), with a learned policy guiding the search (v18).
- **The ARC boundary, mapped on both sides (v19–v27).** Pointed at *real ARC* — a
  benchmark we didn't design — the depth loop's payoff **vanishes**: ARC is hard
  along *breadth* (many distinct shallow concepts), and the airfoil loop is a
  *depth* tool. v22–v27 then exhaust the "bolt a recognizer/selector/code-proposer
  around a frozen small LLM" levers — each bounces off the model's **intrinsic
  induction ceiling** (the scaffold is a real broad search-driven amplifier, but
  the LLM's *naming/selection/teaching* never beats blind search). The honest
  conclusion that opened the next chapter: the remaining lever is **architecture**,
  not scaffolding.

### The incubation engine (`incubation/`, steps 0 → 17)

A propose → aim → verify **creative controller**, built and measured as a clean
mechanism in a controllable synthetic world (then grounded on ARC). Its headline:

> **Creativity = inference-time search over a frozen *causal* world-model + a
> structure-general value, irreducible to reactive weights.**

The world-model is deliberately **task-agnostic** (trained on *interventions* for
full causal effects — it *keeps* the structure a reward-centric model would
discard, which is exactly what *repurposing* a known operator needs). A
structure-general value-to-go aims the search; an exact verifier (act in the world,
re-plan) grounds it. The session's spine, triangulated **three** independent ways —
scaling parameters (step 11), distilling the search into a reactive net (step 12),
and training cycles (step 15) all **fail** to put creativity into weights — is that
the creative transfer lever is *test-time search compute*, not model size. It is
**depth-scaled** (holds to depth-4, step 16) and packaged as a deployable
**Hybrid-C**: one progress-gated controller, a distilled *fast* path for the routine
plus *gated search* for the novel, that spends search compute *exactly* where the
fast net is weak (step 17). Full journal: **[incubation/NOTES.md](incubation/NOTES.md)**.

### The active line (this chapter): a functionally-creative non-LLM ARC solver

The newest work makes the non-LLM creative engine *actually* creative and smokes it
on **ARC-AGI** (ARC-AGI-1 first, then ARC-AGI-2). The method is **DIY-AlphaEvolve**:
agents are the *design-time* variation operator evolving a whole non-LLM solver
(the deployed artifact = search + small trained nets, **no LLM at solve time**).
Creativity is defined operationally and falsifiably — an *unrestricted* grasp of
cause-and-effect (induce the invariant mechanism; cross-train-pair invariance
licenses it; the exact verifier is the held-out intervention) **plus** real-time
*invention* of that mechanism from experience (compose + abstract new rules, not
retrieve from a menu) — and the selection **gate** measures *invented-and-
generalizing* mechanisms (they must survive an invention-OFF ablation), not mere
coverage. Early measured state: an ARC-AGI-1 seed baseline of 24/400 train, 2/400
held-out eval (failures ~93–99% an *expressiveness* ceiling, not search); a
gen-1 of evolved operators roughly doubled the dev solve-rate and **generalizes**
(28/400 held-out eval, 14× the seed) — but honest attribution shows the gains are
generalizing *competence*, while the *creative* mechanisms contributed ~0. The next
departure is genuinely different: **active causal discovery** (intervention +
invariance + non-directed exploration) toward a *learned* grid causal world-model.

Full writeup: **[RESULTS.md](RESULTS.md)**. Chronological notes: **[LOG.md](LOG.md)**.
Roadmap: **[PLAN.md](PLAN.md)**.

## Reproduce

```bash
python3 induct.py        # v0
python3 induct_v1.py     # … the induction line, through induct_v27.py
                         #   (v19/v20/v22+ need the ARC clone at /data/arc)
```

The induction line is pure Python standard library; each script prints its own
table and an honest verdict (including its own caveats). The **incubation** engine
(`incubation/`) uses torch 2.x CPU (`/data/llm/.venv`); the ARC grounding and the
active solver campaign read ARC-AGI-1 at `/data/arc` and ARC-AGI-2 at
`/data/arc-agi-2`. See **[INVENTORY.md](INVENTORY.md)** for every artifact.

## Status

The induction line is a set of **mechanism demonstrations**, not benchmark results
— and they say so; their honest scope (a *depth* tool, not a *breadth* tool) is
mapped on real ARC. The **incubation** engine establishes, on a controllable
substrate, that creative transfer is *inference-time search over a causal
world-model*, irreducible to weights — packaged as the deployable Hybrid-C. The
**active line** is now testing whether a genuinely *creative*, evolved, non-LLM
solver can move ARC-AGI. See PLAN.md.
