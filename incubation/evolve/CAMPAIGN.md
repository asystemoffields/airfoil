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
