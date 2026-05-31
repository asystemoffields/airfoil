# Compression as Generalization — results (v0–v7)

*A small, self-contained line of experiments testing one idea: that a system whose
objective is the **shortest reusable description** of its data will **generalize**,
because the pressure to be short forces it to discover the abstractions that recur.*

All experiments are pure-Python, CPU-only, and run in well under a second each.
The point is not scale — it's to isolate the **mechanism** with honest controls.

---

## The thesis

Capability splits into *knowledge* (large, offloadable) and *reasoning* (a short
program, cheap to reuse). If we make **minimum description length** the objective,
generalization shouldn't need to be designed in — it should fall out, because the
only way to be short across many examples is to find and reuse common structure.
This is the practical shadow of Solomonoff/Kolmogorov: the shortest program that
fits the data is the optimal predictor of the rest. "Within the particular is
contained the universal."

A corollary we wanted to test directly: choosing compression as the objective
doesn't *banish* a learner — it **domesticates** it into a guide for the search
over short programs.

## The method

- **A DSL** of composable operations. Programs are sequences (v0–v6) or
  higher-order **expression trees** (v7).
- **Library learning** = BPE-style greedy abstraction: repeatedly fold the
  most-reused contiguous sub-structure (a substring, or in v7 a **subtree**) into
  a named library primitive, until reuse runs out (frequency < 2).
- **Measurements**, each with **held-out** tasks and **controls**:
  - *description length* — fewest library symbols / tree-nodes to express a program;
  - *two-part MDL in bits* — library cost + data cost;
  - *search cost* — programs enumerated to **solve** a task from examples.
- **Controls** are the spine of the whole thing. A *related* held-out set reuses
  the trained motifs in novel ways; a *disjoint* control shares no structure; a
  *scrambled* control reuses the same primitives in non-motif arrangements. The
  claim is only credible if *related* moves and the controls **don't**.

---

## Results at a glance

| | question | headline | control |
|---|---|---|---|
| **v0** | naive first cut (frequency code) | related 1.27× | control *inflated* 0.50× — code over-specialised; muddy. Motivated a cleaner metric. |
| **v1** | does structure transfer? | **related 2.28×** shorter | both controls **flat (1.00×)** |
| **v2** | is it real (pay for the library)? | train MDL **1.37×** (library pays for itself); related **1.70×** | control **inflates** (bigger codebook taxes unrelated work — honest no-free-lunch) |
| **v3** | does it speed up *solving*, not just describing? | related search **21×** fewer nodes (25,761→1,207) | control ~3× **slower** (branching tax) |
| **v4** | depth generalization + overfitting | **flat 2.25×** at depths 3–6 (trained ≤2) | forcing memorization past generic motifs helps TRAIN 1.71× but novel held-out **unchanged** |
| **v5** | does it survive a real domain? | list-processing pipelines: related **2.28×**, depth-gen flat | controls **flat** — identical to the toy DSL |
| **v6** | the domesticated learner | a learned proposer guides search **another 12.7×** (~**270×** cumulative) | control **0.4×** (guidance mis-prioritizes where no structure exists) |
| **v7** | the tree leap (higher-order) | trees: related **2.03×**, depth-gen ~2.2× | disjoint control **flat (1.00×)**; same-leaf control 1.06× (see caveat) |

---

## The through-line

Read top to bottom, every experiment points the same way, and each answers a
distinct objection to the last:

1. **It transfers (v1).** Abstractions learned from shallow training shorten
   *novel, deeper* compositions they never saw — and do nothing for unrelated work.

2. **It's not an accounting trick (v2).** Counting the library's own bits, total
   description length still falls: the abstractions more than pay for themselves on
   structured data. And honestly: a bigger library slightly *taxes* unrelated work.

3. **It's computation, not just description (v3).** The same library that
   *describes* related tasks more cheaply lets a solver *find* them in 21× fewer
   steps. Abstractions are search accelerators.

4. **It generalizes compositionally (v4).** Trained at depth ≤2, compression stays
   flat out to depth 6 — reuse of generic parts, not interpolation within the
   trained range.

5. **Occam ≡ generalization (v4, the key conceptual result).** BPE's *own* stopping
   point — where recurring structure runs out — is exactly the generalization
   optimum. Compressing *past* it (memorizing whole training compositions) shrinks
   the training description but does nothing for novel held-out work. **Shortness
   that generalizes stops where reuse stops.**

6. **It's domain-agnostic (v5).** The identical story holds for real
   list-processing idioms ("double the evens", "square the positives"), not just
   toy integer ops. The mechanism keys off compositional structure, which real
   programs have.

7. **The learner is domesticated, not banished (v6).** Choosing compression as the
   objective leaves a job for a learner: *guiding* the search. A tiny proposer
   trained on the same data cuts search another 12.7× (≈270× over naive) — and,
   tellingly, makes it *worse* where there's no learned structure to exploit. The
   objective never changed; the learner just points the search.

8. **It lifts to higher-order trees (v7).** Moving from linear pipelines to
   `map`/`filter` over sub-program lambdas — genuine trees — the story survives:
   recurring *subtrees* mined from shallow training shorten novel deeper trees.

**The recurring signature across all of it:** abstractions are a *double-edged*
tool — a large win where structure matches, a real cost where it doesn't. That the
controls reliably *fail* to benefit (and sometimes pay a tax) is what makes the
*related* wins mean something.

---

## Honest limitations

These are clean **mechanism demonstrations**, not benchmark results. In particular:

- **Synthetic, hand-designed corpora.** The motifs are planted; real-world
  structure is messier and rarer. Transfer magnitudes (~2×) are partly set by motif
  length — the robust claim is the *qualitative* pattern (related moves, controls
  don't), not the number.
- **Greedy abstraction.** BPE / greedy subtree-mining is not optimal compression
  (true Kolmogorov complexity is uncomputable); these are tractable approximations.
- **Limited notion of "generalization."** It's compositional *reuse* of learned
  fragments. It does **not** yet test variable binding beyond a single hole,
  recursion, or invention of genuinely novel primitives.
- **No external benchmark yet** (e.g. ARC, held-out list-function suites).
- **v7's tree caveat is real:** the same-primitive control isn't perfectly flat
  (1.06×) because sharing a leaf op means sharing the tiny subtree `dbl(x)` — a
  partial-subtree overlap with no analogue in the linear case. The *disjoint*
  control (no shared structure at all) is the clean test there, and it's exactly flat.
- **v2 used a uniform codebook** for stability; a frequency code is "better" in
  principle but wobbles badly on a tiny corpus (this is exactly the v0 failure).

## Where it goes next

- **v8 — parameterized fragments earn their keep.** Antiunification (fragments with
  holes) is sketched and correct in v7 but doesn't pay on this tiny corpus. Build a
  corpus where a *holed schema* beats every ground fragment — the step from "reuse
  exact subtrees" toward "reuse a *pattern*."
- **A real benchmark.** Graduate to a mini-ARC subset or a held-out list-functions
  suite, where generalization is measured, not assumed — the regime where a 7 GB
  CPU and a 70 B GPU model are *both* weak, so a different kind of system can show
  a real edge.
- **The proposer as a small learned model.** v6's bigram guide could become a
  context-conditioned net — the domesticated learner with more capacity.

---

*Reproduce: `python3 induct.py` (v0), `induct_v1.py` … `induct_v7.py`. Each prints
its own table and an honest verdict. Full chronological notes in `LOG.md`;
roadmap in `PLAN.md`.*
