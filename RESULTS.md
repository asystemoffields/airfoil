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
| **v8** | the verifier (the load-bearing wall) | a 10% false-accept verifier *alone* → **4%** solved; redundancy rescues as `eps^M` (M=3→92%, M=5→100%) | — |
| **v9** | repetition vs diversity | systematic-error verifier: repetition flat **~29%**; *diverse* verifiers (same budget) → **100%** | — |
| **v10** | the loop closes (poisoning) | no gate: skills 29% → depth-3 compositions **10%** (reuse amplifies error); diverse gate → **99/99/99** | — |
| **v11** | reuse a *pattern* (antiunification) | on a non-contiguous template, exact reuse learns nothing; a schema → **1.5×** desc, **51×** less search, generalizes to unseen fillers | exact/BPE: **0** |
| **v12** | generalization **measured** (held-out) | *discovered* (not planted) library solves **100%** of novel depth-3 test-near (base 35%) within budget | test-far (held-out idiom) 25%→50% — the honest boundary |
| **v13** | the loop **bootstraps** (wake-sleep) | round 0→1 discovering idioms lifts the solvable frontier 58→83% train, **46→79% held-out** (fixed budget, frozen base) | but *ungoverned* it bloats → held-out then **declines** (the sleep needs an Occam razor → v14) |
| **v14** | the sleep must **suppress** | an MDL razor at the sleep holds held-out at **79%** with **6** macros | vs ungoverned: **24** macros, declines to 75% — suppression = stability + parsimony |
| **v15** | governance that **climbs** | given recurring phrases, the governed loop discovers a *hierarchy* (idioms→phrases): held-out **25→100%**, deep-solve **~11× cheaper**, stays lean | scope: shows hierarchy *use*, not a clean phrases-vs-idioms isolation (→ v16) |
| **v16** | when is depth *necessary*? (budget sweep) | matched libs + budget sweep: necessity regime is **real** (depth-6 idioms cap 67% within 100k; phrases 100%) | *confounded*: some "deep" tasks collapse to short functions → inflates idiom baseline; clean isolation = v17 |
| **v17** | the necessity window, **clean** | provably-incompressible tasks (each idiom doubles degree): idiom cost ~\|V\|^depth — d6 **unsolved** within 100k — vs phrases near-flat (161→2,108 nodes) | resolves v16: deeper abstraction is **required**, not just cheaper |

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

**Part II — the wall it all rests on (v8–v10).** Everything above assumes a
verifier (run the program, check the examples — *free and perfect* in program
synthesis, noisy and expensive everywhere real). Three results pin down how much
that assumption matters:

9. **A weak verifier is catastrophic *alone* (v8).** A 10% false-accept rate drops
   solving to 4% — there are so many short *wrong* candidates that one fluke
   acceptance ends the search on garbage. But redundancy rescues it *exponentially*
   (`eps^M`): a few independent confirmations restore reliability. You don't need a
   *reliable* verifier — a cheap one plus confirmation suffices.

10. **…but only if the errors are independent (v9).** Real verifiers fail
    *systematically*. Repeating one weak verifier stays flat (~29%); spending the
    same budget on *diverse* verifiers climbs to 100%. The lever is **diversity, not
    volume** — you must *manufacture* the independence v8 assumed for free.

11. **In a reuse system, verification is load-bearing (v10).** A verifier error
    doesn't cost one task — it poisons a *reusable* skill, and the error compounds
    with composition depth (`~accuracy^depth`: 29% skills → 10% at depth 3). A
    diverse-consensus gate at crystallization keeps the library clean (→99%), and
    matters *more* than in one-shot solving because it protects every future reuse
    at once.

**Part III — past exact reuse, and out of the sandbox (v11–v12).**

12. **Reuse the *pattern*, not just the structure (v11).** Exact compression only
    sees contiguous repeats. On a corpus whose shared structure is a *template with
    a varying slot*, exact reuse learns nothing — but antiunification recovers the
    schema, describing novel instances 1.5× shorter and solving them 51× faster by
    *filling one hole*, generalizing to fillers never trained. The step from "reuse
    what recurs verbatim" to "reuse the pattern."

13. **Generalization measured, not assumed (v12).** v1–v11 planted the structure.
    v12 *discovers* the idioms by compressing solutions to training tasks
    (recovering all four, untold), then evaluates on *novel, deeper, held-out*
    tasks against a base baseline: a discovered library lifts solve-rate on novel
    near tasks from 35% to **100%** within a fixed budget, and on far tasks (a
    held-out idiom) only 25%→50% — the honest boundary. Training experience,
    compressed into reusable idioms, extends the solvable frontier to harder novel
    tasks it never saw.

**Part IV — the loop, and its discipline (v13–v14).**

14. **The loop bootstraps (v13).** Closed into wake-sleep, the system compresses
    its own solutions into idioms, and that library makes deeper tasks solvable the
    next round — solvable frontier 58→83% train, 46→79% held-out, on a *fixed*
    budget with a *frozen* base. Competence compounding from experience. But
    ungoverned, the library bloats with junk and held-out *declines* — the loop
    eats itself.

15. **Suppression is the other half (v14).** A two-part-MDL razor at the sleep —
    keep a piece only if it pays for itself — keeps the library lean (6 vs 24
    macros) and the gain stable (held-out holds 79% vs declining to 75%). Most of
    intelligence is inhibition; the sleep's job is forgetting as much as learning.

16. **And, given real structure, it climbs (v15).** With a curriculum containing
    recurring *phrases*, the governed loop discovers a *hierarchy* — idioms, then
    the phrases built on them — riding it to 100% held-out with deep-solving ~11×
    cheaper, still lean. (Honestly scoped: this shows the loop *uses* a hierarchy;
    it doesn't cleanly isolate phrases-vs-idioms — that ablation is v16.)

17. **When is depth *necessary*, not just cheaper? (v16 — methodology + a caught
    confound).** A matched-library budget sweep is the right instrument, and the
    necessity regime is real (depth-6 idioms cap at 67% within 100k; phrases reach
    100%). But it surfaced the shortest-equivalent gremlin at the *task* level:
    some "deep" compositions collapse to short functions, so idioms crack them
    cheaply and inflate the baseline. Cleanly isolating necessity needs an
    incompressibility-controlled task set — the honest debt carried to v17.

18. **The necessity window, cleanly (v17).** Make the tasks *provably*
    incompressible — each idiom doubles polynomial degree, so a depth-K task needs
    ≥K squarings and no shorter program exists. Now the sweep is unambiguous:
    phrases solve at near-flat cost (161→2,108 nodes from depth-4 to depth-6) while
    idiom-only search costs ~\|V\|^depth (depth-6 *unsolved* within 100k). The
    budget at which flat search "catches up" explodes exponentially with depth, so
    for anything deep relative to your compute, deeper abstraction is **required**,
    not merely cheaper. The airfoil thesis, quantified: configuration, not resources.

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
- **Still a synthetic distribution.** v12 is a genuine *held-out* measurement
  (discovered library, novel tasks, real baseline) — but the task distribution is
  still ours, and it's a single solve-then-compress pass, not the full iterative
  wake-sleep, with v11's antiunification not yet wired into the loop. A real
  *external* benchmark (ARC) remains future.
- **v7's tree caveat is real:** the same-primitive control isn't perfectly flat
  (1.06×) because sharing a leaf op means sharing the tiny subtree `dbl(x)` — a
  partial-subtree overlap with no analogue in the linear case. The *disjoint*
  control (no shared structure at all) is the clean test there, and it's exactly flat.
- **v2 used a uniform codebook** for stability; a frequency code is "better" in
  principle but wobbles badly on a tiny corpus (this is exactly the v0 failure).
- **The verifier results assume you *can* build diverse checkers.** v8–v10 show a
  cheap+diverse verifier suffices — but *constructing* genuinely decorrelated
  verifiers in a real domain (not a conveniently partitioned test suite) is itself
  the hard, unsolved part.

## Where it goes next

The compression arc (v1–v7) and the verifier arc (v8–v10) are done. Open threads:

- **Parameterized fragments earn their keep (v11).** Antiunification (fragments with
  holes) is sketched and correct in v7 but doesn't pay on this tiny corpus. Build a
  corpus where a *holed schema* beats every ground fragment — the step from "reuse
  exact structure" to "reuse a *pattern*." This is where real generalization gets
  its teeth (and where the field strains).
- **A real benchmark.** Graduate to a mini-ARC subset or a held-out list-functions
  suite, where generalization is measured, not assumed — the regime where a 7 GB
  CPU and a 70 B GPU model are *both* weak, so a different kind of system can show
  a real edge.
- **Constructing diverse verifiers in real domains.** v9–v10 prove diversity is the
  lever; the open practical question is how to *manufacture* decorrelated verifiers
  (different lenses/evidence/framings) when you can't just partition a test suite.
- **The proposer as a small learned model.** v6's bigram guide could become a
  context-conditioned net — or a frozen small LLM — the domesticated learner with
  more capacity.

---

*Reproduce: `python3 induct.py` (v0), `induct_v1.py` … `induct_v17.py`. Each prints
its own table and an honest verdict. Full chronological notes in `LOG.md`;
roadmap in `PLAN.md`.*
