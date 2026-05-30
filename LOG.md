# Research log — compression-as-generalization

Autonomous run kicked off 2026-05-30 (Alex away for his dad's birthday).
Newest entries at the bottom. Each entry: what I tried, what happened, what next.

---

## v0 — toy unary-int DSL, frequency-code bits  (`induct.py`)
- Built a tiny DSL (unary int ops), defined motifs, solved a train set by search
  (14/14), then BPE-style library learning, measured description length in bits
  under a frequency code (-log2 p) fit on train.
- **Result:** related compressed only 1.27×; control *inflated* (0.50×, i.e. got
  2× more expensive). The frequency code *specialises* to the training
  distribution — base ops absorbed into macros become rare and so the standalone
  codes for them get expensive, penalising the control.
- **Read:** real phenomenon (no-free-lunch of a specialised code) but a muddy
  headline. Also noticed the search "optimises away" planted motif structure by
  finding shorter semantic equivalents — interesting, parked for later.
- **Next:** a cleaner metric where controls stay flat by construction.

## v1 — clean metric: expression length in learned symbols  (`induct_v1.py`)
- Metric = fewest library symbols needed to write a program (DP). A held-out
  program can only shrink if it *reuses* learned structure; an unrelated one
  stays at full length by construction. Train = 4 motifs + all depth-2 pairs;
  held-out = novel **depth-3** compositions; two controls (disjoint-ops and
  same-ops-scrambled-to-avoid-motifs).
- **Result (clean ✓):**
    - related        6.83 → 3.00 symbols  (**2.28× shorter**)
    - ctrl-disjoint  6.17 → 6.17          (1.00×, flat)
    - ctrl-scramble  6.00 → 6.00          (1.00×, flat)
  Learned exactly the motifs (dbl-inc, sqr-inc, dbl-dbl, inc-inc-inc, +inc-inc).
- **Read:** unambiguous structural transfer — abstractions from depth-2 training
  shorten unseen depth-3 compositions, and do nothing for unrelated work. This is
  the core thesis demonstrated with proper controls.
- **Next (v2):** make it bits, not just symbol-count — a two-part MDL that also
  *charges* for the library itself, and show the library cost is amortised across
  tasks (net corpus bits fall). Then v3: does the library make *solving* new tasks
  faster, not just describing them?
