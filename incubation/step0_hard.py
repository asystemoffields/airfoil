#!/usr/bin/env python3
"""
Incubation STEP 0 (hardened) — make RULESWITCH actually induce FIXATION.

v0 was too easy: a single unique marker is trivially content-addressable, so even a
tiny model solved it (train loss -> 0). Hardened on four axes so that a SHALLOW
single-pass model genuinely can't retrieve the decisive token and falls back on the
salient local pattern:

  RULESWITCH-hard (sequence classification, predict the answer digit):
    [MARK t0]  examples(rule t0) ... [MARK t1] examples(rule t1) ... [MARK t_{S-1}]
    examples(rule t_{S-1})  [QUERY] q   =>   answer
  - S segments, each opened by a marker of a RANDOM type (A=+1 / B=+3 mod 10); the
    segment's example pairs follow THAT marker's rule.
  - DECISIVE rule = the FIRST marker's type (primacy). Decoy markers share the same
    types, so "which type is present" is useless — the model must locate the FIRST one.
  - The query sits after the LAST segment, whose examples follow the last marker's
    rule => a strong RECENCY / local LURE pulling toward the wrong answer.
  - first type != last type (so correct != lure). distance = total length (the first
    marker is at the far left).
  Fixation is confirmed if accuracy falls toward the lure as length grows AND errors are
  dominated by the lure (the recency answer) — the model rides the local pattern and
  fails to retrieve the distant decisive marker.

Deliberately SHALLOW (1-layer) tiny transformer. Run with /data/llm/.venv/bin/python.
"""
import torch
import torch.nn as nn

torch.manual_seed(0)

PAD, MARK_A, MARK_B, ARROW, SEP, QUERY = 10, 11, 12, 13, 14, 15
VOCAB = 16
RULE = {MARK_A: 1, MARK_B: 3}
MARKS = [MARK_A, MARK_B]
SEQLEN = 224
DEVICE = "cpu"
DEPTH = 1            # shallow on purpose


def _build(g, seg_min, seg_max, ex_min, ex_max):
    S = int(torch.randint(seg_min, seg_max + 1, (1,), generator=g).item())
    types = [MARKS[int(torch.randint(0, 2, (1,), generator=g).item())] for _ in range(S)]
    # force first != last so correct (primacy) != lure (recency)
    if types[0] == types[-1]:
        types[-1] = MARK_B if types[0] == MARK_A else MARK_A
    toks = []
    for t in types:
        toks.append(t)
        ne = int(torch.randint(ex_min, ex_max + 1, (1,), generator=g).item())
        for _ in range(ne):
            x = int(torch.randint(0, 10, (1,), generator=g).item())
            toks += [x, ARROW, (x + RULE[t]) % 10, SEP]
            if len(toks) > SEQLEN - 6:
                break
        if len(toks) > SEQLEN - 6:
            break
    q = int(torch.randint(0, 10, (1,), generator=g).item())
    toks += [QUERY, q]
    toks = toks[:SEQLEN]
    ans = (q + RULE[types[0]]) % 10
    lure = (q + RULE[types[-1]]) % 10
    return toks, ans, lure


def make_batch(bs, seg_min=2, seg_max=5, ex_min=1, ex_max=6, gen=None):
    g = gen or torch.Generator().manual_seed(torch.randint(0, 1 << 30, (1,)).item())
    X = torch.full((bs, SEQLEN), PAD, dtype=torch.long)
    Y = torch.zeros(bs, dtype=torch.long)
    L = torch.zeros(bs, dtype=torch.long)
    Lure = torch.zeros(bs, dtype=torch.long)
    for i in range(bs):
        toks, ans, lure = _build(g, seg_min, seg_max, ex_min, ex_max)
        X[i, SEQLEN - len(toks):] = torch.tensor(toks)     # LEFT-pad: query at column -1
        Y[i] = ans
        Lure[i] = lure
        L[i] = len(toks)
    return X.to(DEVICE), Y.to(DEVICE), L, Lure.to(DEVICE)


class TinyTransformer(nn.Module):
    def __init__(self, d=64, heads=2, layers=DEPTH):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(SEQLEN, d)
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=4 * d, batch_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d, 10)

    def forward(self, x):
        h = self.emb(x) + self.pos(torch.arange(x.size(1), device=x.device))
        h = self.enc(h)
        return self.head(h[:, -1])     # query answer is at column -1 (left-padded)


def main():
    flush = lambda *a: print(*a, flush=True)
    flush("=" * 74)
    flush("Incubation step 0 (HARD) — does RULESWITCH-hard induce FIXATION? (1-layer)")
    flush("=" * 74)
    model = TinyTransformer().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    lossf = nn.CrossEntropyLoss()
    flush(f"  params={sum(p.numel() for p in model.parameters())/1e3:.0f}K  depth={DEPTH}  seqlen={SEQLEN}")

    model.train()
    for step in range(6000):
        x, y, _, _ = make_batch(128)
        opt.zero_grad()
        loss = lossf(model(x), y)
        loss.backward()
        opt.step()
        if step % 1000 == 0:
            with torch.no_grad():
                acc = (model(x).argmax(1) == y).float().mean().item()
            flush(f"  step {step:4d}  loss {loss.item():.3f}  train-acc {acc*100:.1f}%")

    # eval: accuracy + lure-fraction by sequence length (= distance to the first marker)
    model.eval()
    flush("\n  len-bucket   acc%   among-errors:%=lure   (chance 10%)")
    with torch.no_grad():
        for (emn, emx) in [(1, 1), (2, 3), (4, 6), (8, 10), (14, 18)]:
            x, y, L, lure = make_batch(800, ex_min=emn, ex_max=emx)
            pred = model(x).argmax(1)
            acc = (pred == y).float().mean().item()
            err = pred != y
            lf = (pred[err] == lure[err]).float().mean().item() if err.sum() > 0 else 0.0
            flush(f"   ~{int(L.float().mean()):3d} tok    {acc*100:5.1f}        {lf*100:5.1f}")

    flush("\n  FIXATION if acc falls toward ~10% as length grows AND errors are lure-dominated")
    flush("  (model rides recency/local pattern, fails to retrieve the distant FIRST marker).")


if __name__ == "__main__":
    main()
