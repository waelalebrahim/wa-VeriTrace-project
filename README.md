# VeriTrace

**AI that admits when it doesn't know.** VeriTrace checks every claim in an answer against the documents you trust, cites what holds up, and refuses what it can't prove.

It's the enforcement layer for the principle behind [The I Don't Know Project](https://theidkproject.ai): an AI should only assert what it can ground — and say "I don't know" otherwise.

🔗 **Live demo:** https://wa-veritrace-project.pages.dev/ runs entirely in the browser, no server.

- **Zero dependencies** in the core. Pure Python standard library — no model downloads, no API keys.
- **Tiered confidence**: every claim is graded `HIGH` (cited), `MEDIUM` (verify this), or `LOW` (unprovable).
- **A hard gate**: `gate()` raises `VeriTraceSourceFault` so an ungrounded claim never reaches your users.
- **Pluggable backends**: start lexical, upgrade to semantic embeddings for paraphrase matching.

---

## Install

```bash
pip install -e .
# optional semantic matching:
pip install -e ".[embeddings]"
```

## Quickstart

```python
from veritrace import VeriTrace, VeriTraceSourceFault

vt = VeriTrace()
vt.add_source(
    "The Eiffel Tower is in Paris and was completed in 1889.",
    id="eiffel", name="Fact sheet", date="2025-03-01",
)

answer = "The Eiffel Tower is in Paris. It was built to talk to dolphins."

report = vt.verify(answer)
print(report.summary())          # 2 claims -> 1 high, 0 medium, 1 low
for c in report.claims:
    print(c.tier.value, "|", c.claim)

# Or block ungrounded answers outright:
try:
    vt.gate(answer)
except VeriTraceSourceFault as fault:
    for c in fault.offending:
        print("rejected:", c.claim)
```

Run the examples and tests:

```bash
python examples/quickstart.py
pytest
```

## How the tiers work

| Tier     | Meaning                                        | Citation |
|----------|------------------------------------------------|----------|
| `HIGH`   | Strongly supported by a source                 | yes      |
| `MEDIUM` | Partially supported — ask the reader to verify | yes      |
| `LOW`    | No supporting source found                     | none     |

Thresholds are configurable: `VeriTrace(high_threshold=0.55, medium_threshold=0.18)`. When several sources match, VeriTrace prefers the **newest-dated** document.

## Repo layout

```
veritrace/      the Python library (the actual product)
tests/          test suite
examples/       runnable quickstart
web/            the live browser demo (index.html) — deployed to Cloudflare Pages
```

## Live demo deployment (Cloudflare Pages)

The demo is a single static file with no build step. When connecting this repo in Cloudflare Pages:

- **Framework preset:** None
- **Build command:** _(leave empty)_
- **Build output directory:** `web`

Every push to this repo then auto-deploys the updated demo.

## Honest limitations

This is a V1, and grounding is genuinely hard. What it does **not** do yet:

- The default backend matches on **content overlap**, not deep meaning. A fabrication that reuses a real entity from your sources is flagged `MEDIUM`, not `LOW` — VeriTrace refuses to certify it, but can't fully reject it without a semantic/NLI backend. This is tested on purpose (`test_known_limitation_same_subject_fabrication`).
- Claim segmentation is sentence-level.
- `verify()` runs after generation; streaming interception is on the roadmap.
- Source hashing proves a source is **unchanged**, not that a claim is true.

## Roadmap

- NLI/entailment backend (claim-vs-source, not just overlap) — the big quality unlock
- Sub-sentence claim extraction
- Streaming / incremental verification
- LLM-as-judge backend adapter

## License & attribution

MIT — free to use, copy, modify, and redistribute. Per the MIT terms, the copyright and author notice must travel with the code.

Created by **Wael Alebrahim**.
