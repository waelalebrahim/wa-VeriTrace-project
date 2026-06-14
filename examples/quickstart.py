"""VeriTrace quickstart -- runs with zero dependencies.

    python examples/quickstart.py
"""

from veritrace import VeriTrace, VeriTraceSourceFault, ConfidenceTier

# 1. Load the documents you trust. These are the ONLY things VeriTrace will
#    let an answer be grounded in.
vt = VeriTrace()
vt.add_source(
    "The Eiffel Tower is in Paris and was completed in 1889. "
    "It is 330 metres tall including antennas.",
    id="eiffel-wiki",
    name="Eiffel Tower fact sheet",
    date="2025-03-01",
)

# 2. Verify an answer an LLM produced.
answer = (
    "The Eiffel Tower is in Paris. "
    "It was completed in 1889. "
    "It was originally built as a giant radio for talking to dolphins."
)

result = vt.verify(answer)
print(result.summary())
print("-" * 60)
for c in result.claims:
    line = f"[{c.tier.value.upper():6}] {c.claim}"
    if c.citation:
        line += f"\n         -> {c.citation.source_name}: {c.citation.passage!r}"
    print(line)

print("-" * 60)

# 3. Or run it as a hard gate in front of your UI. Ungrounded claims stop here.
try:
    vt.gate(answer)
except VeriTraceSourceFault as fault:
    print("Blocked before reaching the user.")
    for c in fault.offending:
        print(f"  rejected: {c.claim}")

# ---------------------------------------------------------------------------
# Wiring it after a real LLM call (sketch):
#
#   llm_answer = call_your_model(question, context=docs)
#   try:
#       report = vt.gate(llm_answer, sources=docs)
#       return llm_answer, report          # safe to show
#   except VeriTraceSourceFault:
#       return "I don't know based on the sources I have.", None
# ---------------------------------------------------------------------------
