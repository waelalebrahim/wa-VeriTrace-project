"""Tests for VeriTrace. Run with: pytest"""

from veritrace import (
    ConfidenceTier,
    SourceDocument,
    VeriTrace,
    VeriTraceSourceFault,
)


def test_grounded_claim_is_high_with_citation():
    vt = VeriTrace()
    vt.add_source(
        "The Eiffel Tower is located in Paris and was completed in 1889.",
        id="eiffel",
        name="Encyclopedia entry",
    )
    result = vt.verify("The Eiffel Tower is located in Paris.")
    assert len(result.claims) == 1
    claim = result.claims[0]
    assert claim.tier == ConfidenceTier.HIGH
    assert claim.citation is not None
    assert claim.citation.doc_id == "eiffel"
    assert claim.citation.doc_hash  # integrity hash is attached


def test_fabricated_claim_is_low_and_uncited():
    vt = VeriTrace()
    vt.add_source("The Eiffel Tower is located in Paris.", id="eiffel")
    # A claim that shares nothing with the source must be LOW.
    result = vt.verify("Penguins operate a secret stock exchange on the moon.")
    assert len(result.claims) == 1
    claim = result.claims[0]
    assert claim.tier == ConfidenceTier.LOW
    assert claim.citation is None
    assert not result.grounded


def test_known_limitation_same_subject_fabrication():
    """Documented limitation, not a bug.

    A fabrication that reuses a real entity from the sources ("Eiffel Tower")
    cannot be driven all the way to LOW by a surface-lexical backend -- the
    subject genuinely appears in the source. VeriTrace correctly refuses to
    call it HIGH and lands it at MEDIUM ("verify this"). Catching this fully
    needs a semantic / NLI backend (see README roadmap).
    """
    vt = VeriTrace()
    vt.add_source("The Eiffel Tower is located in Paris.", id="eiffel")
    result = vt.verify("The Eiffel Tower secretly cures every known disease.")
    assert result.claims[0].tier == ConfidenceTier.MEDIUM


def test_mixed_answer_separates_tiers():
    vt = VeriTrace()
    vt.add_source("Water boils at 100 degrees Celsius at sea level.", id="phys")
    result = vt.verify(
        "Water boils at 100 degrees Celsius at sea level. "
        "Drinking it grants the power of flight."
    )
    tiers = {c.tier for c in result.claims}
    assert ConfidenceTier.LOW in tiers
    assert any(c.tier == ConfidenceTier.HIGH for c in result.claims)


def test_no_sources_means_everything_is_low():
    vt = VeriTrace()
    result = vt.verify("Anything at all. Truly anything.")
    assert all(c.tier == ConfidenceTier.LOW for c in result.claims)
    assert not result.grounded


def test_conflict_resolution_prefers_newest_source():
    vt = VeriTrace()
    vt.add_documents(
        [
            SourceDocument(
                id="old",
                text="The company headquarters is in Boston.",
                date="2019-01-01",
            ),
            SourceDocument(
                id="new",
                text="The company headquarters is in Boston.",
                date="2025-01-01",
            ),
        ]
    )
    result = vt.verify("The company headquarters is in Boston.")
    claim = result.claims[0]
    assert claim.citation is not None
    assert claim.citation.doc_id == "new"  # newest wins the tie


def test_gate_raises_on_ungrounded_claim():
    vt = VeriTrace()
    vt.add_source("The sky appears blue due to Rayleigh scattering.", id="sky")
    try:
        vt.gate("The sky is blue. Also, the moon is made of cheese.")
        assert False, "expected VeriTraceSourceFault"
    except VeriTraceSourceFault as e:
        assert e.offending
        assert "cheese" in str(e).lower()


def test_gate_passes_when_all_grounded():
    vt = VeriTrace()
    vt.add_source(
        "Photosynthesis converts sunlight, water, and carbon dioxide into glucose.",
        id="bio",
    )
    result = vt.gate(
        "Photosynthesis converts sunlight, water, and carbon dioxide into glucose."
    )
    assert result.grounded


def test_verify_with_inline_sources():
    vt = VeriTrace()
    result = vt.verify(
        "Mount Everest is the tallest mountain above sea level.",
        sources=["Mount Everest is the tallest mountain above sea level on Earth."],
    )
    assert result.claims[0].tier == ConfidenceTier.HIGH


def test_result_serializes_to_dict():
    vt = VeriTrace()
    vt.add_source("Honey does not spoil due to its low moisture content.", id="honey")
    result = vt.verify("Honey does not spoil due to its low moisture content.")
    d = result.to_dict()
    assert "claims" in d and "grounded" in d
    assert d["claims"][0]["tier"] in {"high", "medium", "low"}
