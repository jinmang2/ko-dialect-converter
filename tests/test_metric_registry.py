from __future__ import annotations

from ko_dialect.evaluation.metric_registry import (
    REGISTRY,
    arrow,
    get_spec,
    glossary_markdown,
    header_label,
    is_higher_better,
    orient,
)


def test_copy_margin_is_higher_better_not_lower():
    # The whole point: correct the recurring "low copy_margin is good" misconception.
    assert is_higher_better("copy_margin") is True
    assert arrow("copy_margin") == "↑"
    # -2.37 (grpo_500) must orient ABOVE -4.97 (SFT)
    assert orient("copy_margin", -2.37) > orient("copy_margin", -4.97)


def test_recon_bleu_higher_better():
    assert is_higher_better("reconstruction_bleu") is True
    assert orient("reconstruction_bleu", 38.4) > orient("reconstruction_bleu", 37.0)


def test_chrf_source_is_lower_better_and_negates():
    assert is_higher_better("chrf_source") is False
    assert arrow("chrf_source") == "↓"
    # lower chrf_source (less copying) orients higher
    assert orient("chrf_source", 50.0) > orient("chrf_source", 80.0)


def test_header_label_appends_arrow():
    assert header_label("copy_margin") == "copy_margin↑"
    assert header_label("chrf_source") == "chrf_source↓"


def test_unregistered_metric_defaults_higher_better():
    spec = get_spec("totally_new_metric")
    assert spec.higher_is_better is True
    assert orient("totally_new_metric", 5.0) == 5.0


def test_all_registered_specs_have_groups():
    valid = {"selection", "dialectness", "surface", "monitoring"}
    for spec in REGISTRY.values():
        assert spec.group in valid
        assert spec.meaning


def test_glossary_markdown_documents_direction():
    md = glossary_markdown(("reconstruction_bleu", "copy_margin"))
    assert "| metric | dir | group | meaning | caveat |" in md
    assert "`copy_margin`" in md
    assert "↑ higher" in md
    # selection metric flagged
    assert "proxy-independent" in md
