"""Pin the speaker-id contract shared by the leakage audit and the disjoint re-split.

These two consumers must extract the *same* speakers from the same id, otherwise
`resplit_speaker_disjoint.py` can certify a split that `audit_corpus_fields.py leakage`
then reports as leaking. That is the whole reason `ko_dialect.ids` exists.
"""

from __future__ import annotations

from ko_dialect.ids import speakers_of


def test_single_speaker_id():
    assert speakers_of("say_set2_collectorgw185_speakergw2034_59_0_26_1") == ["speakergw2034"]


def test_two_speaker_dialogue_returns_both_in_order():
    sample_id = "talk_set1_collectorjj14_speakerjj59_speakerjj60_4_0_121"
    assert speakers_of(sample_id) == ["speakerjj59", "speakerjj60"]


def test_collector_id_is_not_mistaken_for_a_speaker():
    # `collectorgw185` also ends in digits; only the `speaker` prefix may match.
    assert speakers_of("st_set1_collectorgw185_speakergw2034_1") == ["speakergw2034"]


def test_old_format_id_has_no_speaker():
    # Old (2020) ids carry no speaker, so a disjointness claim cannot be made for them.
    assert speakers_of("DZES20000001") == []


def test_none_and_empty_are_safe():
    assert speakers_of(None) == []
    assert speakers_of("") == []


def test_every_supported_region_prefix_parses():
    for prefix in ("gw", "gs", "jj", "jl", "cc"):
        assert speakers_of(f"talk_set1_speaker{prefix}42_0") == [f"speaker{prefix}42"]
