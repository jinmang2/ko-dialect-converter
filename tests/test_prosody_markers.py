from __future__ import annotations

from ko_dialect.data.prosody import (
    PROSODY_MARKER_POLICY,
    PROSODY_MARKER_POLICY_V1,
    add_sentence_final_marker,
    apply_eojeol_markers,
    apply_eojeol_markers_to_text,
    eojeol_markers_aligned,
    eojeol_prosody_markers,
    f0_coefficient_of_variation,
    prosody_marker,
    slice_intonation_by_time,
    strip_markers,
)


def test_strip_markers_removes_all_marker_types():
    for marker in PROSODY_MARKER_POLICY["markers"]:
        assert strip_markers(f"문장{marker}") == "문장"


def test_strip_markers_tidies_whitespace_and_punctuation():
    assert strip_markers("밥 뭇나<UP>?") == "밥 뭇나?"
    assert strip_markers("어데 가노<DOWN>") == "어데 가노"
    assert strip_markers("보통 말<KEEP> 함더<WAVE>") == "보통 말 함더"


def test_strip_markers_noop_on_clean_text():
    assert strip_markers("no markers here") == "no markers here"
    assert strip_markers("") == ""


def test_add_then_strip_roundtrips():
    # adding a marker then stripping returns the original (modulo strip())
    original = "밥 먹었니?"
    marked = add_sentence_final_marker(original, "<UP>")
    assert marked == "밥 먹었니<UP>?"
    assert strip_markers(marked) == original


# --- v2 K-ToBI-grounded marker classification ---------------------------------


def test_prosody_marker_none_for_empty():
    assert prosody_marker(None) is None
    assert prosody_marker({}) is None


def test_v2_rising_is_up():
    # delta above +0.05 => H% rising boundary tone
    assert prosody_marker({"f0_mean": 200, "f0_std": 30, "f0_delta": 0.14}) == "<UP>"


def test_v2_mild_declination_is_keep_not_down():
    # The key v2 fix: ordinary declination (delta ~ -0.2, low dispersion) is level/KEEP,
    # not a falling boundary tone (v1 wrongly tagged this <DOWN>).
    p = {"f0_mean": 192, "f0_std": 41, "f0_delta": -0.198}
    assert prosody_marker(p) == "<KEEP>"
    assert prosody_marker(p, version=1) == "<DOWN>"


def test_v2_steep_fall_is_down():
    # delta below -0.40 => L% steep fall beyond declination
    assert prosody_marker({"f0_mean": 191, "f0_std": 50, "f0_delta": -0.489}) == "<DOWN>"


def test_v2_high_dispersion_is_wave():
    # flat-ish delta but high coefficient of variation => contour / <WAVE>
    p = {"f0_mean": 100, "f0_std": 40, "f0_delta": -0.1}  # cv = 0.40 >= 0.30
    assert f0_coefficient_of_variation(p) >= PROSODY_MARKER_POLICY["wave_threshold_cv"]
    assert prosody_marker(p) == "<WAVE>"


def test_f0_cv_falls_back_to_range_when_std_absent():
    # re-parsed summaries may carry f0_range but not f0_std
    assert f0_coefficient_of_variation({"f0_mean": 100, "f0_range": 50}) == 0.5
    assert f0_coefficient_of_variation({"f0_mean": 0}) == 0.0


def test_v1_policy_unchanged():
    # v1 constants are frozen so the original prosody A/B stays reproducible
    assert PROSODY_MARKER_POLICY_V1["up_threshold_delta_ratio"] == 0.15
    assert PROSODY_MARKER_POLICY_V1["down_threshold_delta_ratio"] == -0.15
    assert PROSODY_MARKER_POLICY["version"] == 2


# --- Phase 2: per-eojeol prosody (time-sliced F0) -----------------------------


def test_slice_intonation_by_time_maps_window_to_indices():
    # 10 samples over [0,1]s => 10 fps; segment [0.2,0.5] -> indices [2,5)
    series = list(range(10))
    assert slice_intonation_by_time(series, 0.0, 1.0, 0.2, 0.5) == [2, 3, 4]
    # out-of-range / degenerate cases
    assert slice_intonation_by_time([], 0.0, 1.0, 0.0, 1.0) == []
    assert slice_intonation_by_time(series, 0.0, 0.0, 0.0, 1.0) == []  # zero duration
    assert slice_intonation_by_time(series, 0.0, 1.0, 0.6, 0.6) == []  # empty window


def test_eojeol_prosody_markers_assigns_per_word():
    # First half rises, second half is flat/low — distinct per-eojeol contours.
    series = [100, 120, 140, 160] + [150, 150, 150, 150]
    eojeols = [
        {"word": "내가", "start_s": 0.0, "end_s": 0.5},
        {"word": "왔다", "start_s": 0.5, "end_s": 1.0},
    ]
    out = eojeol_prosody_markers(series, 0.0, 1.0, eojeols)
    assert [m["word"] for m in out] == ["내가", "왔다"]
    assert out[0]["marker"] == "<UP>"  # rising first eojeol
    assert all("marker" in m and "prosody" in m for m in out)


def test_eojeol_prosody_marker_none_when_too_few_points():
    out = eojeol_prosody_markers([200.0], 0.0, 1.0, [{"word": "음", "start_s": 0.0, "end_s": 1.0}])
    assert out[0]["marker"] is None  # < min_valid_points


def test_apply_eojeol_markers_inlines_markers():
    marked = [
        {"word": "내가", "marker": "<UP>"},
        {"word": "왔다", "marker": None},
        {"word": "함더", "marker": "<KEEP>"},
    ]
    assert apply_eojeol_markers(marked) == "내가<UP> 왔다 함더<KEEP>"


def test_apply_eojeol_markers_to_text_marks_aligned_words():
    text = "밥 뭇나"
    ep = [{"word": "밥", "marker": "<KEEP>"}, {"word": "뭇나", "marker": "<UP>"}]
    assert apply_eojeol_markers_to_text(text, ep) == "밥<KEEP> 뭇나<UP>"
    # marker is faithfully attached to the ORIGINAL words; stripping recovers gold
    assert strip_markers(apply_eojeol_markers_to_text(text, ep)) == text


def test_apply_eojeol_markers_to_text_falls_back_on_misalignment():
    text = "밥 뭇나"  # 2 words
    assert apply_eojeol_markers_to_text(text, [{"word": "밥", "marker": "<UP>"}]) == text  # 1 entry
    assert apply_eojeol_markers_to_text(text, None) == text
    assert apply_eojeol_markers_to_text(text, []) == text


def test_apply_eojeol_markers_to_text_handles_none_marker():
    text = "밥 뭇나"
    ep = [{"word": "밥", "marker": None}, {"word": "뭇나", "marker": "<UP>"}]
    assert apply_eojeol_markers_to_text(text, ep) == "밥 뭇나<UP>"


def test_eojeol_markers_aligned():
    text = "밥 뭇나"
    assert eojeol_markers_aligned(text, [{"word": "밥"}, {"word": "뭇나"}]) is True
    assert eojeol_markers_aligned(text, [{"word": "밥"}]) is False
    assert eojeol_markers_aligned(text, None) is False
