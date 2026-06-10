from __future__ import annotations

from datasets import Dataset

from ko_dialect.data.dataset import build_classification_dataset
from ko_dialect.data.filtering import carries_dialect_marker, norm_levenshtein


def test_norm_levenshtein_bounds():
    assert norm_levenshtein("안녕", "안녕") == 0.0
    assert norm_levenshtein("", "abc") == 1.0
    # one char differs out of 4 -> 0.25
    assert abs(norm_levenshtein("abcd", "abce") - 0.25) < 1e-9
    # near-identical long sentence -> well below 0.1
    s = "뭐 폰이 없음 뭐 택시를 잡아타고 뭐 집으로 가야지 뭐 길을 잃었으면"
    d = s.replace("잃었으면", "잃었으믄")
    assert norm_levenshtein(s, d) < 0.1


def test_carries_dialect_marker():
    assert carries_dialect_marker("뭐라 카노 진짜")
    assert carries_dialect_marker("옷 장사를 하거든예")
    assert not carries_dialect_marker("안녕하세요 반갑습니다")


def _raw(rows):
    return Dataset.from_list(rows)


def test_min_levenshtein_drops_near_standard_dialect():
    raw = _raw(
        [
            # near-identical: 1 char diff in a long sentence -> dropped as dialect
            {
                "standard": "길을 잃었으면 어떡해 정말 큰일이네 그러게",
                "dialect": "길을 잃었으믄 어떡해 정말 큰일이네 그러게",
                "do": "gangwondo",
                "is_identical": False,
            },
            # clearly divergent -> kept
            {
                "standard": "밥 먹었니",
                "dialect": "밥 무읏나",
                "do": "gyeongsangdo",
                "is_identical": False,
            },
        ]
    )
    out = build_classification_dataset(raw, min_norm_levenshtein=0.1, drop_collisions=False)
    texts = out["train"]["text"]
    labels = out["train"]["label"]
    # both standards present as label 0
    assert texts.count("길을 잃었으면 어떡해 정말 큰일이네 그러게") == 1
    # near-identical dialect dropped, divergent dialect kept
    assert "길을 잃었으믄 어떡해 정말 큰일이네 그러게" not in texts
    assert "밥 무읏나" in texts
    assert labels[texts.index("밥 무읏나")] == 2  # gyeongsangdo


def test_denoise_standard_drops_marked_standard():
    raw = _raw(
        [
            {
                "standard": "옷 장사를 하거든예",  # polluted standard
                "dialect": "옷 장사를 하는기라예",
                "do": "gyeongsangdo",
                "is_identical": False,
            },
        ]
    )
    kept = build_classification_dataset(raw, denoise_standard=True, min_norm_levenshtein=None)
    assert "옷 장사를 하거든예" not in kept["train"]["text"]
    not_denoised = build_classification_dataset(
        raw, denoise_standard=False, min_norm_levenshtein=None
    )
    assert "옷 장사를 하거든예" in not_denoised["train"]["text"]


def test_drop_collisions_removes_ambiguous_text():
    raw = _raw(
        [
            # same surface string appears as standard here ...
            {
                "standard": "밥 먹자",
                "dialect": "밥 무우러 가자야",
                "do": "gyeongsangdo",
                "is_identical": False,
            },
            # ... and as a gangwon dialect there -> collision on "밥 먹자"
            {
                "standard": "식사하자 우리 같이 밥 한끼 하자고",
                "dialect": "밥 먹자",
                "do": "gangwondo",
                "is_identical": False,
            },
        ]
    )
    out = build_classification_dataset(
        raw, min_norm_levenshtein=None, denoise_standard=False, drop_collisions=True
    )
    assert "밥 먹자" not in out["train"]["text"]  # both occurrences dropped


def test_labels_are_correct():
    raw = _raw(
        [
            {
                "standard": "안녕하세요 오늘 날씨 좋네요",
                "dialect": "안녕하시소 오늘 날씨 억수로 좋네예",
                "do": "gyeongsangdo",
                "is_identical": False,
            },
        ]
    )
    out = build_classification_dataset(raw, min_norm_levenshtein=0.1)
    pairs = dict(zip(out["train"]["text"], out["train"]["label"]))
    assert pairs["안녕하세요 오늘 날씨 좋네요"] == 0
    assert pairs["안녕하시소 오늘 날씨 억수로 좋네예"] == 2
