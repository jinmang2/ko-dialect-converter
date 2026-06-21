"""Single-source-of-truth for dialect-region labels.

Before this module the label map was copy-pasted in three places (data/dataset.py,
rewards/style.py, evaluation/metrics.py). These tests pin the canonical map, its
backward-compatible ids, and the data-driven ``num_labels_for`` helper that lets a
2-region build stay 3-class while a 5-region build becomes 6-class.
"""

from __future__ import annotations

from ko_dialect.data.labels import (
    DIALECT_LABELS,
    DO_TO_LABEL,
    SUPPORTED_DO,
    num_labels_for,
)


def test_existing_ids_are_stable_and_appended():
    # standard=0, gangwon=1, gyeongsang=2 must never renumber (trained checkpoints
    # depend on them). New regions are appended.
    assert DIALECT_LABELS["standard"] == 0
    assert DIALECT_LABELS["gangwondo"] == 1
    assert DIALECT_LABELS["gyeongsangdo"] == 2
    assert DIALECT_LABELS["jeollado"] == 3
    assert DIALECT_LABELS["jejudo"] == 4
    assert DIALECT_LABELS["chungcheongdo"] == 5


def test_do_to_label_excludes_standard():
    assert "standard" not in DO_TO_LABEL
    assert DO_TO_LABEL == {k: v for k, v in DIALECT_LABELS.items() if k != "standard"}


def test_supported_do_is_all_five_regions():
    assert SUPPORTED_DO == frozenset(
        {"gangwondo", "gyeongsangdo", "jeollado", "jejudo", "chungcheongdo"}
    )


def test_num_labels_is_data_driven_backward_compatible():
    # gangwon/gyeongsang-only build → 3 classes (unchanged from the original pipeline).
    assert num_labels_for([0, 1, 2, 1, 0, 2]) == 3
    # full 5-region build → 6 classes.
    assert num_labels_for([0, 1, 2, 3, 4, 5]) == 6
    # empty defaults to a single class, never crashes.
    assert num_labels_for([]) == 1


def test_dataset_module_reexports_canonical_map():
    # Back-compat: existing imports of ko_dialect.data.dataset.DIALECT_LABELS still work
    # and point at the same object.
    from ko_dialect.data import dataset as d

    assert d.DIALECT_LABELS is DIALECT_LABELS
    assert d.SUPPORTED_DO is SUPPORTED_DO


def test_reward_and_metric_modules_share_one_map():
    from ko_dialect.evaluation.metrics import DO_TO_LABEL as METRICS_MAP
    from ko_dialect.rewards.style import DO_TO_LABEL as STYLE_MAP

    assert METRICS_MAP is DO_TO_LABEL
    assert STYLE_MAP is DO_TO_LABEL
