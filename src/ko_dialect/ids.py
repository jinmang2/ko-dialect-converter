"""Sample-id parsing — the single source of truth for speaker extraction.

`prepare_data.py` builds each row's ``id`` from the AI-Hub filename, e.g.

    say_set2_collectorgw185_speakergw2034_59_0_26_1        (1인발화 — one speaker)
    talk_set1_collectorjj14_speakerjj59_speakerjj60_4_0_121 (2인발화 — two speakers)

so the speaker(s) behind a row are recoverable from the id alone, without re-reading the
raw JSON. Two consumers depend on that and must agree exactly, or a "speaker-disjoint"
claim made by one is not the property the other checks:
``scripts/audit_corpus_fields.py leakage`` and ``scripts/resplit_speaker_disjoint.py``.

Lives at the package root, NOT under ``ko_dialect.data``: that package's ``__init__``
imports the collator, which imports torch. The audit scripts that use this need neither,
and on the 6 GB dev box paying for a torch import to run a regex is not free.
"""

from __future__ import annotations

import re

# Region code is two letters (gw/gs/jj/jl/cc), then a numeric speaker index.
SPEAKER_ID_RE = re.compile(r"(speaker[a-z]{2}\d+)")


def speakers_of(sample_id: str | None) -> list[str]:
    """Every speaker id embedded in ``sample_id``, in order of appearance.

    Returns an empty list for old-format ids, which carry no speaker — callers must
    decide what that means rather than silently treating it as "no leakage".
    """
    return SPEAKER_ID_RE.findall(sample_id or "")
