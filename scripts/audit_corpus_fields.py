#!/usr/bin/env python3
"""Audit the AI-Hub dialect corpus for signal the pipeline is **not** using yet.

`scripts/audit_raw_signal.py` answers "is the dialect↔standard pair worth training on".
This one answers a different question: **what else is in the box?** The 139-x labelling
JSON carries far more than `standard`/`dialect`, and none of it currently reaches
`prepare_data.py`'s output schema:

  transcription.sentences[].{standard, dialect}   <- the only thing we use today
  stt.{recognizer, segments[]}                    <- a REAL ASR hypothesis (Naver Clova),
                                                     timestamp-aligned to the sentences
  annotation.grammarTypes[]                       <- DEC / YNI / WHI / IMP / PRO per sentence
  annotation.intents[] / emotions[]               <- per-sentence dialogue-act & sentiment
  script.{domain, speechType, value}              <- topic domain, elicitation prompt
  speaker[].{birthYear, gender, job, ...}         <- demographics (fairness / disjoint splits)
  audio.{recordDuration, samplingFrequency, ...}  <- speech-rate features without the audio

Two subcommands:

    fields  — scan a raw JSON tree and report what is present, at what rate, with what
              distribution. Also measures how well `stt` aligns to the human sentences
              and whether the ASR hypothesis sits closer to the dialect or the standard
              side (i.e. whether ASR-post-correction is a real task on this data).

    leakage — on a BUILT dataset, check whether speakers are shared across splits. AI-Hub's
              own Training/Validation partition is not speaker-disjoint, so held-out
              metrics are partly measuring memorised speakers.

Usage:
    python scripts/audit_corpus_fields.py fields --data_path data/aihub_samples/71558
    python scripts/audit_corpus_fields.py leakage --raw_dataset_path outputs/dialect_raw_new
"""

from __future__ import annotations

import json
import logging
import os
import statistics
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import fire  # noqa: E402

from ko_dialect.ids import speakers_of  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("audit_corpus_fields")

# Per-sentence annotation blocks, all keyed by sentenceId.
SENTENCE_ANNOTATIONS = ("grammarTypes", "intents", "emotions")


def _time_to_seconds(stamp: str | None) -> float:
    if not stamp:
        return -1.0
    try:
        hours, minutes, seconds = stamp.split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except (ValueError, AttributeError):
        return -1.0


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _top(counter: Counter, k: int = 8) -> dict[str, int]:
    return dict(counter.most_common(k))


def fields(
    data_path: str = "data/aihub_samples/71558",
    max_files: int = 200,
    show_examples: int = 3,
    json_out: str | None = None,
) -> None:
    """Scan raw labelling JSON and report every reusable field.

    Args:
        data_path: Directory containing unzipped AI-Hub labelling JSON.
        max_files: Cap on files scanned (deterministic: sorted order, first N).
        show_examples: How many STT-vs-human triples to print.
        json_out: Optional path to dump the summary.
    """
    root = Path(data_path)
    files = sorted(root.rglob("*.json"))[:max_files]
    if not files:
        raise SystemExit(f"No JSON under {root}. Download a sample first (aihub_fetch.py).")
    logger.info("scanning %d files under %s", len(files), root)

    counters: dict[str, Counter] = {
        name: Counter()
        for name in (
            "grammarTypes",
            "intents",
            "emotions",
            "domain",
            "speechType",
            "gender",
            "birth_decade",
            "job",
            "recognizer",
        )
    }
    presence = Counter()
    durations: list[float] = []
    speakers: set[str] = set()
    stt_pairs: list[tuple[str, str, str]] = []
    n_sentences = 0

    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — malformed AI-Hub JSON is expected and counted
            presence["unparseable_file"] += 1
            continue
        if not isinstance(data, dict):
            presence["unparseable_file"] += 1
            continue

        annotation = data.get("annotation") or {}
        for name in SENTENCE_ANNOTATIONS:
            entries = annotation.get(name) or []
            if entries:
                presence[name] += 1
            for entry in entries:
                counters[name][entry.get("tagType")] += 1

        script = data.get("script") or {}
        if script:
            presence["script"] += 1
            counters["domain"][script.get("domain")] += 1
            counters["speechType"][script.get("speechType")] += 1

        for speaker in data.get("speaker") or []:
            if speaker.get("speakerId"):
                speakers.add(speaker["speakerId"])
            counters["gender"][speaker.get("gender")] += 1
            counters["job"][speaker.get("job")] += 1
            if speaker.get("birthYear"):
                counters["birth_decade"][f"{int(speaker['birthYear']) // 10 * 10}s"] += 1
        if data.get("speaker"):
            presence["speaker"] += 1

        audio = data.get("audio") or {}
        if audio.get("recordDuration"):
            presence["audio"] += 1
            durations.append(float(audio["recordDuration"]))

        sentences = (data.get("transcription") or {}).get("sentences") or []
        n_sentences += len(sentences)

        stt = data.get("stt") or {}
        segments = stt.get("segments") or []
        if segments:
            presence["stt"] += 1
            counters["recognizer"][stt.get("recognizer")] += 1
        for sentence in sentences:
            start = _time_to_seconds(sentence.get("startTime"))
            end = _time_to_seconds(sentence.get("endTime"))
            dialect = (sentence.get("dialect") or "").strip()
            standard = (sentence.get("standard") or "").strip()
            if not (dialect and standard and start >= 0):
                continue
            covered = [
                seg.get("value", "")
                for seg in segments
                if start <= _time_to_seconds(seg.get("startTime")) <= end + 0.01
            ]
            hypothesis = " ".join(v for v in covered if v).strip()
            if hypothesis:
                stt_pairs.append((hypothesis, dialect, standard))

    n_files = len(files)
    print("\n" + "=" * 78)
    print(f"CORPUS FIELD AUDIT  {root}   ({n_files} files, {n_sentences:,} sentences)")
    print("=" * 78)

    print("\n### 필드 존재율 (파일 기준)")
    for name in ("stt", "speaker", "script", "audio", *SENTENCE_ANNOTATIONS):
        print(f"  {name:14s}: {presence[name]:5d}/{n_files}  ({presence[name] / n_files:6.1%})")
    if presence["unparseable_file"]:
        print(f"  {'unparseable':14s}: {presence['unparseable_file']:5d}/{n_files}")

    print("\n### 미사용 라벨 분포")
    for name in ("grammarTypes", "intents", "emotions", "domain", "speechType"):
        print(f"  {name:14s}: {_top(counters[name])}")

    print("\n### 화자 인구통계")
    print(f"  unique speakers : {len(speakers):,}")
    for name in ("gender", "birth_decade"):
        print(f"  {name:14s}: {_top(counters[name])}")
    print(f"  {'job(top5)':14s}: {_top(counters['job'], 5)}")
    if durations:
        print(
            f"  {'recordDuration':14s}: mean={statistics.mean(durations):.1f}s "
            f"median={statistics.median(durations):.1f}s n={len(durations)}"
        )

    summary: dict[str, Any] = {
        "data_path": str(root),
        "files": n_files,
        "sentences": n_sentences,
        "presence": dict(presence),
        "distributions": {k: dict(v) for k, v in counters.items()},
        "unique_speakers": len(speakers),
    }

    print("\n### ASR 후처리 교정(GER) 가능성 — stt vs 사람 전사")
    if not stt_pairs:
        print("  정렬된 STT 가설 없음 → 이 트리에서는 GER 태스크 불가")
    else:
        to_dialect = [_similarity(h, d) for h, d, _ in stt_pairs]
        to_standard = [_similarity(h, s) for h, _, s in stt_pairs]
        closer_to_dialect = sum(1 for a, b in zip(to_dialect, to_standard, strict=True) if a > b)
        print(f"  recognizer      : {_top(counters['recognizer'], 3)}")
        print(f"  정렬된 문장쌍   : {len(stt_pairs):,} / {n_sentences:,} 문장")
        print(
            f"  STT~방언 유사도 : mean={statistics.mean(to_dialect):.3f} "
            f"median={statistics.median(to_dialect):.3f}"
        )
        print(
            f"  STT~표준 유사도 : mean={statistics.mean(to_standard):.3f} "
            f"median={statistics.median(to_standard):.3f}"
        )
        print(f"  STT가 방언쪽에 더 가까운 비율: {closer_to_dialect / len(stt_pairs):.1%}")
        print(
            "\n  읽는 법: 상용 ASR 가설이 방언 쪽에 붙어 있고 표준과 멀수록,\n"
            "  'ASR 출력 -> 표준어' 교정이 실제로 존재하는 태스크라는 뜻이다\n"
            "  (입력이 사람 전사가 아니라 시스템이 실제로 내놓는 문자열이므로)."
        )
        summary["ger"] = {
            "aligned_pairs": len(stt_pairs),
            "sim_to_dialect_mean": statistics.mean(to_dialect),
            "sim_to_standard_mean": statistics.mean(to_standard),
            "closer_to_dialect_ratio": closer_to_dialect / len(stt_pairs),
        }
        for hypothesis, dialect, standard in stt_pairs[:show_examples]:
            print(f"\n  STT  : {hypothesis[:95]}")
            print(f"  방언 : {dialect[:95]}")
            print(f"  표준 : {standard[:95]}")

    if json_out:
        out = Path(json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("wrote %s", out)


def leakage(
    raw_dataset_path: str = "outputs/dialect_raw_new",
    json_out: str | None = None,
) -> None:
    """Check whether speakers are shared between splits of a built dataset.

    AI-Hub ships its own Training/Validation directories, and `prepare_data.py` inherits
    that partition verbatim. If a speaker appears on both sides, held-out scores partly
    reward memorising that speaker's idiolect rather than generalising to a new one.
    """
    from datasets import load_from_disk

    ds_dict = load_from_disk(raw_dataset_path)
    per_split: dict[str, set[str]] = {}
    rows: dict[str, int] = {}
    for split, dataset in ds_dict.items():
        ids = dataset["id"]
        found: set[str] = set()
        for sample_id in ids:
            found.update(speakers_of(sample_id))
        per_split[split] = found
        rows[split] = len(ids)

    print("\n" + "=" * 78)
    print(f"SPEAKER LEAKAGE  {raw_dataset_path}")
    print("=" * 78)
    for split, found in per_split.items():
        print(f"  {split:6s}: {rows[split]:>9,} rows, {len(found):>6,} speakers")
    if not any(per_split.values()):
        print("\n  화자 id를 파싱하지 못했다 (old 포맷은 id에 speaker가 없다).")
        return

    summary: dict[str, Any] = {"raw_dataset_path": raw_dataset_path, "splits": {}}
    splits = list(per_split)
    for i, left in enumerate(splits):
        for right in splits[i + 1 :]:
            shared = per_split[left] & per_split[right]
            denom = min(len(per_split[left]), len(per_split[right])) or 1
            print(
                f"\n  {left} ∩ {right}: {len(shared):,} speakers ({len(shared) / denom:.1%} of {right})"
            )
            if shared:
                print(f"    examples: {sorted(shared)[:6]}")
                print(
                    "    → 이 split으로 잰 held-out 점수는 그만큼 부풀려져 있다.\n"
                    "      화자 분리 재분할은 scripts/resplit_speaker_disjoint.py 참조."
                )
            summary["splits"][f"{left}|{right}"] = {
                "shared": len(shared),
                "ratio": len(shared) / denom,
            }

    if json_out:
        out = Path(json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("wrote %s", out)


if __name__ == "__main__":
    fire.Fire({"fields": fields, "leakage": leakage})
