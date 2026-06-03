from __future__ import annotations

import json
import os
import random
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import StrEnum, auto
from json import JSONDecodeError
from pathlib import Path
from typing import IO, Callable

import fire
from datasets import Dataset, DatasetDict, load_from_disk
from tqdm.auto import tqdm


class SpeechKind(StrEnum):
    READ = auto()
    SAY = auto()
    TALK = auto()
    UNKNOWN = auto()


def get_json_files(
    data_path: os.PathLike | str,
    glob_pattern: str = "**/*.json",
) -> list[Path]:
    data_path = Path(data_path)
    return list(data_path.glob(glob_pattern))


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def t2s(t: str) -> float:
    """'00:00:01.230' -> 1.23"""
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def filter_segments_by_time(segments, start, end):
    s, e = t2s(start), t2s(end)
    return [seg for seg in segments if s <= t2s(seg["startTime"]) <= e + 0.01]


def summarize_intonation(intonations: list[float]) -> dict | None:
    """raw F0 시계열 -> 요약 통계 (None 값/0 값 제외)"""
    valid = [p for p in intonations if p and p > 30]  # 유효 피치만
    if len(valid) < 3:
        return None

    return {
        "f0_mean": round(sum(valid) / len(valid), 2),
        "f0_std": round(
            (sum((p - sum(valid) / len(valid)) ** 2 for p in valid) / len(valid)) ** 0.5, 2
        ),
        "f0_start": round(valid[0], 2),
        "f0_end": round(valid[-1], 2),
        "f0_delta": round((valid[-1] - valid[0]) / valid[0], 3) if valid[0] > 0 else 0,
    }


def get_annotation(data: dict, sent_id: str | None, key: str) -> str | None:
    items = data.get("annotation", {}).get(key, [])
    for it in items:
        if it.get("sentenceId") == sent_id:
            return it.get("tagType")
    return None


# ---------------------------------------------------------------------------
# Main functions
# ---------------------------------------------------------------------------
REGION_MAP = {
    "강원도": "gangwondo",
    "전라도": "jeollado",
    "제주도": "jejudo",
    "충청도": "chungcheongdo",
    "경상도": "gyeongsangdo",
}


def _process_single_json(
    path: os.PathLike | str,
    use_old_format: bool = False,
    encoding: str = "utf-8",
) -> list[dict]:
    path: Path = Path(path)
    with open(path, encoding=encoding) as f:
        s = f.read()
        try:
            data = json.loads(s)
        except JSONDecodeError:

            def clean_json(s: str) -> str:
                s = s.replace("\n", "").replace("\t", "")
                s = re.sub(r", +}", ",}", s)
                s = re.sub(r", +]", ",]", s)
                s = s.replace(",}", "}").replace(",]", "]")
                s = s.replace("'", '"').replace(".,", ",")
                return s

            data = json.loads(clean_json(s))

    if not isinstance(data, dict):
        return []

    if use_old_format:
        return _process_old_single_json(path, data)

    m = re.compile(r"_([가-힣]+도)_").search(str(path))
    region = REGION_MAP.get(m.group(1), m.group(1)) if m else "unknown"
    for part in path.parts:
        if part == "Training":
            split = "train"
            break
        elif part == "Validation":
            split = "valid"
            break

    if path.stem.startswith("st_"):
        speech_kind = SpeechKind.READ.value
    elif path.stem.startswith("say_"):
        speech_kind = SpeechKind.SAY.value
    elif path.stem.startswith("talk_"):
        speech_kind = SpeechKind.TALK.value

    samples = []
    sentences = data.get("transcription", {}).get("sentences", [])
    segments = data.get("transcription", {}).get("segments", [])

    for sentence in sentences:
        standard = (sentence.get("standard") or "").strip()
        dialect = (sentence.get("dialect") or "").strip()

        if not (standard and dialect):
            continue

        # 어절 단위 방언 매핑
        sentence_segments = filter_segments_by_time(
            segments, sentence["startTime"], sentence["endTime"]
        )

        dialect_eojeol_map = [
            {
                "idx": int(segment["orderInFile"]) - 1,
                "dialect": segment["dialect"],
                "standard": segment["standard"],
                "pronunciation": segment.get("pronumciation"),
            }
            for segment in sentence_segments
            if segment.get("standard") is not None
        ]

        # 운율 요약
        prosody = summarize_intonation(sentence.get("intonations", []))

        sentence_id = (
            f"{data.get('fileName', os.path.basename(path))}_{int(sentence.get('sentenceId', 0))}"
        )
        samples.append(
            {
                "id": sentence_id,
                "do": region,
                "split": split,
                "speech_kind": speech_kind,
                "standard": standard,
                "dialect": dialect,
                "is_identical": standard == dialect,
                "dialect_eojeol_map": dialect_eojeol_map,
                "prosody": prosody,
                "intent": get_annotation(data, sentence.get("sentenceId"), "intents"),
                "emotion": get_annotation(data, sentence.get("sentenceId"), "emotions"),
            }
        )
    return samples


def _process_old_single_json(path: os.PathLike | str, data: dict) -> list[dict]:
    # Extract region, split from path string:
    #   한국어 방언 발화({REGION})/{SPLIT}/*.json
    path = Path(path)
    _RE_REGION = re.compile(r"한국어 방언 발화\((.+?)\)")
    _SPLIT_MAP = {"Training": "train", "Validation": "valid"}
    m = _RE_REGION.search(str(path))
    region = REGION_MAP.get(m.group(1), m.group(1)) if m else "unknown"
    split = _SPLIT_MAP.get(path.parent.name, path.parent.name.lower())

    speech_kind = SpeechKind.READ.value

    # Construct samples
    samples = []
    for utterance in data["utterance"]:
        standard = (utterance.get("standard_form") or "").strip()
        dialect = (utterance.get("dialect_form") or "").strip()

        if not (standard and dialect):
            continue

        dialect_eojeol_map = [
            {
                "idx": int(eojeol["id"]),
                "dialect": eojeol["eojeol"],
                "standard": eojeol["standard"],
                "pronunciation": None,
            }
            for eojeol in utterance["eojeolList"]
            if eojeol.get("standard") is not None
        ]
        samples.append(
            {
                "id": utterance["id"],
                "do": region,
                "split": split,
                "speech_kind": speech_kind,
                "standard": standard,
                "dialect": dialect,
                "is_identical": standard == dialect,
                "dialect_eojeol_map": dialect_eojeol_map,
                "intent": None,
                "emotion": None,
            }
        )
    return samples


def prepare_dialect_dataset(
    files: list[Path],
    output_dir: Path,
    **kwargs,
) -> DatasetDict:
    verbose = kwargs.pop("verbose", False)
    tmp_dir = output_dir / "_jsonl_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # file_handles: split name → open file handle, created on first encounter
    file_handles: dict[str, IO[str]] = {}
    failed: list[dict] = []
    n_written: int = 0

    def _run_parallel(fn: Callable, max_workers: int | None = None, **fn_kwargs):
        nonlocal n_written
        encoding = fn_kwargs.get("encoding", "utf-8")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(fn, f, **fn_kwargs): f for f in files}
            pbar = tqdm(total=len(files), desc="Processing", disable=not verbose)
            for future in as_completed(future_to_file):
                src = future_to_file[future]
                try:
                    result = future.result()
                    if not result:
                        failed.append({"file": str(src), "error": "empty or invalid"})
                        continue
                    for sample in result:
                        split = sample["split"]
                        if split not in file_handles:
                            file_handles[split] = open(
                                tmp_dir / f"{split}.jsonl", "a", encoding=encoding
                            )
                        file_handles[split].write(json.dumps(sample, ensure_ascii=False) + "\n")
                        n_written += 1
                except Exception as e:
                    failed.append({"file": str(src), "error": str(e)})
                    if verbose:
                        tqdm.write(f"[Error] {src}: {e}")
                finally:
                    pbar.update(1)
            pbar.close()

    try:
        _run_parallel(fn=_process_single_json, **kwargs)
    finally:
        for fh in file_handles.values():
            fh.close()

    total = len(files)
    print(
        f"Done: {n_written} samples from {total - len(failed)}/{total} files"
        f" ({len(failed)} failed)"
    )

    if failed:
        failed_path = output_dir / "failed_files.json"
        with open(failed_path, "w", encoding="utf-8") as f:
            json.dump(failed, f, ensure_ascii=False, indent=2)
        tqdm.write(f"[Warning] {len(failed)} failed files → {failed_path}")

    dataset = DatasetDict(
        {split: Dataset.from_json(str(tmp_dir / f"{split}.jsonl")) for split in file_handles}
    )
    save_path = output_dir / "dialect_raw"
    dataset.save_to_disk(str(save_path))
    print(f"Saved {sum(len(v) for v in dataset.values())} samples → {save_path}")

    shutil.rmtree(tmp_dir)
    return dataset


def main(
    data_path: os.PathLike | str = None,
    output_dir: os.PathLike | str = None,
    use_old_format: bool = False,
    max_workers: int | None = None,
    verbose: bool = False,
    speedrun: bool = False,
    n_samples: int | None = None,
    encoding: str = "utf-8",
    **kwargs,
) -> None:
    data_path = Path(
        data_path or Path(__file__).parents[1] / "data"
    ) # fmt: skip
    output_dir = Path(
        output_dir or Path(__file__).parents[1] / "outputs"
    ) # fmt: skip

    if n_samples or speedrun:
        n_samples = n_samples or 100

    if speedrun:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_dir / f"speedrun_{n_samples}_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    if max_workers is None:
        cpu_count = os.cpu_count() or 4
        max_workers = min(16, cpu_count)

    # Prevent "Unexpected UTF-8 BOM (decode using utf-8-sig)" JSONDecodeError
    if use_old_format:
        encoding = "utf-8-sig"

    files = get_json_files(data_path)
    if not files:
        raise FileNotFoundError(f"No JSON files found under {data_path!r}")
    if speedrun:
        files = random.sample(files, k=min(n_samples, len(files)))

    dataset = prepare_dialect_dataset(
        files,
        output_dir,
        max_workers=max_workers,
        verbose=verbose,
        encoding=encoding,
        use_old_format=use_old_format,
    )


if __name__ == "__main__":
    fire.Fire(main)
