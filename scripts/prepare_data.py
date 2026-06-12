from __future__ import annotations

import json
import os
import random
import re
import shutil
import signal
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import suppress
from enum import StrEnum, auto
from pathlib import Path

import fire
from datasets import Dataset, DatasetDict
from tqdm.auto import tqdm

from ko_dialect.data import prosody as data_prosody

PROSODY_MARKER_POLICY = data_prosody.PROSODY_MARKER_POLICY
prosody_marker = data_prosody.prosody_marker
summarize_intonation = data_prosody.summarize_intonation

# ---------------------------------------------------------------------------
# 🔥 High-Performance JSON Engine Auto-Fallback Setup
# ---------------------------------------------------------------------------
_USING_ORJSON = True
try:
    import orjson

    load_json = orjson.loads
    dump_json = orjson.dumps

    def dump_jsonl(x, **kwargs):
        return orjson.dumps(x) + b"\n"

except ImportError:
    _USING_ORJSON = False

    def _get_dump_jsonl_fn(fn: Callable) -> Callable:
        def dump_jsonl(x, **kwargs):
            ensure_ascii = kwargs.pop("ensure_ascii", False)
            return fn(x, ensure_ascii=ensure_ascii, **kwargs) + "\n"

        return dump_jsonl

    try:
        import ujson

        load_json = ujson.loads
        dump_json = ujson.dumps
        dump_jsonl = _get_dump_jsonl_fn(ujson.dumps)
    except ImportError:
        import json

        load_json = json.loads
        dump_json = json.dumps
        dump_jsonl = _get_dump_jsonl_fn(json.dumps)


class SpeechKind(StrEnum):
    READ = auto()
    SAY = auto()
    TALK = auto()
    UNKNOWN = auto()


def get_json_files(data_path: os.PathLike | str, glob_pattern: str = "**/*.json") -> list[Path]:
    return list(Path(data_path).glob(glob_pattern))


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def t2s(t: str) -> float:
    """'00:00:01.230' -> 1.23"""
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def get_annotation(data: dict, sent_id: str | None, key: str) -> str | None:
    items = data.get("annotation", {}).get(key, [])
    for it in items:
        if it.get("sentenceId") == sent_id:
            return it.get("tagType")
    return None


def write_json_file(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
            data = load_json(s)
        except Exception:

            def clean_json(s: str) -> str:
                s = s.replace("\n", "").replace("\t", "")
                s = re.sub(r", +}", ",}", s)
                s = re.sub(r", +]", ",]", s)
                s = s.replace(",}", "}").replace(",]", "]")
                s = s.replace("'", '"').replace(".,", ",")

                # Ignore error case ( e.g., ],,"transcriptionAnnotations" )
                # TL_02._경상도_01._1인발화_따라말하기/st_set1_collectorgs100_speakergs442_54_10.json
                s = re.sub(r'([}\]])\s*,\s*,\s*(?="[^"]+"\s*:)', r"\1,", s)
                return s

            try:
                data = load_json(clean_json(s))
            except Exception:
                return []

    if not isinstance(data, dict):
        return []

    if use_old_format:
        return _process_old_single_json(path, data)

    m = re.compile(r"_([가-힣]+도)_").search(str(path))
    region = REGION_MAP.get(m.group(1), m.group(1)) if m else "unknown"
    split = "unknown"
    for part in path.parts:
        if part == "Training":
            split = "train"
            break
        elif part == "Validation":
            split = "valid"
            break

    speech_kind = SpeechKind.UNKNOWN.value
    if path.stem.startswith("st_"):
        speech_kind = SpeechKind.READ.value
    elif path.stem.startswith("say_"):
        speech_kind = SpeechKind.SAY.value
    elif path.stem.startswith("talk_"):
        speech_kind = SpeechKind.TALK.value

    sentences = data.get("transcription", {}).get("sentences", [])
    segments = data.get("transcription", {}).get("segments", [])
    for segment in segments:
        # Ignore error cases ( "startTime": None, "endTime": None )
        # TL_01._강원도_01._1인발화_따라말하기/st_set1_collectorgw132_speakergw2151_72_11.json
        # VL_02._경상도_01._1인발화_따라말하기/st_set3_collectorgs198_speakergs1537_27_9.json
        # VL_02._경상도_01._1인발화_따라말하기/st_set3_collectorgs244_speakergs2933_29_8.json
        st_raw = segment.get("startTime")
        segment["_start_seconds"] = t2s(st_raw) if st_raw else -1.0

    samples = []
    for sentence in sentences:
        standard = (sentence.get("standard") or "").strip()
        dialect = (sentence.get("dialect") or "").strip()

        if not (standard and dialect):
            continue

        # 어절 단위 방언 매핑 (Double-ptr)
        s_time, e_time = t2s(sentence["startTime"]), t2s(sentence["endTime"])
        sentence_segments = [
            segment for segment in segments if s_time <= segment["_start_seconds"] <= e_time + 0.01
        ]
        standard_words = (sentence.get("standard") or "").split()
        dialect_words = (sentence.get("dialect") or "").split()
        s_ptr, d_ptr = 0, 0

        dialect_eojeol_map = []
        for seg in sentence_segments:
            seg_std_str = seg.get("standard")
            seg_dia_str = seg.get("dialect") or ""

            # Prevent ghost space edge cases
            # ( e.g., st_set2_collectorgs444_speakergs3971_38_11_1, st_set1_collectorgs384_speakergs4099_23_5_1, st_set1_collectorgs384_speakergs3052_25_8_1 )
            if seg_std_str is not None:
                seg_std_str = seg_std_str.strip()
            seg_dia_str = seg_dia_str.strip()

            seg_dia_words = seg_dia_str.split()

            # 1. 표준어 맵핑이 존재하는 사투리 변경 세그먼트인 경우
            if seg_std_str is not None:
                seg_std_words = seg_std_str.split()

                # 표준어 단어 배열에서 현재 세그먼트 단어들의 시작 위치 검색
                match_s = -1
                for i in range(s_ptr, len(standard_words) - len(seg_std_words) + 1):
                    if standard_words[i : i + len(seg_std_words)] == seg_std_words:
                        match_s = i
                        break

                # 사투리 단어 배열에서 현재 세그먼트 단어들의 시작 위치 검색
                match_d = -1
                for j in range(d_ptr, len(dialect_words) - len(seg_dia_words) + 1):
                    if dialect_words[j : j + len(seg_dia_words)] == seg_dia_words:
                        match_d = j
                        break

                # 양쪽 다 정확한 위치를 찾은 경우에만 맵에 주입
                if match_s != -1 and match_d != -1:
                    # 단어 수가 일치하는 일반적인 사투리 변경 (1:1 unrolling)
                    if len(seg_std_words) == len(seg_dia_words):
                        for k in range(len(seg_std_words)):
                            dialect_eojeol_map.append(
                                {
                                    "standard_idx": match_s + k,
                                    "dialect_idx": match_d + k,
                                    "dialect": seg_dia_words[k],
                                    "standard": seg_std_words[k],
                                    "pronunciation": seg.get("pronunciation"),
                                }
                            )
                    else:
                        # 축약/늘림 현상 발발 시 (ex: 가 버리고 -> 가삐고) [N:M 매핑]
                        dialect_eojeol_map.append(
                            {
                                "standard_idx": match_s,
                                "dialect_idx": match_d,
                                "dialect": seg_dia_str,
                                "standard": seg_std_str,
                                "pronunciation": seg.get("pronunciation"),
                            }
                        )

                    # 매칭된 길이만큼 포인터를 정확하게 전진
                    s_ptr = match_s + len(seg_std_words)
                    d_ptr = match_d + len(seg_dia_words)
            else:
                # 2. 표준어 필드가 없는 '동일 어절' 세그먼트인 경우
                # 포인터 싱크를 유지하기 위해 단어가 매칭되는 지점만큼만 포인터들을 전진
                match_s = -1
                for i in range(s_ptr, len(standard_words) - len(seg_dia_words) + 1):
                    if standard_words[i : i + len(seg_dia_words)] == seg_dia_words:
                        match_s = i
                        break
                match_d = -1
                for j in range(d_ptr, len(dialect_words) - len(seg_dia_words) + 1):
                    if dialect_words[j : j + len(seg_dia_words)] == seg_dia_words:
                        match_d = j
                        break

                if match_s != -1:
                    s_ptr = match_s + len(seg_dia_words)
                if match_d != -1:
                    d_ptr = match_d + len(seg_dia_words)

        # 운율 요약 (sentence-level) + 어절 단위 운율 (Phase-2: F0 series sliced per segment)
        intonations = sentence.get("intonations", [])
        prosody = summarize_intonation(intonations)
        marker = prosody_marker(prosody)

        eojeols = []
        for seg in sentence_segments:
            seg_dia = (seg.get("dialect") or "").strip()
            seg_st, seg_et = seg.get("startTime"), seg.get("endTime")
            if not (seg_dia and seg_st and seg_et):
                continue
            eojeols.append({"word": seg_dia, "start_s": t2s(seg_st), "end_s": t2s(seg_et)})
        dialect_eojeol_prosody = data_prosody.eojeol_prosody_markers(
            intonations, s_time, e_time, eojeols
        )

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
                "prosody_marker": marker,
                "dialect_eojeol_prosody": dialect_eojeol_prosody,
                "intent": get_annotation(data, sentence.get("sentenceId"), "intents"),
                "emotion": get_annotation(data, sentence.get("sentenceId"), "emotions"),
            }
        )
    return samples


def _process_old_single_json(path: os.PathLike | str, data: dict) -> list[dict]:
    # Extract region, split from path string:
    #   한국어 방언 발화({REGION})/{SPLIT}/*.json
    path = Path(path)
    re_region = re.compile(r"한국어 방언 발화\((.+?)\)")
    split_map = {"Training": "train", "Validation": "valid"}
    m = re_region.search(str(path))
    region = REGION_MAP.get(m.group(1), m.group(1)) if m else "unknown"
    split = split_map.get(path.parent.name, path.parent.name.lower())

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
                "speech_kind": SpeechKind.READ.value,
                "standard": standard,
                "dialect": dialect,
                "is_identical": standard == dialect,
                "dialect_eojeol_map": dialect_eojeol_map,
                "intent": None,
                "emotion": None,
            }
        )
    return samples


def _process_chunk_worker(
    file_chunk: list[Path],
    tmp_dir: Path,
    **fn_kwargs,
) -> list[dict]:
    encoding = fn_kwargs.get("encoding", "utf-8")
    local_handles = {}
    local_failed = []

    mode = "ab" if _USING_ORJSON else "a"
    open_kwargs = {} if _USING_ORJSON else {"encoding": encoding}

    pid = os.getpid()
    for split in ["train", "valid", "unknown"]:
        local_handles[split] = open(tmp_dir / f"{split}_worker_{pid}.jsonl", mode, **open_kwargs)

    for f in file_chunk:
        try:
            result = _process_single_json(f, **fn_kwargs)
            if not result:
                local_failed.append({"file": str(f), "error": "empty or invalid"})
                continue
            for sample in result:
                split = sample.get("split", "unknown")
                local_handles[split].write(dump_jsonl(sample))
        except Exception as e:
            local_failed.append({"file": str(f), "error": str(e)})

    for fh in local_handles.values():
        fh.close()
    return local_failed


def _init_worker() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)


# ---------------------------------------------------------------------------
# Main Pipeline Orchestrator
# ---------------------------------------------------------------------------


def prepare_dialect_dataset(
    files: list[Path],
    output_dir: Path,
    **kwargs,
) -> DatasetDict:
    start_time = time.perf_counter()

    verbose = kwargs.pop("verbose", False)
    use_old_format = kwargs.get("use_old_format", False)

    tmp_dir = output_dir / "_jsonl_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    failed: list[dict] = []

    def _cleanup_resources(delete_tmp: bool = False):
        if delete_tmp and tmp_dir.exists():
            with suppress(Exception):
                shutil.rmtree(tmp_dir)

    def _run_parallel(max_workers: int | None = None, chunk_size: int = 1000, **fn_kwargs):
        max_workers = max_workers or os.cpu_count() or 4

        # 태스크 균등 분할
        chunks = [files[i : i + chunk_size] for i in range(0, len(files), chunk_size)]
        executor = ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker)

        futures = {
            executor.submit(_process_chunk_worker, chunk, tmp_dir, **fn_kwargs): len(chunk)
            for i, chunk in enumerate(chunks)
        }
        pbar = tqdm(total=len(files), desc="Processing Files", disable=not verbose)

        try:
            for future in as_completed(futures):
                local_failed = future.result()
                if local_failed:
                    failed.extend(local_failed)
                pbar.update(futures[future])
        except KeyboardInterrupt:
            tqdm.write(
                "\n🚨 [정지] 사용자가 중단했습니다. 자식 프로세스 청소 후 즉시 강제 종료합니다..."
            )
            executor.shutdown(wait=False, cancel_futures=True)
            _cleanup_resources(delete_tmp=True)
            os._exit(1)
        finally:
            pbar.close()
            executor.shutdown(wait=True)

    try:
        _run_parallel(**kwargs)

        if failed:
            failed_path = output_dir / "failed_files.json"
            with open(failed_path, "w", encoding="utf-8") as f:
                if _USING_ORJSON:
                    f.write(orjson.dumps(failed, option=orjson.OPT_INDENT_2).decode("utf-8"))
                else:
                    import json

                    json.dump(failed, f, ensure_ascii=False, indent=2)
            tqdm.write(f"[Warning] {len(failed)} failed files → {failed_path}")

        # 🔄 [Reduce]
        splits_found = []
        for split in ["train", "valid"]:
            worker_files = list(tmp_dir.glob(f"{split}_worker_*.jsonl"))
            if not worker_files:
                continue
            splits_found.append(split)

            with open(tmp_dir / f"{split}.jsonl", "wb") as master_f:
                for wf in worker_files:
                    with open(wf, "rb") as f:
                        shutil.copyfileobj(f, master_f)  # 커널 레벨 고속 버퍼 스트리밍
                    wf.unlink()

        dataset = DatasetDict(
            {split: Dataset.from_json(str(tmp_dir / f"{split}.jsonl")) for split in splits_found}
        )
        save_path = output_dir / f"dialect_raw_{'old' if use_old_format else 'new'}"
        dataset.save_to_disk(str(save_path))
        write_json_file(
            save_path.with_name(f"{save_path.name}_manifest.json"),
            {
                "dataset": save_path.name,
                "source_format": "old" if use_old_format else "new",
                "prosody_marker_policy": (None if use_old_format else PROSODY_MARKER_POLICY),
            },
        )

        elapsed = time.perf_counter() - start_time
        print(f"Saved {sum(len(v) for v in dataset.values())} samples → {save_path}")
        print(f"⏱️ Total Elapsed Time: {int(elapsed // 60)}m {elapsed % 60:.2f}s")
        return dataset
    finally:
        _cleanup_resources(delete_tmp=True)


def main(
    data_path: os.PathLike | str = None,
    output_dir: os.PathLike | str = None,
    use_old_format: bool = False,
    max_workers: int | None = None,
    verbose: bool = False,
    speedrun: bool = False,
    n_samples: int | None = None,
    chunk_size: int = 1000,
    encoding: str = "utf-8",
    **kwargs,
) -> None:
    data_path = Path(data_path or Path(__file__).parents[1] / "data")
    output_dir = Path(output_dir or Path(__file__).parents[1] / "outputs")

    if n_samples or speedrun:
        speedrun = True
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

    prepare_dialect_dataset(
        files,
        output_dir,
        max_workers=max_workers,
        verbose=verbose,
        encoding=encoding,
        use_old_format=use_old_format,
        chunk_size=chunk_size,
    )


if __name__ == "__main__":
    fire.Fire(main)
