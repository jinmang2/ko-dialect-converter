#!/usr/bin/env python3
"""wandb_doctor — diagnose "metrics not showing up in W&B" problems.

This is a *teaching* script as much as a tool. It was distilled from a real
debugging session where a training run showed up in the W&B dashboard with its
config + summary, but every history panel said "There's no data for the
selected runs." The culprit turned out to be a **server-side metrics-ingestion
incident**, not the training code — but proving that required peeling back
several layers of the W&B stack.

The key idea this script teaches: **separate the client from the server.**

    [your code] --wandb.log()--> [local .wandb transaction log]
                                          |
                                          | filestream upload (HTTP)
                                          v
                                 [W&B backend] --> summary (fast path)
                                               --> history (metered path)
                                               --> console logs

If the *local* transaction log contains your metrics but the *server* won't
return them, the client did its job and the problem is downstream (quota,
auth, or a platform incident). If the local log is empty, the bug is in your
code/config. This script lets you inspect every box in that diagram.

Subcommands
-----------
  versions                      wandb/transformers versions + package origin
                                wandb/transformers 버전 + 패키지 진위(METADATA) + wandb-core 바이너리
  local   <run_dir>             parse the local .wandb log, dump history rows
                                로컬 .wandb 트랜잭션 로그 파싱 → 클라이언트가 실제로 기록했나
  server  <entity/proj/run_id>  query the backend: summary vs history mismatch
                                백엔드 조회 → summary 있음 + lastHistoryStep 증가 + scan_history 0행 핑거프린트
  compare <refA> <refB>         diff a "working" run against a "broken" one
                                정상 run vs 깨진 run 비교
  account                       entity / plan / usage via GraphQL
                                entity/플랜/usage GraphQL
  status                        fetch the public W&B status page (incidents)
                                status.wandb.com JSON API로 진행 중 장애 조회
  probe                         create a tiny live run and verify retention
                                일회용 run 로깅 후 즉시 read-back (지금 로깅 되나?)
  diagnose <entity/proj/run_id> orchestrate the above into a single verdict
                                versions→local→server→status 한 방에 + 판정 가이드

Run any subcommand with -h for details. Nothing here mutates your runs
(except `probe`, which creates—and by default deletes—one throwaway run).

Stdlib corners worth studying (grep for "LEARN:"): netrc, itertools.islice,
urllib.request, and the W&B internals wandb.sdk.internal.datastore +
wandb.proto.wandb_internal_pb2.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys
import time
import urllib.request
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Small output helpers (no external deps — keep this script runnable anywhere).
# ---------------------------------------------------------------------------


def _h(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 4} {title} {'=' * max(4, 72 - len(title))}")


def _kv(key: str, value: Any) -> None:
    print(f"  {key:<28} {value}")


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def read_wandb_key(host: str = "api.wandb.ai") -> str | None:
    """Return the W&B API key from ~/.netrc *without ever printing it*.

    LEARN: ``netrc`` parses the classic ``~/.netrc`` file that tools like curl,
    git, and wandb use to store per-host credentials. Format:

        machine api.wandb.ai
          login user
          password <40-char-key>

    ``authenticators(host)`` returns a ``(login, account, password)`` triple,
    so the key is index 2. We return it so callers can re-auth programmatically,
    but we deliberately never echo it to stdout.
    """
    import netrc  # stdlib; imported lazily so `versions` works even if absent

    path = os.path.expanduser("~/.netrc")
    if not os.path.exists(path):
        return None
    try:
        auth = netrc.netrc(path).authenticators(host)
    except Exception:
        return None
    return auth[2] if auth else None


# ---------------------------------------------------------------------------
# 1) versions / package authenticity
# ---------------------------------------------------------------------------


def cmd_versions(_args: argparse.Namespace) -> None:
    """Print wandb/transformers versions and confirm the package is genuine.

    Why this matters: a surprising version number (e.g. an internal fork) can
    masquerade as the real library. Checking the dist-info METADATA's
    Project-URL / Author-email tells you whether you're on the real PyPI build.
    """
    _h("versions")
    try:
        import wandb

        _kv("wandb", wandb.__version__)
        # Locate the wandb-core binary: modern wandb ships a Go service that
        # actually does the uploading. If it's missing, logging silently no-ops.
        core = Path(wandb.__file__).parent / "bin" / "wandb-core"
        _kv("wandb-core binary", f"{core} ({'present' if core.exists() else 'MISSING'})")
    except Exception as exc:  # pragma: no cover - environment dependent
        _kv("wandb", f"IMPORT FAILED: {exc!r}")
    try:
        import transformers

        _kv("transformers", transformers.__version__)
    except Exception as exc:  # pragma: no cover
        _kv("transformers", f"IMPORT FAILED: {exc!r}")

    # Confirm the installed wandb is the genuine article via its METADATA.
    try:
        from importlib.metadata import metadata

        md = metadata("wandb")
        _kv("wandb author", md.get("Author-email"))
        urls = [v for k, v in md.items() if k == "Project-URL"]
        _kv("wandb project-urls", "; ".join(urls) or "(none — suspicious!)")
    except Exception as exc:  # pragma: no cover
        _kv("metadata", f"unavailable: {exc!r}")


# ---------------------------------------------------------------------------
# 2) local — parse the .wandb transaction log
# ---------------------------------------------------------------------------


def _find_wandb_file(run_dir: str) -> str | None:
    """Locate the ``run-<id>.wandb`` transaction log inside a run directory."""
    p = Path(run_dir)
    # Accept either the run dir, the parent `wandb/` dir, or a direct .wandb file
    if p.is_file() and p.suffix == ".wandb":
        return str(p)
    hits = sorted(glob.glob(str(p / "*.wandb"))) or sorted(glob.glob(str(p / "**" / "*.wandb")))
    return hits[0] if hits else None


def iter_local_records(wandb_file: str) -> Iterator[tuple[str, Any]]:
    """Yield ``(record_type, record)`` for every record in a .wandb log.

    LEARN: the ``.wandb`` file is a LevelDB-style append-only transaction log.
    W&B exposes a reader at ``wandb.sdk.internal.datastore.DataStore`` and the
    record schema as protobuf messages in ``wandb.proto.wandb_internal_pb2``.

      - ``DataStore.open_for_scan(path)`` opens the log for sequential reads.
      - ``scan_data()`` returns the next record's raw bytes (or None at EOF).
      - ``Record.ParseFromString(bytes)`` decodes those bytes.
      - ``Record.WhichOneof("record_type")`` tells you which kind it is:
        history / summary / config / stats / output_raw / run / exit / ...

    This is the *ground truth* of what your client recorded locally — before
    anything touches the network.
    """
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal import datastore

    ds = datastore.DataStore()
    ds.open_for_scan(wandb_file)
    while True:
        raw = ds.scan_data()
        if raw is None:
            break
        rec = pb.Record()
        rec.ParseFromString(raw)
        yield rec.WhichOneof("record_type"), rec


def history_row_to_dict(history_msg: Any) -> dict[str, str]:
    """Flatten a protobuf history record into a plain ``{key: value_json}`` dict.

    Modern wandb stores metric keys as ``nested_key`` lists (e.g.
    ``["train", "loss"]`` shown as ``train/loss``) with the value in
    ``value_json`` (a JSON-encoded string). Older/simple keys use ``key``.
    """
    out: dict[str, str] = {}
    for item in history_msg.item:
        name = "/".join(item.nested_key) if list(item.nested_key) else item.key
        out[name] = item.value_json
    return out


def cmd_local(args: argparse.Namespace) -> None:
    """Inspect the LOCAL transaction log: did the client actually record data?"""
    _h(f"local: {args.run_dir}")
    wf = _find_wandb_file(args.run_dir)
    if not wf:
        print(f"  no .wandb file found under {args.run_dir!r}")
        return
    _kv(".wandb file", wf)
    _kv("size", f"{os.path.getsize(wf):,} bytes")

    counts: dict[str, int] = {}
    shown = 0
    for rtype, rec in iter_local_records(wf):
        counts[rtype] = counts.get(rtype, 0) + 1
        if rtype == "history" and shown < args.rows:
            row = history_row_to_dict(rec.history)
            # Keep only the interesting metric-ish keys for readability
            metrics = {
                k: v
                for k, v in row.items()
                if any(s in k for s in ("loss", "acc", "f1", "global_step", "lr", "learning_rate"))
            }
            print(f"  history[{shown}] {metrics or row}")
            shown += 1

    _h("local record-type counts")
    for rtype, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        _kv(rtype, f"{n:,}")
    # Interpretation hint
    if counts.get("history"):
        print("\n  => Client DID record history locally. If the server shows none,")
        print("     the problem is downstream (server ingestion / quota / incident).")
    else:
        print("\n  => No local history records. The bug is in your CODE/CONFIG")
        print("     (wandb never received metrics — check report_to / wandb.log).")
    if counts.get("output_raw", 0) > 50_000:
        print(f"\n  NOTE: {counts['output_raw']:,} console records — tqdm/progress-bar")
        print("        spam. Unrelated to metrics; set disable_tqdm=True / WANDB_CONSOLE=off.")


# ---------------------------------------------------------------------------
# 3) server — query the backend and detect the summary/history mismatch
# ---------------------------------------------------------------------------


def _take(iterable: Iterable[Any], n: int) -> list[Any]:
    """Return the first ``n`` items of any iterable.

    LEARN: ``itertools.islice(iterable, n)`` lazily takes a prefix without
    materializing the whole sequence. ``scan_history()`` is a generator that
    streams rows from the server one page at a time, so islice avoids pulling
    the entire history just to peek at the first few rows.
    """
    return list(itertools.islice(iterable, n))


def probe_server_run(ref: str, rows: int = 5) -> dict[str, Any]:
    """Return a structured snapshot of how the SERVER sees a run.

    The diagnostic signal is the mismatch between three endpoints:
      - run.summary             -> last value of each metric (fast write path)
      - run.lastHistoryStep     -> how many history steps the server *counted*
      - run.scan_history()      -> the actual retrievable time-series rows
      - run.history()           -> sampled pandas view (different endpoint)

    "summary populated + lastHistoryStep advancing + scan_history empty" is the
    fingerprint of a metered-history problem (ingestion lag, quota, incident).
    """
    import wandb

    api = wandb.Api()
    run = api.run(ref)
    summary_keys = [k for k in run.summary.keys() if not k.startswith("_")]
    scan_rows = _take(run.scan_history(), rows)
    try:
        hist_df = run.history()
        hist_shape = tuple(getattr(hist_df, "shape", ()))
    except Exception as exc:  # pragma: no cover
        hist_shape = (f"error: {exc!r}",)
    return {
        "ref": ref,
        "state": run.state,
        "summary_keys": summary_keys,
        "last_history_step": run.lastHistoryStep,
        "scan_history_rows": len(scan_rows),
        "scan_history_sample": scan_rows[:3],
        "history_df_shape": hist_shape,
    }


def _verdict(snap: dict[str, Any]) -> str:
    has_summary = bool(snap["summary_keys"])
    steps = snap["last_history_step"]
    rows = snap["scan_history_rows"]
    if rows > 0:
        return "OK — server is returning history rows."
    if has_summary and (isinstance(steps, int) and steps >= 0):
        return (
            "SUMMARY-ONLY — server has summary + counts history steps but returns\n"
            "      zero rows. Classic server-side metered-history problem:\n"
            "      ingestion lag / incident / quota. NOT your code (verify with `local`)."
        )
    if not has_summary:
        return "EMPTY — server has no summary either. Run may have failed at init."
    return "INCONCLUSIVE — inspect manually."


def cmd_server(args: argparse.Namespace) -> None:
    _h(f"server: {args.ref}")
    snap = probe_server_run(args.ref, rows=args.rows)
    _kv("state", snap["state"])
    _kv("summary keys", snap["summary_keys"][:12])
    _kv("lastHistoryStep", snap["last_history_step"])
    _kv("scan_history rows", snap["scan_history_rows"])
    _kv("history() shape", snap["history_df_shape"])
    for r in snap["scan_history_sample"]:
        print(f"    row: { {k: v for k, v in r.items() if v is not None} }")
    print(f"\n  VERDICT: {_verdict(snap)}")


# ---------------------------------------------------------------------------
# 4) compare — diff a working run vs a broken one
# ---------------------------------------------------------------------------


def cmd_compare(args: argparse.Namespace) -> None:
    _h("compare (working vs broken)")
    for label, ref in (("A", args.ref_a), ("B", args.ref_b)):
        snap = probe_server_run(ref, rows=3)
        print(f"\n  [{label}] {ref}")
        _kv("  state", snap["state"])
        _kv("  lastHistoryStep", snap["last_history_step"])
        _kv("  scan_history rows", snap["scan_history_rows"])
        _kv("  has summary", bool(snap["summary_keys"]))
    print("\n  If one run returns rows and the other doesn't despite identical code")
    print("  and wandb version, the difference is *when* they ran — i.e. server-side.")


# ---------------------------------------------------------------------------
# 5) account — entity / plan / usage via GraphQL
# ---------------------------------------------------------------------------


def cmd_account(args: argparse.Namespace) -> None:
    """Query the account/entity via GraphQL.

    LEARN: wandb.Api() wraps a GraphQL endpoint. The newer public API exposes
    ``api._service_api.execute_graphql(query)``. You can ask for the viewer's
    orgs/plans and per-entity ``storageBytes`` / ``computeHours``.

    CAVEAT learned the hard way: ``entity.available`` is NOT a quota flag — it
    means "is this entity name free to register". An over-limit entity does not
    flip it. Don't infer quota from ``available``; use storage/usage numbers and
    cross-check by logging a tiny probe to a different entity.
    """
    _h("account")
    import wandb

    api = wandb.Api()
    sa = api._service_api

    def gql(label: str, query: str) -> None:
        try:
            print(f"  {label}: {sa.execute_graphql(query)}")
        except Exception as exc:
            print(f"  {label}: ERR {exc!r}")

    gql("viewer", "query { viewer { username entity admin } }")
    gql(
        "orgs/plans",
        "query { viewer { organizations { name subscriptions { plan { name } } } } }",
    )
    if args.entity:
        gql(
            f"entity({args.entity})",
            f'query {{ entity(name:"{args.entity}") {{ name available storageBytes computeHours }} }}',
        )


# ---------------------------------------------------------------------------
# 6) status — fetch the public W&B status page (statuspage.io JSON API)
# ---------------------------------------------------------------------------


def cmd_status(_args: argparse.Namespace) -> None:
    """Fetch unresolved incidents from status.wandb.com.

    LEARN: statuspage.io-hosted pages expose a stable JSON API at
    ``/api/v2/...`` — far more reliable to parse than scraping HTML.
    ``urllib.request`` (stdlib) is enough; no requests dependency needed.
    """
    _h("W&B platform status")
    base = "https://status.wandb.com/api/v2"
    for name, url in (
        ("overall", f"{base}/status.json"),
        ("unresolved incidents", f"{base}/incidents/unresolved.json"),
    ):
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                data = json.load(resp)
        except Exception as exc:
            print(f"  {name}: fetch failed: {exc!r}")
            continue
        if name == "overall":
            _kv("description", data.get("status", {}).get("description"))
        else:
            incidents = data.get("incidents", [])
            if not incidents:
                print("  no unresolved incidents 🎉")
            for inc in incidents:
                print(f"  • [{inc.get('status')}] {inc.get('name')}")
                print(f"    impact={inc.get('impact')} updated={inc.get('updated_at')}")
                print(f"    {inc.get('shortlink')}")


# ---------------------------------------------------------------------------
# 7) probe — create a tiny live run and verify retention end-to-end
# ---------------------------------------------------------------------------


def cmd_probe(args: argparse.Namespace) -> None:
    """Log a handful of points to a throwaway run, then read them back.

    This is the end-to-end "is logging working *right now*" test. If the probe's
    history comes back empty, current online logging is broken regardless of
    your training code.
    """
    _h("live probe")
    import wandb

    # Optionally force a fresh session with the key from ~/.netrc (no echo).
    if args.relogin:
        key = read_wandb_key()
        if key:
            wandb.login(key=key, relogin=True, force=True, verify=True)
            print("  relogin: verified")

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        name="wandb-doctor-probe",
        reinit="finish_previous",
        # console="off" avoids tqdm spam polluting the probe; silent quiets init.
        settings=wandb.Settings(silent=True, console="off"),
    )
    ref = f"{run.entity}/{args.project}/{run.id}"
    for step in range(1, args.points + 1):
        wandb.log({"probe/value": 1.0 / step, "probe/step": step})
    run.finish()
    print(f"  logged {args.points} points to {ref}")

    print(f"  waiting {args.wait}s for ingestion...")
    time.sleep(args.wait)
    snap = probe_server_run(ref, rows=args.points)
    _kv("lastHistoryStep", snap["last_history_step"])
    _kv("scan_history rows", snap["scan_history_rows"])
    print(f"\n  VERDICT: {_verdict(snap)}")

    if args.delete:
        try:
            wandb.Api().run(ref).delete()
            print(f"  cleaned up throwaway run {ref}")
        except Exception as exc:
            print(f"  could not delete {ref}: {exc!r}")


# ---------------------------------------------------------------------------
# 8) diagnose — orchestrate everything into one verdict
# ---------------------------------------------------------------------------


def cmd_diagnose(args: argparse.Namespace) -> None:
    """Run the full funnel: versions -> local -> server -> status."""
    cmd_versions(args)
    if args.run_dir:
        cmd_local(argparse.Namespace(run_dir=args.run_dir, rows=4))
    cmd_server(argparse.Namespace(ref=args.ref, rows=5))
    cmd_status(args)
    _h("how to read this")
    print(
        "  1. `local` shows history records but `server` returns none  -> server-side\n"
        "     (check `status`; if an ingestion incident is open, just wait — no loss).\n"
        "  2. `local` shows NO history records                         -> your code\n"
        "     (report_to not set, wandb.log never called, run crashed at init).\n"
        "  3. Both fine                                                -> it's working;\n"
        "     the dashboard panel may just need its X-axis set to train/global_step."
    )


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="wandb_doctor",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("versions", help="wandb/transformers versions + authenticity").set_defaults(
        func=cmd_versions
    )

    sp = sub.add_parser("local", help="parse a local .wandb log")
    sp.add_argument("run_dir", help="run dir, wandb/ dir, or a *.wandb file")
    sp.add_argument("--rows", type=int, default=4, help="history rows to print")
    sp.set_defaults(func=cmd_local)

    sp = sub.add_parser("server", help="query the backend for a run")
    sp.add_argument("ref", help="entity/project/run_id")
    sp.add_argument("--rows", type=int, default=5)
    sp.set_defaults(func=cmd_server)

    sp = sub.add_parser("compare", help="diff a working run vs a broken run")
    sp.add_argument("ref_a", help="entity/project/run_id (e.g. known-good)")
    sp.add_argument("ref_b", help="entity/project/run_id (e.g. broken)")
    sp.set_defaults(func=cmd_compare)

    sp = sub.add_parser("account", help="entity/plan/usage via GraphQL")
    sp.add_argument("--entity", help="entity name to inspect storage/usage for")
    sp.set_defaults(func=cmd_account)

    sub.add_parser("status", help="fetch W&B status page incidents").set_defaults(func=cmd_status)

    sp = sub.add_parser("probe", help="live logging probe + read-back")
    sp.add_argument("--entity", default=None)
    sp.add_argument("--project", default="wandb-doctor")
    sp.add_argument("--points", type=int, default=5)
    sp.add_argument("--wait", type=int, default=12, help="seconds to wait before read-back")
    sp.add_argument("--relogin", action="store_true", help="force fresh login from ~/.netrc")
    sp.add_argument("--no-delete", dest="delete", action="store_false", help="keep the run")
    sp.set_defaults(func=cmd_probe, delete=True)

    sp = sub.add_parser("diagnose", help="run the full funnel and print a verdict")
    sp.add_argument("ref", help="entity/project/run_id")
    sp.add_argument("--run-dir", default=None, help="local run dir for the same run")
    sp.set_defaults(func=cmd_diagnose)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
