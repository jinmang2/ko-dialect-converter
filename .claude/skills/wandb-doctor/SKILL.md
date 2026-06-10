---
name: wandb-doctor
description: >
  Diagnose "metrics not showing up in Weights & Biases" problems — when a run
  appears in the dashboard with config/summary but the history panels say
  "There's no data for the selected runs", or charts are empty. Separates a
  client/code bug from a server-side issue (ingestion incident, quota, auth)
  using the scripts/wandb_doctor.py funnel. Trigger when someone says wandb
  logging "isn't working", metrics/charts are missing, "summary만 보인다",
  "history가 안 올라간다", or a training run logs but panels stay empty.
---

# wandb-doctor

A reusable workflow for diagnosing **"my metrics aren't in W&B"** problems.
Backed by `scripts/wandb_doctor.py` in this repo (run any subcommand with `-h`).

## The one mental model

Logging flows through layers. Find the **first layer that's broken**:

```
[your code] --wandb.log()--> [local .wandb log] --filestream--> [W&B backend]
   report_to?                  ground truth of                   summary  (fast path)
   wandb.log called?           what was recorded                 history  (metered path)
                                                                  console  logs
```

- Metrics in the **local** log but **not on the server** → downstream
  (ingestion incident / quota / auth). **Not your code.**
- **Nothing** in the local log → your **code/config** (report_to, wandb.log,
  init crash).
- Both fine → it's working; the dashboard panel may just need its **X-axis**
  set to `train/global_step`.

## Decision tree (run these in order)

1. **`versions`** — confirm wandb/transformers versions and that the installed
   `wandb` is genuine (METADATA Project-URL/author) with a `wandb-core` binary.

2. **`local <run_dir>`** — parse the local `run-*.wandb` transaction log.
   - History records present → the **client did its job**. Go to step 3.
   - No history records → stop; fix the training code (`report_to=["wandb"]`,
     ensure `Trainer.log`/`wandb.log` actually runs, run didn't crash at init).

3. **`server <entity/project/run_id>`** — query the backend. The smoking-gun
   fingerprint is **summary populated + `lastHistoryStep` advancing +
   `scan_history` returns 0 rows**. That means metered history isn't being
   retained while summary (separate write path) is.

4. **`status`** — hit `status.wandb.com`'s JSON API for open incidents. An
   active "Metric ingestion delayed" / "Degraded Performance" on the Metrics
   Ingestion Pipeline explains a summary-only symptom with **no data loss** —
   it backfills once the backlog drains.

5. **`compare <good_ref> <bad_ref>`** — if an older run logged fine and a newer
   one didn't *with identical code and wandb version*, the only variable is
   **when** they ran → server-side.

6. **`probe [--relogin]`** — end-to-end "is logging working right now": log a
   few points to a throwaway run and read them back. Empty read-back ⇒ current
   online logging is broken regardless of training code. `--relogin` rules out
   stale auth; an enterprise-org `--entity` rules out personal-account quota.

7. **`account [--entity NAME]`** — orgs/plans and per-entity storage/usage.

`diagnose <ref> [--run-dir DIR]` runs versions→local→server→status in one shot.

## Hard-won gotchas (don't re-learn these)

- **Stringified console metrics are a red herring.** Seeing
  `{'loss': '1.098'}` (quoted) in the console is HF Transformers'
  `ProgressCallback` building a *display-only* dict with `f"{v:.4g}"`. The real
  `logs` dict passed to `WandbCallback` still holds floats.
- **`scan_history(keys=[...])` can return [] even when data exists** in some
  wandb versions — query **without** the `keys=` filter, and cross-check with
  `run.history()` and `run.lastHistoryStep`.
- **`entity.available` is NOT a quota flag.** It means "is this entity name
  free to register." An over-limit entity does not flip it. Use
  `storageBytes`/`computeHours` and a cross-entity `probe` instead.
- **`lastHistoryStep` advancing ≠ history retrievable.** The server can count
  steps it received while the queryable history table stays empty (ingestion
  lag). Always confirm with actual rows.
- **Huge `output_raw` counts** in `local` (tens of thousands) are tqdm/progress
  bar spam captured as console logs — harmless to metrics. Quiet with
  `TrainingArguments(disable_tqdm=True)` or `WANDB_CONSOLE=off`.

## What you're learning along the way

`scripts/wandb_doctor.py` is commented for study (`grep "LEARN:"`):
- `netrc` — read the W&B API key from `~/.netrc` without printing it.
- `itertools.islice` — peek at the first N rows of a streaming generator
  (`scan_history`) without pulling the whole history.
- `urllib.request` — fetch the statuspage.io JSON API (no `requests` needed).
- `wandb.sdk.internal.datastore.DataStore` + `wandb.proto.wandb_internal_pb2` —
  read the raw `.wandb` LevelDB-style transaction log and decode `Record`
  protobufs (history/summary/config/output_raw/...). This is the ground truth
  of what the client recorded locally.
- `wandb.Api()` + `api._service_api.execute_graphql(...)` — the public read API
  and its GraphQL escape hatch for viewer/entity/usage.

## Quick reference

```bash
python scripts/wandb_doctor.py versions
python scripts/wandb_doctor.py local  wandb/run-YYYYMMDD_HHMMSS-<id>
python scripts/wandb_doctor.py server entity/project/<run_id>
python scripts/wandb_doctor.py status
python scripts/wandb_doctor.py compare entity/proj/<good> entity/proj/<bad>
python scripts/wandb_doctor.py probe --relogin            # creates+deletes a run
python scripts/wandb_doctor.py account --entity <name>
python scripts/wandb_doctor.py diagnose entity/proj/<id> --run-dir wandb/run-...
```
