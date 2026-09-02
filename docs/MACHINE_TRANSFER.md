# MACHINE_TRANSFER.md — 다른 PC로 옮기기 (Tailscale)

> 2026-08-04 실측. 실행체는 [`scripts/transfer_to_host.sh`](../scripts/transfer_to_host.sh)
> (기본 dry-run). 이관 후 재개 절차는 [`DATA_EXPANSION_RUNBOOK.md`](DATA_EXPANSION_RUNBOOK.md).

---

## 0. 왜 골라 보내야 하나

작업 트리는 **79 GB**인데 git이 들고 있는 건 **72 MB**(size-pack)뿐이다. 나머지는 전부
gitignore된 산출물이고, 그중 대부분은 **파생물**이다 — 데이터셋은 raw에서 다시 만들어지고,
optimizer 상태는 동일 런을 이어서 돌릴 게 아니면 쓸모가 없다.

| 티어 | 크기 | 내용 | 판단 |
|---|---|---|---|
| **code** | ~80 MB | git 트리 + configs + notebooks + **git이 놓치는 파일** | 필수 |
| **results** | **3.31 GB** | 모델·어댑터·GRPO 체크포인트 어댑터·`eval_logs` | **필수(재생성 불가)** |
| **raw** | ~46 GB | `data/aihub/` (v1_2020 + v2_2022, 7개 데이터셋 라벨 zip+압축해제) | 선택 — 아래 §3 |
| 안 보냄 | ~40 GB | optimizer/rng/scheduler(**18 GB, 384파일**), `outputs/datasets*`, `outputs/dialect_raw*`, 구 `classifier`, speedrun, 캐시, `llama.cpp` | 재생성/재클론 |

**핵심 절감 두 가지**
- `optimizer.pt`가 체크포인트마다 붙어 있다. 분류기 체크포인트 463 MB 중 **301 MB가 optimizer**.
  `classifier_clean` 루트에 최종 모델(151 MB + 토크나이저 11 MB)이 이미 있으므로 체크포인트
  24개(11 GB)는 통째로 뺀다.
- 구 `outputs/classifier`(17 GB)는 macro-F1 0.776으로 `classifier_clean`(0.947)에 밀려 폐기됐다.

> `outputs/grpo_*/checkpoint-*`는 **남긴다.** `eval_grpo_checkpoints.py`가 과최적화 지점을
> 찾을 때 그 어댑터들을 스윕하고, optimizer만 빼면 런당 ~90 MB다.

---

## 0.5 WSL2에서 "지웠는데 용량이 안 줄어요"

**휴지통 문제가 아니다.** WSL2의 `ext4.vhdx`는 커지기만 하고 **자동으로 줄지 않는다**. 안에서
파일을 지워도 호스트 C: 여유 공간은 그대로다. 2026-08-04 실측:

| | 값 |
|---|---|
| WSL 내부 (`df -h /`) | 삭제 전 427G 사용 → 삭제 후 **360G** (67G 확보) |
| `ext4.vhdx` 실제 크기 | **434G** (내부 사용량과 무관하게 유지) |
| 호스트 C: | 933G 중 911G 사용, **여유 23G** — 변화 없음 |

확보한 공간을 호스트로 돌려주려면 **수동 압축**이 필요하다. `wsl --shutdown`이 이 세션도
종료시키므로 작업이 끝난 뒤 Windows에서 직접 실행할 것.

```powershell
# 1) WSL 종료 (실행 중인 세션 전부 끊김)
wsl --shutdown

# 2-A) WSL 2.0+ 권장 — 앞으로는 자동으로 줄어들게 만든다
wsl --manage Ubuntu-24.04 --set-sparse true

# 2-B) 즉시 압축 (Hyper-V 모듈 있을 때)
Optimize-VHD -Path "C:\Users\jinma\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu24.04LTS_79rhkp1fndgsc\LocalState\ext4.vhdx" -Mode Full

# 2-C) Home 에디션 등 Optimize-VHD 없을 때
diskpart
  select vdisk file="C:\Users\jinma\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu24.04LTS_79rhkp1fndgsc\LocalState\ext4.vhdx"
  attach vdisk readonly
  compact vdisk
  detach vdisk
  exit
```

> 배포판 2개가 등록돼 있다(`Ubuntu24.04LTS` 434G, 레거시 `UbuntuonWindows` 14G).
> 후자도 오늘 수정된 흔적이 있어 **사용 중으로 보이므로 지우지 말 것.**

### 리포 밖 캐시가 더 크다 (2026-08-04 실측)

`ko_dialect/`만 줄여도 한계가 있다. 홈 디렉터리 쪽이 훨씬 크다:

| 경로 | 크기 | 비고 |
|---|---|---|
| `~/.cache/uv` | 81G | `uv cache prune` (미사용분만). 다른 프로젝트 설치 중엔 `clean` 금지 |
| `~/miniconda3` | 68G | envs 61G + pkgs 12G. env 정리 후 `conda clean -a` |
| `~/.cache/huggingface` | 60G | 모델 캐시 — 필요한 모델 확인 후 선별 삭제 |
| `~/.cache/pip` | 13G | `pip cache purge` 안전 |
| `~/.npm` | 3.0G | `npm cache clean --force` |

**이미 정리한 것**: 홈에 있던 `139-1.중·노년층_...(강원도,_경상도)` **20G** — `raw_data/new_dialect`와
JSON 341,043개가 md5까지 동일한 완전 중복(+원본 zip 12개, 재다운로드 가능)이라 삭제했다.

---

## 1. 보내기 전에 — git 정리

`.git`이 **1.1 GB**인데 그중 985 MB가 loose object다(팩은 72 MB). 압축하면 ~900 MB가 사라진다.

```bash
git gc --aggressive --prune=now
du -sh .git        # 1.1G -> ~80M 예상
```

## 2. 전송

```bash
tailscale status                                   # 대상 호스트 이름 확인
scripts/transfer_to_host.sh <host> --tier code     # dry-run
scripts/transfer_to_host.sh <host> --tier code --go
scripts/transfer_to_host.sh <host> --tier results --go
```

`--partial`이 켜져 있어 중간에 끊겨도 이어받는다. 대용량은 `--tier raw`를 따로,
가급적 유선/안정 구간에서.

## 3. `data/aihub/` — 보낼까 다시 받을까

| | 전송 | 재다운로드 |
|---|---|---|
| 바이트 | ~46 GB | 10.1 GB (zip) + 압축해제 |
| 리스크 | 없음 | **AI-Hub 데이터 승인 만료** — 2026-08-04에 실제로 한 번 막혔다 |
| 시간 | Tailscale 대역폭 의존 | 약 12분(15 MB/s 실측) + 압축해제 |

**zip만 보내면 된다** (1.3 GB v1 + 8.8 GB v2). 받는 쪽에서
`python scripts/aihub_fetch.py extract`로 복원되고, 압축해제분 ~36 GB는 전송할 필요가 없다.
승인이 유효하면 재다운로드도 12분이라 부담이 적다 — 절차는
[`DATA_EXPANSION_RUNBOOK.md`](DATA_EXPANSION_RUNBOOK.md) §3.

> ⚠️ v1(118~122) 5지역 중 실제로 쓸 값이 있는 건 **강원·제주**뿐이다
> (경상/전라/충청은 어절 신호 2%대). 용량이 부담되면 그 둘만 보내도 된다 —
> 근거는 [`DATA_EXPANSION_RUNBOOK.md`](DATA_EXPANSION_RUNBOOK.md) §1.2.

---

## 4. git이 놓치는 파일 (이관 시 유실 주의)

`--tier code`가 이 둘을 명시적으로 포함한다. **git만 clone하면 사라진다.**

| 파일 | 상태 | 왜 |
|---|---|---|
| `.env.local` | gitignored | AI-Hub API 키. 전송은 하되 절대 커밋 금지 |
| `configs/training/deepspeed_zero2_offload.json` | **gitignored (사고)** | `.gitignore:12`의 포괄적 `*.json` 규칙이 **설정 파일**을 잡아먹는다. CLAUDE.md는 "config면 커밋" 규칙이므로 예외 등록이 맞다 |

수정하려면:
```gitignore
*.json
!configs/**/*.json          # 설정 파일은 예외
```

## 5. 커밋 사고 위험 — 지금 상태

아래 4개는 **gitignore되지 않은 채 untracked**라, `git add .` 한 번이면 그대로 커밋된다.

```
?? KoDialect_Portfolio.pdf          860K   (개인 참고자료)
?? KoDialect_Portfolio_draft.pdf    892K   (생성물)
?? project_pt_myunghoonjin.pdf      4.6M   (개인 참고자료)
?? ondevice/MEMO.md                 4.0K   (면접 메모, 로컬 전용)
```

의도적으로 gitignore하지 않은 것으로 이해하고 **건드리지 않았다.** 다만 `git add .`은 피하고
`git add <path>`로 명시 추가하거나, 방침이 바뀌면 `.gitignore`에 넣는 편이 안전하다.

---

## 6. 도착 후 재생성

안 보낸 것들을 여기서 되살린다.

```bash
# 1) 환경
make venv && make install && make install-unsloth

# 2) 파생 데이터 (zip만 받았으면 extract 먼저)
python scripts/aihub_fetch.py extract                      # 또는 fetch --version all (재다운로드)
python scripts/prepare_data.py --data_path data/aihub/v2_2022 --output_dir outputs
python scripts/prepare_data.py --data_path data/aihub/v1_2020 --output_dir outputs --use_old_format
python scripts/stage0_build_datasets.py --raw_dataset_path outputs/dialect_raw_new

# 3) 게이트 — 반드시
python scripts/audit_raw_signal.py --raw_dataset_path outputs/dialect_raw_new
python scripts/audit_corpus_fields.py leakage --raw_dataset_path outputs/dialect_raw_new

# 4) 화자 분리 뷰 (선택, 권장)
python scripts/resplit_speaker_disjoint.py --raw_dataset_path outputs/dialect_raw_new

# 5) llama.cpp (GGUF/온디바이스 작업 시에만)
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && cmake -B build && cmake --build build -j
```

**재빌드하면 자동으로 얻는 것**: `prepare_data.py`가 이제 `stt_hypothesis` / `grammar_type` /
`domain` / `speaker_*` 6컬럼을 뽑는다. 139-1 라벨에 해당 필드가 **100% 존재**함을
확인했으므로(300파일 스캔), **재다운로드 없이** ASR 후처리 교정·문장유형 슬라이스·도메인 홀드아웃
실험이 열린다 — [`DATASET_TASK_MAP.md`](DATASET_TASK_MAP.md) 실험 A·D·E·G.

## 7. 검증 체크리스트

```bash
python -m pytest tests/ -q -m "not gpu"       # 250 passed 기대
python -m ruff check . && python -m ruff format --check .
ls outputs/eval_logs | wc -l                   # 42개 기대 — 모든 실험 결과의 원본
python scripts/eval_leaderboard.py --all_regions --n 50   # 수치 재현되는지
```

`outputs/eval_logs/`(1.9 MB)는 이 프로젝트에서 **가장 값싸고 가장 대체 불가능한** 자산이다.
`docs/RESULTS.md`가 여기서 생성되고 리더보드/유의성 수치의 원본이다. 제일 먼저 확인할 것.
