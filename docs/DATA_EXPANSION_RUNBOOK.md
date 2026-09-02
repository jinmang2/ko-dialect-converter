# DATA_EXPANSION_RUNBOOK.md — 5지역 확장 · 운율 · 단문 도메인 갭 교정

> **이 문서 하나로 타 PC에서 재개 가능.** 이 PC(RTX 2060 6GB / RAM 7GB)에서는 조사·검증·코드
> 준비만 했고, 실제 대용량 다운로드와 학습은 이 런북대로 타 PC에서 진행한다.
> 작성일 **2026-08-04**. 관련: [`OLD_DIALECT_EXPANSION.md`](OLD_DIALECT_EXPANSION.md) ·
> [`EXPERIMENTS.md`](EXPERIMENTS.md) · [`DESIGN_DECISIONS.md`](DESIGN_DECISIONS.md) ·
> [`REFERENCES.md`](REFERENCES.md)

---

## 0. 한 줄 목표

**2지역(강원·경상) → 5지역**으로 확장하고, ①어절 단위 운율 마커(구현됐지만 한 번도 검증 안 된
경로)를 A/B로 닫고, ②실사용 데모에서 드러난 **짧은 문장 실패**를 데이터 레벨에서 교정한다.

---

## 1. 오늘 측정으로 확정된 사실 (전부 재현 가능)

### 1.1 실사용 데모에서 드러난 문제
`outputs/sft_merged` fp16 + `grpo_500`/`grpo_arm2` 어댑터로 실측(peak VRAM 1068 MiB).

- **긴 구어체(학습 도메인) dia→std는 거의 정답**: `하그덩→하거든요`, `온나→와라`, `돼가→돼서`.
- **짧은 단문(직접 작성, out-of-corpus)은 급격히 붕괴**:

  | 입력 | 출력 | 실패 유형 |
  |---|---|---|
  | 밥 뭇나? | 밥 뭇냐 | 미변환 |
  | 억수로 마이 왔데이. | 많이 많이 와요 | 강조부사·시제 소실 |
  | 마카 다 모였소? | **만치** 다 모였소? | 오역 |
  | 일 다 했스요? | 일 다 했으니까 **이제는 밥 먹고 나와요** | 환각 |
  | 우리 시댁이 창녕인데 | 우리 시댁이 **창(^)(** 있는데 | 고유명사 degeneration |

  `(^)` 는 코퍼스 arrow에 존재하지 않음(grep 확인) → **데이터 잔재가 아니라 디코딩 붕괴**.

### 1.2 원인 — 데이터가 아니라 **샘플링** 문제 (`scripts/audit_raw_signal.py`)

```bash
python scripts/audit_raw_signal.py --raw_dataset_path outputs/dialect_raw_new --split train --n 20000
```

| 코퍼스 | 지역 | 어절신호↑ | pair신호 | 어절수 p50 | **학습대상 중 ≤5어절** | F0 |
|---|---|---|---|---|---|---|
| **v2** (139-1) | gangwondo | 11.03% | 44.7% | 12 | **3.1%** | 100% |
| **v2** | gyeongsangdo | 14.24% | 66.9% | 11 | **4.1%** | 99.6% |
| **v2** | OVERALL | **12.65%** | 55.8% | 12 | **3.7%** | 99.8% |
| **v1** (2020) | gangwondo | **14.29%** | 58.8% | 5 | **56.2%** | 0% |
| **v1** | jejudo | **13.77%** | 41.2% | 5 | **48.3%** | 0% |
| **v1** | gyeongsangdo | 2.57% | 15.9% | 8 | 15.7% | 0% |
| **v1** | jeollado | 2.54% | 15.0% | 7 | 22.2% | 0% |
| **v1** | chungcheongdo | 2.09% | 15.1% | 9 | 11.7% | 0% |
| **v1** | OVERALL | 5.91% | 29.2% | 6 | **41.4%** | 0% |

**결론 A — 단문 실패는 예견된 결과**: `filter_identical` 이후 학습 대상 중 ≤5어절이 **3.7%**.
모델은 짧은 입력을 사실상 본 적이 없다. 데이터가 없어서가 아니라 **분포가 한쪽으로 쏠려서**다.
(참고: v2 `read`(따라말하기) 192,027행은 96.3%가 `is_identical` → 필터에서 전량 증발한다.)

**결론 B — v1 "2~4% 신호라 폐기" 판정은 부분적으로 틀렸다.** 그 숫자는 경상/전라/충청
(2.09~2.57%)에만 해당하고, **강원 14.29% · 제주 13.77%** 는 v2 전체(12.65%)보다도 높다.
그리고 v1은 **≤5어절이 학습대상의 41.4%** — 정확히 v2가 비어 있는 구간이다.
→ **v1은 폐기 대상이 아니라 v2의 상보재**다.

**재현 검증**: 오늘 새로 받은 v1 샘플(제주 121 / 경상 119 Validation 라벨)을 처음부터 빌드해
독립 측정한 결과 **경상 2.13% · 제주 13.36%** — 로컬 기존 수치(2.57 / 13.77)와 일치.
지역 간 신호 격차는 **처리 아티팩트가 아니라 실제 코퍼스 특성**이다.

### 1.3 139-2(71558) 스키마 호환성 — 검증 완료 ✅

제주 2인발화 라벨 샘플(9 MB, 188 JSON)을 실제로 받아 `prepare_data.py`로 빌드 → **1,601행**.

- 문장 키가 `standard`/`dialect` 로 **139-1과 동일** → 신포맷 파서 무수정 재사용 가능.
- `intonations`(F0 시계열) **존재** → 3개 신규 지역도 운율 마커 생성됨.
  생성 확인 컬럼: `prosody`, `prosody_marker`, `dialect_eojeol_prosody`.
- 지역 정규식 `_([가-힣]+도)_` 이 `VL_03._제주도_03._2인발화` 에 정상 매칭.
- 품질: non-identical 266/300(88.7%), 어절 매핑 풍부.
  예) `엇일→없을`, `일헷주만은→일했지만`, `겅허긴→그렇긴`, `호꼼→조금`, `닮아예→같아요`

> ⚠️ 주석 노이즈 1건 관측: `#이름#은 → 하지만` (익명화 토큰이 어절 매핑에 잘못 정렬).
> 빈도 낮으나 5지역 빌드 후 `dialect_eojeol_map`에서 `#...#` 포함 항목 비율을 한 번 재볼 것.

### 1.4 오늘 고친 버그 2개 (`scripts/prepare_data.py`)

| # | 증상 | 원인 | 영향 |
|---|---|---|---|
| B1 | 단일 split 코퍼스 처리 시 `StopIteration` 크래시 | 워커가 train/valid/unknown 핸들을 **무조건** 열어 빈 `train_worker_*.jsonl`이 남고, reduce가 이를 유효 split으로 오인 | **샘플 다운로드·지역별 증분 빌드가 전부 불가**했음 |
| B2 | 새로 받은 v1이 `do="unknown"`, `split=쓰레기` | 지역 정규식이 `한국어 방언 발화(제주도)`(공백·"데이터" 없음)만 매칭. 실제 aihubshell 산출물은 `016.한국어_방언_발화_데이터(제주도)/.../2.Validation/...` | stage0의 `SUPPORTED_DO` 필터에서 **전량 조용히 폐기** (에러 없음) |

수정: reduce에서 0바이트 워커 파일 제외 + 전부 비면 명확한 `ValueError`. 지역/split은 경로
컴포넌트를 **뒤에서부터** 스캔해 해석(구 레이아웃 하위호환 테스트 통과). 관련 테스트 44개 통과.

---

## 2. 데이터 인벤토리 (2026-08-04 실측)

조회는 API 키 없이 가능. **다운로드는 키 + 유효한 데이터 승인**이 필요하다.

```bash
python scripts/aihub_fetch.py datasets --grep 방언      # 데이터셋 키 찾기
python scripts/aihub_fetch.py plan --datasetkey 71558   # 라벨 vs 음성, 용량, filekey
```

| datasetkey | 이름 | 라벨(텍스트) | 원천(음성) | **비식별화 재업로드** | 상태 |
|---|---|---|---|---|---|
| **71517** | 139-1 중·노년층(**강원, 경상**) | 4.3 GB | 101 GB | **없음 (0/24)** | ✅ 2026-08-04 재다운로드 |
| **71558** | 139-2 중·노년층(**충청, 전라, 제주**) | **4.5 GB** | 150 GB | **전부 (36/36)** | ✅ 2026-08-04 확보 |
| 118~122 | 한국어 방언 발화 v1 (5개 도) | 각 0.09~0.36 GB | 각 ~0.6 TB | **전부 (44/44 등)** | ✅ 2026-08-04 확보 |

**2026-08-04: 7개 전량 라벨 확보 완료.** 합계 10.1 GB(zip) → `data/aihub/{v1_2020,v2_2022}`.
음성은 받지 않았다(§2 마지막 불릿). 승인은 이날 **유효**했고 실측 대역폭은 약 15 MB/s.
71517은 비식별화판이 없어 로컬본이 최신이었지만, 출처를 하나로 맞추려고 재다운로드한 뒤
`diff_trees`로 기존 `raw_data/new_dialect`와 대조했다:

```
shared names 341,043 / only in A 0 / only in B 0 / content differs 0  → IDENTICAL
```

sha256 전수 일치를 확인하고 **`raw_data/`를 삭제했다**(15 GB 회수). 코퍼스 경로는 이제
`data/aihub/` 하나뿐이다.

### 2.1 비식별화(de-identification) 재업로드 — 반드시 확인할 것

AI-Hub가 2026년 7월 중·하순경 **개인정보 비식별화 처리본**으로 일부 데이터셋을 재업로드했다.
파일트리의 `(비식별화완료)` 접두어로 판별되며, 2026-08-04 전수 확인 결과:

```
118 강원 v1   44/44 ✅    121 제주 v1   41/41 ✅    71517 (139-1)   0/24 ❌ 없음
119 경상 v1   14/14 ✅    122 충청 v1   20/20 ✅    71558 (139-2)  36/36 ✅
120 전라 v1   45/45 ✅
```

- **v1(118~122)과 139-2는 반드시 비식별화판을 써야 한다.** 그 이전에 받은 사본은 폐기 대상.
  → 로컬 `raw_data/old_dialect`(18 GB, 비식별화 이전)와 파생물 `outputs/dialect_raw_old`(4.4 GB)는
  **2026-08-04에 삭제했다.** 같은 날 비식별화판으로 **전량 재확보**했다(§2 표).
- **139-1은 비식별화판이 아직 없다.** 나중에 재업로드되면 `diff_trees`로 현재 사본과 대조한 뒤
  교체할 것 — 그 검사가 이 커맨드가 존재하는 이유다.
- 판별 커맨드:
  ```bash
  python scripts/aihub_fetch.py tree --datasetkey 118 | grep -c 비식별화완료
  ```

### 2.2 지역 구성 (자주 헷갈림)

| 코퍼스 | 포함 지역 |
|---|---|
| 139-1 (71517) | **강원 · 경상** |
| 139-2 (71558) | **충청 · 전라 · 제주** |
| v1 (118~122) | 강원 · 경상 · 전라 · 제주 · 충청 (5개 전부, 데이터셋 5개로 분리) |

즉 중·노년층(139) 계열은 **강원/경상이 먼저**, 충청/전라/제주가 139-2로 추가된 구조다.
139-1 + 139-2를 합쳐야 5개 도가 완성된다.

- **음성은 이번 범위 밖.** F0가 라벨 JSON에 있어 운율 실험에 음성이 필요 없다.
- 71558 라벨 4.5 GB 예상 행수: 71517이 4.2 GB → 691,520행(≈165k행/GB)이므로 **약 74만 행**.
  확장 후 `dialect_raw_new`는 **140만 행 규모**가 된다(디스크·RAM 예산에 반영할 것).

---

## 3. 타 PC 실행 절차

### Step 0 — 환경
```bash
git checkout feat/ondevice-galaxy      # 또는 확장용 새 브랜치
printf 'AIHUB_API_KEY=<키>\n' > .env.local     # .gitignore 로 차단됨 (.env.*)
ls ~/aihubshell || echo "aihub.or.kr > API 이용안내에서 내려받아 ~/aihubshell 로"
python scripts/aihub_fetch.py plan --datasetkey 71558   # 키 없이 동작 = 사전 점검
```
> 다운로드가 `데이터 승인 유효기간이 도래하였습니다`로 실패하면 **키 문제가 아니라 승인 만료**다.
> aihub.or.kr에서 해당 데이터셋을 재신청·승인받아야 한다(오늘 이 벽에 한 번 막혔다).

### Step 1 — 라벨 다운로드 (10.1 GB, 음성 0 GB) — **2026-08-04 완료**
```bash
python scripts/aihub_fetch.py fetch --version all --dry_run   # 먼저 확인
python scripts/aihub_fetch.py fetch --version all             # 다운로드 + 압축해제
```
목적지는 `src/ko_dialect/corpora.py` 레지스트리가 결정한다 — `--dest`를 손으로 주지 말 것.
`data/aihub/v1_2020`(구포맷) / `data/aihub/v2_2022`(신포맷)로 갈리는 게 Step 2의 전제다.
`--kind source`(음성 150 GB)는 기본 가드(`allow_large_mb=6000`)가 **거부**한다. 의도적으로만 푼다.

### Step 2 — raw 빌드 (5지역)
```bash
# v2 = 139-1(강원·경상) + 139-2(충청·전라·제주) → 한 트리, 한 번에 5개 도.
python scripts/prepare_data.py --data_path data/aihub/v2_2022 --output_dir outputs
#  → outputs/dialect_raw_new (5개 도) + dialect_raw_new_manifest.json

# v1은 구포맷이라 반드시 별도 실행 (한 트리에 두 포맷을 섞지 말 것 — 파서 분기가 트리 단위다).
python scripts/prepare_data.py --data_path data/aihub/v1_2020 --use_old_format --output_dir outputs
```
- 두 커맨드의 `--data_path`가 곧 레지스트리의 `version`이다. 디렉토리를 섞으면 잘못된 파서가
  `do="unknown"`을 만들고 stage0이 조용히 전량 폐기한다(§1.4 B2). `tests/test_corpora.py`가
  "한 버전 = 한 포맷" 불변식을 고정한다.
- 전량 빌드 전에 `--n_samples 200` speedrun으로 지역·split 해석을 먼저 확인할 것(싸고 빠르다).
- 실패 파일은 `outputs/failed_files.json`. AI-Hub JSON 일부는 원래 깨져 있다(정상).

### Step 3 — 게이트: 신호 감사 (다음 단계로 넘어가기 전 필수)
```bash
python scripts/audit_raw_signal.py --raw_dataset_path outputs/dialect_raw_new \
    --split train --n 20000 --json_out outputs/eval_logs/signal_raw_new_5region.json
```
**통과 기준**: 신규 3지역이 각각 ① `do`가 `unknown`이 아님 ② F0 커버리지 > 95%
③ 어절신호 > 5%. 하나라도 어긋나면 파싱이 잘못된 것이니 **여기서 멈춘다**
(§1.4 B2가 정확히 이 게이트가 없어서 놓칠 뻔한 종류의 실패다).

### Step 4 — stage0 (6-class)
```bash
python scripts/stage0_build_datasets.py --raw_dataset_path outputs/dialect_raw_new
```
`labels.py`가 이미 jeolla=3 / jeju=4 / chungcheong=5를 갖고 있고 `num_labels_for()`가
데이터 기반이라 **자동으로 6-class**가 된다. 기존 3-class 체크포인트는 id append-only라 그대로 유효.

### Step 5 — 분류기 재학습 (GRPO 스타일 보상)
```bash
# stage2는 Hydra다 (stage0/prepare_data의 fire 스타일과 다름 — `--` 를 붙이지 말 것)
python scripts/stage2_train_classifier.py data.cls_dataset_path=outputs/datasets/classifier
```
현행 macro-F1 0.947(3-class). **6-class에서는 당연히 떨어진다** — 특히 충청↔표준,
전라↔경상 혼동을 볼 것. 보상으로 쓰기 전 per-class recall을 확인하고, 낮은 클래스가 있으면
GRPO 스타일 보상의 신뢰구간이 그만큼 넓어진다는 걸 기록해 둘 것.

### Step 6 — SFT (§4의 단문 교정 포함)
### Step 7 — GRPO → Step 8 — 평가
```bash
python scripts/eval_leaderboard.py --all_regions --n 150
```
`--all_regions`는 데이터에 있는 지역을 자동 발견하므로 5지역이 그대로 잡힌다.
per-region + 가중 OVERALL, Pareto, Koehn 유의성까지 기존 프레임 그대로.

---

## 4. 단문 도메인 갭 교정 설계 (우선순위 순)

측정된 원인은 하나다: **학습대상의 3.7%만 ≤5어절**. 레버 세 개를 이 순서로 시도한다.

### L1. v1 단문 혼합 (가장 근거가 강함) ★
v1 강원(14.29% 신호, ≤5어절 56.2%)·제주(13.77%, 48.3%)를 v2에 섞는다.
- **왜 이게 맞나**: 신호 세기는 v2와 대등하고 길이 분포는 정확히 상보적. 새 데이터를 구할
  필요도 없다(이미 `outputs/dialect_raw_old` 보유).
- **주의**: v1은 F0가 **없다**(prosody 0%). 운율 실험과 섞으면 마커 커버리지가 깨진다.
  → **운율 A/B와 단문 혼합은 분리된 arm으로 돌릴 것.**
- 시작 비율 제안: 학습대상 중 ≤5어절 비중이 **3.7% → 15% 내외**가 되도록 v1에서 샘플링.
  비율 자체를 스윕(10/15/25%)하고 `eval_leaderboard`의 proxy-independent 지표로 선택.
- 경상/전라/충청 v1은 신호 2%대라 **넣지 말 것**(복사학습 유도).

### L2. 길이 계층 샘플링
`filter_identical` 이후 길이 버킷(≤5 / 6–12 / 13–25 / 26+)별로 상한·하한을 두어
짧은 버킷이 최소 비중을 확보하게 한다. v1을 안 쓰고도 부분적 개선이 가능하지만,
v2의 ≤5어절 학습대상 절대량 자체가 적어 L1보다 천장이 낮다.

### L3. 평가에 길이 버킷 리포팅 추가
교정 여부를 **측정 가능**하게 만드는 단계. `compare_sft_grpo.py`에 이미 stratified bucket이
있으므로, 리더보드에도 `≤5어절` 서브셋 점수를 별도 컬럼으로 노출한다.
→ 이게 없으면 L1/L2의 효과가 OVERALL 평균에 묻힌다.

> **정직성 메모**: 위 세 레버는 이 repo의 측정에 근거한 **자체 설계**이며 특정 논문의 권고가
> 아니다. 인용을 붙이지 말 것 ([`REFERENCES.md`](REFERENCES.md)의 검증 규범).

---

## 5. 어절 단위 운율(prosody) A/B — 미검증 경로 닫기

### 현황
- 문장 단위 마커 A/B는 **완료**: recon_bleu 유의하게 상승(+3.9/+6.4, p<.001)하지만 방언성
  축은 평탄 → *충실도 정규화기지 방언성 레버가 아님* ([`EXPERIMENTS.md`](EXPERIMENTS.md) §2).
- **어절 단위(`prosody_mode="eojeol"`)는 코드·테스트만 있고 데이터셋으로 빌드된 적이 없다.**
  `dialect_raw_prosody_v2`에도 `dialect_eojeol_prosody` 컬럼이 없다.
  → 이번 확장에서 처음으로 실측한다.

### 실행
현행 `prepare_data.py`는 raw 빌드 시 `prosody` / `prosody_marker` / `dialect_eojeol_prosody`
**세 컬럼을 모두 생성**한다(오늘 제주 샘플로 확인). 기존 `dialect_raw_new`에 앞의 두 개가 없는
건 더 오래된 코드로 빌드됐기 때문이다 → **재빌드하면 별도 `dialect_raw_prosody_*` 트리가 불필요**하다.

```bash
# stage0는 fire. 출력 트리를 분리해야 기존 datasets/를 덮지 않는다.
python scripts/stage0_build_datasets.py --raw_dataset_path outputs/dialect_raw_new \
    --prosody_mode eojeol --output_dir outputs/datasets_prosody_eojeol
python scripts/stage1_sft.py data.sft_dataset_path=outputs/datasets_prosody_eojeol/sft
python scripts/eval_prosody_ab.py   # 기존 A/B 하니스 재사용
```

### 반드시 함께 볼 것
1. **정렬 실패율** — `eojeol_markers_aligned()`가 1:1이 아니면 그 행은 plain으로 폴백한다.
   폴백 비율이 높으면 A/B가 사실상 무효다. stage0가 split별로 카운트해 보고한다.
2. **알려진 결함(수정 말고 기록)**: `both_directions=True` + prosody_mode 조합에서
   dia→std의 **소스**에 마커가 붙는다. 실제 추론 입력에는 마커가 없으므로 그 방향은 OOD로
   학습된다(`dataset.py`에 KNOWN ISSUE로 명시). 문장 단위 A/B와 조건을 맞추려고 유지 중이니,
   **어절 A/B도 std2dia 방향으로만 결론을 내릴 것.**
3. 마커 정책은 K-ToBI 경계성조 근거(Jun 2000/2005, [`REFERENCES.md`](REFERENCES.md) A11).
   `PROSODY_MARKER_POLICY_VERSION=2`와 raw manifest 버전이 어긋나면 stage0가 막는다.

---

## 6. 리스크 / 함정

| 리스크 | 대응 |
|---|---|
| AI-Hub 승인 만료 (오늘 실제로 막힘) | 다운로드 전 `plan`으로 조회 → `download --dry_run` → 실제 다운. 실패 메시지로 키/승인 구분 |
| 실수로 음성 150 GB 다운로드 | `aihub_fetch.py`의 `allow_large_mb` 가드(기본 6 GB)가 거부. `--kind source`는 명시해야만 선택됨 |
| **압축해제가 "성공"인데 0 파일** | 139-1(71517) zip은 멤버 이름이 `/talk_….json`(**선행 슬래시**)이다. `out_dir / "/x"`는 pathlib에서 루트로 리셋되므로, 멤버 단위 추출은 경로를 반드시 정규화해야 한다(`extractall`은 자동으로 해준다). 2026-08-04에 12개 아카이브가 전부 "ok"를 찍고 341,043개 중 0개를 푼 적이 있다 → `tests/test_aihub_extract.py`가 고정. **추출 직후 JSON 개수를 항상 확인할 것** |
| 5지역 raw 140만 행 → RAM | `audit_raw_signal.py`는 지역별 stride 샘플링으로 저메모리. stage0/학습은 타 PC 사양에 맞춰 `chunk_size` 조정 |
| 6-class 분류기 품질 저하 → 보상 신뢰도 하락 | Step 5에서 per-class recall 확인. 낮으면 해당 지역 GRPO 결론은 보류 |
| v1 혼합이 F0 커버리지를 깨뜨림 | 단문 혼합 arm과 운율 arm을 **분리** (§4 L1) |
| 지역 확장 후 "단일 승자 없음"이 더 심해짐 | 예상된 결과. Pareto + per-region 유지, 억지로 한 줄 세우지 말 것 (기존 결론과 동일) |

---

## 7. 이 PC에서 만들어 둔 것 (타 PC에서 바로 쓰는 자산)

| 경로 | 내용 |
|---|---|
| `src/ko_dialect/corpora.py` | 7개 데이터셋 레지스트리(지역·포맷·비식별화 여부·디스크 위치). 다운로드 목적지의 단일 출처 |
| `scripts/aihub_fetch.py` | 조회/계획/`fetch`(세대 단위 일괄)/`extract`(멱등 압축해제)/`diff_trees`(sha256 대조)/inspect. 용량 가드 내장 |
| `tests/test_corpora.py` | "한 버전 = 한 포맷", 지역명↔`SUPPORTED_DO` 정합성 등 조용히 깨지는 불변식 고정 |
| `scripts/audit_raw_signal.py` | 신호·길이·운율 커버리지 감사. §3 Step 3 게이트의 실행체 |
| `scripts/prepare_data.py` | 버그 2건 수정(§1.4) — 단일 split 빌드 가능, 신규 다운로드 레이아웃 인식 |
| `outputs/eval_logs/signal_raw_{new,old}.json` | 오늘 측정한 기준선 (확장 후 비교 대상) |
| `data/aihub_samples/` | 검증에 쓴 샘플(71558 제주 9 MB, v1 121/119 라벨). gitignore 대상 |
| `.gitignore` | `.env.*` 차단 추가 (`.env.local`이 추적될 수 있던 상태였음) |

---

## 8. 재개 첫 명령

라벨 7종은 **이미 `data/aihub/`에 있다**(2026-08-04). 이 PC에서 재개한다면 Step 1은 건너뛴다.

```bash
git checkout feat/ondevice-galaxy
python scripts/aihub_fetch.py extract                          # 멱등 — 이미 풀렸으면 no-op
python scripts/prepare_data.py --data_path data/aihub/v2_2022 --output_dir outputs  # §3 Step 2
# 데이터가 없는 새 PC라면 먼저:
#   printf 'AIHUB_API_KEY=<키>\n' > .env.local
#   python scripts/aihub_fetch.py fetch --version all
```
