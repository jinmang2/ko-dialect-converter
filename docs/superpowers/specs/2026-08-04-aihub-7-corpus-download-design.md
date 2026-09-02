# AI-Hub 7개 방언 코퍼스 라벨 일괄 확보 · 경로 단일화 — 설계

작성일 **2026-08-04**. 관련: [`../../DATA_EXPANSION_RUNBOOK.md`](../../DATA_EXPANSION_RUNBOOK.md)

---

## 1. 목표

AI-Hub 한국어 방언 **7개 데이터셋**(v1 지역별 5개 + v2 2개)의 **라벨(텍스트)** 을 전부 받아
포맷별로 정리하고, 흩어져 있던 코퍼스 경로 지식을 소스코드의 단일 레지스트리로 모은다.

**범위 밖**: 원천(음성) 다운로드, 전량 raw 빌드(140만 행), 신호 감사, stage0/분류기 재학습.

---

## 2. 확정된 사실 (2026-08-04 실측)

조회는 키 없이 가능. 다운로드 승인은 **유효함**(118 라벨 36 MB 실제 수신, ~15 MB/s).

| key | 코퍼스 | 지역 | 라벨 | 파일 | 비식별화 |
|---|---|---|---|---|---|
| 118 | v1/2020 | 강원 | 0.22 GB | 2 | ✅ |
| 119 | v1/2020 | 경상 | 0.33 GB | 2 | ✅ |
| 120 | v1/2020 | 전라 | 0.09 GB | 2 | ✅ |
| 121 | v1/2020 | 제주 | 0.36 GB | 2 | ✅ |
| 122 | v1/2020 | 충청 | 0.34 GB | 3 | ✅ |
| 71517 | v2/2022 (139-1) | 강원·경상 | 4.3 GB | 12 | ❌ 없음 |
| 71558 | v2/2022 (139-2) | 충청·전라·제주 | 4.5 GB | 18 | ✅ |

합계 약 **10.1 GB**(압축), 추출 후 약 36 GB 예상. 여유 디스크 613 GB.

호스트는 RTX 2060 / RAM 7 GB — 런북이 "타 PC에서 하라"고 한 작업이지만, 라벨만 받으면
디스크·네트워크 작업이라 이 PC에서 가능하다. RAM이 걸리는 건 전량 빌드뿐이고 그건 범위 밖이다.

---

## 3. 레이아웃

```
data/aihub/
  v1_2020/     013(강원) 014(경상) 015(전라) 016(제주) 017(충청)   ← 118~122, 구포맷
  v2_2022/     139-1(강원·경상)  139-2(충청·전라·제주)            ← 71517, 71558, 신포맷
data/aihub_samples/    (기존 검증 샘플 유지)
```

**version == 디렉토리**가 핵심 불변식이다. `prepare_data.py`의 파서 분기(`--use_old_format`)는
**트리 단위**라, 구포맷 JSON이 신포맷 트리에 섞이면 잘못된 파서가 조용히 쓰레기를 만든다
(런북 §1.4 B2와 같은 계열의 실패). 디렉토리 경계를 포맷 경계와 일치시켜 이 사고를 구조적으로 막는다.

zip은 추출 후에도 제자리에 남긴다 — 재다운로드 없이 재추출이 가능하고, 10 GB는 613 GB 대비 무시 가능.

`raw_data/`는 71517 대조 검증 후 삭제한다(15 GB 회수).

---

## 4. 소스코드 경로 관리 — `src/ko_dialect/corpora.py` 신설

현재 코퍼스 지식이 두 곳으로 갈라져 있고 서로 드리프트할 수 있다:
`aihub_fetch.py`의 `DIALECT_DATASETS`(설명 문자열뿐), `prepare_data.py:651`의 하드코딩 기본 경로.

```python
@dataclass(frozen=True)
class Corpus:
    datasetkey: str
    title: str
    regions: tuple[str, ...]     # labels.py의 정규 이름 (gangwondo, ...)
    version: str                 # "v1_2020" | "v2_2022"  == 디렉토리 이름
    old_format: bool             # → prepare_data --use_old_format
    deidentified: bool           # 71517만 False
    label_gb: float

CORPORA: dict[str, Corpus]              # 7개
def data_root() -> Path                 # $KO_DIALECT_DATA_ROOT 우선, 없으면 <repo>/data
def aihub_root() -> Path                # data_root()/"aihub"
def tree_root(version) -> Path          # aihub_root()/version
def corpora_for(version=None, old_format=None) -> list[Corpus]
def uses_old_format(version) -> bool    # 버전 내 포맷이 섞이면 예외
```

`ids.py`와 같은 이유로 **패키지 루트**에 둔다: `ko_dialect.data`를 import하면
`__init__`이 collator→torch를 끌어오는데, 다운로드/조회 스크립트에 torch는 순수 낭비다.
같은 이유로 `labels.py`를 import하지 않고 지역명을 문자열로 두되, **테스트가**
`SUPPORTED_DO`의 부분집합인지 검증해 드리프트를 막는다(테스트는 torch 비용을 내도 된다).

경로는 import 시점이 아니라 **호출 시점**에 해석한다 — 그래야 테스트가 환경변수로 격리된다.

소비자: `aihub_fetch.py`(`DIALECT_DATASETS`·dest 기본값), `prepare_data.py`(기본 `data_path`).

---

## 5. `aihub_fetch.py` 추가 커맨드 3개

| 커맨드 | 이유 |
|---|---|
| `extract` | 트리 아래 모든 zip을 멱등 압축해제. **현재 없는 기능** — `inspect`가 최대 2개만 푸는 게 전부라 전량 다운로드 후 수작업으로 풀어야 하는 구멍이 있었다 |
| `fetch --version` | 레지스트리 기준 라벨 일괄 다운로드+추출. dest를 사람이 타이핑하지 않게 해서 §3의 포맷 섞임을 구조적으로 차단 |
| `diff_trees --a --b` | JSON basename→sha256 매니페스트 비교. 71517 대조용이자, 139-1 비식별화판이 나중에 올라오면 다시 쓸 검사 |

---

## 6. 실행 순서

1. v1 5개 → `v1_2020/`, 71558·71517 → `v2_2022/` 다운로드 + 추출 (~10 GB, 약 12분)
2. `diff_trees raw_data/new_dialect ↔ data/aihub/v2_2022/139-1...` — 동일성 판정
3. **동일할 때만** `raw_data/` 삭제. 불일치하면 삭제하지 않고 차이를 보고한다
4. 게이트: 트리별 `prepare_data.py --n_samples 200` speedrun — 포맷 분기 / `do` 해석 /
   split 해석 확인. 런북 §1.4 B2가 "조용히 `do=unknown` → stage0에서 전량 폐기"였으므로
   전량 빌드 전에 여기서 막는다

---

## 7. 검증

- `tests/test_corpora.py` 신설: 레지스트리 정합성(7개, 지역 5개 커버, version↔포맷 일관성,
  지역명이 `SUPPORTED_DO` 부분집합), 경로 해석의 환경변수 오버라이드
- 기존 테스트 전량 통과 (`prepare_data` 기본 경로 변경의 회귀 확인)
- `ruff check` + `ruff format --check` (CI가 두 개 다 본다)

## 8. 문서 갱신

`DATA_EXPANSION_RUNBOOK.md` §2·§3·§7·§8, `docs/MACHINE_TRANSFER.md`, `scripts/README.md`의
`raw_data` 참조.

---

## 9. 리스크

| 리스크 | 대응 |
|---|---|
| 71517 재다운로드본이 로컬과 불일치 | §6-3에서 삭제 보류 + 차이 보고. 로컬 15 GB는 그때까지 온전 |
| 다운로드 중 승인 만료 | 실패 메시지로 키/승인 구분(`aihub_fetch._run`이 이미 감지). 재신청 필요 |
| 추출 후 디스크 | 피크 약 61 GB (구 15 + zip 10 + 추출 36). 613 GB 대비 여유 |
| v1 구포맷 파싱 결함 | §6-4 speedrun 게이트. old 포맷은 `utf-8-sig` + `clean_old_transcript` 경로 |

---

## 10. 실행 결과 (2026-08-04)

7개 전량 확보. `data/aihub/v1_2020` 15 GB(JSON 27,796) + `data/aihub/v2_2022` 41 GB(JSON 699,175).

**설계 단계에서 예상하지 못한 결함 1건** — 멤버 단위 추출의 경로 정규화 누락.
139-1(71517) zip은 모든 멤버가 `/talk_….json`처럼 **선행 슬래시를 가진 절대경로**로 저장돼
있다. `out_dir / "/x.json"`은 pathlib에서 좌변이 버려져 `/x.json`이 되므로, zip-slip 가드가
전 멤버를 "탈출 시도"로 거부했다. 결과는 **12개 아카이브 전부 "ok", 추출 0/341,043개** —
실패가 성공처럼 보이는 형태였다. `ZipFile.extractall`이 대신 해주던 정규화를 멤버 단위로
직접 추출하면서 잃은 것이다.

수정 후 341,043개 정상 추출. `tests/test_aihub_extract.py` 8개로 고정했고, 그 과정에서
플래그 없는 UTF-8 이름을 CP949로 강제 해석하던 2차 결함도 함께 잡았다(리눅스제 zip을
오히려 깨뜨릴 수 있었다).

**게이트 결과**

| 트리 | 표본 | 행 | `do=unknown` | 비고 |
|---|---|---|---|---|
| `v1_2020` | 300 파일 | 108,763 | 0 | split train/valid 정상 |
| `v2_2022` | 400 파일 | 837 | 0 | **한 트리에서 5개 도 전부**, F0 커버리지 99.86%, prosody 3컬럼 생성 |

**71517 대조**: `shared 341,043 / only-in-A 0 / only-in-B 0 / differs 0` → IDENTICAL.
sha256 전수 일치를 확인한 뒤 `raw_data/`(15 GB)를 삭제했다.

**검증**: 테스트 271개 통과(신규 21개), `ruff check` + `ruff format --check` 통과.
