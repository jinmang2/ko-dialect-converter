# KoDialect On-Device — Build Brief (Claude Code 핸드오프)

> 한 줄: 기존 KoDialect의 `SFT-merged 0.5B` 를 **Galaxy S25 Ultra에서 실측**하여,
> §8.3 각주("실속도는 GGUF Q4_K_M 경로")의 미완 주장을 **측정으로 닫는다.**
> 가치는 "폰에서 돌렸다"가 아니라 **측정 기반 경량화/추론 의사결정**(Pareto · prefill/decode 분해 · 부호-뒤집힘 서사)에 있다.

---

## 0. Context (기존 자산 — 재활용)

- 모델: `Qwen2.5-0.5B-Instruct` 기반, **QLoRA SFT → merge** 한 16-bit `sft_merged`.
- 기존 export: `scripts/export_gguf.py` 가 merge/convert/quantize/serve 보유 (GGUF Q4_K_M, llama.cpp).
- 기존 평가 harness: `evaluation/metrics.py` (chrF, recon_bleu), `significance.py` (Koehn paired bootstrap), `leaderboard.py` (Pareto).
- 데스크탑 측정 선례(§8.3): RTX 2060 fp16 vs bnb NF4 — **NF4가 0.5× 느림**(Turing에 4bit dequant 커널 없음).

## 1. Goal (JD 매핑: 자동차 도메인 LLM · 경량화 · 추론 최적화 · 온디바이스)

오프라인 차량 음성 명령 전처리(방언→표준어 정규화) sLLM 을 S25 Ultra에서 end-to-end 구동.
무선 음영지역/프라이버시/레이턴시 때문에 on-device 가 요구된다는 자동차 시나리오로 프레이밍.

## 2. Hardware Target

| 항목 | 값 |
|---|---|
| SoC | Snapdragon 8 Elite for Galaxy (SM8750-AC, 3nm) |
| CPU | Oryon V2: 2×4.47GHz (Phoenix-L) + 6×3.53GHz (Phoenix-M) |
| GPU | Adreno 830 @ ~1200MHz |
| NPU | Hexagon (전세대 대비 +40%) |
| RAM | LPDDR5X 12GB |
| 비고 | sustained 성능 약함 — Wildlife Extreme 1분 후 ~20% throttle → **thermal 곡선이 측정 소재** |

---

## 3. Scope

### Tier 0 — 재활용 구동 (0.5d)
- 기존 `Q4_K_M.gguf` 를 폰에서 구동, 데모 녹화 1개.
- 측정: decode tok/s, TTFT, peak RSS, on-disk MB.

### Tier 1 — 온디바이스 양자화 Pareto (+1~1.5d) ★핵심
- k-quant 스윕: `Q8_0 / Q5_K_M / Q4_K_M / Q4_K_S` (+가능하면 `Q3_K_M`, `IQ4_XS`).
- 각 variant 폰 실측: (on-disk MB × decode tok/s × prefill tok/s × peak RSS) + 품질(chrF / recon_bleu vs gold).
- 산출물: **size–speed–quality Pareto**. §8.3 데스크탑 비교의 모바일 GGUF 확장.
- **핵심 서사 검증**: 2060에선 NF4가 느렸는데(dequant 커널 없음), 모바일 ARM에선 llama.cpp k-quant 커널 + decode가 memory-bandwidth bound라 **Q4가 fp16/Q8보다 빠를 것**으로 예측(PREDICTED). 부호 뒤집히면 "커널 지원이 속도를 지배"(§8.4 결론)를 반대 하드웨어에서 재입증.

### Tier 2 — 온디바이스 KV cache 관리 실측 (합격 후 +3~5d) ★재프레이밍
> 프레이밍 변경: "자동차 RAG 긴-context NPU prefill 데모"는 측정 불가 시나리오라 폐기.
> 대신 **모바일 안에서 KV cache를 직접 관리/측정** → "긴 sys prompt면 KV cache 어떻게 관리?" 면접 답변과 1:1.
> CPU llama.cpp 만으로 측정 가능 (NPU/GPU 런타임 비교는 드롭 또는 각주).

KV cache 관리 4층 (Qwen2.5-0.5B 앵커):
1. **GQA (구조적)**: query 14 / KV 2 head → KV cache MHA 대비 7×↓. KV/token ≈ 2(K,V)×24L×2kv×64×2B(f16) ≈ **12 KB/tok** → 4K≈50MB, 32K≈0.39GB.
2. **System-prompt prefix KV 재사용**: 고정 sys prompt KV 1회 prefill 후 재사용 (prompt cache / slot prefix / `--keep`). 매 턴 re-prefill 제거.
3. **KV cache 양자화**: `--cache-type-k/v q8_0|q4_0`, **`-fa` 필수**(FA 없으면 매 attention dequant → 느려짐). q8_0=안전(ppl<0.1), q4_0=ppl+0.2~0.25 & 긴 ctx에서 dequant 오버헤드로 느려질 수 있음. V가 K보다 민감. K/V 타입 매칭해야 fused FA path(불일치 시 경고 없이 느린 fallback — silent failure).
4. **긴 세션**: context shifting + sink(`n_keep`). 최근 128~256 tok f16 유지 트릭. 서버 PagedAttention(멀티유저용)은 단일유저 모바일엔 덜 중요 — 구분해서 서술.

측정 실험 (CPU llama.cpp, 폰):
- **Exp A** — prefix cache ON/OFF → 긴 sys prompt **TTFT 절감**(amortization).
- **Exp B** — KV cache `f16 / q8_0 / q4_0` (`-fa` on, K/V 타입 매칭) → **peak RSS × max context × 품질(chrF/recon_bleu)**.
  - 킬러: 문헌은 "KV quant 손실 작다"지만 **0.6B급 소형은 저하 큼**(TurboQuant 논의). → 본인 0.5B에서 weight quant vs KV quant 민감도를 실측 확정.
- **Exp C** — KV/token vs context length 곡선 (GQA 7× 효과 시각화).

> 정직성: 0.5B KV quant 민감도·prefix cache TTFT 절감폭은 모두 MEASURED 로 확정 (문헌 인용은 출발점일 뿐). thermal/에너지는 선택, NPU는 미수행.

---

## 4. Tech Stack

- **Tier 0/1**: llama.cpp (Termux, cmake 빌드 — ARM NEON/dotprod/i8mm 자동 감지), 내장 `llama-bench` + 커스텀 로깅.
- **품질 채점(분리)**: 폰은 생성만 → stdout/JSON 캡처 → `adb pull` → **데스크탑 기존 harness 로 chrF/recon_bleu 채점**. (폰에 평가환경 안 만듦.)
- **Tier 2**: 동일 llama.cpp(CPU) + KV cache 관리 플래그 — `-fa`, `--cache-type-k/v`, prompt cache/`--keep`, `--ctx-size`, `n_keep`. 별도 런타임(GPU/NPU) 불요.

---

## 4b. 실행(시동) & 데모 — 네이티브 앱 빌드 불요

전략: Android 앱 개발 안 함. `llama-server`가 이미 HTTP API + 내장 webui + timing(tok/s) 제공 → 그 위에 "복붙→수치" 정적 페이지 1장만 얹는다.

**측정 트랙 vs 데모 트랙 분리**: 정밀 측정(Tier1/2 = `llama-bench` + JSONL median)과 데모(복붙→라이브 tok/s)는 같은 server를 쓰는 별개 트랙. **포폴 표 수치는 항상 `llama-bench` median**, 데모 숫자는 녹화용(체감)으로만.

### 시동 (Termux, S25 Ultra)
```bash
pkg install git cmake clang
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build && cmake --build build -j      # ARM NEON/dotprod/i8mm 자동 감지
# 모델: adb push 또는 Termux 다운로드 → ./models/kodialect-Q4_K_M.gguf
./build/bin/llama-server \
  -m models/kodialect-Q4_K_M.gguf \
  -c 4096 -fa on \
  --cache-type-k q8_0 --cache-type-v q8_0 \   # Tier2 KV-quant 실험 시 토글
  --host 127.0.0.1 --port 8080
```
- 폰 브라우저 `http://127.0.0.1:8080` → 내장 webui 즉시 사용.
- 노트북 녹화: `adb forward tcp:8080 tcp:8080` 후 데스크탑 브라우저 접속.

### 데모 UI (`demo/index.html` — 정적 1파일, 빌드 없음)
복붙 입력 → `/completion`(stream) → 출력 + 실시간 수치. 수치 출처: 응답 `timings` 객체.
- TTFT = 첫 토큰 도착(client-side) / prefill = `timings.prompt_per_second` / decode = `timings.predicted_per_second` / tokens = `prompt_n`·`predicted_n`.
- peak RSS = Termux `VmHWM`(/proc/PID/status) 별도 폴링(라이브 UI엔 선택).

스타터 스케치 (Claude Code가 SSE 파싱/에러처리/스타일 보강 — 정확한 timings 키는 server 응답으로 확인):
```html
<!-- demo/index.html : llama-server 복붙→실행→tok/s (정적, 빌드 X) -->
<input id="ep" value="http://127.0.0.1:8080" style="width:100%">
<textarea id="in" placeholder="방언 문장 붙여넣기" style="width:100%;height:6em"></textarea>
<button onclick="run()">실행</button>
<pre id="out"></pre><div id="m"></div>
<script>
async function run(){
  const ep=document.getElementById("ep").value, t0=performance.now();
  let tFirst=null, out="", buf="";
  const r=await fetch(ep+"/completion",{method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({prompt:document.getElementById("in").value,n_predict:256,stream:true})});
  const rd=r.body.getReader(), dec=new TextDecoder();
  while(true){
    const {done,value}=await rd.read(); if(done) break;
    buf+=dec.decode(value,{stream:true});
    let nl; while((nl=buf.indexOf("\n"))>=0){
      const line=buf.slice(0,nl); buf=buf.slice(nl+1);
      if(!line.startsWith("data:")) continue;
      const j=JSON.parse(line.slice(5));
      if(j.content){ if(tFirst===null) tFirst=performance.now();
        out+=j.content; document.getElementById("out").textContent=out; }
      if(j.stop && j.timings){ const T=j.timings;
        document.getElementById("m").textContent=
          `TTFT ${(tFirst-t0).toFixed(0)}ms | prefill ${T.prompt_per_second?.toFixed(1)} tok/s `+
          `| decode ${T.predicted_per_second?.toFixed(1)} tok/s | gen ${T.predicted_n} tok`; }
    }
  }
}
</script>
```

---

## 5. Repo 구조 (신규 — 기존 repo에 추가)

```
ondevice/
  quantize/        # export_gguf 재활용 래퍼: variant 스윕 생성 (Q8_0/Q5_K_M/Q4_K_M/Q4_K_S/...)
  serve/
    launch_server.sh # termux llama-server 시동 (모델/-fa/cache-type/ctx 파라미터화)
    mem_logger.sh    # /proc/PID/status VmHWM 폴링 → peak RSS JSONL
  demo/
    index.html       # 정적 복붙→실행→tok/s 데모 (빌드 없음, /completion stream)
  bench/
    run_bench.sh   # adb push + termux llama-bench + 커스텀 prompt 셋 실행
    logger.py      # 측정 → JSONL (스키마 §6)
    parse_bench.py # llama-bench stdout → 구조화
  eval/
    score_offline.py  # 폰 출력 캡처 → 데스크탑 chrF/recon_bleu (기존 metrics 재사용)
  report/
    pareto.py      # size×speed×quality Pareto (기존 leaderboard 패턴 재사용)
    plots.py       # Pareto / prefill-decode 분해 / (Tier2) thermal 곡선
  data/
    eval_prompts.jsonl   # 강원·경상 고정 평가셋 (n=150/지역, 기존과 동일 인덱스)
docs/
  ONDEVICE_DECISIONS.md  # 의사결정 카드 (§7)
  ONDEVICE_RESULTS.md    # 실측 표 + Pareto + 정직성 표기
```

## 6. 측정 스키마 (logger → JSONL, 1 row = 1 run)

```json
{
  "variant": "Q4_K_M",
  "runtime": "llama.cpp-cpu",
  "device": "SM-S938N",
  "n_prompt_tokens": 64,
  "n_gen_tokens": 128,
  "prefill_tok_s": null,
  "decode_tok_s": null,
  "ttft_ms": null,
  "peak_rss_mb": null,
  "ondisk_mb": null,
  "cache_type_k": "f16",
  "cache_type_v": "f16",
  "flash_attn": false,
  "n_ctx": 4096,
  "prefix_cache": false,
  "chrF": null,
  "recon_bleu": null,
  "thermal_series": [],
  "n_threads": 6,
  "notes": "PREDICTED|MEASURED"
}
```

- 반복 ≥5회 median 보고. 워밍업 1회 버림.
- 품질은 동일 고정 평가셋(강원/경상 각 n=150)으로, **데스크탑 출력과 폰 출력 동치성**도 1줄 확인(양자화로 품질 얼마 깎였나).

## 7. 의사결정 카드 (docs/ONDEVICE_DECISIONS.md 에 채울 것)

- **O1 측정 선행**: "돌아간다"는 자명 → 가치는 Pareto·분해·부호뒤집힘. 세일즈 금지.
- **O2 k-quant only (Tier0/1)**: bnb 4bit 아님. GGUF k-quant 가 모바일 ARM 최적 커널 보유.
- **O3 품질 채점 분리**: 폰=생성, 데스크탑=채점. 재현성/단일출처 유지.
- **O4 Tier2 = KV cache 관리 실측**: 시나리오 데모 폐기, "긴 sys prompt KV 관리" 면접 답변과 1:1. prefix 재사용(TTFT) + cache 양자화(메모리/품질) + GQA 구조. 서버 PagedAttention과 온디바이스를 구분 서술.
- **O5 thermal**: S25U sustained 약함 → 연속추론 곡선을 자동차(장시간) 서사 차별점으로.

## 8. Deliverables

1. `ONDEVICE_RESULTS.md`: variant×(size/speed/quality) 표 + Pareto + (부호뒤집힘 확인 여부) + 정직성 표기.
2. Pareto plot (PNG/SVG) + prefill/decode 분해 bar.
3. 데모 녹화(폰 화면, 방언 입력→표준어 출력, tok/s 오버레이).
4. 슬라이드 2~3장(기존 PT 부록): 제약→측정→Pareto→정직한 결론.
5. (Tier2, 후속) KV cache 관리 결과: prefix cache TTFT 절감 + cache f16/q8_0/q4_0 (RSS×max-ctx×품질, 0.5B 민감도) + KV/token 곡선.

## 9. Acceptance Criteria

- [ ] ≥4개 variant 폰 실측 완료, 반복 median, JSONL 기록.
- [ ] decode tok/s · TTFT · peak RSS · on-disk MB · chrF · recon_bleu 6개 축 모두 채움.
- [ ] Pareto 프런티어 산출(지배당하지 않은 variant 집합).
- [ ] **부호뒤집힘 가설**(Q4 vs fp16 decode 속도) MEASURED 로 확정 — 맞든 틀리든 결과 기록.
- [ ] 모든 예측치는 `PREDICTED` 태그, 실측은 `MEASURED` 태그로 구분.
- [ ] 데모 녹화 1개, 재현 명령 1줄.

## 10. 정직성 표기 (기존 포폴 습관 유지)

- decode tok/s 추정·NPU 배수는 모두 PREDICTED (roofline/문헌) → 실측 전까지 단정 금지.
- 0.5B 소형이라 "돌렸다" 임팩트 약함을 명시하고, 가치를 측정 깊이로 정당화.
- Tier 2 미수행분은 "예측 + 근거"로만 보고, 한 것처럼 쓰지 않음.

## 11. Risks / Mitigations

- Termux 빌드 삽질 → prebuilt GGUF 로더 앱(PocketPal/ChatterUI류)로 우회 가능.
- adb/USB 권한 → 무선 디버깅 fallback.
- NPU(Tier2) 변환 난이도(ExecuTorch `.pte`/QNN) → GPU(MLC) 먼저, NPU 후순위.
- 품질 동치성 깨짐(폰 토크나이저/샘플링 차이) → seed·greedy 고정, 데스크탑과 동일 디코딩 파라미터.
