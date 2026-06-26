# ondevice/ — KoDialect On-Device 측정 하니스

Galaxy S25 Ultra(Snapdragon 8 Elite)에서 `sft_merged` 0.5B 를 **실측**해
BRIEF §8.3 각주("실속도는 GGUF Q4_K_M 경로")를 측정으로 닫는다.

```
ondevice/
  quantize/sweep.py        # export 재활용: f16 GGUF → k-quant 변형 스윕
  serve/launch_server.sh   # (폰) llama-server 시동 — model/-fa/cache-type/ctx 파라미터화
  serve/mem_logger.sh      # (폰) /proc/PID VmHWM 폴링 → peak RSS JSONL
  demo/index.html          # (정적) 복붙→/completion stream→라이브 tok/s
  bench/run_bench.sh       # (폰) llama-bench 속도(pp/tg) → JSON
  bench/parse_bench.py     # llama-bench JSON/markdown → 구조화 (순수함수)
  bench/logger.py          # 속도+size+RSS+config → §6 스키마 JSONL
  bench/schema.py          # §6 측정 스키마 단일 출처
  eval/make_eval_prompts.py# (데스크탑) 고정 평가셋 → data/eval_prompts.jsonl (ChatML)
  eval/gen_capture.py      # (데스크탑, forwarded) 폰 서버 구동 → 출력 캡처
  eval/score_offline.py    # (데스크탑) chrF/recon_bleu 채점 + 스키마 병합
  report/pareto.py         # size×speed×quality Pareto 프런티어
  report/plots.py          # Pareto 산점도 + prefill/decode bar (PNG/SVG)
  data/eval_prompts.jsonl  # 강원·경상 고정 평가셋 (make_eval_prompts 가 생성)
```

데스크탑 스크립트는 repo의 conda 환경(예: `balaenoptera`)에서 실행
(`datasets`/`sacrebleu`/`matplotlib` 필요). 폰 스크립트는 Termux + llama.cpp.

---

## 액션 플랜 — 네가 직접 할 일 (순서대로)

### 0단계 · 데스크탑: 모델 변형 만들기 (~10분, GPU 불요)
```bash
conda activate balaenoptera
# llama.cpp 가 없으면: git clone https://github.com/ggml-org/llama.cpp && (cd llama.cpp && cmake -B build && cmake --build build -j)
python ondevice/quantize/sweep.py --merged outputs/sft_merged \
    --variants Q8_0,Q5_K_M,Q4_K_M,Q4_K_S --out_dir ondevice/models
# 고정 평가셋 생성 (한 번)
python ondevice/eval/make_eval_prompts.py --regions gangwondo,gyeongsangdo --n 150
```

### 1단계 · 폰 시동 확인 (★ 먼저 이게 되는지) — Termux
```bash
pkg install git cmake clang
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && cmake -B build && cmake --build build -j
# 모델 옮기기: (노트북) adb push ondevice/models/kodialect-Q4_K_M.gguf /sdcard/Download/
#            (Termux) cp /sdcard/Download/kodialect-Q4_K_M.gguf ~/llama.cpp/models/
~/ko_dialect/ondevice/serve/launch_server.sh -m models/kodialect-Q4_K_M.gguf
# → "serve: http://127.0.0.1:8080" 뜨면 성공. 폰 브라우저로 접속해 내장 webui 확인.
```

### 2단계 · 데모 (복붙→tok/s 뜨는지)
```bash
# (노트북) 폰 서버를 노트북 브라우저로: adb forward tcp:8080 tcp:8080
# demo/index.html 을 브라우저로 열고 endpoint=http://127.0.0.1:8080,
# 지역 선택 후 방언 문장 붙여넣기 → 실행. TTFT/prefill/decode 칩이 뜨면 성공.
# (녹화: 폰 화면, 방언 입력→표준어 출력, tok/s 오버레이 — Deliverable 3)
```

### 3단계 · 정밀 측정 (variant 스윕, JSONL)
```bash
# (폰) 각 variant 속도 — 메모리도 같이 보려면 다른 셸에서 mem_logger 를 띄움
for V in Q8_0 Q5_K_M Q4_K_M Q4_K_S; do
  ~/ko_dialect/ondevice/serve/mem_logger.sh "$(pgrep -f llama-bench | head -1)" 0.5 logs/$V.rss.jsonl &
  ~/ko_dialect/ondevice/bench/run_bench.sh -m models/kodialect-$V.gguf -v $V -r 5 -t 6
done
# (노트북) 스키마 JSONL 기록 (bench json + rss + 모델파일):
for V in Q8_0 Q5_K_M Q4_K_M Q4_K_S; do
  python ondevice/bench/logger.py --variant $V \
    --bench_log ondevice/bench/logs/bench_${V}_p64_n128.json \
    --model_path ondevice/models/kodialect-$V.gguf \
    --rss_json ondevice/bench/logs/$V.rss.jsonl --device SM-S938N
done
# (노트북) 품질: 폰 서버 띄운 채 forwarded 로 생성 캡처 → 채점
for V in Q8_0 Q5_K_M Q4_K_M Q4_K_S; do   # 각 V 로 launch_server 재시동 후
  python ondevice/eval/gen_capture.py --variant $V
  python ondevice/eval/score_offline.py --variant $V
done
```

### 4단계 · 리포트 (Pareto)
```bash
python ondevice/report/pareto.py     # 표 + 프런티어 (★)
python ondevice/report/plots.py      # figs/pareto.{png,svg}, figs/prefill_decode.{png,svg}
# 결과를 docs/ONDEVICE_RESULTS.md 표에 채우고 PREDICTED→MEASURED 로 교체
```

---

## 측정 트랙 vs 데모 트랙 (혼동 금지)
- **측정**(포폴 표): `llama-bench` median → `logger.py` → `pareto.py`. 재현 가능.
- **데모**(녹화): `demo/index.html` 라이브 tok/s. 체감용. 표에 쓰지 않음.

## Acceptance (BRIEF §9)
- [ ] ≥4 variant 폰 실측, median, JSONL.
- [ ] decode tok/s · TTFT · peak RSS · on-disk MB · chrF · recon_bleu 6축 모두 채움.
- [ ] Pareto 프런티어 산출.
- [ ] 부호뒤집힘 가설 MEASURED 확정 (맞든 틀리든).
- [ ] PREDICTED/MEASURED 태그 구분.
- [ ] 데모 녹화 1개 + 재현 1줄.
```
