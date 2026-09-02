## 텍스트 기반 모델에 운율을 녹이는 아이디어

1. 운율 마커 토큰화: 피치 패턴을 기호로 추상화해서 토큰 사이에 삽입
```
원본:    "밥 먹었어?"
운율 추가: "밥↗ 먹었어↘?"
        또는 "<H>밥</H> <L>먹었어</L>"
```
피치 통계로 어절별 라벨링
- `↗` (rising): 어절 평균 F0가 다음 어절보다 +20Hz
- `↘` (falling): -20Hz 이상 낮음
- `〰` (sustained): 변동 적음
- `↑↑` (sharp rise): 강조

2. 종결어미 운율만 살리기
운율 정보는 보통 문장 끝에 의미가 집중됨 (의문/평서 구분 등). 종결어미 직전의 피치 변화만 마커로
```
"밥 먹었어<UP>?"     ← 의문 (피치 상승)
"갔다 왔다<DOWN>."   ← 평서 (피치 하강)
"이거 뭐고<KEEP>?"   ← 경상도식 (끝까지 유지)
```

3. SpeechType 자체를 활용
`Read` 데이터만 골라서 학습하면 어휘 차이 풍부. `Speak` 데이터는 운율 마커 학습용으로 분리
```
Pipeline A: Read 데이터 → 어휘 변환 학습
Pipeline B: Speak 데이터 → 운율 마커 학습
최종: 두 어뎁터를 결합
```

## 가치?
- TTS 입력 전 운율 힌트 생성기
- 챗봇 대답 톤 컨트롤 (경상도 화난 톤 vs 부드러운 톤)
- 방언 더빙용 스크립트 자동 생성
- 음성합성 파이프라인 보조 모듈

```python
# 어절별 F0 통계
def get_prosody_marker(segment_intonations):
    pitches = [p for p in segment_intonations if p > 0]
    if not pitches:
        return ""

    start, end = pitches[0], pitches[-1]
    delta = (end - start) / start * 100

    if delta > 15:
        return "↗"  # 상승
    elif delta < -15:
        return "↘"  # 하강
    elif max(pitches) - min(pitches) > 50:
        return "〜"  # 변동
    return ""
```

## 최종 모델
```
입력:  "이거 뭐야?"
출력1: "이거 뭐고?"           ← 어휘만
출력2: "이거 뭐고↗?"          ← 어휘 + 운율
출력3: "이거 뭐고〜↗?"        ← 어휘 + 운율 + 강조
```