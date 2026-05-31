## Summary

<!-- 변경 내용을 1-3줄로 요약해주세요 -->

## Type of Change

- [ ] Bug fix
- [ ] New feature / enhancement
- [ ] Refactor / cleanup
- [ ] Data pipeline change
- [ ] Model / training change
- [ ] Evaluation / metrics change
- [ ] Serving / export change
- [ ] CI/CD / tooling
- [ ] Documentation

## Related Issues

<!-- Closes #xxx / Related to #xxx -->

## Changes

<!-- 무엇을 왜 바꿨는지 설명 -->

## Test Plan

- [ ] `pytest tests/ -x -q -m "not gpu"` 통과
- [ ] `ruff check . && black --check . && isort --check .` 통과
- [ ] GPU 관련 변경이라면: 로컬 RTX 2060에서 직접 테스트
- [ ] GGUF export 관련이라면: 변환 및 추론 동작 확인

## Hardware Notes (ML changes only)

<!-- RTX 2060 제약 관련 사항 (VRAM, fp16 강제, 배치 크기 등) -->

## Checklist

- [ ] bf16 사용하지 않음 (fp16 또는 int4만 사용)
- [ ] 대용량 파일 (*.pt, *.bin, *.gguf) 커밋하지 않음
- [ ] 하이퍼파라미터를 Python 코드가 아닌 `configs/*.yaml`에 정의함
- [ ] 새 public 함수에 타입 힌트 추가
