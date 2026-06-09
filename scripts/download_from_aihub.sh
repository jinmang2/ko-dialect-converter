#!/bin/bash

# 기본값 설정
DATASET_KEY=""
DEST_DIR="."
API_KEY=""
AIHUBSHELL_OPT_PATH=""

# 실행 옵션 파싱 (-k: 데이터셋키, -d: 목적지디렉토리, -a: API키, -p: aihubshell 경로)
while getopts "k:d:a:p:" opt; do
  case $opt in
    k) DATASET_KEY="$OPTARG" ;;
    d) DEST_DIR="$OPTARG" ;;
    a) API_KEY="$OPTARG" ;;
    p) AIHUBSHELL_OPT_PATH="$OPTARG" ;;
    *) echo "사용법: $0 -k <dataset_key> [-d <dest_dir>] [-a <api_key>] [-p <aihubshell_path>]"; exit 1 ;;
  esac
done

# 필수 인자 체크
if [ -z "$DATASET_KEY" ]; then
    echo "❌ 에러: 데이터셋 키(-k)는 필수 항목입니다."
    exit 1
fi

# --- [aihubshell 위치 탐색 우선순위] ---
# 1. 인자 입력 (-p) -> 2. 환경변수 ($AIHUBSHELL_PATH) -> 3. 현재 디렉토리 -> 4. 홈 디렉토리 (~/)
PRE_PATH="${AIHUBSHELL_OPT_PATH:-$AIHUBSHELL_PATH}"

if [ -n "$PRE_PATH" ] && [ -f "$PRE_PATH" ]; then
    AIHUBSHELL_FINAL="$(realpath "$PRE_PATH")"
elif [ -f "./aihubshell" ]; then
    AIHUBSHELL_FINAL="$(pwd)/aihubshell"
elif [ -f "$HOME/aihubshell" ]; then
    AIHUBSHELL_FINAL="$HOME/aihubshell"
elif command -v aihubshell &> /dev/null; then
    AIHUBSHELL_FINAL="aihubshell"
else
    echo "❌ 에러: aihubshell 스크립트를 찾을 수 없습니다."
    echo "지정 경로, 현재 폴더, 또는 홈 디렉토리($HOME)를 확인해 주세요."
    exit 1
fi

# API Key 옵션 문자열 구성
API_OPT=""
if [ -n "$API_KEY" ]; then
    API_OPT="-aihubapikey $API_KEY"
fi

echo "=================================================="
echo "✔ 사용 중인 aihubshell: $AIHUBSHELL_FINAL"
echo "▶ 1단계: 라벨링 데이터 파일키(filekey) 추출 중..."
echo "=================================================="

# 파일 트리 구조 읽기 및 TL_, VL_ 키 추출
FILE_KEYS=$(bash "$AIHUBSHELL_FINAL" -mode l -datasetkey "$DATASET_KEY" $API_OPT | \
            awk -F'|' '/TL_|VL_/ {gsub(/[[:space:]]/, "", $3); print $3}' | \
            paste -sd, -)

if [ -z "$FILE_KEYS" ]; then
    echo "❌ 라벨링 파일을 찾지 못했거나 API Key/네트워크 상태를 확인하세요."
    exit 1
fi

echo "✔ 추출된 파일키: $FILE_KEYS"

echo -e "\n=================================================="
echo "▶ 2단계: 다운로드 목적지 설정 및 실행"
echo "=================================================="

# 목적지 폴더로 이동 (지정하지 않으면 현재 위치인 ~/ko_dialect 유지)
mkdir -p "$DEST_DIR"
cd "$DEST_DIR" || exit 1
echo "✔ 현재 다운로드 진행 경로: $(pwd)"

# 최종 다운로드 실행
bash "$AIHUBSHELL_FINAL" -mode d -datasetkey "$DATASET_KEY" -filekey "$FILE_KEYS" $API_OPT