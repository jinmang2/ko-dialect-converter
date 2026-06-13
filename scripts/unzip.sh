#!/bin/bash

TARGET_DIR="."
REMOVE_ZIP=false
QUIET=false

# 옵션 파싱
while getopts "d:rq" opt; do
  case $opt in
    d) TARGET_DIR="$OPTARG" ;;
    r) REMOVE_ZIP=true ;;
    q) QUIET=true ;;
    *) echo "사용법: $0 [-d <대상_디렉토리>] [-r] [-q]"; exit 1 ;;
  esac
done

log_msg() {
    if [ "$QUIET" = false ]; then
        echo "$1"
    fi
}

if [ ! -d "$TARGET_DIR" ]; then
    echo "❌ 에러: '$TARGET_DIR' 디렉토리가 존재하지 않습니다." >&2
    exit 1
fi

log_msg "=================================================="
log_msg "▶ 데이터셋 압축 해제 및 구조 정규화 작업 시작"
log_msg "   - 대상 경로: $(realpath "$TARGET_DIR")"
log_msg "=================================================="

ZIP_COUNT=$(find "$TARGET_DIR" -type f -name "*.zip" | wc -l)
if [ "$ZIP_COUNT" -eq 0 ]; then
    log_msg "ℹ️ 압축을 해제할 .zip 파일이 없습니다."
    exit 0
fi

find "$TARGET_DIR" -type f -name "*.zip" | sort | while read -r zip_path; do
    EXTRACT_DIR=$(dirname "$zip_path")
    ZIP_FILE=$(basename "$zip_path")
    ZIP_NAME_NO_EXT="${ZIP_FILE%.zip}"
    TARGET_EXTRACT_DIR="$EXTRACT_DIR/$ZIP_NAME_NO_EXT"
    
    # 1. 고속 스킵 로직 (이미 풀려있으면 패스)
    if [ -d "$TARGET_EXTRACT_DIR" ] && [ -n "$(ls -A "$TARGET_EXTRACT_DIR" 2>/dev/null)" ]; then
        log_msg "⏩ [$ZIP_FILE] 이미 압축 해제됨 (스킵)"
        if [ "$REMOVE_ZIP" = true ]; then
            rm "$zip_path"
            log_msg "   🗑️  원본 ZIP 파일 제거 완료"
        fi
        log_msg "--------------------------------------------------"
        continue
    fi
    
    log_msg "▶ [$ZIP_FILE] 압축 해제 중..."
    mkdir -p "$TARGET_EXTRACT_DIR"
    
    # 2. 압축 해제 (출력을 변수에 먼저 담아 파이프라인으로 인한 Exit Code 유실 방지)
    UNZIP_OUT=$(unzip -o -q "$zip_path" -d "$TARGET_EXTRACT_DIR" 2>&1)
    UNZIP_EXIT_CODE=$?
    
    # 3. 종료 코드 검증 (0: 완전 성공, 1: 경로 제거 등 단순 경고 포함 성공)
    if [ $UNZIP_EXIT_CODE -eq 0 ] || [ $UNZIP_EXIT_CODE -eq 1 ]; then
        
        # 보기 싫은 절대 경로 Warning 텍스트만 필터링해서 화면에서 완전히 삭제
        if [ "$QUIET" = false ]; then
            FILTERED_OUT=$(echo "$UNZIP_OUT" | grep -v "stripped absolute path spec" | grep -v "^$")
            if [ -n "$FILTERED_OUT" ]; then
                echo "$FILTERED_OUT"
            fi
        fi

        # 4. 껍데기 절대 경로 추적 및 파괴 로직
        CURRENT_DIR="$TARGET_EXTRACT_DIR"
        while true; do
            ITEM_COUNT=$(ls -A "$CURRENT_DIR" | wc -l)
            if [ "$ITEM_COUNT" -eq 1 ]; then
                SUB_ITEM=$(ls -A "$CURRENT_DIR")
                if [ -d "$CURRENT_DIR/$SUB_ITEM" ]; then
                    CURRENT_DIR="$CURRENT_DIR/$SUB_ITEM"
                else
                    break
                fi
            else
                break
            fi
        done
        
        # 알맹이만 '제목 폴더' 바로 밑으로 이동
        if [ "$CURRENT_DIR" != "$TARGET_EXTRACT_DIR" ]; then
            (
                cd "$CURRENT_DIR" || exit
                find . -maxdepth 1 -mindepth 1 -exec mv {} "$TARGET_EXTRACT_DIR/" \;
            )
            FIRST_SHELL=$(echo "$CURRENT_DIR" | sed "s|^$TARGET_EXTRACT_DIR/||" | cut -d'/' -f1)
            if [ -n "$FIRST_SHELL" ] && [ -d "$TARGET_EXTRACT_DIR/$FIRST_SHELL" ]; then
                rm -rf "$TARGET_EXTRACT_DIR/$FIRST_SHELL"
            fi
        fi
        
        log_msg "   ✔ 정규화 완료: $TARGET_EXTRACT_DIR"
        
        if [ "$REMOVE_ZIP" = true ]; then
            rm "$zip_path"
            log_msg "   🗑️  원본 파일 삭제 완료"
        fi
    else
        # 진짜 치명적 에러 (Exit Code 2 이상)인 경우에만 실패 처리
        echo "   ❌ 실패: $ZIP_FILE 압축 해제 치명적 에러 (Exit Code: $UNZIP_EXIT_CODE)" >&2
        if [ -n "$UNZIP_OUT" ]; then
            echo "$UNZIP_OUT" >&2
        fi
        rm -rf "$TARGET_EXTRACT_DIR"
    fi
    log_msg "--------------------------------------------------"
done

log_msg "🎉 모든 작업이 성공적으로 완료되었습니다."