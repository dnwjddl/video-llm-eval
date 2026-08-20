#!/usr/bin/env bash
# debug/ 아래의 진단 출력 파일들을 GitHub에 푸시 — Claude가 직접 읽을 수 있게
set -e
cd "$(dirname "$0")/.."
git add debug/*_output.txt
git commit -m "debug output" || echo "커밋할 변경 없음"
git push
echo "푸시 완료 — 이제 Claude에게 '보냈어'라고 말하면 됩니다."
