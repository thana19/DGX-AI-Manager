#!/bin/bash
# deploy AI Server v2 จาก Mac ขึ้น DGX — rsync แล้ว restart :9001
# ห้ามแตะ ~/aiserver (hub เดิม) และ :9000 เด็ดขาด
set -euo pipefail
HOST="${DGX_SSH:-dgx}"
DEST="${DGX_DEST:-aiserver2}"
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "▶ rsync → $HOST:~/$DEST"
rsync -a --delete \
  --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
  --exclude '.pytest_cache' --exclude '.DS_Store' \
  "$DIR"/ "$HOST:$DEST/"

echo "▶ ติดตั้ง dependency (venv python3.12)"
ssh "$HOST" "cd $DEST && { [ -x .venv/bin/python ] || python3.12 -m venv .venv; } && .venv/bin/pip -q install -r requirements.txt"

echo "▶ restart :9001"
ssh "$HOST" "bash $DEST/run.sh restart"
sleep 3
ssh "$HOST" "bash $DEST/run.sh status"
