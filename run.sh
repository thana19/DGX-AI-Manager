#!/bin/bash
# AI Server v2 launcher — start/stop/restart/status (default :9001)
# พอร์ตปรับด้วย AISERVER2_PORT · ห้ามชนกับ hub เดิม :9000
DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${AISERVER2_PORT:-9001}"; export AISERVER2_PORT="$PORT"
LOG="$HOME/.aiserver2/logs/server.log"; mkdir -p "$(dirname "$LOG")"
UV="$DIR/.venv/bin/uvicorn"; [ -x "$UV" ] || UV="$(command -v uvicorn)"
[ -x "$UV" ] || { echo "ไม่พบ uvicorn — สร้าง venv ก่อน: python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt"; exit 1; }
case "${1:-restart}" in
  stop)
    pkill -f "uvicorn server.main:app.*--port $PORT" 2>/dev/null || true; echo "หยุด AI Server v2 :$PORT แล้ว";;
  start|restart)
    pkill -f "uvicorn server.main:app.*--port $PORT" 2>/dev/null || true; sleep 1
    # aria2 ต้องขึ้นก่อน server เสมอ — downloads.py สั่งงานผ่าน RPC ของมัน
    bash "$DIR/services/aria2.sh" start || echo "⚠️ aria2 ไม่ขึ้น — ดาวน์โหลดจะใช้ไม่ได้ (ส่วนอื่นยังทำงาน)"
    export ARIA2_RPC_SECRET="$(bash "$DIR/services/aria2.sh" secret)"
    export ARIA2_RPC_URL="http://127.0.0.1:${ARIA2_RPC_PORT:-6800}/jsonrpc"
    # detach สองชั้นแบบเดียวกับ v1 — กัน ssh แขวนเพราะ subshell ถือ stdout pipe ไว้
    ( cd "$DIR" && setsid "$UV" server.main:app --host 0.0.0.0 --port "$PORT" > "$LOG" 2>&1 < /dev/null & ) > /dev/null 2>&1 < /dev/null
    echo "เริ่ม AI Server v2 :$PORT (log: $LOG)";;
  status)
    curl -s -m 2 "http://127.0.0.1:$PORT/api/health" >/dev/null && echo UP || echo DOWN;;
esac
