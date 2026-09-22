#!/bin/bash
# aria2 daemon ของ AI Server v2 — โหมด RPC ให้ server/downloads.py สั่งงาน
# แยกพอร์ต/secret ของตัวเอง ไม่ยุ่งกับ aria2c ที่ hub เดิม (v1) spawn เป็นครั้ง ๆ
PORT="${ARIA2_RPC_PORT:-6800}"
STATE="$HOME/.aiserver2"
SECRET_FILE="$STATE/aria2-secret"
SESSION="$STATE/aria2.session"
LOG="$STATE/logs/aria2.log"
mkdir -p "$STATE/logs"

# secret สร้างครั้งเดียวต่อเครื่อง — สิทธิ์ 600 เพราะใครอ่านได้ = สั่งดาวน์โหลดอะไรก็ได้
if [ ! -s "$SECRET_FILE" ]; then
  (umask 077; head -c 24 /dev/urandom | base64 | tr -d '\n/+=' > "$SECRET_FILE")
fi
SECRET="$(cat "$SECRET_FILE")"

running() { pgrep -f "aria2c.*--rpc-listen-port=$PORT" >/dev/null 2>&1; }

case "${1:-start}" in
  start)
    if running; then echo "aria2 :$PORT ทำงานอยู่แล้ว"; exit 0; fi
    command -v aria2c >/dev/null || { echo "ไม่พบ aria2c — ติดตั้งก่อน: sudo apt install -y aria2"; exit 1; }
    touch "$SESSION"
    # --continue + session file = ปิดเครื่องแล้วโหลดต่อได้ · file-allocation=none คือค่าที่ v1 ใช้จริงบน GB10
    # --log-level เป็นคนละตัวกับ --console-log-level และ default ของมันคือ debug — ต้องตั้ง warn ด้วย
    # ไม่งั้นไฟล์ log โตไม่หยุดเพราะ downloads.py poll RPC ทุกไม่กี่วิ (เจอจริง 2026-09-03: aria2.log 17 GB)
    aria2c --enable-rpc --rpc-listen-port="$PORT" --rpc-listen-all=false \
           --rpc-secret="$SECRET" --daemon=true --continue=true \
           --file-allocation=none --max-tries=5 --retry-wait=10 \
           --max-concurrent-downloads=1 --split=4 --max-connection-per-server=4 \
           --save-session="$SESSION" --input-file="$SESSION" \
           --save-session-interval=30 --auto-save-interval=30 \
           --console-log-level=warn --log-level=warn --log="$LOG" >/dev/null 2>&1
    sleep 1
    running && echo "เริ่ม aria2 :$PORT (log: $LOG)" || { echo "เริ่ม aria2 ไม่สำเร็จ — ดู $LOG"; exit 1; }
    ;;
  stop)
    pkill -f "aria2c.*--rpc-listen-port=$PORT" 2>/dev/null || true
    echo "หยุด aria2 :$PORT แล้ว";;
  status)
    running && echo UP || echo DOWN;;
  secret)
    echo "$SECRET";;
esac
