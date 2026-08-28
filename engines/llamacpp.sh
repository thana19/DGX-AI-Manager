#!/bin/bash
# ใช้: PORT=8000 engines/llamacpp.sh <model.gguf> [args เพิ่ม เช่น -md mtp.gguf --spec-type draft-mtp]
set -e
source "$(dirname "$0")/common.sh"
MODEL="$1"; shift || true
[ -f "$MODEL" ] || { echo "ไม่พบไฟล์โมเดล: $MODEL"; exit 1; }

# หา binary: env > build ในบ้าน > PATH — ไม่เจอให้ฟ้องชัดๆ (ไม่ปล่อยให้ nohup ตายเงียบ)
BIN="${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}"
[ -x "$BIN" ] || BIN="$(command -v llama-server || true)"
[ -x "$BIN" ] || { echo "ไม่พบ llama-server ในเครื่อง — ติดตั้งจากหน้า AI Server (ซอฟต์แวร์) หรือ build llama.cpp ก่อน"; exit 1; }

stop_engine_port "$PORT"
# binary ที่ก๊อปข้ามเครื่องมี RUNPATH ฝัง path เครื่องต้นทาง — ชี้ lib ไปโฟลเดอร์ของ binary เองเสมอ
export LD_LIBRARY_PATH="$(dirname "$BIN")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# reasoning-format auto (ค่ามาตรฐาน): ความคิดออกเป็นฟิลด์ reasoning_content — thClaws/client ที่ต่อตรง :8000
# แสดงกล่อง Thinking ได้ถูกต้อง (บทเรียน: 'none' ทำ <think> ฝังใน content แล้วโดน markdown ฝั่ง client กลืนหาย)
nohup "$BIN" -m "$MODEL" \
  --host 0.0.0.0 --port "$PORT" -c "${CTX:-65536}" -ngl 99 --jinja \
  --reasoning-format "${REASONING_FORMAT:-auto}" --metrics "$@" \
  > "$LOG_DIR/llamacpp-$PORT.log" 2>&1 &
WAIT_PID=$! wait_health 180
