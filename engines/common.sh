#!/bin/bash
# AI Server — ยูทิลกลางของ engine ทุกตัว
# กติกาเหล็ก: LLM หลักเสิร์ฟบน :8000 เสมอ (thClaws และ client ในเครื่องผูกพอร์ตนี้)
# โมเดลเสริมโหลดเพิ่มได้ที่ :8001, :8002, ... ภายใต้งบแรม (GPU manager ใน hub คุม)

LOG_DIR="$HOME/.aiserver/logs"
mkdir -p "$LOG_DIR"
PORT="${PORT:-8000}"

stop_engine_port() {
  # หยุดเฉพาะตัวที่ถือพอร์ตนี้ (vLLM container ผูก :8000 เท่านั้น)
  # สำคัญ: ทุกคำสั่งต้อง || true — พอร์ตว่างคือเรื่องปกติ ห้ามให้ set -e ฆ่าสคริปต์
  local port="${1:-$PORT}"
  if [ "$port" = "8000" ]; then docker rm -f aiserver-vllm >/dev/null 2>&1 || true; fi
  fuser -k "$port/tcp" 2>/dev/null || true
  sleep 3
}

free_ram_for_big_model() {
  # ก่อนโหลดโมเดลใหญ่ (vLLM/Q8): ปลด ollama ออกจากแรม
  curl -s -m 3 http://127.0.0.1:11434/api/generate -d '{"model":"","keep_alive":0}' >/dev/null 2>&1
}

wait_health() {
  # รอจนพอร์ตตอบ (สูงสุด $1 วินาที, default 180)
  local timeout="${1:-180}" t=0
  until curl -s -m 2 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q '"id"'; do
    sleep 3; t=$((t+3))
    [ "$t" -ge "$timeout" ] && echo "TIMEOUT" && return 1
    # container/process ที่รอตายไปแล้ว (ถูกหยุด/OOM) → เลิกรอทันที ไม่ต้องรอจนหมดเวลา
    if [ -n "${WAIT_CONTAINER:-}" ] && [ "$t" -ge 6 ] && ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$WAIT_CONTAINER"; then
      echo "DIED: container $WAIT_CONTAINER หายไประหว่างรอ (ดู log)"; return 1
    fi
    if [ -n "${WAIT_PID:-}" ] && ! kill -0 "$WAIT_PID" 2>/dev/null; then
      echo "DIED: process $WAIT_PID ตายระหว่างรอ (ดู log)"; return 1
    fi
  done
  echo "READY"
}
