#!/bin/bash
# ใช้: engines/vllm.sh <model-dir> [args เพิ่ม]   (env MODEL_ID = ชื่อโมเดลใน hub → serve ชื่อนั้น + "auto" ให้ LiteLLM ส่งผ่านตรงได้)
# บทเรียนที่ฝังไว้: ต้องเคลียร์แรมก่อน (ollama/worker) และ util 0.55 คือค่าที่ผ่านจริงบน GB10
# tool calling: agent client (thClaws) ส่ง tools ทุก request → vLLM ต้องเปิด --enable-auto-tool-choice + parser ตรงตระกูลโมเดล
#   (qwen3_xml สำหรับ Qwen3.x — ไม่เปิด = 400 ทุก request ที่มี tools; llama.cpp ไม่ต้องเพราะใช้ jinja ของโมเดล)
#   override ต่อโมเดลด้วย env VLLM_TOOL_PARSER / VLLM_REASONING_PARSER (reasoning parser แยก thinking ออกเป็น reasoning_content)
set -e
source "$(dirname "$0")/common.sh"
MODEL_DIR="$1"; shift || true
[ -d "$MODEL_DIR" ] || { echo "ไม่พบโฟลเดอร์โมเดล: $MODEL_DIR"; exit 1; }

PORT=8000   # vLLM ผูกพอร์ตหลักเท่านั้น (โหมดเสิร์ฟหลายผู้ใช้ ควรได้ทั้งเครื่อง)
stop_engine_port 8000
free_ram_for_big_model
# vLLM จอง gpu-memory-utilization 0.55 ของทั้งเครื่องตอน start — ถ้าแรมของตัวเดิมยังคืนไม่หมด (container เพิ่งถูกลบ)
# จะตายทันที exit 1 → รอจน MemAvailable ≥ 58% (กันชนเล็กน้อย) สูงสุด 90 วิ
NEED_KB=$(( $(awk '/MemTotal/{print $2}' /proc/meminfo) * 58 / 100 ))
for _ in $(seq 1 30); do
  AVAIL_KB=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  [ "$AVAIL_KB" -ge "$NEED_KB" ] && break
  sleep 3
done
echo "แรมว่าง $((AVAIL_KB/1048576)) GB (ต้องการ ≥ $((NEED_KB/1048576)) GB สำหรับ vLLM 0.55)"
# image: ใช้ aiserver-vllm:26.07 (base NVIDIA + xgrammar ที่ tool calling ใช้ได้) — build ให้เองครั้งแรก (~1-2 นาที ต้องมีเน็ต)
# build ไม่ได้ (ไม่มีเน็ต) → ใช้ image ดิบ: แชทปกติได้ แต่ request ที่มี tools จะ 500
IMAGE="aiserver-vllm:26.07"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "build image $IMAGE (base + xgrammar fix สำหรับ tool calling)..."
  docker build -q -t "$IMAGE" "$(dirname "$0")/vllm-image" >/dev/null 2>&1 || { echo "⚠️ build ไม่สำเร็จ — ใช้ image ดิบ (tool calling ใช้ไม่ได้)"; IMAGE="nvcr.io/nvidia/vllm:26.07-py3"; }
fi
docker run -d --name aiserver-vllm --gpus all --ipc=host \
  -p 8000:8000 -v "$(dirname "$MODEL_DIR")":/models \
  "$IMAGE" \
  vllm serve "/models/$(basename "$MODEL_DIR")" \
  --served-model-name "${MODEL_ID:-$(basename "$MODEL_DIR")}" auto \
  --enable-auto-tool-choice --tool-call-parser "${VLLM_TOOL_PARSER:-qwen3_xml}" \
  --reasoning-parser "${VLLM_REASONING_PARSER:-qwen3}" \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --max-num-seqs 4 --gpu-memory-utilization 0.55 --max-model-len 32768 "$@" \
  > "$LOG_DIR/vllm.log" 2>&1
WAIT_CONTAINER=aiserver-vllm wait_health 600
