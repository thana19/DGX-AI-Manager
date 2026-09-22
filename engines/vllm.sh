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

# flag ตาม env: ใช้ "${VAR-default}" (ไม่มี ":") ไม่ใช่ "${VAR:-default}" — ห้ามพลาดจุดนี้
#   ":-" มองค่าว่าง ("") เหมือน "ไม่ได้ตั้ง" แล้วเอา default มาแทน ⇒ ปิด parser/MTP ไม่ได้เลยเพราะ
#   "-" เท่านั้นที่แยก "ไม่ได้ตั้ง env" (unset → ใช้ default เดิม) ออกจาก "ตั้งเป็นค่าว่างเพื่อปิดชัดเจน"
TOOL_PARSER="${VLLM_TOOL_PARSER-qwen3_xml}"
REASONING_PARSER="${VLLM_REASONING_PARSER-qwen3}"
SPECULATIVE="${VLLM_SPECULATIVE-mtp}"   # default เปิด (mtp) ตามเดิม — โมเดลที่ไม่มี MTP head ต้องสั่งปิดเองผ่าน engine_env

# สร้าง flag array ก่อน docker run — ห้ามเขียน "[ -n "$X" ] && ARR+=(...)" เดี่ยว ๆ เพราะสคริปต์รัน
# ด้วย set -e: ถ้าเงื่อนไขเป็นเท็จ "&&" คืน exit code 1 แล้วสคริปต์ตายทั้งสายทันที ต้องใช้ if/then เท่านั้น
EXTRA_FLAGS=()
if [ -n "$TOOL_PARSER" ]; then
  EXTRA_FLAGS+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER")
fi
if [ -n "$REASONING_PARSER" ]; then
  EXTRA_FLAGS+=(--reasoning-parser "$REASONING_PARSER")
fi
if [ "$SPECULATIVE" = "mtp" ]; then
  EXTRA_FLAGS+=(--speculative-config '{"method":"mtp","num_speculative_tokens":3}')
fi

# env ของ vLLM ต้องส่งเข้า container ด้วย -e — ตั้งไว้บน host เฉย ๆ ไปไม่ถึงตัว vLLM ที่รันข้างใน
#   DeepGEMM: บน GB10 แปลง scale-factor layout ของ FP8 block-quant ไม่ได้ ⇒ RuntimeError
#   "Unknown SF transformation" ตั้งแต่ process_weights_after_loading แล้ว EngineCore ตายทั้งตัว
#   (เจอจริง 2026-09-03 กับ qwen3-8b-fp8 · ปิดแล้วขึ้นได้ใน 120 วิ ทั้ง chat และ tools ปกติ)
#   โมเดลที่อยากลองเปิด: ตั้ง VLLM_USE_DEEP_GEMM=1 ใน engine_env ของ entry นั้น
ENV_FLAGS=(-e "VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM-0}")
# VLLM_* ตัวอื่นที่ engine_env ตั้งไว้ ส่งต่อเข้า container ให้หมด — ยกเว้น 3 ตัวที่เป็น flag ของสคริปต์นี้เอง
for _v in ${!VLLM_@}; do
  case "$_v" in
    VLLM_TOOL_PARSER|VLLM_REASONING_PARSER|VLLM_SPECULATIVE|VLLM_USE_DEEP_GEMM) continue ;;
  esac
  ENV_FLAGS+=(-e "$_v=${!_v}")
done

# "docker run -d" พิมพ์แค่ container ID ออก stdout — redirect ทับ log ตรง ๆ ไฟล์จึงเหลือแค่ id
# (เจอจริง: หน้าเว็บโชว์ error เป็น hash 64 ตัวแทน log จริง จนตามสาเหตุไม่ได้)
# → เก็บ id ไว้ แล้วดึง log จริงจาก docker logs -f มาเขียนลงไฟล์แทน
if ! CID=$(docker run -d --name aiserver-vllm --gpus all --ipc=host \
  -p 8000:8000 -v "$(dirname "$MODEL_DIR")":/models "${ENV_FLAGS[@]}" \
  "$IMAGE" \
  vllm serve "/models/$(basename "$MODEL_DIR")" \
  --served-model-name "${MODEL_ID:-$(basename "$MODEL_DIR")}" auto \
  "${EXTRA_FLAGS[@]}" \
  --max-num-seqs 4 --gpu-memory-utilization 0.55 --max-model-len "${CTX:-32768}" "$@" \
  2> "$LOG_DIR/vllm.log"); then
  echo "docker run ไม่สำเร็จ:"; cat "$LOG_DIR/vllm.log"; exit 1
fi
# stream log ลงไฟล์แบบ detach — setsid + ปิด fd ครบ กัน subshell ถือ pipe ไว้แล้วทำ ssh แขวน (บทเรียนเดียวกับ run.sh)
( setsid docker logs -f "$CID" > "$LOG_DIR/vllm.log" 2>&1 < /dev/null & ) > /dev/null 2>&1 < /dev/null
WAIT_CONTAINER=aiserver-vllm wait_health 600
