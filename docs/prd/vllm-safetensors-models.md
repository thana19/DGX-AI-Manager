# PRD — เพิ่มโมเดล safetensors (vLLM) จาก Hugging Face

สถานะ: รอ confirm · วันที่ 2026-09-01 · ต่อจากเฟส 1.5

## Problem Statement

- วาง `Qwen/Qwen3-8B-FP8` ในช่อง "เพิ่มโมเดล" แล้วหน้าเว็บขึ้น `arch: ไม่ทราบ · ctx_train: -` และ "ไม่พบ quant ในรีโปนี้" — ดูเหมือนพัง แต่จริง ๆ คือยังไม่รองรับ
- สาเหตุ: เส้นทาง "เพิ่มโมเดล" เป็น GGUF-only ทั้งเส้น — `hf.group_quants()` (server/hf.py:154) ข้ามไฟล์ที่ไม่ลงท้าย `.gguf` · `main.py:264-272` อ่าน arch/ctx จาก GGUF header เท่านั้น · `main.py:259,274` ตรึง engine เป็น `llamacpp`
- เครื่องนี้มี vLLM เป็น engine อยู่แล้ว (catalog มี entry `qwen38-nvfp4-vllm`) แต่ต้องให้ vendor เขียน entry มือเท่านั้น ผู้ใช้เพิ่มเองไม่ได้

## Solution (ภาพรวม)

ทำให้ `/api/models/resolve` แยกได้ 2 รูปแบบ repo แล้วตอบพร้อมบอก engine/format:

| format | ที่มาของ arch/ctx | engine | quant |
|---|---|---|---|
| `gguf` | GGUF header ผ่าน HTTP Range (ของเดิม ไม่แตะ) | llamacpp | ตามกติกาเดิม |
| `safetensors` | `config.json` ของ repo | vllm | 1 กลุ่ม = ทั้ง repo |

## ข้อเท็จจริงที่ verify แล้ว (2026-09-01 — ยิงของจริง ไม่ใช่เดา)

1. `Qwen/Qwen3-8B-FP8` = safetensors ล้วน 12 ไฟล์ 9.45 GB ไม่มี `.gguf` เลย (HF API `?blobs=true`)
2. `config.json` ของ repo นั้นมี `architectures: ["Qwen3ForCausalLM"]` · `model_type: qwen3` · `max_position_embeddings: 40960` · `quantization_config.quant_method: fp8`
3. **`max_position_embeddings` อาจซ้อนอยู่ใน `text_config`** — โมเดล NVFP4 ที่รันอยู่บนเครื่อง (`~/models/nvfp4/qwen3.8-27b-nvfp4/config.json`) มีค่าอยู่ที่ `text_config.max_position_embeddings = 262144` ไม่ใช่ระดับบนสุด ⇒ parser ต้องดูทั้งสองชั้น
4. รายการไฟล์ที่ vendor curate ไว้ใน `catalog.yaml:106-117` = ไฟล์ทั้ง repo **ยกเว้น** `.gitattributes` และ `README.md` (เทียบกับ HF API ของ `unsloth/Qwen3.8-27B-NVFP4` แล้วตรงกันเป๊ะ) ⇒ กติกาจัดกลุ่ม safetensors ใช้แบบนี้ได้เลย
5. โมเดลที่มี MTP มีไฟล์ `model_mtp.safetensors` จริงในโฟลเดอร์ (NVFP4 มี · FP8 ไม่มี) ⇒ ตรวจ MTP จากรายชื่อไฟล์ได้ก่อนดาวน์โหลด
6. เครื่องเป็น GB10 compute capability 12.1 (Blackwell) — FP8 e4m3 รองรับในระดับฮาร์ดแวร์

## จุดอ่อนของเดิมที่ต้องแก้ไปด้วย (พี่หนุ่ม confirm แล้ว)

| # | อาการ | ที่มา |
|---|---|---|
| 1 | `--speculative-config mtp` ใส่ทุกโมเดลแบบตายตัว — โมเดลที่ไม่มี MTP head จะโหลดไม่ขึ้น | engines/vllm.sh:38 |
| 2 | `ctx` ที่ UI ส่งมาถูกเมิน — vllm ตรึง `--max-model-len 32768` | engines/vllm.sh:39 · main.py:807 ตั้ง env CTX ไว้แต่ไม่มีใครอ่าน |
| 3 | `is_ready()` ของโมเดลแบบโฟลเดอร์เช็คแค่ "โฟลเดอร์ไม่ว่าง" — ดาวน์โหลดค้างกลางทางก็ขึ้นว่าพร้อม แล้ว activate พัง | server/catalog.py:212-214 |

## User Stories

1. ในฐานะเจ้าของเครื่อง อยากวาง repo id ของโมเดล FP8/NVFP4 แล้วระบบบอกได้ว่า arch อะไร ctx เท่าไหร่ ขนาดกี่ GB แรมพอไหม — เหมือนที่ทำได้กับ GGUF
2. อยากกด "เพิ่มเข้าคลัง" แล้วได้ entry ที่ถูกต้องพร้อมดาวน์โหลด โดยไม่ต้องแก้ YAML เอง
3. อยากให้ระบบตั้ง parser/MTP ให้เองตามตระกูลโมเดล เพราะจำไม่ได้ว่าโมเดลไหนใช้ตัวไหน

## Implementation Decisions

| # | การตัดสินใจ | เหตุผล |
|---|---|---|
| 1 | **arch/ctx ของ safetensors อ่านจาก `config.json`** — ยิง `https://huggingface.co/{repo}/resolve/main/config.json` ตรง ๆ (ไฟล์เล็กหลักร้อย byte) · parser ต้องอ่าน `architectures[0]`, `model_type`, `max_position_embeddings` **ทั้งระดับบนสุดและใน `text_config`** | เป็นแหล่งเดียวที่มีข้อมูลจริง · การซ้อนใน text_config เจอจริงกับโมเดลที่รันอยู่ |
| 2 | **1 repo safetensors = 1 quant group** — key มาจาก `quantization_config.quant_method` (uppercase เช่น `FP8`, `NVFP4`) ไม่มีก็ใช้ `BF16`/`safetensors` | root-level repo ไม่มีทั้งโฟลเดอร์ย่อยและ token quant ในชื่อไฟล์ กติกาข้อ 3/4 ของ group_quants ใช้ไม่ได้ |
| 3 | **ไฟล์ที่รวมในกลุ่ม = ทุกไฟล์ ยกเว้น `.gitattributes`, `README.md`, `LICENSE*`, `*.md`, ไฟล์ภาพ** | ตรงกับที่ vendor curate ไว้เองใน catalog.yaml (ข้อเท็จจริงข้อ 4) |
| 4 | **`model_mtp.safetensors` ต้องอยู่ในกลุ่มหลัก ห้ามถูกจับเป็น companion** | `companion_kind()` เห็นคำว่า mtp แล้วจะแยกออกเป็น draft ซึ่งเป็นกติกาของ llama.cpp (`-md`) · vLLM ต้องการไฟล์นี้อยู่ในโฟลเดอร์โมเดล |
| 5 | **`QuantGroup` เพิ่มฟิลด์ `format: "gguf" \| "safetensors"`** และ response ของ resolve เพิ่ม `engine` กับ `format` | frontend ต้องรู้ว่ากำลังดูอะไรถึงจะสร้าง payload ถูก |
| 6 | **compat ของ safetensors เช็คด้วย `check_arch(arch, "vllm", info=engines.detect("vllm"), requires={"vllm_image": <image ที่ตรวจเจอ>})`** | engines.py:416 บังคับว่า vllm ต้องมี `requires` ถึงจะได้ `OK` ไม่งั้นค้างที่ `UNKNOWN` ตลอด |
| 7 | **เพิ่มฟิลด์ใหม่ 1 ตัวใน `ModelEntry`: `engine_env: dict[str, str] \| None`** — main.py เอาไป `env.update()` ก่อนเรียก engine script | `ModelEntry` เป็น `extra="forbid"` เพิ่ม field ไม่ได้ถ้าไม่ประกาศ · ตัวเดียวครอบคลุมทั้ง parser/MTP และรองรับ engine อื่นในอนาคต ไม่ต้องเพิ่ม field รายตัว |
| 8 | **`engines/vllm.sh` เปลี่ยนมาสร้าง flag ตาม env แบบมีเงื่อนไข** — `VLLM_TOOL_PARSER` / `VLLM_REASONING_PARSER` / `VLLM_SPECULATIVE` · ใช้ `${VAR-default}` (ไม่มี colon) เพื่อแยก "ไม่ได้ตั้ง" ออกจาก "ตั้งเป็นค่าว่างเพื่อปิด" · `--max-model-len "${CTX:-32768}"` | แก้จุดอ่อน 1 กับ 2 พร้อมกัน โดยไม่พึ่งสมมติฐานว่า argparse ของ vLLM ให้ flag ท้ายชนะ (ยังไม่ verify — เลี่ยงดีกว่า) |
| 9 | **default ของ MTP ใน vllm.sh คงเป็น "เปิด" ตามเดิม** (`${VLLM_SPECULATIVE-mtp}`) · โมเดลใหม่ที่ตรวจแล้วไม่มี MTP → resolve ต้องส่ง `engine_env: {VLLM_SPECULATIVE: ""}` มาปิดให้ชัดเจน · `catalog.yaml` ไม่ต้องแก้ | พี่หนุ่มเลือกไม่แตะพฤติกรรมของโมเดล NVFP4 ที่ใช้งานได้อยู่ · ทำให้การแยก "ไม่ได้ตั้ง" กับ "ตั้งเป็นค่าว่าง" (ข้อ 8) กลายเป็นแกนหลักของกลไกนี้ ห้ามใช้ `${VAR:-default}` เด็ดขาด |
| 10 | **เลือก parser ตาม `model_type` ด้วยตารางแม็ป** (`qwen3`/`qwen3_5` → `qwen3_xml` + `qwen3` · `llama` → `llama3_json` · `mistral` → `mistral` · `deepseek_v3` → `deepseek_v3`) · **ตระกูลที่ไม่รู้จัก = ไม่ใส่ parser เลย** (แชทได้ปกติ tool calling ใช้ไม่ได้ และหน้าเว็บต้องบอกผู้ใช้) | พี่หนุ่มเลือก "รองรับทุกตระกูล" · เดา parser ผิดอันตรายกว่าไม่ใส่ เพราะ request ที่มี tools จะ 400/500 ทั้งหมด |
| 11 | **path ของ entry vllm = โฟลเดอร์** `~/models/<quant ตัวเล็ก>/<ชื่อ repo>` เช่น `~/models/fp8/qwen3-8b-fp8` | vllm.sh mount โฟลเดอร์แม่ (`-v $(dirname "$MODEL_DIR"):/models`) ⇒ โมเดลต้องอยู่ลึกอย่างน้อย 1 ชั้น |
| 12 | **`is_ready()` ของ folder entry เข้มขึ้น**: โฟลเดอร์ไม่ว่าง **และ** ไม่มีไฟล์ `.aria2` ค้าง **และ** ถ้ามี `dl` ต้องมีครบทุก basename | แก้จุดอ่อน 3 · ใช้กติกาเดียวกับฝั่ง GGUF ที่กัน `.aria2` อยู่แล้ว |
| 13 | **แก้บั๊กที่เจอระหว่างทาง: `state.resolveRepoId` เก็บข้อความดิบที่ผู้ใช้พิมพ์ ไม่ใช่ `resolved_repo_id`** ⇒ ถ้าวาง URL เต็ม URL ดาวน์โหลดที่สร้างจะผิด | index.html:1253 ใช้ที่ 1386 · แก้เลยเพราะกระทบ payload ที่งานนี้สร้าง |

## Modules ที่ต้องแก้

| ไฟล์ | สิ่งที่ทำ | ประมาณการ |
|---|---|---|
| `server/hf.py` | เพิ่ม `fetch_config()`, `group_weights()`, ฟิลด์ `format` ใน `QuantGroup`, ตารางแม็ป parser · **ไม่แตะ `group_quants()` เดิม** | ปานกลาง |
| `server/main.py` | resolve แยกสองทาง · response เพิ่ม `engine`/`format`/`requires`/`engine_env` · `env.update(entry.engine_env)` ตอน activate | ปานกลาง |
| `server/catalog.py` | ฟิลด์ `engine_env` ใน `ModelEntry` · `is_ready()` folder เข้มขึ้น | เล็ก |
| `engines/vllm.sh` | flag แบบมีเงื่อนไขจาก env + `--max-model-len` จาก CTX | เล็ก |
| `server/static/index.html` | payload builder แยกตาม `format` · carry `requires`/`engine_env` · แสดง format/engine ในตาราง · เตือนเมื่อไม่มี parser · แก้บั๊ก resolveRepoId | ปานกลาง |
| `tests/` + `tests/fixtures/` | fixture safetensors ใหม่ + test ตามด้านล่าง | ปานกลาง |

## Test Coverage ที่ตั้งใจ

- `test_hf.py`: `fetch_config()` อ่าน arch/ctx จากทั้งระดับบนสุดและ `text_config` · `group_weights()` คัดไฟล์ถูก (ตัด README/.gitattributes, เก็บ `model_mtp.safetensors` ไว้ในกลุ่ม) · key มาจาก quant_method · repo GGUF เดิมยังผ่านกติกาเดิมทุกข้อ (regression)
- `test_main.py`: resolve repo safetensors → ได้ `format: "safetensors"`, `engine: "vllm"`, 1 quant, arch/ctx จาก config · repo GGUF → ยังตอบเหมือนเดิมเป๊ะ · activate ส่ง `engine_env` เข้า env จริง · **repo ที่ไม่มี `model_mtp.safetensors` → resolve ต้องคืน `engine_env` ที่มี `VLLM_SPECULATIVE: ""` (ปิด MTP)** · repo ที่มี → ไม่ต้องส่งคีย์นี้ (ปล่อยให้ default เปิด)
- `test_vllm_sh` (bash, ถ้าคุ้ม): ยืนยันว่า `VLLM_SPECULATIVE=""` ทำให้ไม่มี `--speculative-config` ใน command ส่วนไม่ตั้งค่าเลยยังมี — จุดนี้คือหัวใจของข้อ 8/9 พลาดแล้วโมเดลโหลดไม่ขึ้น
- `test_catalog.py`: `ModelEntry` รับ `engine_env` · `is_ready()` = False เมื่อมี `.aria2` ค้าง / ไฟล์ตาม `dl` ไม่ครบ · entry เดิมทั้ง 9 ตัวยัง lint ผ่าน
- fixture ใหม่: `tests/fixtures/hf_qwen3-8b-fp8_blobs.json` (จาก response จริง) + `tests/fixtures/config_qwen3_fp8.json`, `config_nested_text_config.json`
- **ไม่มี test ฝั่ง frontend** (โปรเจกต์นี้ไม่มี harness) — ตรวจด้วย `node --check` กับสายตา

## Out of Scope

- ดาวน์โหลด/activate โมเดล vLLM จริงบนเครื่องในงานนี้ — พี่หนุ่มสั่งทดสอบถึงหน้า resolve พอ (การ activate ต้องยึดพอร์ต :8000 และเคลียร์แรม กระทบ llama.cpp ที่ลูกค้าใช้อยู่บน :8001)
- engine `ds4` (ยังไม่มี `engines/ds4.sh` ในโปรเจกต์)
- tensor-parallel / หลาย GPU
- `POST /api/engines/upgrade` สำหรับ vLLM image (ตอนนี้รับเฉพาะ llamacpp)
- การเรียงผลค้นหาที่ดัน GGUF ขึ้นก่อน (hf.py:355) — ยังคงไว้ตามเดิม
- โมเดล multimodal ที่ต้องการ preprocessor เพิ่มเติมนอกเหนือไฟล์ใน repo

## ความเสี่ยงที่ยังเหลือ

- ยังไม่ได้พิสูจน์ว่า vLLM 26.07 บน GB10 รัน FP8 block-quant (`weight_block_size [128,128]`) ได้จริง — ฮาร์ดแวร์รองรับ แต่ต้องทดสอบตอน activate จริงในงานถัดไป
- ตารางแม็ป parser ครอบคลุมเท่าที่รู้ ณ วันนี้ ตระกูลใหม่ต้องมาเติมเอง

> อ้างอิงศัพท์จาก CONTEXT.md · ADR 0001 (registry split) ยังใช้ได้ไม่ขัดกับงานนี้ — entry ที่ผู้ใช้เพิ่มยังไปอยู่ใน user-models.json เหมือนเดิม
