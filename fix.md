# fix.md

## [2026-08-29 00:50] args ที่มี ~ ไม่ถูกขยาย ทำให้ draft model โหลดไม่ขึ้น

- **อาการ**: `POST /api/activate` ล้มเหลว log ขึ้น `load_model: failed to load draft model, '~/models/gguf/qwen3.8-27b/mtp-Qwen3.8-27B-Q8_0.gguf'`
- **สาเหตุ**: `main.py` ส่ง args เป็น argv ตรง ๆ ให้ `subprocess` ไม่ผ่าน shell ⇒ ไม่มีใครขยาย `~` ให้ (`catalog.expand()` ขยายให้เฉพาะ `path` ไม่ได้แตะ token ใน `args`)
- **วิธีแก้**: `os.path.expanduser()` ทุก token หลัง `shlex.split(entry.args)` ใน `server/main.py`
- **ยืนยันแล้ว**: 2026-08-29 01:00 — โหลด `qwen38-ud-q2-mtp` ขึ้น :8001 สำเร็จ log ขึ้น `common_speculative_init_result: loading draft model ...` และวัด draft acceptance ได้ 0.625

## [2026-08-29 00:57] ตัวตรวจ "ใครถือแรมอยู่" เงียบเพราะ llama.cpp ใช้ mmap

- **อาการ**: ด่านตรวจแรมปฏิเสธถูกต้องแล้ว แต่บอกไม่ได้ว่าต้องหยุดอะไรก่อน
- **สาเหตุ**: `_ram_hogs()` กรองด้วย RSS ≥ 5GB แต่ llama.cpp โหลดโมเดลแบบ mmap ⇒ RSS โชว์แค่ ~3.1GB ทั้งที่กินแรมจริง ~95GB (`free` แสดง used 98GB) ตัวกรองจึงตัดทิ้ง
- **วิธีแก้**: เลิกกรองด้วย RSS — มี process `llama-server`/`ds4-server` รันอยู่ = ผู้ต้องสงสัยเสมอ · ดึงชื่อโมเดลจาก `-m` และพอร์ตจาก `--port` มาบอกผู้ใช้
- **ยืนยันแล้ว**: 2026-08-29 00:58 — ข้อความเปลี่ยนเป็น "…ตอนนี้ GLM-5.3-Flash-UD-IQ1_S-00001-of-00003.gguf บนพอร์ต :8000 ถือแรมอยู่ — หยุดตัวนั้นก่อนแล้วลองใหม่"

## [2026-08-29 07:53] quant MXFP4 หายทั้งตัว + draft head ถูกนับเป็น quant

- **อาการ**: `ggml-org/gpt-oss-120b-GGUF` แสดงแค่ 2 quant ขนาด 0.8GB กับ 1.6GB — ตัวโมเดลจริง 63.4GB ไม่โผล่เลย
- **สาเหตุ**: `_QUANT_RE` ไม่รู้จัก `MXFP4` ⇒ `gpt-oss-120b-MXFP4.gguf` ถูกข้ามเงียบ ๆ · และ `eagle3-*` (speculative draft head) ไม่ได้อยู่ใน `_COMPANION_PREFIXES` ⇒ ถูกจับเป็น quant ปกติ ผู้ใช้เห็น "Q8_0 0.8GB" แล้วเลือกไปโหลดจะได้ draft head แทนโมเดลจริง
- **วิธีแก้**: ขยาย `_QUANT_RE` รองรับ `(MX|NV)?FP\d+` และ `TQ` · เพิ่ม `eagle3-`/`eagle-`/`draft-` ใน `_COMPANION_PREFIXES` · ต่อมาขยายอีกเป็นการจับ keyword ที่ไหนก็ได้ในชื่อไฟล์ (`companion_kind()`) เพื่อจับ `FastMTP` ที่ไม่ได้ขึ้นต้นด้วย `mtp-`
- **ยืนยันแล้ว**: 2026-08-29 08:00 — `MXFP4 63.4GB` โผล่เป็น quant เดียว · `eagle3-*` 2 ไฟล์ไปอยู่ companions · มี test 3 ตัวล็อกไว้ (เคยถูก revert ไปครั้งหนึ่ง)

## [2026-08-29 10:29] โหลดโมเดลสำเร็จแต่หน้าเว็บดูเหมือนพัง

- **อาการ**: กด "โหลดขึ้นแรม" แล้วกล่อง log เต็มไปด้วยข้อความ `model has unused tensor blk.64...` ผู้ใช้เข้าใจว่า error
- **สาเหตุ**: โมเดลที่มี MTP layer ฝังในไฟล์ (`blk.NN.nextn.*`) ทำให้ llama.cpp เตือน `unused tensor` 15 บรรทัดรวด ซึ่งเป็นบรรทัดท้าย log พอดี · UI โชว์ log ดิบตรง ๆ · ความจริง log จบด้วย `model loaded` + `listening` และ `grep` หา error ได้ 0 บรรทัด
- **วิธีแก้**: `/api/activate` คืน `summary` (ctx · slots · multimodal · warnings ที่กรอง `unused tensor` ออก) · UI โชว์เป็นข้อความเดียว log ดิบย้ายไปใน `<details>`
- **ยืนยันแล้ว**: 2026-08-29 10:35 — ขึ้นว่า `✅ โหลดสำเร็จ พอร์ต 8001 · ctx 262,144 · 4 slot · อ่านภาพได้`

## [2026-08-29 13:21] snippet ที่ให้ผู้ใช้ copy ได้ [object Object]

- **อาการ**: modal "วิธีใช้" สร้างตัวอย่างโค้ดโดยหยิบ `gateway.models[0]`
- **สาเหตุ**: หลังเพิ่ม flag `live` แล้ว `models[]` เปลี่ยนจาก `list[str]` เป็น `list[dict]` แต่ฝั่ง UI ยังหยิบแบบเดิม · และต่อให้หยิบ `.id` ได้ก็ยังอาจได้ตัวที่ `live:false`
- **วิธีแก้**: ใช้ `gateway.recommended_model` แทน · dropdown ของ playground ไม่ใส่ gateway เลยถ้าไม่มีโมเดลที่ใช้ได้จริง
- **ยืนยันแล้ว**: 2026-08-29 13:30 — copy คำสั่งจาก modal ไปรันจริง ได้คำตอบ "ปารีส"

## [2026-08-29 18:47] โหลดโมเดลใหม่แล้ว chat ในหน้าเว็บใช้ไม่ได้

- **อาการ**: โหลด Qwen3.8-Flash-Next ขึ้น `:8001` สำเร็จ แต่ playground ไม่มีปลายทาง gateway ให้เลือก
- **สาเหตุ**: กติกา `live` เทียบ **ชื่อ** gateway กับชื่อไฟล์ของ instance · hub เดิมตั้งชื่อใน LiteLLM ไม่แน่นอน — โมเดลก่อนหน้าตั้งตามชื่อไฟล์ (บังเอิญตรง) แต่ตัวนี้ตั้งตาม `id` ใน catalog (`qwen3.8-flash-next-udq2` vs `Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf`) ⇒ ไม่ตรง ⇒ `live=false` ทุกตัว ⇒ `recommended_model=None`
- **วิธีแก้**: เลิกเทียบชื่อ — ถาม LiteLLM ที่ `/model/info` ซึ่งคืน `litellm_params.api_base` แล้วเทียบ **พอร์ตปลายทางจริง** กับพอร์ตของ instance ที่รันอยู่ · `/model/info` ใช้ไม่ได้ → ถอยไปเทียบชื่อแบบเดิม
- **ยืนยันแล้ว**: 2026-08-29 18:55 — `qwen3.8-flash-next-udq2` ขึ้น ✅ ใช้ได้ · ทดสอบ chat ในหน้าเว็บตอบ "ดอกราชพฤกษ์" 27.5 tok/s
