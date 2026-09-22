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

## [2026-09-03 15:15] AI Server v2 (:9001) ไม่ขึ้นเองตอนบูตเครื่อง

- **อาการ**: หลังเปิดเครื่อง DGX เข้า `http://gx10-6214.tail3f5086.ts.net:9001/` ไม่ได้ ต้องสั่ง start ด้วยมือทุกครั้ง ขณะที่ hub เดิม `:9000` ขึ้นเองปกติ
- **สาเหตุ**: crontab ของ user `dgx` บน DGX มีเฉพาะ entry ของ v1 (`:9000`) เท่านั้น — ทั้ง `@reboot` และ watchdog รายนาที ไม่มีบรรทัดใดชี้ไป `/home/dgx/aiserver2/run.sh` เลย ไม่ใช่ปัญหาของ `run.sh` (ทดสอบสั่ง start ด้วยมือแล้วขึ้นปกติ health 200) และเครื่องไม่มี systemd unit ของ aiserver ทั้ง system และ user scope — autostart ของโปรเจกต์นี้อาศัย cron อย่างเดียว
- **วิธีแก้**: สำรอง crontab เดิมไว้ที่ `/home/dgx/crontab.bak.20260903-1513` แล้วเติม 2 บรรทัดต่อท้าย crontab ของ user `dgx` ล้อรูปแบบเดียวกับ v1:
  ```
  @reboot sleep 30 && AISERVER2_PORT=9001 bash /home/dgx/aiserver2/run.sh start
  * * * * * curl -sf -m5 http://127.0.0.1:9001/api/health >/dev/null 2>&1 || (sleep 8; curl -sf -m5 http://127.0.0.1:9001/api/health >/dev/null 2>&1) || AISERVER2_PORT=9001 bash /home/dgx/aiserver2/run.sh start >/dev/null 2>&1
  ```
  (`sleep 30` ตั้งให้ห่างจากของ v1 ที่ใช้ `sleep 25` · watchdog เช็ค `/api/health` ของ v2 ไม่ใช่ `/api/status` ของ v1)
- **ยืนยันแล้ว**: ฆ่า process uvicorn ของ `:9001` ทิ้ง แล้ว watchdog กู้กลับมาเองภายใน 25 วินาที (health 200) และเรียกจากภายนอกผ่าน tailscale `http://gx10-6214.tail3f5086.ts.net:9001/` ได้ 200
- **ข้อควรรู้ที่เจอระหว่างแก้**: ถ้าจะสั่ง `pkill -f "uvicorn server.main:app.*--port 9001"` ผ่าน ssh บรรทัดเดียว pattern จะไป match ตัว shell ของคำสั่งเองทำให้ ssh ตาย exit 255 — ให้ใช้ `bash /home/dgx/aiserver2/run.sh stop` แทน
- **สิ่งที่ยังไม่ได้พิสูจน์ตรง ๆ**: ยังไม่ได้ reboot เครื่องจริงเพื่อทดสอบ `@reboot` (ยังไม่ได้ขออนุญาตเจ้าของเครื่อง) — แต่ watchdog รายนาทีเป็นตาข่ายรองอยู่แล้ว ต่อให้ `@reboot` พลาด ระบบจะขึ้นเองภายในราว 1 นาทีหลังบูต

## [2026-09-03 15:41] โหลด qwen3-8b-fp8 ไม่ขึ้น และ error ในหน้าเว็บโชว์เป็น hash

- **อาการ**: กดโหลด `qwen3-8b-fp8` แล้วหน้าเว็บขึ้น "โหลด qwen3-8b-fp8 ขึ้นแรมไม่สำเร็จ (พอร์ต 8001): 401acf608e3cfe4aec064dc03f8f8939607c3c9124fc467359da1c42df8db5bb" — เป็น hash 64 ตัวไม่มีข้อความ error ใด ๆ ให้ตามต่อ
- **สาเหตุ**: มี 2 ชั้นซ้อนกัน
  1. ต้นเหตุจริงที่โหลดไม่ขึ้น — vLLM ตายตั้งแต่ `process_weights_after_loading` ด้วย `RuntimeError: Assertion error (deepgemm-src/csrc/apis/layout.hpp:59): Unknown SF transformation` · DeepGEMM (เปิดเป็น default `VLLM_USE_DEEP_GEMM=True` ใน image) แปลง scale-factor layout ของ FP8 block-quant บน GB10 ไม่ได้ ⇒ EngineCore ตายทั้งตัว ไม่เกี่ยวกับแรมหรือไฟล์โมเดลเสีย
  2. เหตุที่ตามต่อไม่ได้ — `engines/vllm.sh` เขียน `docker run -d ... > "$LOG_DIR/vllm.log" 2>&1` แต่ `docker run -d` พิมพ์แค่ **container ID** ออก stdout ⇒ `vllm.log` เหลือ 65 ไบต์เป็น container id ล้วน · `main.py` อ่านไฟล์นี้ไปโชว์เป็นข้อความ error ⇒ ผู้ใช้เห็น hash · log จริงอยู่ใน `docker logs` ซึ่งไม่มีใครอ่าน
     (แถมยังพบว่า `docker run` ไม่มี `-e` เลย ⇒ env ที่ตั้งใน `engine_env` ของ entry ตกอยู่แค่ระดับ shell ไปไม่ถึง vLLM ในคอนเทนเนอร์)
- **วิธีแก้** (แก้ที่ `engines/vllm.sh` 3 จุด แล้ว deploy):
  1. ปิด DeepGEMM เป็นค่าตั้งต้น — `ENV_FLAGS=(-e "VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM-0}")` · เปิดกลับต่อโมเดลได้ด้วย `VLLM_USE_DEEP_GEMM=1` ใน `engine_env`
  2. ส่ง `VLLM_*` ทุกตัวจาก env เข้าคอนเทนเนอร์ด้วย `-e` (ยกเว้น `VLLM_TOOL_PARSER` / `VLLM_REASONING_PARSER` / `VLLM_SPECULATIVE` ที่เป็น flag ของสคริปต์เอง ไม่ใช่ env ของ vLLM)
  3. เลิก redirect stdout ของ `docker run -d` ทับไฟล์ log — เก็บ container id ไว้ในตัวแปร (`if ! CID=$(docker run -d ... 2> "$LOG_DIR/vllm.log"); then` เพื่อให้ error ของ docker เองยังลงไฟล์) แล้ว stream log จริงด้วย `( setsid docker logs -f "$CID" > "$LOG_DIR/vllm.log" 2>&1 < /dev/null & )` แบบ detach ปิด fd ครบตามบทเรียนเดิมของ run.sh
- **ยืนยันแล้ว**: 2026-09-03 15:41 — สั่งผ่าน API จริง `POST /api/activate {"id":"qwen3-8b-fp8","port":8000,"allow_main_port":true}` ได้ `{"ok":true,"port":8000}` ใช้เวลาราว 110 วินาที · `/api/instances` เห็น instance `up:true` · chat ตอบ "4" (finish_reason=stop) · request ที่มี tools ได้ 200 (tool parser ยังทำงาน) · `~/.aiserver/logs/vllm.log` มี log จริง 178 บรรทัด ไม่ใช่ hash เดี่ยวอีกแล้ว

## [2026-09-03 15:48] aria2.log โตถึง 17 GB

- **อาการ**: `~/.aiserver2/logs/aria2.log` โตถึง 17 GB และยังโตต่อเนื่อง ทั้งที่ไม่มี download ค้างอยู่เลย (ทุก job สถานะ done) — เจอตอนไล่ปัญหาอื่นแล้วบังเอิญเห็นขนาดไฟล์
- **สาเหตุ**: aria2 แยก log level เป็นสองตัวคนละหน้าที่ — `--console-log-level` (จอ) กับ `--log-level` (ไฟล์ที่ระบุด้วย `--log`) · `services/aria2.sh` ตั้งแต่ `--console-log-level=warn` ไว้ตัวเดียว ส่วน `--log-level` ไม่ได้ตั้ง จึงใช้ค่า default ของ aria2 คือ **debug** (ยืนยันจาก `aria2c --help=#advanced` บนเครื่องจริง) ⇒ ไฟล์เก็บ debug ทุกบรรทัด และ `downloads.py` poll RPC ทุกไม่กี่วินาที ไฟล์เลยโตไม่หยุด
- **วิธีแก้**: เติม `--log-level=warn` ต่อจาก `--console-log-level=warn` ใน `services/aria2.sh` · ส่งไฟล์ขึ้น DGX · `aria2.sh stop` → ลบไฟล์ log เดิมทิ้ง → `aria2.sh start`
- **ยืนยันแล้ว**: 2026-09-03 15:48 — อ่าน `/proc/<pid>/cmdline` ของ aria2 ที่รันอยู่จริง เห็น `--console-log-level=warn` และ `--log-level=warn` ครบทั้งคู่ · ไฟล์ log จาก 17 GB เหลือ 0 ไบต์ · เฝ้าดู 120 วินาทีพร้อม poll `/api/downloads` ซ้ำ ๆ ไฟล์ยังคง 0 ไบต์ ไม่โตอีก · `/api/downloads` ยังคุยกับ aria2 ผ่าน RPC ได้ปกติ (secret เดิมใช้ได้ ไม่ต้อง restart server)

## [2026-09-22 21:35] GGUF repo ที่ชื่อไฟล์ลงท้ายด้วยชื่อ variant ถูกมองว่า "ไม่ใช่ GGUF" + repo gated ดาวน์โหลดไม่ได้

- **อาการ**: วาง `Navin-Models/Qwen3.8-Flash-Next-Uncensored-AD-4.27-GGUF` ในช่องตรวจสอบ → `arch: ไม่ทราบ` และ "repo นี้ไม่ใช่ GGUF และไม่ใช่ safetensors — ยังไม่รองรับ" ทั้งที่ repo มี .gguf 68 ไฟล์
- **สาเหตุ** (2 ข้อ):
  1. `server/hf.py` `_QUANT_RE` บังคับให้ quant token อยู่ท้าย stem แต่ไฟล์ของ repo นี้อยู่ที่ root และลงท้ายด้วยชื่อ variant `-main-00001-of-00034.gguf` / `-mainline-00001-of-00033.gguf` → `key is None` → ถูกข้ามทุกไฟล์ → `groups == []` → main.py ตกไปข้อความ "ไม่รู้จักรูปแบบ" (ยืนยันด้วยการรัน `group_quants()` กับ siblings จริง ได้ `[]`)
  2. repo เป็น gated="auto" — HF API list ไฟล์ได้โดยไม่ต้อง token แต่ resolve URL ตอบ `401 GatedRepo` ทั้งตอนอ่าน header และดาวน์โหลด · เดิม `main.py` hardcode `"gated": False`, `gguf.fetch_header()` ไม่รับ token, และ aria2 `addUri` ไม่ส่ง Authorization → ใส่ token ในช่อง UI ก็โหลดไม่ได้อยู่ดี
- **วิธีแก้**: `hf.group_quants()` — ไฟล์ root ที่ไม่มี quant token ใช้ stem ทั้งก้อนเป็น key (สอดคล้องกับกติกาโฟลเดอร์ย่อย) + `_IGNORE_KEYWORDS=("imatrix",)` กันไฟล์ calibration · `hf.is_gated()` ส่งค่าจริงจาก HF API · `hf.save_token()/load_token()` เก็บ token ที่ `~/.aiserver2/hf_token` (0600) เมื่อผู้ใช้ใส่ตอนตรวจสอบ · `gguf.fetch_header(token=)` ส่ง Bearer (httpx ตัด header ทิ้งเองเมื่อ 302 ข้ามออริจินไป CDN) · `downloads.Aria2Client.add_uri(headers=)` + `DownloadManager(token_loader=)` ส่ง `Authorization` ให้ aria2 เฉพาะ URL host `huggingface.co` · UI: แสดง `data.message` ในเส้นทางปกติ, ปรับข้อความ 🔒, dedupe id ไม่ให้ยาวซ้ำชื่อ repo
- **ยืนยันแล้ว**: 2026-09-22 21:35 — deploy `:9001` แล้ว `POST /api/models/resolve` คืน `gated: true` · quant 2 กลุ่ม `…-AD-4.27-mainline` (33 shard, 94.5 GB, fits_ram true) และ `…-AD-4.27-main` (34 shard, 97.3 GB) · mmproj อยู่ใน companions · มี message บอกให้ใส่ token · repo ไม่ gated (`unsloth/GLM-5.3-Flash-GGUF`) ยังได้ arch `glm5next` 7 quant เหมือนเดิม · unit test 328/328 (ใหม่ 29 ตัว) · ⚠️ ชุด `main` 34 shard ต้องใช้ llama.cpp จาก PR #28243 ใช้กับ build ปกติไม่ได้ ผู้ใช้ต้องเลือก `mainline`

## [2026-09-22 22:05] สถานะดาวน์โหลดค้าง "active 0%" ไม่ยอมเป็น error — refresh() ล้มทั้งก้อนเพราะ gid เก่า

- **อาการ**: กดดาวน์โหลด repo gated โดยไม่มี token → aria2 ล้ม (errorCode 24 Authorization failed.) ทั้ง 34 ไฟล์ แต่ UI/`GET /api/downloads` ยังโชว์ job เป็น active 0% error null และคิวไม่เดินต่อ
- **สาเหตุ**: `DownloadManager.refresh()` วนทุก job รวม 5 job ที่ DONE ไปแล้วซึ่ง gid หายจาก aria2 ตั้งแต่ aria2 restart 2026-09-03 · aria2 ตอบ HTTP 400 + body `{"error":{"code":1,"message":"GID … is not found"}}` · `Aria2Client.call()` เรียก `raise_for_status()` ก่อนอ่าน body → ได้ `httpx.HTTPStatusError` ไม่ใช่ `Aria2Error` → หลุด `except Aria2Error` ใน refresh → refresh abort ทั้งก้อน · `list_downloads()` กลืน exception → ไม่มี job ไหนอัปเดต, `_advance_queue`/`_save_state` ไม่ถูกเรียก (ยืนยันด้วย dry-run บน DGX: traceback ชี้ที่ `call()` 400 และ tellStatus 5 gid เก่าตอบ GID not found)
- **วิธีแก้**: `call()` อ่าน JSON body หา `"error"` ก่อน แล้วแปลงทุกกรณี (JSON-RPC error / HTTP non-2xx / httpx connect-timeout) เป็น `Aria2Error` เสมอ · `refresh()` ข้าม job ที่จบแล้ว (DONE/ERROR/CANCELLED) ไม่ถาม aria2 · gid ที่ aria2 ตอบ "not found" บน job ที่ยังไม่จบ → file ERROR ข้อความ "aria2 ไม่รู้จักงานนี้แล้ว (aria2 อาจถูก restart) — กดดาวน์โหลดใหม่"
- **ยืนยันแล้ว**: 2026-09-22 22:05 — deploy `:9001` แล้ว `GET /api/downloads` โชว์ job 48852d8c เป็น error "Authorization failed." ทันที คิวเดินต่อไป job ถัดไปเอง · unit test 342/342 (ใหม่ 14)

## [2026-09-22 22:05] กดดาวน์โหลด repo gated โดยไม่มี HF token แล้วล้มเงียบ

- **อาการ**: เพิ่ม `Navin-Models/Qwen3.8-Flash-Next-Uncensored-AD-4.27-GGUF` เข้าคลังแล้วกดดาวน์โหลดทั้งที่ยังไม่เคยใส่ token ตอนตรวจสอบ → aria2 401 ทุกไฟล์ ผู้ใช้ไม่รู้ว่าต้องทำอะไร
- **สาเหตุ**: `POST /api/downloads` submit ให้ aria2 ทันทีโดยไม่เช็คว่า URL huggingface.co นี้ต้องการ token ไหม · บน DGX ยังไม่มี `~/.aiserver2/hf_token`
- **วิธีแก้**: ก่อน submit ถ้า URL แรกเป็น huggingface.co และไม่มี token ที่จำไว้ → `hf.probe_gated_url()` ยิง HEAD (ไม่ตาม redirect, timeout 10s) ถ้า 401/403 → ตอบ 400 "repo นี้ติด gate — ใส่ HF token ในช่อง 'ตรวจสอบ HF repo' แล้วกดตรวจสอบก่อน ระบบจะจำ token ไว้ดาวน์โหลด" · เน็ตพัง/timeout ปล่อยผ่านให้ aria2 รายงานเอง
- **ยืนยันแล้ว**: 2026-09-22 22:05 — ยิง `POST /api/downloads` ของ entry mainline บน `:9001` โดยไม่มี token ได้ 400 พร้อมข้อความข้างต้น · job ที่ล้ม 2 ตัว (main/mainline) cancel แล้ว รอพี่หนุ่มใส่ token แล้วโหลด mainline ใหม่

## [2026-09-23 06:20] มี token แล้วยังโหลดไม่ได้ (403) แต่ระบบไม่บอกเหตุผล + แท็บดาวน์โหลดนับ job ที่จบแล้วผิด + arch ว่างขึ้น "ต้องอัป engine"

- **อาการ**: ใส่ HF token แล้ว (บันทึกที่ `~/.aiserver2/hf_token`) กดโหลดได้ job ที่ error "The response status is not successful. status=403" ไม่รู้ว่าต้องทำอะไร · job ที่ error/cancelled ยังนับอยู่ในแท็บ "กำลังดำเนินการ (5)" · การ์ดโมเดลที่เพิ่มตอนยังอ่าน header ไม่ได้ (arch = "") ขึ้น "⚠️ ต้องอัป engine — upgrade_llamacpp" ทั้งที่ควรเป็น "ไม่ทราบ"
- **สาเหตุ** (3 ข้อ):
  1. HF ตอบ 403 `x-error-message: … you are not in the authorized list` = token ใช้ได้แต่บัญชียังไม่ได้กด Agree ที่หน้า repo (หรือ fine-grained token ไม่มีสิทธิ์ gated repos) · guard ใน `create_download()` ตรวจเฉพาะกรณีไม่มี token
  2. `renderDownloads()` ใน index.html ใช้ `state !== "done"` เป็น "ยังไม่จบ" → error/cancelled ค้างในแท็บแรกตลอด
  3. `engines.check_arch()` เช็ค `arch is None` เท่านั้น สตริงว่างหลุดไปข้อ 8 "ไม่อยู่ใน arch list" → NEEDS_UPGRADE
- **วิธีแก้**: `hf.probe_download_access(url, token=)` HEAD ด้วย token ที่จำไว้ คืนข้อความจาก `x-error-message` เมื่อ 401/403 · `create_download()` ตรวจทุกครั้งสำหรับ URL huggingface.co: ไม่มี token → 400 บอกให้ใส่ token · มี token แต่ HF ปฏิเสธ → 400 แนบข้อความ HF + บอกให้กด 'Agree and access repository' และตรวจสิทธิ์ token · UI: `DOWNLOAD_FINISHED_STATES = done/error/cancelled` ไปแท็บ "จบแล้ว" · `check_arch()` normalize `""`/ช่องว่าง → None (ข้อ 5 UNKNOWN)
- **ยืนยันแล้ว**: 2026-09-23 06:20 — deploy `:9001` แล้ว `POST /api/downloads` entry mainline ตอบ 400 "HF ปฏิเสธ (403): Access to model … not in the authorized list … — ต้องกด 'Agree and access repository' …" · `GET /api/models` entry Navin ทั้งสอง compat `unknown` action None · unit test 351/351 (ใหม่ 9) · ⚠️ ณ เวลานี้บัญชีของพี่หนุ่มยังไม่ได้กด Agree (HEAD ด้วย token ยัง 403) — ต้องทำก่อนจึงโหลดได้
