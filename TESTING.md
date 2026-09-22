# TESTING.md

## [2026-08-29 00:42] AI Server v2 เฟส 1 — deploy :9001 + ทดสอบ end-to-end บน DGX

### ภาพรวม

| รายการ | ผล |
|---|---|
| Flow ที่ทดสอบ | 7/7 ผ่าน |
| Unit test | 186/186 ผ่าน |
| Bug ที่พบระหว่างทดสอบ | 2 ตัว (แก้แล้วทั้งคู่ — ดู `fix.md`) |

### Checklist แต่ละ flow

| ข้อ | สิ่งที่ทดสอบ | ผล | หลักฐาน |
|---|---|---|---|
| 1 | Deploy + `GET /api/health` | ✅ | ตอบ `ram_total_gb 121.7` · `disk_free_gb 2647` |
| 2 | `GET /api/models` — คลังโมเดล 9 ตัว สถานะบนดิสก์ตรงจริง | ✅ | เช่น GLM-5.3 93.1GB, Qwen3.8-Flash-Next 78.9GB, Qwen3.8 Q8 31.8GB — ตรงกับไฟล์จริงบนดิสก์ |
| 3 | `GET /api/engines` อ่านจากเครื่องจริง | ✅ | llamacpp build 10696 รองรับ 2341 arch · vllm `aiserver-vllm:26.07` · ds4 ไม่ได้ติดตั้ง |
| 4 | **เคสหลัก** `POST /api/models/resolve` ด้วย `unsloth/GLM-5.3-Flash-GGUF` | ✅ | อ่าน arch `glm5next` + `ctx_train 1,048,576` + 7 quant พร้อมขนาดจริง (93.1GB → 199.7GB) + `fits_ram` ถูกต้อง (3 ตัวแรกพอ 4 ตัวหลังไม่พอ) + เจอ companion `mmproj-BF16/F16` — ทั้งหมดนี้ก่อนดาวน์โหลดไฟล์จริงแม้แต่ไบต์เดียว |
| 5 | ดาวน์โหลดจริง (`mmproj-F16.gguf` 1.13GB) หยุดที่ 25.5% → โหลดต่อ → จบ 100% | ✅ | sha256 ตรงกับที่ HF ประกาศเป๊ะ (`96ccc182...99ed27`) · ความเร็ว ~11.8 MB/s |
| 6 | `POST /api/activate` โหลด `qwen38-ud-q2-mtp` ขึ้น `:8001` ctx 32768 | ✅ | สำเร็จใน 18 วินาที · `/v1/models` ตอบ · ยิง chat จริงตอบ "กรุงเทพฯ" ถูกต้อง แยก `reasoning_content` ได้ · draft model (MTP) โหลดติด acceptance 62.5% |
| 7 | hub เดิม `:9000` ไม่กระทบตลอดการทดสอบ | ✅ | HTTP 200 ทุกครั้งที่เช็ค |

### Guard ที่ทดสอบว่าปฏิเสธถูกต้อง

| Guard | ผล |
|---|---|
| activate ที่ `:8000` โดยไม่ส่ง `allow_main_port` | 409 |
| activate โมเดลที่ยังโหลดไฟล์ไม่ครบ | 409 |
| activate ตอนแรมไม่พอ | 409 พร้อมบอกว่าอะไรถือแรมอยู่ |
| activate id ที่ไม่มีในคลัง | 404 |
| `GET /api/downloads` ตอน aria2 ไม่ทำงาน | ยังตอบ 200 ไม่พังทั้ง endpoint |

### Deploy log

| เวลา | action | result |
|---|---|---|
| 2026-08-29 00:42 | rsync + สร้าง venv python3.12 + start aria2 :6800 + start uvicorn :9001 | UP |
| 2026-08-29 00:52 | redeploy หลังแก้บั๊ก 2 ตัว | UP |
| 2026-08-29 00:58 | redeploy หลังแก้ `_ram_hogs` | UP |

### หมายเหตุ

- ระหว่างทดสอบ hub เดิมโหลด GLM-5.3 ขึ้น `:8000` เอง กินแรม 98GB ทำให้ข้อ 6 ติด — ได้รับอนุญาตจากพี่หนุ่มให้หยุดชั่วคราว ทดสอบจบแล้ว**โหลดกลับด้วยคำสั่งเดิมเป๊ะ** (จดไว้ที่ `~/.aiserver2/glm-8000-cmdline.txt` บน DGX) ยืนยันว่า `:8000` กลับมา HTTP 200 พร้อมโมเดลเดิม
- ของทดสอบที่สร้างเองลบหมดแล้ว (`~/models/gguf/_v2test` และ entry `test-mmproj-glm53`)

## [2026-08-31 09:24] ทดสอบต่อเนื่องระหว่างใช้งานจริง (2026-08-29 → 08-31)

### ภาพรวม

| รายการ | ผล |
|---|---|
| Unit test | 252/252 ผ่าน |
| ทดสอบบนเครื่องจริง | ทุกฟีเจอร์ผ่าน |
| Bug ที่พบและแก้ | 6 ตัว (ดู `fix.md`) |

### Checklist

| ข้อ | สิ่งที่ทดสอบ | ผล | หลักฐาน |
|---|---|---|---|
| 1 | วาง `https://github.com/openai/gpt-oss?utm_source=chatgpt.com` | ✅ | ได้รายการค้นหา 8 รายการ GGUF ขึ้นก่อน · กดเลือก `unsloth/gpt-oss-20b-GGUF` → arch `gpt-oss` ctx 131,072 · 16 quant แรมพอทุกตัว compat ok |
| 2 | `ggml-org/gpt-oss-120b-GGUF` | ✅ | `MXFP4 63.4GB` โผล่ถูกต้อง · `eagle3-*` ไปอยู่ companions |
| 3 | ปุ่มหยุด instance | ✅ | หยุดจริง คืนแรม 52 → 4 GB · guard ปฏิเสธถูกทั้ง 3 เคส (พอร์ต 8000 ไม่ยืนยัน → 409 · พอร์ต 9999 → 400 · พอร์ตว่าง → ok=false) |
| 4 | ธีม + gauge 6 ตัว | ✅ | ตรวจด้วยตาเทียบกับ `:9100` · ค่าตรงกัน · caption `เพดาน 90°C` / `เหลือ 2.7 TB` |
| 5 | `/api/endpoints` | ✅ | แยก `live` ถูก · copy คำสั่งจาก modal ไปรันจริงได้คำตอบ "ปารีส" · ตัวที่ระบบเตือนว่าตาย (`qwen38`) ยิงแล้ว error จริงตามที่บอก |
| 6 | playground สตรีม | ✅ | ถาม "ประเทศไทยมีกี่จังหวัด" ตอบ "77 จังหวัด" · 54 token · 8.4 วิ · 6.4 tok/s · กล่อง "กำลังคิด" แยกพับไว้ |
| 7 | vision | ✅ | ส่งภาพวงกลมแดง โมเดลตอบ "วงกลม สีแดง" ถูกต้อง (พิสูจน์ว่า mmproj ใช้งานได้) |
| 8 | สวิตช์ปิดโหมดคิด | ✅ | ปิดแล้วถาม "แม่น้ำที่ยาวที่สุดในไทย" ตอบ "แม่น้ำเจ้าพระยา" ตรง ๆ ไม่มีกล่องคิด · 4 token · 1.1 วิ |
| 9 | หลังแก้ gateway matching | ✅ | Flash-Next ตอบ "ดอกราชพฤกษ์" · 86 token · 3.1 วิ · **27.5 tok/s** |
| 10 | ตอบยาว | ✅ | `max_tokens 3000` ขอบทความ 800 คำ ได้ 4,294 ตัวอักษร (970 token) `finish=stop` ไม่ถูกตัด |
| 11 | `:9000` และ hub เดิมไม่กระทบตลอด | ✅ | HTTP 200 ทุกครั้ง |

### Deploy log

| เวลา | action | result |
|---|---|---|
| 2026-08-29 07:53 | deploy | UP |
| 2026-08-29 10:09 | deploy | UP |
| 2026-08-29 12:44 | deploy | UP |
| 2026-08-29 13:21 | deploy | UP |
| 2026-08-29 14:10 | deploy | UP |
| 2026-08-29 18:47 | deploy | UP |

### หมายเหตุ

- ทดสอบ vision ผ่าน API โดยสร้างภาพ PNG วงกลมแดงขึ้นมาเอง (ไม่ได้ใช้ภาพของผู้ใช้)
- เคย stop โมเดลของพี่หนุ่มเพื่อทดสอบ 1 ครั้ง (ขออนุญาตก่อน) แล้วโหลดกลับด้วยคำสั่งเดิมเป๊ะ

## [2026-09-22 21:35] รองรับ GGUF repo ที่ไม่มี quant token ในชื่อไฟล์ + gated repo

### ภาพรวม

| รายการ | ผล |
|---|---|
| Unit test | 328/328 ผ่าน (เดิม 299 + ใหม่ 29) |
| Bug ที่พบและแก้ | 1 ตัว (ดู `fix.md` entry `[2026-09-22 21:35]`) |

### Checklist

| ข้อ | สิ่งที่ทดสอบ | ผล | หลักฐาน |
|---|---|---|---|
| 1 | Unit test ทั้งชุด | ✅ | 328/328 ผ่าน (เดิม 299 + ใหม่ 29: group_quants fallback, is_gated, save/load token 0600, fetch_header ส่ง Authorization, aria2 add_uri header, `_start_job` ส่ง header เฉพาะ host huggingface.co, resolve endpoint gated with/without token) |
| 2 | resolve local (TestClient, ไม่มี token) กับ repo จริง | ✅ | gated true · 2 quant · mmproj companion · message |
| 3 | Deploy `:9001` (`bash deploy.sh`) | ✅ | health ok `2.0.0-phase1` |
| 4 | resolve บน `:9001` ไม่มี token | ✅ | gated true · mainline 33 shard 94.5 GB fits_ram true · main 34 shard 97.3 GB · compat `unknown` (reason: อ่าน header ไม่สำเร็จ) · message ขึ้น |
| 5 | Regression `unsloth/GLM-5.3-Flash-GGUF` บน `:9001` | ✅ | gated false · arch glm5next · ctx 1,048,576 · quant ครบ |
| 6 | ยังไม่มีไฟล์ `~/.aiserver2/hf_token` บน DGX | ✅ | ยังไม่มีใครใส่ token — ถูกต้อง |

### Deploy log

| เวลา | action | result |
|---|---|---|
| 2026-09-22 21:30 | `bash deploy.sh` → rsync + restart `:9001` | UP |

### หมายเหตุ

- ยังไม่ได้ทดสอบขั้นดาวน์โหลดจริงด้วย token (ต้องใช้ token ที่ยอมรับ gate แล้ว — พี่หนุ่มจะใส่เองในช่อง UI)
- ชุด `main` ใช้กับ llama.cpp build ปกติไม่ได้ ให้เลือก `mainline`

## [2026-09-22 22:05] hotfix ดาวน์โหลด: refresh ทน gid เก่า + กัน repo gated ไม่มี token

### ภาพรวม

| รายการ | ผล |
|---|---|
| Unit test | 342/342 ผ่าน (เดิม 328 + ใหม่ 14) |
| Bug ที่พบและแก้ | 2 ตัว (ดู `fix.md` entry `[2026-09-22 22:05]` ทั้งสอง) |

### Checklist

| ข้อ | สิ่งที่ทดสอบ | ผล | หลักฐาน |
|---|---|---|---|
| 1 | Unit test ทั้งชุด | ✅ | 342/342 ผ่าน (ใหม่ 14: `call()` 400-JSON/500/connect → `Aria2Error` · `refresh()` ข้าม job จบแล้ว · gid not found → error · advance queue · `probe_gated_url` 401/403/200/302/exception · `create_download` 400/มี token/ไม่ใช่ HF) |
| 2 | Deploy `:9001` | ✅ | UP |
| 3 | `GET /api/downloads` job `48852d8c` | ✅ | error "Authorization failed." ทันที (ก่อนแก้ค้าง active 0%) |
| 4 | `POST /api/downloads/48852d8c/cancel` | ✅ | ok |
| 5 | `POST /api/downloads` mainline ไม่มี token | ✅ | 400 พร้อมข้อความบอกให้ใส่ token |
| 6 | cancel job `79121c2b` ที่คิวดันเริ่มไปโดยไม่มี token | ✅ | cancel สำเร็จ |

### Deploy log

| เวลา | action | result |
|---|---|---|
| 2026-09-22 22:02 | `bash deploy.sh` | UP |

### หมายเหตุ

- ยังไม่ได้ทดสอบดาวน์โหลดจริงด้วย token (พี่หนุ่มจะใส่เอง)
- เอกสารวิธีหา token: huggingface.co/settings/tokens (Read) + กด Agree ที่หน้า repo
