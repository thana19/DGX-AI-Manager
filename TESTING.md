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
