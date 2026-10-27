# DGX AI Manager

**ภาษาไทย** · [English](README.en.md)

เว็บจัดการโมเดล LLM และ engine บน DGX Spark (GB10) — เปิดหน้าเดียวจบ: ดาวน์โหลดโมเดล จัดคลัง โหลดขึ้นแรม และลองคุยได้ทันที

![DGX AI Manager — แดชบอร์ด](docs/screenshots/01-dashboard.png)

## ฟีเจอร์

### 🎛️ แดชบอร์ดสถานะเครื่อง
- Gauge 6 ตัวแถวบนสุด (CPU · RAM · GPU ฯลฯ) อัปเดตสด ธีมเดียวกับ DGX Spark Monitor
- สี gauge 3 ระดับ: ฟ้า ≤60% · เหลือง >60–80% · แดง >80% ของค่าสูงสุด

### 🧠 โมเดลที่โหลดอยู่
- เห็น instance ที่รันจริงทุกพอร์ต พร้อมปุ่มหยุดจากหน้าเว็บ
- ตรวจจับ speculative decoding (FastMTP) และกู้อัตโนมัติเมื่อ draft model โหลดไม่ขึ้น
- ตอนโหลดโมเดลเลือกพอร์ต 8001–8009 (บอกว่าพอร์ตไหนใช้อยู่/ว่าง) และ ctx ได้ (ปุ่มลัด 8K/32K/64K/128K/max, เตือนเมื่อเกินเพดานของโมเดล)

### 💬 Playground ลองคุย
- คุยกับโมเดลที่รันอยู่ได้ทันทีจากหน้าเดียวกัน สตรีมคำตอบแบบเรียลไทม์
- กล่อง "กำลังคิด" แยกพับเก็บได้ · มีสวิตช์ปิดโหมดคิดเพื่อให้ตอบเร็วขึ้น

### 📚 คลังโมเดล
- รวมโมเดลจาก catalog ของระบบและโมเดลที่ผู้ใช้เพิ่มเองไว้ในที่เดียว
- เห็นขนาดไฟล์และสถานะพร้อมใช้ · สั่งดาวน์โหลดหรือโหลดขึ้นแรมได้จากการ์ดเดียว

### ➕ เพิ่มโมเดลจาก Hugging Face
- วางลิงก์ HF รูปแบบไหนก็ได้ ระบบค้นหาและ resolve repo ให้เอง
- เลือก quant ได้พร้อมเห็นขนาดไฟล์จริงก่อนตัดสินใจโหลด
- อ่านสถาปัตยกรรมโมเดลจาก GGUF header ด้วย HTTP Range request — รู้ก่อนโหลดว่า engine รันได้ไหม โดยไม่ต้องดาวน์โหลดทั้งไฟล์
- เช็คความเข้ากันกับ engine จากไฟล์ engine ที่ติดตั้งจริง ไม่พึ่งตารางที่จดมือ
- ปุ่ม 🔍 ตรวจสอบใหม่ บนการ์ดที่ arch/ctx ว่าง — อ่านจากไฟล์ในเครื่อง (หรือจาก HF ถ้ายังไม่โหลด) แล้วเติมให้ โดยไม่แตะ catalog

### ⬇ ตัวจัดการดาวน์โหลด
- คิวดาวน์โหลดผ่าน aria2 — progress ความเร็ว และ ETA รายไฟล์
- หยุดชั่วคราว / ทำต่อ / ยกเลิก ได้ระหว่างโหลด
- แท็บแยก **กำลังดำเนินการ / จบแล้ว** — งานที่เสร็จแล้วย้ายแท็บอัตโนมัติ และการ์ดในคลังโมเดลพร้อมให้โหลดขึ้นแรมทันที
- ปุ่มล้างงานดาวน์โหลดที่ไม่สำเร็จ (error/ยกเลิก) ออกจากรายการ โดยไม่ลบไฟล์บนดิสก์

### 🔌 วิธีใช้ในตัว
- Modal รวมตัวอย่างการเรียก API สำหรับต่อกับ client ภายนอก

## 📸 ตัวอย่างหน้าจอ

### 💬 Playground
คุยกับโมเดลที่รันอยู่ · สตรีมข้อความ · กล่องความคิดพับได้ · สถิติ token/s ด้านล่าง

![Playground](docs/screenshots/02-chat.png)

### 📚 คลังโมเดล
การ์ดแสดงสถานะพร้อมใช้ · engine/arch/ctx/ขนาด · ปุ่มโหลดขึ้นแรม · เตือนเมื่อต้องอัปเกรด engine

![คลังโมเดล](docs/screenshots/03-library.png)

### ➕ เพิ่มจาก Hugging Face
ใส่ repo id (เช่น `unsloth/Qwen3-8B-GGUF`) → แสดง arch/ctx_train · ทุก quant พร้อมขนาด · แรมพอหรือไม่ · เข้ากับ engine หรือเปล่า

![เพิ่มโมเดล](docs/screenshots/04-add-model.png)

### ⬇ ตัวจัดการดาวน์โหลด
แท็บ กำลังดำเนินการ / จบแล้ว · ความคืบหน้ารายไฟล์

![ดาวน์โหลด](docs/screenshots/05-downloads.png)

## 🚀 วิธีใช้งาน

### ติดตั้งและรัน
ต้องมี: DGX Spark (หรือ Linux arm64 + NVIDIA GPU) · Python 3.12 · aria2 (`aria2c`) · llama.cpp `llama-server` และ/หรือ Docker (สำหรับ vLLM)

```bash
git clone https://github.com/thana19/DGX-AI-Manager.git
cd DGX-AI-Manager
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
bash run.sh start      # เริ่ม aria2 (RPC :6800) + เว็บที่ :9001
bash run.sh status     # UP / DOWN
bash run.sh stop
```

จากนั้นเปิด `http://<DGX-host>:9001` · เปลี่ยนพอร์ตได้ด้วย env `AISERVER2_PORT` · state/log เก็บที่ `~/.aiserver2/` (log: `~/.aiserver2/logs/server.log`)

### Deploy จาก Mac
`bash deploy.sh` — rsync ไป `dgx:~/aiserver2` · ติดตั้ง dependency ใน venv · restart :9001 · ปรับได้ด้วย env `DGX_SSH` (ssh host, default `dgx`) และ `DGX_DEST` (default `aiserver2`)

### ใช้งานบนหน้าเว็บ
1. **เพิ่มโมเดล** — ที่ ➕ เพิ่มโมเดล วาง repo id/ลิงก์ HF → 🔍 ตรวจสอบ → เลือก quant (ดูขนาด, แรมพอ ✅ และ COMPAT) → ➕ เพิ่มเข้าคลัง · repo แบบ gated ให้ใส่ HF token ในช่อง token
2. **ดาวน์โหลด** — ติดตามที่ ⬇ ดาวน์โหลด · หยุดชั่วคราว/ต่อ/ยกเลิกได้ · งานที่เสร็จย้ายไปแท็บ จบแล้ว
3. **โหลดขึ้นแรม** — ที่ 📚 คลังโมเดล กด ▶ โหลดขึ้นแรม เลือกพอร์ต 8001–8009 และ ctx (preset 8K/32K/64K/128K/max) → ปรากฏใน 🧠 โมเดลที่โหลดอยู่
4. **ลองคุย** — ที่ 💬 ลองคุย เลือกปลายทาง (gateway :4000 หรือพอร์ตตรง) พิมพ์แล้วส่ง · ปิดโหมดคิดเพื่อให้ตอบเร็วขึ้น
5. **หยุด** — กดปุ่ม ⏹ หยุด บนการ์ดโมเดลที่โหลดอยู่

### เรียกจากโปรแกรมอื่น
โมเดลที่โหลดอยู่เปิด API แบบ OpenAI-compatible — ผ่าน LiteLLM gateway ที่ `:4000` หรือตรงที่พอร์ตของ instance (8001–8009) · กดปุ่ม 🔌 บนการ์ดโมเดลที่โหลดอยู่เพื่อดูตัวอย่างพร้อมคัดลอก ตัวอย่าง:

```bash
curl http://<DGX-host>:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"สวัสดี"}]}'
```

หมายเหตุ: llama-server ใช้ path เต็มของไฟล์โมเดลเป็น id · ใส่ `"auto"` ก็ใช้ได้

### รันเทสต์
```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest
```

## Engine ที่รองรับ

- **llama.cpp** (llama-server) — GGUF ทุก quant
- **vLLM** (docker) — NVFP4 พร้อม tool calling

## เบื้องหลัง

- Backend: Python 3.12 · FastAPI + uvicorn
- Frontend: HTML/JS ไฟล์เดียว ไม่มี build step
- Test 253 ตัว รันได้โดยไม่ต้องต่อเน็ต (fixture จาก response จริง)