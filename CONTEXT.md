# AI Server v2 — CONTEXT

## โปรเจกต์นี้คืออะไร

AI Server = ผลิตภัณฑ์ของ AIServer.in.th — ศูนย์รวม AI บน DGX Spark (ASUS Ascent GX10) ที่ลูกค้าซื้อไปแล้วเปิดหน้าเว็บเดียวใช้งานได้ทุกอย่าง (LLM, gen ภาพ/เสียง/วิดีโอ, agent)

v2 = เขียนใหม่ทั้งตัวเพื่อแทน hub เดิม (v0.2.88)

**เหตุผล:** dev repo เดิมหาย — `release.sh` ต้องการ git repo ที่มี `aiserver/` + `tools/lint-models.py` แต่หาไม่เจอทั้งบน Mac และ DGX ⇒ ปล่อยอัปเดตให้ลูกค้าไม่ได้อีกแล้ว

ส่งมอบเป็นเฟส — **เฟส 1 = Model & Engine Manager** รันที่พอร์ต 9001 คู่กับ hub เดิม :9000 (hub เดิม = ทางถอย)

ตอนนี้ยังไม่มีเครื่องลูกค้าใช้งานจริง — มีแต่เครื่องพี่หนุ่มเอง ⇒ **ไม่ต้องทำ migration / backward compat**

## คำศัพท์

ใช้คำเหล่านี้ให้ตรงกันทั้งโค้ดและเอกสาร

| คำ | ความหมาย |
|---|---|
| **catalog** | รายการโมเดลที่ระบบรู้จัก = `catalog.yaml` (vendor curate) + `user-models.json` (ผู้ใช้เพิ่มเอง) รวมกัน |
| **model entry** | 1 รายการใน catalog — มี id, path, dl, engine, requires ฯลฯ |
| **engine** | โปรแกรมที่รันโมเดล: `llamacpp` (llama-server) · `vllm` (docker) · `ds4` |
| **instance** | engine ที่กำลังรันอยู่จริงบนพอร์ตหนึ่ง (เช่น llama-server บน :8001) |
| **arch** | ชื่อสถาปัตยกรรมโมเดลใน GGUF header (`general.architecture`) เช่น `qwen4exp`, `glm5next`, `qwen35` — ตัวชี้ขาดว่า llama-server รุ่นไหนรันได้ |
| **build** | เลข build ของ llama-server เช่น `b10353` |
| **pack** | ไฟล์ tarball ที่ vendor แจกผ่าน `dl.php` (llama-pack-gb10, ds4-pack-gb10) |
| **studio pack** | ชุดโมเดลของ ComfyUI (voice / lipsync / h3 / image) — **เฟสถัดไป ไม่อยู่ในเฟส 1** |
| **download job** | งานดาวน์โหลดไฟล์โมเดล 1 ชุด มีสถานะ/progress แยกจากการโหลดขึ้นแรม |
| **activate** | โหลดโมเดลขึ้นแรมให้ engine เสิร์ฟบนพอร์ตหนึ่ง (คนละเรื่องกับ download) |

## ข้อเท็จจริงของเครื่อง (verify แล้ว 2026-08-28)

- DGX Spark GB10 · host `gx10-6214` · ssh alias `dgx` (เข้าถึงผ่าน Tailscale)
- แรม unified 128GB (ใช้ได้จริง ~121.7GB) — GPU กับ CPU ใช้ก้อนเดียวกัน ⇒ LLM กับ ComfyUI แย่งกัน
- โมเดลอยู่ที่ `~/models/` (gguf/ และ nvfp4/) · llama.cpp ที่ `~/llama.cpp/build/bin/`
- llama-server ปัจจุบัน: **build 10696** (`version: 0.3.0-dev (build 10696, commit 1f0a36a35)`) · vLLM image `aiserver-vllm:26.07`
- hub เดิม: `~/aiserver` (ไม่ใช่ git · auto-update เขียนทับ) · state ที่ `~/.aiserver/` · log ที่ `~/.aiserver/logs/`
- v2: `~/aiserver2` · state ที่ `~/.aiserver2/` (แยกกันเด็ดขาด)

## พอร์ตในเครื่อง

| พอร์ต | บริการ |
|---|---|
| 4000 | LiteLLM gateway — ประตู API ถาวรของ client ภายนอก |
| 8000 | **LLM หลัก** — llama-server หรือ vLLM (กติกาเหล็ก: thClaws และ client ในเครื่องผูกพอร์ตนี้) |
| 8001-8002 | LLM เสริม (โหลดหลายโมเดลพร้อมกันได้ภายใต้งบแรม) |
| 8188 | ComfyUI backend |
| 8189 / 8191 / 8192 | Video Studio / Voice Studio / Story Studio |
| 8470 | thClaws agent |
| 9000 | hub เดิม (v0.2.88) |
| **9001** | **AI Server v2 (เฟส 1)** |

## ข้อค้นพบที่ระบบ v2 พึ่งพา (verify แล้ว 2026-08-28 — มีหลักฐานจริง ไม่ใช่การเดา)

1. **HF รองรับ HTTP Range** — `curl -H "Range: bytes=0-1048575"` ตอบ `206` พร้อม header `content-range: bytes 0-1048575/9828981664` ⇒ ได้ทั้ง header ของไฟล์และขนาดเต็มโดยไม่ต้องโหลดจริง
2. **อ่าน arch จาก GGUF header ได้จาก byte แรก ๆ** — `general.architecture` และ `<arch>.context_length` อยู่ต้น KV เสมอ อ่านได้ตั้งแต่ 256KB แรก (แต่ **KV ทั้งหมดอ่านไม่จบใน 1MB** เพราะ token list ยาว ⇒ parser ต้องหยุดทันทีที่ได้ field ที่ต้องการ ห้าม require ว่าต้องอ่านครบ)
3. **`libllama.so` เก็บรายชื่อ arch ที่รองรับไว้ในไฟล์** — `strings libllama.so.0.3.0 | grep -x <arch>` ⇒ เช็ค compat ได้แบบชี้ขาดจาก engine ที่ติดตั้งจริง ไม่ต้องพึ่งตารางที่คนจดเอง (`llama-server` เองเป็น wrapper บาง ๆ arch ไม่ได้อยู่ในนั้น)
4. **`note:` ของ v1 สะกด arch ผิด** — จดไว้ว่า `qwen4_exp` แต่ของจริงในไฟล์คือ `qwen4exp` ⇒ **นี่คือเหตุผลที่ต้องอ่านจากไฟล์จริง ห้ามเชื่อข้อความที่คนเขียน**
5. **HF API `?blobs=true` ให้ขนาด + `lfs.sha256` ทุกไฟล์** — ใช้ทั้งโชว์ขนาดก่อนโหลด และ verify ไฟล์หลังโหลดเสร็จ
6. **ไฟล์ quant อยู่ในโฟลเดอร์ย่อย** (`UD-IQ1_S/`, `MTP/`, `BF16/`) ⇒ จัดกลุ่ม quant ต้องดู path ไม่ใช่แค่ชื่อไฟล์

## กติกาเหล็ก (บทเรียนจากสนามจริงบน GB10 — ห้ามละเมิด)

- LLM หลักเสิร์ฟบน **:8000 เสมอ**
- สลับโมเดล = หยุดตัวเก่าก่อนเสมอ แล้ว health-check ก่อนประกาศว่าพร้อม
- ก่อนโหลดโมเดลใหญ่ (vLLM): ปลด ollama ออกจากแรม · `gpu-memory-utilization 0.55`
- vLLM ต้องเปิด `--enable-auto-tool-choice --tool-call-parser qwen3_xml` ไม่งั้น request ที่มี tools ตอบ 400
- ห้ามใส่ `-c` ใน `args:` ของ model entry — llamacpp.sh ใส่ `-c` จาก ctx ให้แล้ว จะทับค่าที่ผู้ใช้ตั้ง
- **เฟส 1 ห้ามแตะ :9000 และห้ามแตะ instance ที่รันอยู่บน :8000**

## Stack

Python 3.12 · FastAPI + uvicorn · PyYAML (ไม่เขียน parser เอง — ของเดิมเขียนมือแล้วเปราะ ดูคอมเมนต์เตือนใน `docs/models.yaml.v1-reference`) · pydantic สำหรับ validate schema · pytest

เขียนบน Mac ที่ `~/Documents/dgx/aiserver-v2` → deploy ด้วย rsync ไป `dgx:~/aiserver2`
