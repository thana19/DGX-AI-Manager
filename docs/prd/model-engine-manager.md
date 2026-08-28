# PRD — Model & Engine Manager (AI Server v2 เฟส 1)

อัปเดต: 2026-08-28 23:30

## Problem Statement

- hub เดิมโหลดได้เฉพาะโมเดลที่ vendor ใส่ไว้ใน `models.yaml` — ผู้ใช้เพิ่มเองไม่ได้เลย
- ความสัมพันธ์ "โมเดลนี้ต้องใช้ engine รุ่นไหน" เป็นแค่ข้อความใน `note:` ให้คนอ่านเอง ตัวอย่างจริงจาก registry เดิม (`docs/models.yaml.v1-reference`):
  - `qwen3.8-flash-next-udq2` — note: "⚠️ ต้องอัป llama-server รุ่นใหม่ก่อน (🧰 ซอฟต์แวร์ → รองรับ arch qwen4_exp; รุ่นเก่าโหลดไม่ได้)"
  - `glm-5.3-flash-udq1` — note: "⚠️ ชอบคิดยาว → ใส่ system prompt 'ตอบตรงๆ สั้นๆ' ให้ตอบตรง · ต้องอัป llama-server รุ่นใหม่ก่อน (🧰 ซอฟต์แวร์ → รองรับ glm5next; รุ่นเก่าโหลดไม่ได้)"
- ผลคือ: กด "โหลด" → ดาวน์โหลด 79-87GB → llama-server เก่าโหลดไม่ขึ้น → ผู้ใช้ไม่รู้ว่าเกิดอะไร เสียเวลาและเน็ตฟรี
- ปุ่ม "โหลด" ปุ่มเดียวทำ 2 อย่างพร้อมกัน (ดาวน์โหลด + โหลดขึ้นแรม) → มองไม่เห็นว่าตอนนี้ระบบทำอะไรอยู่ หยุด/ต่อไม่ได้

## Solution

ระบบตอบให้ได้**ก่อนดาวน์โหลด**ว่า "โมเดลตัวนี้เครื่องนี้รันได้ไหม" ด้วย 2 ฝั่งที่อ่านจากของจริงทั้งคู่:

- **ฝั่งโมเดล** — อ่าน `general.architecture` จาก GGUF header ผ่าน HTTP Range เพียง ~256KB (ยืนยันแล้ว: HF ตอบ `206`)
- **ฝั่ง engine** — อ่านรายชื่อ arch ที่รองรับออกจาก `libllama.so` ของ engine ที่ติดตั้งอยู่จริง

เทียบสองฝั่งนี้ = คำตอบที่ชี้ขาด ไม่ต้องเชื่อข้อความที่คนจดไว้

5 module หลัก:

| module | หน้าที่ |
|---|---|
| `catalog` | รวม registry vendor + user, validate schema, CRUD |
| `hf` | คุยกับ Hugging Face API เพื่อ resolve repo id → quant list |
| `gguf` | อ่าน arch จาก GGUF header ผ่าน byte range |
| `engines` | ตรวจ engine ที่มี, ตัดสิน compat, อัป/rollback |
| `downloads` | คิวดาวน์โหลดผ่าน aria2 RPC พร้อม pause/resume/cancel |

## User Stories

1. ในฐานะเจ้าของเครื่อง ฉันอยากวาง HF repo id แล้วเห็นรายการ quant + ขนาด เพื่อเลือกตัวที่แรมพอ
2. ในฐานะเจ้าของเครื่อง ฉันอยากรู้ก่อนกดดาวน์โหลดว่า engine ที่มีรันได้ไหม เพื่อไม่เสียเน็ต 80GB ฟรี
3. ในฐานะเจ้าของเครื่อง ฉันอยากให้ระบบอัป engine ให้เองในคลิกเดียวเมื่อโมเดลใหม่ต้องการ
4. ในฐานะเจ้าของเครื่อง ฉันอยากเห็นความคืบหน้าการดาวน์โหลด + หยุด/ต่อได้ เพราะไฟล์ใหญ่ใช้เวลาหลายชั่วโมง
5. ในฐานะเจ้าของเครื่อง ฉันอยากให้ "ดาวน์โหลด" กับ "โหลดขึ้นแรม" เป็นคนละปุ่ม เพราะบางทีอยากโหลดไฟล์ไว้ก่อนแล้วค่อยรัน
6. ในฐานะผู้ใช้สายลึก ฉันอยากแก้ engine/ctx/args เองได้ผ่านโหมด advanced เมื่อระบบเดาผิด

## Implementation Decisions

| # | การตัดสินใจ | เหตุผล | ทางเลือกที่ไม่เอา |
|---|---|---|---|
| 1 | แยก `catalog.yaml` (vendor) ออกจาก `~/.aiserver2/user-models.json` (ผู้ใช้) | vendor push catalog ทับได้โดยไม่ลบของที่ผู้ใช้เพิ่มเอง | ไฟล์เดียวรวมกัน (ของผู้ใช้จะหายตอนอัปเดต) |
| 2 | ใช้ PyYAML + pydantic แทน parser เขียนมือ | parser เดิมเปราะจนต้องมีคอมเมนต์เตือนในไฟล์ว่า "ห้ามใส่ inline #" และ `lint-models.py` ที่กันไว้ก็หายไปแล้ว | เขียน parser เอง (ซ้ำรอยเดิม) |
| 3 | อ่าน GGUF header ผ่าน HTTP Range 1MB | ตอบ compat ได้โดยไม่ต้องโหลดไฟล์จริง | โหลดก่อนแล้วค่อยรู้ (คือปัญหาต้นเรื่อง) |
| 4 | **compat ชั้นหลัก = ถาม engine ที่ติดตั้งจริง** — อ่าน arch จาก GGUF header ↔ `strings libllama.so \| grep -x <arch>` | ชี้ขาดและอัปเดตตัวเองตาม engine ที่ลง ไม่มีตารางให้ล้าสมัย · **หลักฐาน:** `note:` ของ v1 จดว่า `qwen4_exp` แต่ของจริงคือ `qwen4exp` ⇒ ตารางที่คนจดผิดได้ | ตารางที่คนดูแลเป็นแหล่งความจริงหลัก |
| 4b | ชั้นสำรอง 3 ชั้น: (ก) `requires:` ที่ vendor ประกาศใน entry (ใช้กับ vLLM/ds4 ที่ไม่มี arch list ให้อ่าน) · (ข) `engine-compat.yaml` แม็ป arch → pack เวอร์ชันที่มี arch นั้น (ใช้ตอบว่า "ต้องอัปเป็นตัวไหน") · (ค) เรียนจาก error log `unknown model architecture` ตอนรันจริง | ชั้นหลักตอบได้แค่ "ได้/ไม่ได้" — ชั้นสำรองตอบว่า "แล้วต้องทำยังไงต่อ" | ใช้ชั้นเดียว (ครอบคลุมไม่พอ) |
| 5 | aria2c RPC แทน spawn ทีละไฟล์ | ได้ progress/pause/resume/cancel จริง | spawn `aria2c` ต่อไฟล์แล้วให้ UI เดาความคืบหน้าจากขนาดไฟล์บนดิสก์ (ของเดิม) |
| 6 | คิวดาวน์โหลด 1 job ต่อครั้ง | เน็ตกับดิสก์เป็นคอขวด รันขนานไม่ได้เร็วขึ้น | รันขนานหลาย job |
| 7 | reuse `engines/*.sh` เดิมทั้งดุ้น | bash พวกนี้ผ่านสนามจริงบน GB10 แล้ว มีบทเรียนฝังอยู่ในคอมเมนต์ เขียนใหม่มีแต่เสีย | เขียน engine script ใหม่ |
| 8 | อัป engine ต้องเก็บ binary เก่าไว้ rollback | แบบเดียวกับที่ `install-tools.sh` ทำกับ thclaws | อัปทับตรง ๆ ไม่เก็บของเก่า |
| 9 | รันพอร์ต 9001 คู่ hub เดิม | ทางถอยคือแค่ปิด :9001 ระบบเดิมไม่เคยถูกแตะ | แทนที่ hub เดิมทันที |

## Modules

| ไฟล์ | หน้าที่ | test อะไร |
|---|---|---|
| `server/catalog.py` | รวม registry 2 แหล่ง, validate ด้วย pydantic, CRUD user models | merge, validate ฟิลด์ผิด, sharded path, ตรวจไฟล์ครบบนดิสก์ |
| `server/hf.py` | เรียก HF API list ไฟล์+ขนาด, จัดกลุ่ม quant, รวม shard, ตรวจ gated, จับ mmproj/mtp/dflash | จาก fixture JSON ที่บันทึกจาก response จริง |
| `server/gguf.py` | parse GGUF header จาก byte range | fixture header จริงของ Qwen3.8 และ GLM-5.3 |
| `server/engines.py` | ตรวจ engine ที่มี, compat 3 ชั้น, อัป/rollback engine | compat matrix, parse `llama-server --version`, learn จาก log |
| `server/downloads.py` | job queue + aria2 RPC | state machine, resume, ยกเลิก |
| `server/software.py` | เช็ค/ติดตั้ง software (ยกจาก `install-tools.sh.v1-reference`) | parse ผลเช็คเวอร์ชัน |
| `server/main.py` | FastAPI routes + เสิร์ฟ static | — |
| `server/static/index.html` | UI 3 ส่วน (คลังโมเดล / เพิ่มโมเดล / ดาวน์โหลด) | — |

## API (เฟส 1)

| method | path | คำอธิบาย |
|---|---|---|
| GET | `/api/health` | health check |
| GET | `/api/models` | รายการ catalog (vendor + user รวมแล้ว) |
| POST | `/api/models/resolve` | รับ HF repo id → คืน quant list + arch + compat |
| POST | `/api/models` | เพิ่มเข้า user-models |
| DELETE | `/api/models/{id}` | ลบออกจาก user-models |
| POST | `/api/downloads` | สั่งดาวน์โหลด |
| GET | `/api/downloads` | สถานะ/ความคืบหน้า download job ทั้งหมด |
| POST | `/api/downloads/{id}/pause\|resume\|cancel` | ควบคุม job |
| GET | `/api/engines` | engine ที่มีในเครื่อง + เวอร์ชัน |
| POST | `/api/engines/upgrade` | อัป engine |
| POST | `/api/activate` | โหลดโมเดลขึ้นแรมบน engine/พอร์ตที่ระบุ |
| GET | `/api/software` | เช็คสถานะ software ที่ต้องมี |
| POST | `/api/software/install` | ติดตั้ง software |

## Test coverage ที่ต้องมี

- unit test ทุก module ที่มี logic (ไม่ต้อง test route ที่แค่ต่อสาย)
- fixture จาก response จริงของ HF (บันทึกไว้ใน `tests/fixtures/`) — ไม่ยิงเน็ตตอนรัน test
- ไม่ mock GGUF parser — ใช้ header จริง

## Out of Scope (เฟสถัดไป)

gauges/health dashboard · LiteLLM config sync · cloud BYOK keys · per-instance sampling settings · discovery endpoint + API keys · ระบบ self-update/license · studio packs (ComfyUI) · Voice/Story/Video Studio · ระบบ auth บนหน้าเว็บ
