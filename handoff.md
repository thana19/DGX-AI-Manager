# handoff.md

## [2026-08-28 22:23 → 2026-08-29 01:05] สร้าง AI Server v2 + ส่งมอบเฟส 1

- **สิ่งที่ทำ**:
  - สร้าง repo ใหม่ทั้งตัว
  - เอกสารพื้นฐาน: `CONTEXT.md` · `docs/prd/model-engine-manager.md` · `docs/adr/0001-registry-split.md` · `html-plan/model-engine-manager-2026-08-28_2330.html` (เวอร์ชันออนไลน์: https://claude.ai/code/artifact/e793f65d-bacb-4322-adcd-3e8929dfdfdd)
  - 8 โมดูล TDD รวม 186 test ผ่าน
  - Deploy `:9001` บน DGX (`dgx:~/aiserver2`)
  - ทดสอบ end-to-end 7/7 ผ่าน (ดู `TESTING.md`)
  - แก้บั๊ก 2 ตัวที่เจอตอนทดสอบ (ดู `fix.md`)

- **สถานะระบบล่าสุด (2026-08-29 01:05)**:
  - AI Server v2 `:9001` — ✅ ทำงานปกติ
  - aria2 RPC `:6800` — ✅ ทำงานปกติ
  - hub เดิม `:9000` — ✅ ทำงานปกติ
  - `:8000` = GLM-5.3-Flash (hub เดิมโหลดไว้ คืนสภาพเดิมหลังทดสอบแล้ว)
  - แรมใช้ 97GB ว่าง 24GB

- **วิธีใช้งาน/ดูแล**:
  - `bash deploy.sh` — deploy จาก Mac ไป DGX
  - `bash run.sh start|stop|status` — คุม service บน DGX
  - `bash services/aria2.sh start|status` — คุม aria2
  - log อยู่ที่ `~/.aiserver2/logs/` บน DGX

- **งานค้าง / ควรทำ session ถัดไป**: ดูหัวข้อ "สิ่งที่รู้แล้วว่าต้องทำในเฟสถัดไป" ใน `plan.md` + เริ่มเฟส 2 (gauges/health dashboard · service start/stop · studio packs)

- **Suggested Skills**: `new-project-setup` (ก่อนเริ่มเฟส 2) · `project-hygiene` (ก่อน git push) · `remember`

## [2026-08-29 07:20 → 2026-08-31 09:24] ใช้งานจริงแล้วแก้ตามที่เจอ + ทำส่วน "วิธีใช้โมเดล"

ต่อจากเฟส 1 ที่ส่งมอบไปแล้ว รอบนี้พี่หนุ่มใช้งานจริงแล้วเจอปัญหาทีละอย่าง แล้วแก้ตามไปเรื่อย ๆ — 9 commit ใหม่ (`dd06a10` → `9ca4506`) · **252 test ผ่าน** (จากเดิม 186)

- **สิ่งที่เพิ่มเข้ามา**:

| ฟีเจอร์ | ทำอะไร |
|---|---|
| ค้นหา HF จากลิงก์ใด ๆ | วางลิงก์ GitHub/ข้อความลอย ๆ ก็ได้ — normalize แล้วค้น HF ให้เลือก · `normalize_repo_id()` + `search_models()` |
| ปุ่มหยุด instance | `server/instances.py` (ใหม่) · `GET /api/instances` · `POST /api/instances/{port}/stop` · guard พอร์ต 8000 และช่วง 8000-8009 |
| ธีม + gauge 6 ตัว | ธีมเดียวกับ DGX Spark Monitor · `GET /api/metrics` proxy จาก `:9100` (ไม่เปิด CORS เรียกตรงไม่ได้) |
| ปุ่ม 🔌 วิธีใช้ | `GET /api/endpoints` · modal 4 แท็บ (curl/env · Claude Code · opencode · Python/JS) copy ได้ |
| ช่องลองคุย (playground) | สตรีม SSE ยิงตรงจากเบราว์เซอร์ · แยกกล่อง "กำลังคิด" · แนบรูปทดสอบ vision · หยุดกลางคันได้ · โชว์ tok/s |
| สวิตช์ปิดโหมดคิด | `chat_template_kwargs.enable_thinking=false` · วัดจริง 106 → 4 token · default `max_tokens` 512 → 2048 |

- **สถานะระบบล่าสุด (2026-08-31 09:24)**:
  - v2 `:9001` ✅ · aria2 `:6800` ✅ · hub เดิม `:9000` ✅
  - `:8001` = `Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf` ctx 262,144 (มี mmproj อ่านภาพได้)
  - แรมใช้ 61 GB / ว่าง 60 GB
  - 252 test ผ่าน · 9 commit ใหม่ · **ยังไม่ push git** (ยังเป็น local repo ไม่มี remote)

- **สิ่งที่รู้แล้วว่ายังขาด (งานเฟสถัดไป)** — รายการเต็มอยู่ใน `plan.md` · ที่เพิ่มรอบนี้:
  - หน้าเว็บไม่บอกเพดาน `ctx` ข้างช่อง `max_tokens` ⇒ ตั้งเกินแล้วไม่รู้ตัวจนกว่าจะ error (พี่หนุ่มถามเรื่องนี้ตอนท้าย session ยังไม่ได้ทำ)
  - สวิตช์ปิดโหมดคิดมีผลเฉพาะใน playground · client ข้างนอก (opencode ฯลฯ) ต้องส่ง `chat_template_kwargs` เอง
  - `POST /api/engines/upgrade` ยังไม่เคยทดสอบกับ pack จริง

- **Suggested Skills**: `project-hygiene` (ก่อน git push) · `new-project-setup` (ก่อนเริ่มเฟส 2) · `remember`
