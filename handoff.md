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

## [2026-09-03 15:11 → 15:58] ซ่อม 3 เรื่องที่เจอตอนใช้งานจริง

session นี้ไม่ได้เพิ่มฟีเจอร์ เป็นการไล่ซ่อมปัญหาที่พี่หนุ่มเจอตอนใช้งานจริง 3 เรื่อง แก้ที่ shell script ล้วน ไม่แตะโค้ด Python จึงไม่มี test ใหม่

- **สิ่งที่ทำ**:

| เรื่อง | ต้นเหตุ | แก้ที่ไหน |
|---|---|---|
| `:9001` ไม่ขึ้นเองตอนบูตเครื่อง | crontab ของ user `dgx` บน DGX มีแต่ entry ของ hub เดิม `:9000` ไม่มีบรรทัดไหนชี้ไป `aiserver2` เลย และเครื่องไม่มี systemd unit ของ aiserver — autostart อาศัย cron อย่างเดียว | crontab บน DGX (ไม่ใช่ไฟล์ใน repo) เติม `@reboot` + watchdog รายนาที · สำรอง crontab เดิมไว้ที่ `/home/dgx/crontab.bak.20260903-1513` |
| โหลด `qwen3-8b-fp8` ไม่ขึ้น + error โชว์เป็น hash | DeepGEMM แปลง scale-factor layout ของ FP8 block-quant บน GB10 ไม่ได้ ⇒ EngineCore ตาย · และ `docker run -d` พิมพ์แค่ container ID ทับไฟล์ log ทำให้ error ที่ผู้ใช้เห็นเป็น hash | `engines/vllm.sh` 3 จุด (ปิด DeepGEMM เป็น default · ส่ง `VLLM_*` เข้า container ด้วย `-e` · stream `docker logs -f` ลงไฟล์แทน) |
| `aria2.log` โตถึง 17 GB | `--log-level` (ของไฟล์) เป็นคนละตัวกับ `--console-log-level` และ default เป็น debug | `services/aria2.sh` เติม `--log-level=warn` |

  รายละเอียดครบทั้ง 3 เรื่องอยู่ใน `fix.md` (entry เวลา 15:15, 15:41, 15:48)

- **สถานะระบบล่าสุด (2026-09-03 15:58)**:
  - AI Server v2 `:9001` ✅ (health 200) · aria2 `:6800` ✅ · hub เดิม `:9000` ✅ (status 200)
  - `:8000` = `qwen3-8b-fp8` โหลดอยู่ผ่าน vLLM (up) — โหลดค้างไว้จาก session นี้ตอนทดสอบ ถ้าจะโหลดโมเดลอื่นต้องหยุดตัวนี้ก่อน
  - แรมใช้ 113 GB ว่าง 7 GB จาก 121 GB (vLLM จอง `gpu-memory-utilization 0.55`)
  - log: `aria2.log` 0 ไบต์ · `~/.aiserver/logs/vllm.log` 119 KB (เป็น log จริงของ vLLM แล้ว)
  - git: มี 3 ไฟล์แก้ค้างยังไม่ commit — `engines/vllm.sh` · `services/aria2.sh` · `fix.md` (commit ล่าสุด `837ac82`) · repo ยังเป็น local ไม่มี remote

- **งานค้าง / ควรทำ session ถัดไป**:
  - ยังไม่ได้ reboot เครื่องจริงเพื่อพิสูจน์ `@reboot` ของ cron ที่เพิ่งเติม (watchdog รายนาทีทดสอบแล้วว่ากู้เองได้ใน 25 วินาที เป็นตาข่ายรองอยู่)
  - commit 3 ไฟล์ที่ค้างอยู่
  - งานเฟสถัดไปตามเดิม — ดูรายการใน `plan.md` และหัวข้อ "สิ่งที่รู้แล้วว่ายังขาด" ของ session ก่อนหน้าในไฟล์นี้

- **Suggested Skills**: `project-hygiene` (ก่อน git push) · `remember`

## [2026-09-22 18:06 → 2026-09-22 21:35] รองรับ GGUF repo ไม่มี quant token + ดาวน์โหลด gated repo

- **สิ่งที่ทำ**:
  - แก้บั๊ก GGUF repo ที่ไฟล์อยู่ root ลงท้ายด้วยชื่อ variant (`main`/`mainline`) ไม่ถูกจับเป็น quant + รองรับดาวน์โหลด repo ที่ gated ด้วย HF token (ดู `fix.md` entry `[2026-09-22 21:35]`)
  - เพิ่ม unit test 29 ตัว รวมเป็น 328/328 ผ่าน + ทดสอบ end-to-end บน `:9001` ครบ (ดู `TESTING.md` entry `[2026-09-22 21:35]`)
  - Deploy `:9001` แล้ว

- **สถานะระบบล่าสุด (2026-09-22 21:35)**:
  - AI Server v2 `:9001` ✅ deploy งานนี้แล้ว
  - llama.cpp build 10696 รองรับ arch `qwen4exp` ของ Qwen3.8-Flash-Next
  - ยังไม่มีไฟล์ `~/.aiserver2/hf_token` บน DGX (ยังไม่มีใครใส่ token)

- **งานค้าง / ควรทำ session ถัดไป**:
  1. ทดสอบดาวน์โหลด `…-AD-4.27-mainline` จริงด้วย HF token (ต้องยอมรับ gate บนเว็บ HF ก่อน) แล้วดูว่า aria2 ไม่ 401 · ถ้าโหลดเสร็จลองโหลดขึ้นแรมพร้อม `--mmproj` (mmproj ไม่ถูกเพิ่มอัตโนมัติ)
  2. ยังไม่มี UI ดู/ลบ token ที่จำไว้ (`~/.aiserver2/hf_token`) — ถ้าต้องการ ให้เพิ่ม
  3. commit/push งานนี้ (ยังไม่ได้ commit; working tree มีของ session ก่อนค้างอยู่ด้วย: `engines/vllm.sh`, `services/aria2.sh`)

- **Suggested Skills**: `project-hygiene` · `remember`

## [2026-09-22 21:31 → 2026-09-22 22:05] hotfix ดาวน์โหลด repo gated + เตรียมงาน upgrade

- **สิ่งที่ทำ**: hotfix 2 บั๊ก (ดู `fix.md` 2 entry `[2026-09-22 22:05]`) · deploy `:9001` · explore ครบ 3 ด้านสำหรับฟีเจอร์ "upgrade llama.cpp / engine / software" (ผลสรุปอยู่ท้าย plan file ของ session — ยังไม่ได้ brainstorm/PRD)

- **สถานะระบบล่าสุด (2026-09-22 22:05)**:
  - `:9001` deploy hotfix แล้ว
  - job ดาวน์โหลด Navin 2 ตัว cancel แล้ว
  - ยังไม่มี `hf_token` บน DGX

- **งานค้าง / ควรทำ session ถัดไป**:
  1. พี่หนุ่มใส่ HF token → ตรวจสอบ → โหลด `…-AD-4.27-mainline` (ห้ามเลือก `main`) → ลบ entry `main` ออกจากคลัง
  2. **ฟีเจอร์ upgrade** (คำขอของพี่หนุ่ม 2026-09-22 21:31): brainstorm → PRD → HTML plan → implement; ของเดิมมี `POST /api/engines/upgrade` (llamacpp เท่านั้น, ไม่เคยยิง pack จริง), `software.py` upgrade/rollback, ไม่มี UI ซอฟต์แวร์, vLLM/ds4/ComfyUI ไม่มี path upgrade; คำถามเปิด: dl.php ยังใช้ได้ไหม/มี endpoint บอกรุ่นล่าสุดไหม, scope ครอบคลุมอะไรบ้าง, restart engine หลังอัปหรือไม่, auth ของปุ่ม upgrade
  3. commit/push งาน hotfix

- **Suggested Skills**: `superpowers:brainstorming` (ต่อเรื่อง upgrade) · `new-project-setup` · `project-hygiene`

## [2026-09-22 22:17 → 2026-09-23 06:20] follow-up ดาวน์โหลด repo gated (403) + UI แท็บ + arch ว่าง

- **สิ่งที่ทำ**: อ้าง `fix.md` entry `[2026-09-23 06:20]` และ `TESTING.md` entry `[2026-09-23 06:20]` · deploy `:9001`
- **สถานะระบบล่าสุด (2026-09-23 06:20)**: token ของพี่หนุ่มอยู่บน DGX แล้ว แต่ HF ยัง 403 เพราะยังไม่ได้กด Agree บนหน้า repo · entry Navin 2 ตัวในคลัง (main ควรลบ) arch ว่าง — หลังโหลดเสร็จ/หรือลบแล้วเพิ่มใหม่หลังตรวจสอบด้วย token จะได้ arch qwen4exp
- **งานค้าง / ควรทำ session ถัดไป**:
  1. พี่หนุ่มกด Agree (บัญชี anaht) + เช็คสิทธิ์ token → กดโหลด mainline → ลบ entry main
  2. ฟีเจอร์ upgrade (ยังไม่เริ่ม brainstorm — ดูหัวข้อก่อนหน้า)
  3. commit/push งานนี้
  4. (ไอเดีย) ปุ่ม "ตรวจสอบใหม่" บนการ์ดเพื่อเติม arch ให้ entry ที่ arch ว่าง
- **Suggested Skills**: `superpowers:brainstorming` · `project-hygiene`

## [2026-09-23 07:30 → 2026-09-23 07:40] ปุ่มล้างงานดาวน์โหลดที่ไม่สำเร็จ

- **สิ่งที่ทำ**: เพิ่ม API `POST /api/downloads/clear` → `DownloadManager.clear_failed()` · เพิ่มปุ่ม 🧹 ในแถวแท็บดาวน์โหลด · อ้าง `TESTING.md` entry `[2026-09-23 07:40]`
- **สถานะ**: `:9001` deploy แล้ว · HF token ของพี่หนุ่มยัง 403 — ยังไม่ได้กด Agree บนหน้า repo
- **งานค้าง / ควรทำ session ถัดไป**:
  1. พี่หนุ่มกดล้าง 5 รายการเอง
  2. กด Agree บน HF แล้วโหลด mainline
  3. commit/push
  4. ฟีเจอร์ upgrade (ยังไม่เริ่ม brainstorm)
- **Suggested Skills**: `superpowers:brainstorming` · `project-hygiene`
