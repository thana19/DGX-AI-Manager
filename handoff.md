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
