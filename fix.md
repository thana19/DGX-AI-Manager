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
