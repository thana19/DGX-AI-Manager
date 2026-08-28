# ADR 0001 — แยก catalog ของ vendor ออกจากโมเดลที่ผู้ใช้เพิ่มเอง

## Status

Accepted · 2026-08-28

## Context

hub เดิมมี `models.yaml` ไฟล์เดียว ที่ vendor push ทับผ่าน `dl.php` ได้ (ดูกลไก `refresh_models_from_server` ของ v1) ถ้าให้ผู้ใช้เพิ่มโมเดลลงไฟล์เดียวกัน ของที่เพิ่มจะหายทุกครั้งที่ vendor push

## Decision

ใช้ 2 ไฟล์ schema เดียวกัน:

- `catalog.yaml` ใน repo (vendor, read-only สำหรับผู้ใช้)
- `~/.aiserver2/user-models.json` (ผู้ใช้, vendor ไม่แตะ)

`catalog.py` merge ทั้งสองไฟล์ตอนอ่าน — id ชนกัน = ของผู้ใช้ชนะ (แต่ขึ้นป้ายเตือนใน UI)

## Consequences

**ดี**
- ผู้ใช้เพิ่มเองปลอดภัย ไม่หายตอน vendor push
- vendor push ได้อิสระ ไม่ต้องกลัวทับของผู้ใช้
- ลบ user model ไม่กระทบ catalog

**เสีย**
- ต้องมี merge logic + กติกา id ชนกัน
- ต้อง validate 2 รูปแบบไฟล์ (YAML/JSON) ด้วย schema เดียวกัน
