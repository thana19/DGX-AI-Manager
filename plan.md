# plan.md — AI Server v2

อัปเดต: 2026-08-29 01:05

## เป้าหมายรวม

เขียน AI Server ใหม่ทั้งตัวแทน hub เดิม (v0.2.88) ส่งมอบเป็นเฟส เหตุผลที่ต้องขึ้น repo ใหม่: dev repo เดิมของ `release.sh` หายไป หาไม่เจอทั้ง Mac และ DGX ⇒ ปล่อยอัปเดตให้ลูกค้าไม่ได้อีกแล้ว

## ตารางเฟส

| เฟส | ขอบเขต | สถานะ |
|---|---|---|
| 1 | Model & Engine Manager (:9001) | ✅ เสร็จ 2026-08-29 |
| 2 | gauges/health dashboard · service start/stop · studio packs | ⏳ |
| 3 | LiteLLM sync · cloud BYOK · per-instance settings · discovery + API keys | ⏳ |
| 4 | ระบบ self-update/license + release pipeline แล้วสลับพอร์ตเป็น :9000 แทน hub เดิม | ⏳ |

## เฟส 1 — ขั้นตอนที่ทำไปแล้ว

| ขั้น | ผล |
|---|---|
| CONTEXT.md | เสร็จ |
| ซักค้าน plan | เสร็จ |
| PRD | เสร็จ |
| HTML plan | เสร็จ (confirm แล้ว) |
| 8 โมดูล TDD | เสร็จ 186 test ผ่าน |
| Deploy :9001 | เสร็จ |
| ทดสอบ | 7/7 ผ่าน |

อ้างอิง (ห้าม copy เนื้อหาซ้ำ): `docs/prd/model-engine-manager.md` · `docs/adr/0001-registry-split.md` · `html-plan/model-engine-manager-2026-08-28_2330.html` · `TESTING.md`

## สิ่งที่รู้แล้วว่าต้องทำในเฟสถัดไป

- ยังไม่มี endpoint หยุด instance (`/api/instances/{port}/stop`) — เฟส 1 บอกได้แค่ว่าต้องหยุดอะไร แต่หยุดให้ไม่ได้
- `POST /api/engines/upgrade` เขียนโค้ดครบแล้วแต่ยังไม่ได้ทดสอบกับ pack จริง (เครื่องนี้ build 10696 ใหม่พอ ไม่มีเหตุให้อัป)
- `read_supported_archs` คืน 2341 token ซึ่งเป็น superset ของ arch จริง — false OK เป็นไปได้ (จะตกไปใช้กลไกเรียนจาก log) ถ้าอยากแม่นกว่านี้ต้องหาวิธีดึงเฉพาะตาราง arch
- ds4 ยังไม่มี `engines/ds4.sh` ในโปรเจกต์
