# ผลสำรวจก่อนทำฟีเจอร์ upgrade llama.cpp / engine / software (2026-09-22)

ที่มา: พี่หนุ่มขอฟีเจอร์ "upgrade llamacpp กับ software และ inference ตัวอื่นใน dgx" เมื่อ 2026-09-22 21:31 · สถานะ: ยังไม่ได้ brainstorm/PRD — เอกสารนี้คือข้อมูลตั้งต้น

## ของที่มีอยู่แล้วใน v2

- `POST /api/engines/upgrade` (`main.py:459`) รับเฉพาะ `llamacpp` → เรียก `software.upgrade_llamacpp()`
- `software.upgrade_llamacpp()` (`software.py:225-309`) ทำงานดังนี้:
  - license อ่านจาก `~/.aiserver2/license` → fallback `~/.aiserver/license`
  - mid = `sha256(machine-id + GPU uuid)[:16]` เหมือน v1
  - โหลด pack ด้วย `GET https://aiserver.in.th/dl.php?f=llama-pack-gb10.tar.gz&code=&mid=`
  - backup ของเก่าเป็น `bin.bak-<ts>`
  - extract ทับ `~/llama.cpp/build/bin`
  - re-detect engine หลังอัป
  - มี auto-rollback ถ้าล้ม
  - มี prune ของเก่า
- `rollback_llamacpp()` มีโค้ดอยู่ แต่**ไม่มี endpoint/UI** เรียกใช้
- unit test ของ upgrade มี 7 ตัว แต่**ไม่เคยยิง pack จริง**
- `engines.check_arch` มี 8 กติกา + กลไก `learned_unsupported`
- `engine-compat.yaml`: field `upgrade_hint` เป็น null หมดทุกตัว, field `verified`/`vllm.image` ไม่มีโค้ดอ่านค่าเหล่านี้เลย
- `GET /api/software` มีอยู่ (คืน 16 รายการจาก v1) แต่**ไม่มี UI ใช้งาน**
- catalog note ชี้ไปหน้า "🧰 ซอฟต์แวร์" ซึ่ง v2 ไม่มีหน้านี้
- UI ปัจจุบันมีแค่ปุ่ม "⬆ อัป engine" บนการ์ดโมเดลที่ `needs_upgrade` (`index.html:1072,1165`) — ไม่มี progress bar / ไม่มี confirm dialog
- ฝั่ง vLLM: `action=upgrade_vllm_image` **ไม่มี implementation**
- โฟลเดอร์ `engines/vllm-image/` **ไม่มีอยู่ในรีโป** → build image ล้ม → ต้องใช้ raw image แทน ทำให้ tool calling พัง
- ฝั่ง ds4: **ไม่มีไฟล์** `engines/ds4.sh`
- `POST /api/software/install` มีระบุใน PRD แต่**ไม่เคยถูก implement**
- `plan.md` เฟส 4 กำหนดไว้ว่าเป็น self-update / license / release pipeline

## ของจริงบน DGX

- OS: Ubuntu 26.04.1 aarch64
- GPU: GB10, driver 580.173.02
- **ไม่มี `nvcc`** ติดตั้งอยู่บนเครื่อง
- llama.cpp เป็น pack tarball วางที่ `~/llama.cpp/build/bin` ปัจจุบัน build 10696
  - ไฟล์ `PACK_VERSION` เขียนค่า `b10488` ซึ่ง**เก่าผิด** (ไม่ตรงกับ build จริง)
  - มีไฟล์ `.so` เก่าค้างอยู่ 4 รุ่น ไม่เคยถูกลบ + มี `bin.bak-20260823` ค้างอยู่ด้วย
- `llama-server` ที่รันอยู่จริงที่ `:8001` **ถูก start ด้วยมือ** (ไม่ได้ผ่าน `engines/llamacpp.sh`)
- v1 hub เวอร์ชัน 0.2.142 อยู่ที่ `~/aiserver` — `install.sh` ทำหน้าที่เป็น updater ตรวจ ed25519 signature ของ `latest.json` + ตรวจ sha256
- v2 อยู่ที่ `~/aiserver2` — **ไม่มี** VERSION file หรือ updater ของตัวเอง (มีแค่ `deploy.sh` + cron watchdog รันทุก 1 นาที)
- ds4 มี pack อยู่แต่**ไม่ได้รัน**
- vLLM รันเป็น docker image `aiserver-vllm:26.07`
- ollama เวอร์ชัน 0.32.14 รันผ่าน systemd (ตัว installer pin ไว้ที่ 0.31.1 → ถ้า reinstall จะกลายเป็น downgrade)
- LiteLLM เวอร์ชัน 1.91.2 รันใน venv ของ v1 (**มี process รั่วอยู่ 3 ตัว**)
- ComfyUI container core เวอร์ชัน 0.33.4 (pin ไว้ที่ v0.37.0 คือมี update รอ; `custom_nodes` เป็นของ user root ทำให้ update ล้มได้)
- aria2 เวอร์ชัน 1.37 ติดตั้งผ่าน apt
- thclaws เวอร์ชัน 0.120.0
- uv เวอร์ชัน 0.12.0
- **ไม่มี systemd unit ควบคุม stack ทั้งหมด** (ใช้ cron `@reboot` + watchdog แทน — อาจเกิด race condition ตอน upgrade)

## กลไก v1

- `dl.php?f=<pack>&code=<license>&mid=` **ไม่มี manifest หรือ version list** — ชื่อไฟล์ทำหน้าที่เป็น version pointer แทน โดยเทียบ sha8 กับค่าที่เก็บใน `~/.aiserver/packs.json`
- v1 hub self-update เขียนทับทั้ง tree ของโปรแกรม → ทำให้ patch 4 จุดที่เคยใส่ไว้หายไป (ต้องกู้ด้วย `dgx-repatch`)
- v2 ได้ดูดซับ patch เหล่านั้นกลายเป็นโค้ดจริงในตัวแล้ว (ไม่ต้อง repatch ซ้ำ)
- `verify_sha` ของ v1 มีบั๊ก: ข้ามการตรวจแบบเงียบ ๆ เมื่อเจอ 404
- `release.sh` ของ dev repo หายไป → ตอนนี้ปล่อยอัปเดตให้ลูกค้าไม่ได้

## คำถามเปิดที่ต้องถามพี่หนุ่ม

- `dl.php` ยังมีชีวิตอยู่ไหม / มี endpoint ที่บอก "เวอร์ชันล่าสุด" ไหม (ถ้าไม่มี = เช็ค update ก่อนโหลดจริงไม่ได้)
- scope ของฟีเจอร์นี้ควรครอบคลุมแค่ llama.cpp อย่างเดียว หรือรวม vLLM image + ds4 + ComfyUI + litellm/ollama + ตัว v2 เองด้วย
- หลัง upgrade ควร restart engine อัตโนมัติ หรือปล่อยให้ผู้ใช้สั่งเอง (กฎเหล็ก: ห้ามแตะ `:8000` ที่กำลังรันอยู่)
- จำเป็นต้องมีหน้า UI "ซอฟต์แวร์" แยกต่างหากไหม
- ปุ่ม upgrade บนพอร์ตที่เปิดอยู่ ควรมี auth ป้องกันอย่างไร
