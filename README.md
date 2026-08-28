# AI Server v2

เขียนใหม่แทน AI Server hub เดิม (v0.2.88 · `:9000`) — ส่งมอบเป็นเฟส

- **เฟส 1 — Model & Engine Manager** (กำลังทำ) · รันที่ `:9001` คู่กับ hub เดิม
- อ่านก่อนเริ่ม: [`CONTEXT.md`](CONTEXT.md) · [PRD เฟส 1](docs/prd/model-engine-manager.md) · [ADR 0001](docs/adr/0001-registry-split.md)
- สรุปแผนแบบหน้าเว็บ: [`html-plan/model-engine-manager-2026-08-28_2330.html`](html-plan/model-engine-manager-2026-08-28_2330.html)

## โครงสร้าง

```
server/      FastAPI app (catalog · hf · gguf · engines · downloads · software)
engines/     bash เดิมจาก v1 ที่ผ่านสนามจริงบน GB10 แล้ว — reuse ทั้งดุ้น
tests/       pytest + fixture จาก response จริงของ HF (ไม่ยิงเน็ตตอน test)
docs/        PRD · ADR · ไฟล์อ้างอิงจาก v1 (*.v1-reference)
```

## รัน (dev บน Mac)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pytest
.venv/bin/uvicorn server.main:app --port 9001
```

## deploy ขึ้น DGX

```bash
bash deploy.sh        # rsync → dgx:~/aiserver2 แล้ว restart :9001
```

**ห้ามแตะ** `:9000` (hub เดิม = ทางถอย) และ instance ที่รันอยู่บน `:8000`
