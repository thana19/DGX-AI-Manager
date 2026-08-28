"""server/main.py — FastAPI app ของ AI Server v2 (เฟส 1 · พอร์ต 9001)

รวมโมดูล catalog/hf/gguf/engines/downloads/software เป็น HTTP API ให้ server/static/index.html เรียกใช้
สัญญา JSON ของทุก endpoint ตายตัว — ดู docs/prd/model-engine-manager.md และ task ที่สั่งงานไฟล์นี้
ห้ามเปลี่ยนชื่อ field เพราะ UI ฝั่งหน้าเว็บเขียนตามนี้เป๊ะ
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import catalog, downloads, engines, gguf, hf, paths, software

APP_VERSION = "2.0.0-phase1"
DEFAULT_PORT = int(os.environ.get("AISERVER2_PORT", "9001"))

# LLM หลักเสิร์ฟบน :8000 เสมอ — กติกาเหล็กจาก CONTEXT.md (thClaws และ client ในเครื่องผูกพอร์ตนี้)
_MAIN_LLM_PORT = 8000
_KNOWN_ENGINES = ("llamacpp", "vllm", "ds4")

app = FastAPI(title="AI Server v2", version=APP_VERSION)

_static_dir = paths.repo("server", "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


def _gb(n: float) -> float:
    """byte → GB ทศนิยม 1 ตำแหน่ง (ฐาน 1e9 — ตรงกับ hf.py/ตัวเลขที่ HF โชว์ผู้ใช้)"""
    return round(n / 1e9, 1)


# ---------------------------------------------------------------------------
# DownloadManager — singleton ระดับโมดูล (สร้างแบบ lazy เพื่อให้ test override env ก่อนได้)
# ---------------------------------------------------------------------------

_download_manager: downloads.DownloadManager | None = None


def _build_download_manager() -> downloads.DownloadManager:
    url = os.environ.get("ARIA2_RPC_URL", "http://127.0.0.1:6800/jsonrpc")
    secret = os.environ.get("ARIA2_RPC_SECRET", "")
    aria2 = downloads.Aria2Client(url, secret)
    return downloads.DownloadManager(aria2)


def get_download_manager() -> downloads.DownloadManager:
    global _download_manager
    if _download_manager is None:
        _download_manager = _build_download_manager()
    return _download_manager


def reset_download_manager() -> None:
    """ใช้เฉพาะ test — บังคับให้ get_download_manager() สร้างใหม่รอบถัดไป (อ่าน env/state_path ปัจจุบัน)"""
    global _download_manager
    _download_manager = None


# ---------------------------------------------------------------------------
# helper ร่วม
# ---------------------------------------------------------------------------


def _compat_dict(result: engines.CompatResult) -> dict[str, Any]:
    return {"status": result.status.value, "reason": result.reason, "action": result.action}


def _check_entry_compat(entry: catalog.ModelEntry, info_cache: dict[str, engines.EngineInfo]) -> engines.CompatResult:
    if entry.engine not in _KNOWN_ENGINES:
        return engines.CompatResult(engines.Compat.UNKNOWN, f"engine '{entry.engine}' ไม่รู้จัก", None)
    if entry.engine not in info_cache:
        info_cache[entry.engine] = engines.detect(entry.engine)
    return engines.check_arch(entry.arch, entry.engine, info=info_cache[entry.engine], requires=entry.requires)


def _job_to_api(job: downloads.DownloadJob) -> dict[str, Any]:
    return {
        "id": job.id,
        "model_id": job.model_id,
        "state": job.state.value,
        "percent": round(job.percent, 1),
        "done_bytes": job.done_bytes,
        "total_bytes": job.total_bytes,
        "speed_bps": job.speed_bps,
        "eta_seconds": job.eta_seconds,
        "files": [
            {
                "dest": f.dest, "state": f.state.value,
                "done_bytes": f.done_bytes, "total_bytes": f.total_bytes, "error": f.error,
            }
            for f in job.files
        ],
    }


def _download_targets(entry: catalog.ModelEntry) -> list[tuple[str, str]]:
    """จับคู่ entry.dl (URL) กับ path ปลายทางบนดิสก์ — reuse ตรรกะ path เดียวกับที่ catalog ใช้ตัดสิน is_ready/disk_bytes"""
    urls = entry.dl or []
    if catalog._is_folder_entry(entry):
        directory = catalog.expand(entry)
        return [(url, os.path.join(directory, url.rsplit("/", 1)[-1])) for url in urls]

    dests = catalog.shard_paths(entry) + catalog._extra_gguf_from_args(entry.args)
    return list(zip(urls, dests))


# ---------------------------------------------------------------------------
# /api/health
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "version": APP_VERSION,
        "port": DEFAULT_PORT,
        "ram_total_gb": software.mem_total_gb(),
        "disk_free_gb": software.disk_free_gb(os.path.expanduser("~")),
    }


# ---------------------------------------------------------------------------
# /api/models
# ---------------------------------------------------------------------------


@app.get("/api/models")
def list_models() -> dict[str, Any]:
    entries = catalog.load_all()
    info_cache: dict[str, engines.EngineInfo] = {}

    models = []
    for e in entries:
        compat = _check_entry_compat(e, info_cache)
        models.append({
            "id": e.id, "name": e.name, "engine": e.engine, "path": e.path,
            "arch": e.arch, "source": e.source, "source_repo": e.source_repo,
            "ctx": e.ctx, "ctx_train": e.ctx_train, "need_gb": e.need_gb, "note": e.note,
            "file": os.path.basename(catalog.expand(e)),
            "ready": catalog.is_ready(e),
            "size_gb": _gb(catalog.disk_bytes(e)),
            "has_dl": bool(e.dl),
            "compat": _compat_dict(compat),
        })
    return {"models": models}


class ResolveReq(BaseModel):
    repo_id: str
    token: str | None = None


@app.post("/api/models/resolve")
def resolve_model(req: ResolveReq) -> dict[str, Any]:
    try:
        repo_json = hf.fetch_repo(req.repo_id, token=req.token)
    except hf.GatedRepoError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    except hf.RepoNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    files = hf.list_files(repo_json)
    groups = hf.group_quants(files)

    ram_gb = software.mem_total_gb()
    llamacpp_info = engines.detect("llamacpp")

    # arch เป็นสมบัติของโมเดล ไม่ใช่ของ quant — ดึง GGUF header ครั้งเดียวจากไฟล์แรกของ quant เล็กสุด
    arch: str | None = None
    ctx_train: int | None = None
    if groups:
        smallest = min(groups, key=lambda g: g.total_bytes)
        url = hf.resolve_url(req.repo_id, smallest.files[0].path)
        try:
            gguf_info, _total = gguf.fetch_header(url)
            arch = gguf_info.arch
            ctx_train = gguf_info.context_length
        except Exception:
            arch, ctx_train = None, None

    compat = engines.check_arch(arch, "llamacpp", info=llamacpp_info)
    compat_out = _compat_dict(compat)

    quants = []
    for g in groups:
        total_gb = _gb(g.total_bytes)
        fits_ram = None if ram_gb is None else (total_gb * 1.05 + 3 <= ram_gb)
        quants.append({
            "key": g.key,
            "total_bytes": g.total_bytes,
            "total_gb": total_gb,
            "shard_count": g.shard_count,
            "files": [{"path": f.path, "size": f.size, "sha256": f.sha256} for f in g.files],
            "fits_ram": fits_ram,
            "compat": compat_out,
        })

    companions = [{"path": c.path, "size_gb": _gb(c.size)} for c in groups[0].companions] if groups else []

    return {
        "repo_id": req.repo_id,
        "gated": False,
        "arch": arch,
        "ctx_train": ctx_train,
        "quants": quants,
        "companions": companions,
    }


@app.post("/api/models")
def add_model(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        entry = catalog.add_user_model(payload)
    except Exception as e:  # pydantic ValidationError จาก ModelEntry — field ผิด/ขาด
        raise HTTPException(status_code=400, detail=f"ข้อมูลโมเดลไม่ถูกต้อง: {e}") from e
    return {"ok": True, "model": entry.model_dump()}


@app.delete("/api/models/{model_id}")
def delete_model(model_id: str) -> dict[str, Any]:
    if not catalog.remove_user_model(model_id):
        raise HTTPException(status_code=404, detail=f"ไม่พบโมเดล id={model_id} ใน user-models (ลบได้เฉพาะโมเดลที่ผู้ใช้เพิ่มเอง)")
    return {"ok": True}


# ---------------------------------------------------------------------------
# /api/downloads
# ---------------------------------------------------------------------------


class DownloadReq(BaseModel):
    model_id: str


@app.post("/api/downloads")
def create_download(req: DownloadReq) -> dict[str, Any]:
    entries = {e.id: e for e in catalog.load_all()}
    entry = entries.get(req.model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"ไม่พบโมเดล id={req.model_id}")
    if not entry.dl:
        raise HTTPException(status_code=400, detail=f"โมเดล {entry.id} ไม่มีลิงก์ดาวน์โหลดอัตโนมัติ (dl)")

    targets = _download_targets(entry)
    job = get_download_manager().submit(entry.id, targets)
    return {"ok": True, "job": _job_to_api(job)}


@app.get("/api/downloads")
def list_downloads() -> dict[str, Any]:
    mgr = get_download_manager()
    try:
        mgr.refresh()
    except Exception:
        pass  # aria2 ต่อไม่ได้ — ตอบ jobs เท่าที่มีอยู่ ไม่ทำให้ endpoint พัง
    return {"jobs": [_job_to_api(j) for j in mgr.jobs()]}


@app.post("/api/downloads/{job_id}/{action}")
def control_download(job_id: str, action: str) -> dict[str, Any]:
    mgr = get_download_manager()
    handlers = {"pause": mgr.pause, "resume": mgr.resume, "cancel": mgr.cancel}
    handler = handlers.get(action)
    if handler is None:
        raise HTTPException(status_code=404, detail=f"ไม่รู้จัก action: {action} (ต้องเป็น pause/resume/cancel)")
    return {"ok": handler(job_id)}


# ---------------------------------------------------------------------------
# /api/engines
# ---------------------------------------------------------------------------


@app.get("/api/engines")
def list_engines() -> dict[str, Any]:
    out = []
    for info in engines.detect_all():
        out.append({
            "name": info.name, "installed": info.installed, "version": info.version,
            "build": info.build, "archs": sorted(info.archs), "detail": info.detail,
        })
    return {"engines": out}


class UpgradeReq(BaseModel):
    engine: str


@app.post("/api/engines/upgrade")
def upgrade_engine(req: UpgradeReq) -> dict[str, Any]:
    if req.engine != "llamacpp":
        raise HTTPException(status_code=400, detail=f"เฟส 1 อัปอัตโนมัติได้เฉพาะ llamacpp (ขอมา: {req.engine})")
    result = software.upgrade_llamacpp()
    return {"ok": result.ok, "message": result.message, "old_build": result.old_build, "new_build": result.new_build}


# ---------------------------------------------------------------------------
# /api/software
# ---------------------------------------------------------------------------


@app.get("/api/software")
def list_software() -> dict[str, Any]:
    items = software.check_all()
    return {"software": [
        {"name": i.name, "installed": i.installed, "version": i.version, "tier": i.tier, "detail": i.detail}
        for i in items
    ]}


# ---------------------------------------------------------------------------
# /api/activate
# ---------------------------------------------------------------------------


class ActivateReq(BaseModel):
    id: str
    port: int = 8001
    ctx: int | None = None
    allow_main_port: bool = False


def _read_tail(text: str, n: int = 2000) -> str:
    return text[-n:] if text else text


def _activate_log_path(engine_name: str, port: int) -> str | None:
    if engine_name == "llamacpp":
        return os.path.join(paths.log_dir(), f"llamacpp-{port}.log")
    if engine_name == "vllm":
        return os.path.join(paths.log_dir(), "vllm.log")
    return None


_RAM_SAFETY_GB = 8  # กันชนให้ OS + buffer (ค่าเดียวกับ SAFETY_GB ของ v1)


def _ram_hogs() -> str:
    """engine ที่กำลังรันอยู่ — เอาไปบอกผู้ใช้ว่าต้องหยุดอะไรก่อน

    ⚠️ ห้ามกรองด้วย RSS: llama.cpp โหลดโมเดลแบบ mmap ⇒ RSS โชว์แค่ ~3GB
    ทั้งที่กินแรมจริง 95GB (เจอกับตัวจริงบน GB10) — มี engine รันอยู่ = ผู้ต้องสงสัยเสมอ
    """
    try:
        out = subprocess.run(
            ["ps", "-eo", "args"], capture_output=True, text=True, timeout=10
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""
    hogs = []
    for line in out.splitlines():
        if "llama-server" not in line and "ds4-server" not in line:
            continue
        if line.startswith("ps ") or " grep " in line:
            continue
        port = re.search(r"--port\s+(\d+)", line)
        model = re.search(r"-m\s+(\S+)", line)
        name = os.path.basename(model.group(1)) if model else "โมเดล"
        hogs.append(f"{name} บนพอร์ต :{port.group(1) if port else '?'}")
    if not hogs:
        return ""
    return " และ ".join(hogs)


@app.post("/api/activate")
def activate(req: ActivateReq) -> dict[str, Any]:
    if req.port == _MAIN_LLM_PORT and not req.allow_main_port:
        raise HTTPException(
            status_code=409,
            detail=(
                "พอร์ต 8000 คือ LLM หลักที่ thClaws และ client ในเครื่องผูกอยู่เสมอ (กติกาเหล็ก) — "
                "ถ้าตั้งใจสลับโมเดลหลักจริง ๆ ให้ส่ง allow_main_port เป็น true"
            ),
        )

    entries = {e.id: e for e in catalog.load_all()}
    entry = entries.get(req.id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"ไม่พบโมเดล id={req.id}")

    if not catalog.is_ready(entry):
        raise HTTPException(status_code=409, detail=f"โมเดล {entry.id} ยังดาวน์โหลดไม่ครบ — ดาวน์โหลดให้เสร็จก่อนโหลดขึ้นแรม")

    info: engines.EngineInfo | None = None
    if entry.engine in _KNOWN_ENGINES:
        info = engines.detect(entry.engine)
        compat = engines.check_arch(entry.arch, entry.engine, info=info, requires=entry.requires)
    else:
        compat = engines.CompatResult(engines.Compat.UNKNOWN, f"engine '{entry.engine}' ไม่รู้จัก", None)

    if compat.status == engines.Compat.NEEDS_UPGRADE:
        raise HTTPException(status_code=409, detail=compat.reason)

    # ด่านแรม: โมเดลใหญ่กว่าแรมที่เหลือ = ตายตอนโหลดแน่นอน บอกก่อนดีกว่าปล่อยให้ OOM
    # (กันชน 8GB สำหรับ OS + buffer — ค่าเดียวกับ SAFETY_GB ของ v1)
    avail = software.mem_available_gb()
    if avail is not None:
        want = catalog.need_gb(entry, req.ctx)
        if want > avail - _RAM_SAFETY_GB:
            busy = _ram_hogs()
            raise HTTPException(
                status_code=409,
                detail=(
                    f"แรมไม่พอ — {entry.id} ต้องการราว {want:.0f} GB แต่ว่างอยู่ {avail:.0f} GB "
                    f"(กันไว้ให้ระบบ {_RAM_SAFETY_GB} GB)"
                    + (f" · ตอนนี้ {busy} ถือแรมอยู่ — หยุดตัวนั้นก่อนแล้วลองใหม่" if busy else "")
                ),
            )

    script = paths.repo("engines", f"{entry.engine}.sh")
    if not os.path.isfile(script):
        raise HTTPException(status_code=400, detail=f"ไม่พบสคริปต์ engine: {script}")

    env = dict(os.environ)
    env["PORT"] = str(req.port)
    if req.ctx is not None:
        env["CTX"] = str(req.ctx)
    env["MODEL_ID"] = entry.id

    # ~ ใน args ต้องขยายเอง — ส่งเป็น argv ตรง ๆ ไม่ผ่าน shell จึงไม่มีใครขยายให้
    # (เจอจริง: -md ~/models/.../mtp-xxx.gguf → llama-server หา draft model ไม่เจอ)
    args_list = [catalog.expand(entry)] + [
        os.path.expanduser(tok) for tok in shlex.split(entry.args or "")
    ]

    try:
        proc = subprocess.run(
            [script, *args_list], env=env, capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail=f"รัน {entry.engine}.sh เกินเวลา (600 วินาที) — เช็ค log เอง") from None

    proc_output = (proc.stdout or "") + (proc.stderr or "")
    log_path = _activate_log_path(entry.engine, req.port)
    log_tail = _read_tail(_read_log_file(log_path) or proc_output)

    success = proc.returncode == 0 and "READY" in proc_output
    if not success:
        unsupported_arch = engines.learn_from_log(log_tail + "\n" + proc_output)
        if unsupported_arch and entry.engine == "llamacpp" and info is not None:
            engines.record_unsupported(unsupported_arch, info.build)
        raise HTTPException(
            status_code=500,
            detail=f"โหลด {entry.id} ขึ้นแรมไม่สำเร็จ (พอร์ต {req.port}): {log_tail[-500:] or proc_output[-500:] or 'ไม่มี log'}",
        )

    return {"ok": True, "port": req.port, "log": log_tail}


def _read_log_file(log_path: str | None) -> str | None:
    if not log_path or not os.path.isfile(log_path):
        return None
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


# ---------------------------------------------------------------------------
# static UI
# ---------------------------------------------------------------------------


@app.get("/")
def index() -> FileResponse:
    return FileResponse(paths.repo("server", "static", "index.html"))
