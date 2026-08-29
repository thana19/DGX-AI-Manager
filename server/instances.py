"""server/instances.py — ดู instance ที่รันอยู่จริง (llama-server / ds4-server / vLLM container) และหยุดมันได้

instance = engine ที่กำลังรันอยู่จริงบนพอร์ตหนึ่ง (ดูคำศัพท์ใน CONTEXT.md — คนละเรื่องกับ server/engines.py
ที่ตอบว่า "engine ที่ติดตั้งในเครื่องรองรับ arch อะไรบ้าง")

⚠️ ห้ามใช้ RSS ตัดสินอะไร: llama.cpp โหลดโมเดลแบบ mmap ⇒ RSS ที่ ps รายงานเป็นแค่ไม่กี่ GB
ทั้งที่กินแรมจริงหลักสิบ GB (เจอกับตัวจริงบน GB10) — rss_gb ในนี้มีไว้ "แสดงเฉย ๆ" ไม่ใช่ตัวตัดสินว่าโหลดสำเร็จ/ล้มเหลว
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, replace

import httpx

# แยก --port / -m / -c ออกจาก args ของ ps — ตัวอย่างจริงที่ต้องแยกได้ถูก (บันทึกไว้ใน task ที่สั่งงานไฟล์นี้):
# "... /home/dgx/llama.cpp/build/bin/llama-server -m /home/dgx/models/gguf/X/Y-Q8_K_P.gguf
#      --host 0.0.0.0 --port 8001 -c 262144 -ngl 99 --jinja --reasoning-format auto --metrics"
_PORT_RE = re.compile(r"--port\s+(\d+)")
_MODEL_RE = re.compile(r"-m\s+(\S+)")
_CTX_RE = re.compile(r"-c\s+(\d+)")

_VLLM_PORT = 8000  # vLLM ผูก :8000 เท่านั้น (ดู engines/vllm.sh)
_VLLM_CONTAINER = "aiserver-vllm"

_STOP_POLL_INTERVAL_SEC = 1
_STOP_POLL_TIMEOUT_SEC = 15


@dataclass
class Instance:
    port: int
    engine: str  # "llamacpp" | "vllm" | "ds4"
    pid: int | None  # vLLM = None (อยู่ใน container คนละ PID namespace)
    model_file: str  # basename ของไฟล์/โฟลเดอร์โมเดล
    model_path: str  # path เต็มจาก -m
    ctx: int | None  # จาก -c
    rss_gb: float  # RSS ที่ ps บอก — มีไว้แสดงเฉย ๆ (ดู warning หัวไฟล์)
    up: bool  # /v1/models ตอบไหม


# ---------------------------------------------------------------------------
# helper รัน process จริง — แยกไว้เป็นฟังก์ชันเล็ก ๆ เพื่อให้ test monkeypatch ทีละตัวได้
# ---------------------------------------------------------------------------


def _ps_output() -> str:
    try:
        return subprocess.run(
            ["ps", "-eo", "pid,rss,args"], capture_output=True, text=True, timeout=10,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _docker_ps_names() -> set[str]:
    """ชื่อ container ที่กำลังรันอยู่ — คืนเซตว่างถ้าไม่มี docker หรือรันไม่สำเร็จ"""
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return set()
    if result.returncode != 0:
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _vllm_model_info() -> tuple[str, str]:
    """เดา model path ของ container vLLM จาก docker inspect (best effort — อ่านไม่ได้ก็คืนค่าว่าง)

    engines/vllm.sh สั่ง: docker run ... "$IMAGE" vllm serve "/models/<model-dir>" ... ⇒ token หลัง "serve" คือ path
    """
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{json .Config.Cmd}}", _VLLM_CONTAINER],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "", ""
    if result.returncode != 0:
        return "", ""
    try:
        cmd = json.loads(result.stdout.strip())
    except (ValueError, TypeError):
        return "", ""
    for i, tok in enumerate(cmd or []):
        if tok == "serve" and i + 1 < len(cmd):
            path = cmd[i + 1]
            return path, os.path.basename(path)
    return "", ""


def _probe_up(port: int) -> bool:
    try:
        r = httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


def _docker_rm_vllm() -> None:
    try:
        subprocess.run(
            ["docker", "rm", "-f", _VLLM_CONTAINER], capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass


def _fuser_kill(port: int) -> None:
    try:
        subprocess.run(
            ["fuser", "-k", f"{port}/tcp"], capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------


def _parse_ps_line(line: str) -> Instance | None:
    line = line.strip()
    if not line:
        return None
    parts = line.split(None, 2)
    if len(parts) < 3:
        return None
    pid_str, rss_str, rest = parts
    if not pid_str.isdigit() or not rss_str.isdigit():
        return None  # แถวหัวตาราง ("PID RSS COMMAND") หรือแถวแปลก ๆ

    if "llama-server" in rest:
        engine = "llamacpp"
    elif "ds4-server" in rest:
        engine = "ds4"
    else:
        return None

    port_m = _PORT_RE.search(rest)
    if port_m is None:
        return None  # ไม่มี --port แปลว่าไม่ได้เสิร์ฟ HTTP บนพอร์ตไหน — ข้ามไป (เลือกตามที่ task อนุญาต)
    port = int(port_m.group(1))

    model_m = _MODEL_RE.search(rest)
    model_path = model_m.group(1) if model_m else ""
    model_file = os.path.basename(model_path) if model_path else ""

    ctx_m = _CTX_RE.search(rest)
    ctx = int(ctx_m.group(1)) if ctx_m else None

    rss_gb = round(int(rss_str) / 1048576, 2)  # ps -eo rss = KiB

    return Instance(
        port=port, engine=engine, pid=int(pid_str),
        model_file=model_file, model_path=model_path,
        ctx=ctx, rss_gb=rss_gb, up=False,
    )


def scan(*, ps_output: str | None = None, probe: bool = True) -> list[Instance]:
    """สแกน process llama-server / ds4-server + container vLLM

    ps_output ส่งเข้ามาได้เพื่อให้ test ไม่ต้องพึ่งเครื่องจริง — ไม่ส่งมาจะรัน `ps` จริง
    probe=True → ยิง GET /v1/models ของทุกพอร์ตที่เจอ เพื่อเช็คว่า up จริงไหม (timeout 2 วิ/พอร์ต)
    """
    text = ps_output if ps_output is not None else _ps_output()

    found: list[Instance] = []
    for line in text.splitlines():
        inst = _parse_ps_line(line)
        if inst is not None:
            found.append(inst)

    if _VLLM_CONTAINER in _docker_ps_names():
        model_path, model_file = _vllm_model_info()
        found.append(Instance(
            port=_VLLM_PORT, engine="vllm", pid=None,
            model_file=model_file, model_path=model_path,
            ctx=None, rss_gb=0.0, up=False,
        ))

    if probe:
        found = [replace(inst, up=_probe_up(inst.port)) for inst in found]

    return found


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------


def stop(port: int) -> tuple[bool, str]:
    """หยุด instance บนพอร์ตนั้น — คืน (สำเร็จไหม, ข้อความไทย)

    พอร์ต 8000 → ลอง `docker rm -f aiserver-vllm` ก่อนเสมอ (แบบเดียวกับ stop_engine_port ใน engines/common.sh
    — ปลอดภัยแม้ไม่มี container จริง เพราะแค่ไม่มีอะไรให้ลบ) จากนั้น `fuser -k <port>/tcp` แล้วรอพอร์ตว่างจริง
    """
    current = {i.port: i for i in scan(probe=False)}
    inst = current.get(port)
    if inst is None:
        return False, f"ไม่มี instance บนพอร์ต {port} อยู่แล้ว"

    if port == _VLLM_PORT:
        _docker_rm_vllm()
    _fuser_kill(port)

    deadline = time.monotonic() + _STOP_POLL_TIMEOUT_SEC
    while time.monotonic() < deadline:
        if not any(i.port == port for i in scan(probe=False)):
            label = inst.model_file or inst.engine
            return True, f"หยุด {label} บนพอร์ต {port} แล้ว"
        time.sleep(_STOP_POLL_INTERVAL_SEC)

    return False, f"สั่งหยุดพอร์ต {port} แล้ว แต่รอ {_STOP_POLL_TIMEOUT_SEC} วินาทีพอร์ตยังไม่ว่าง — เช็คด้วยมือ"
