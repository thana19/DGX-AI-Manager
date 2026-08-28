"""server/software.py — เช็คสถานะ software พื้นฐาน + อัป/rollback llama.cpp engine

SOFTWARE list และคำสั่งเช็คเวอร์ชันยกมาจาก v1 (`scripts/install-tools.sh.v1-reference`
และหน้า /api/software เดิม) — คำสั่งพวกนี้ผ่านสนามจริงมาแล้ว ไม่เขียนใหม่

license_code()/machine_id() ต้องได้สูตรเดียวกับที่ v1 ใช้ตอนออก license/ยิง dl.php
(sha256 ของ machine-id ต่อด้วย GPU uuid เอา 16 ตัวแรก) ไม่งั้น server ฝั่ง dl.php
จะไม่รู้จัก mid นี้และปฏิเสธการดาวน์โหลด
"""
from __future__ import annotations

import contextlib
import glob
import hashlib
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
from dataclasses import dataclass

import httpx

from . import engines, paths

# (ชื่อ, คำสั่งเช็คเวอร์ชัน, tier) — รันด้วย `bash -lc "set -o pipefail; <cmd>"` เสมอ
# (บทเรียนจาก v1: ไม่ใส่ pipefail แล้ว "| head" จะกลบ exit code ของตัวจริง)
SOFTWARE: list[tuple[str, str, str]] = [
    ("git", "git --version", "base"),
    ("docker", "docker --version", "base"),
    ("nvcc", "nvcc --version | tail -1", "base"),
    ("python3", "python3 --version", "base"),
    ("pip3", "pip3 --version", "base"),
    ("uv", "uv --version", "base"),
    ("curl", "curl --version | head -1", "base"),
    ("ffmpeg", "ffmpeg -version | head -1", "base"),
    ("jq", "jq --version", "base"),
    ("tmux", "tmux -V", "base"),
    ("aria2c", "aria2c --version | head -1", "base"),
    ("ollama", "ollama --version", "engine"),
    (
        "llama-server",
        "LD_LIBRARY_PATH=~/llama.cpp/build/bin ~/llama.cpp/build/bin/llama-server --version 2>&1 | head -1",
        "engine",
    ),
    ("vllm (docker image)", "docker images --format '{{.Repository}}:{{.Tag}}' | grep vllm | head -1", "engine"),
    ("litellm (gateway)", "test -x ~/aiserver/venv/bin/litellm -o -x ~/litellm-env/bin/litellm && echo 'พร้อมใช้'", "engine"),
    ("thclaws", "thclaws --version", "app"),
]

_UPGRADE_BASE_URL = "https://aiserver.in.th/dl.php"


@dataclass
class SoftwareItem:
    """สถานะของ software 1 ตัวที่เช็คจากเครื่องจริง"""

    name: str
    installed: bool
    version: str | None
    tier: str
    detail: str


def check_all() -> list[SoftwareItem]:
    """รันคำสั่งเช็คเวอร์ชันของ software ทุกตัวใน SOFTWARE ทีละตัว (timeout 15 วิ/ตัว)"""
    items: list[SoftwareItem] = []
    for name, cmd, tier in SOFTWARE:
        try:
            result = subprocess.run(
                ["bash", "-lc", f"set -o pipefail; {cmd}"],
                capture_output=True, text=True, timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            items.append(
                SoftwareItem(name=name, installed=False, version=None, tier=tier, detail=f"รันคำสั่งเช็คไม่สำเร็จ: {e}")
            )
            continue

        lines = (result.stdout or result.stderr or "").strip().splitlines()
        ok = result.returncode == 0 and bool(lines)
        version = lines[0][:200] if ok else None
        detail = version if ok else "ไม่พบ หรือเช็คเวอร์ชันไม่ผ่าน"
        items.append(SoftwareItem(name=name, installed=ok, version=version, tier=tier, detail=detail))
    return items


def _meminfo_gb(key: str) -> float | None:
    """อ่านค่าจาก /proc/meminfo (KiB) → GB — ไม่มี /proc/meminfo (เช่นบน Mac) คืน None"""
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith(key):
                    return round(int(line.split()[1]) / 1048576, 1)
    except OSError:
        return None
    return None


def mem_available_gb() -> float | None:
    """แรมที่ขอใช้ได้จริงตอนนี้ — MemAvailable ไม่ใช่ MemFree (buff/cache คืนได้ นับรวมด้วย)"""
    return _meminfo_gb("MemAvailable")


def mem_total_gb() -> float | None:
    """อ่าน MemTotal จาก /proc/meminfo (KiB) → GB — ไม่มี /proc/meminfo (เช่นบน Mac) คืน None"""
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    return round(kb / 1048576, 1)
    except OSError:
        return None
    return None


def disk_free_gb(path: str) -> float | None:
    """เนื้อที่ว่างของ path ที่ระบุ — คืน None ถ้าอ่านไม่ได้ (path ไม่มีอยู่จริง ฯลฯ)"""
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        return None
    return round(usage.free / (1024**3), 1)


# ---------------------------------------------------------------------------
# license / machine id — ต้องตรงสูตรกับ v1 เป๊ะ (ดู docstring หัวไฟล์)
# ---------------------------------------------------------------------------


def license_code() -> str | None:
    """อ่าน ~/.aiserver2/license ก่อน ไม่มีค่อย ~/.aiserver/license (ของ v1 — เครื่องอัปเกรดจาก v1 ไม่ต้องขอ license ใหม่)"""
    for p in (paths.state("license"), os.path.expanduser("~/.aiserver/license")):
        try:
            with open(p, encoding="utf-8") as f:
                code = f.read().strip()
        except OSError:
            continue
        if code:
            return code
    return None


def machine_id() -> str:
    """sha256(machine-id ต่อด้วย GPU uuid) เอา 16 ตัวแรก — สูตรเดียวกับ install.sh/hub v1"""
    try:
        with open("/etc/machine-id", encoding="utf-8") as f:
            mid = f.read().rstrip("\n")
    except OSError:
        mid = ""

    gpu_uuid = ""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        lines = result.stdout.strip().splitlines()
        if lines:
            gpu_uuid = lines[0]
    except (OSError, subprocess.TimeoutExpired):
        pass

    return hashlib.sha256((mid + gpu_uuid).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# อัป/rollback llama.cpp engine
# ---------------------------------------------------------------------------


@dataclass
class UpgradeResult:
    ok: bool
    message: str
    old_build: int | None = None
    new_build: int | None = None
    backup_dir: str | None = None


def _is_dev_checkout(bin_dir: str) -> bool:
    """bin_dir ปกติคือ <root>/build/bin — <bin_dir>/../../.git มีอยู่ = เครื่องนี้ build llama.cpp เอง (v1 กันไว้แบบเดียวกัน)"""
    root = os.path.dirname(os.path.dirname(os.path.normpath(bin_dir)))
    return os.path.isdir(os.path.join(root, ".git"))


@contextlib.contextmanager
def _bin_dir_override(bin_dir: str):
    """ชี้ env LLAMA_BIN_DIR ไปที่ bin_dir ระหว่าง block นี้ชั่วคราว แล้วคืนค่าเดิม

    ใช้เพื่อ reuse engines.detect("llamacpp") ตรง ๆ (ไม่ต้อง copy ตรรกะตรวจ build/arch ซ้ำ)
    กับ bin_dir ที่ผู้เรียก upgrade_llamacpp/rollback_llamacpp ระบุมา ซึ่งอาจไม่ใช่ค่า default ของเครื่อง
    (เช่นตอน test ที่ชี้ไป tmp_path)
    """
    old = os.environ.get("LLAMA_BIN_DIR")
    os.environ["LLAMA_BIN_DIR"] = bin_dir
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("LLAMA_BIN_DIR", None)
        else:
            os.environ["LLAMA_BIN_DIR"] = old


def _detect_llamacpp_at(bin_dir: str) -> engines.EngineInfo:
    with _bin_dir_override(bin_dir):
        return engines.detect("llamacpp")


def _find_latest_backup(bin_dir: str) -> str | None:
    candidates = sorted(glob.glob(f"{bin_dir}.bak-*"))
    return candidates[-1] if candidates else None


def _prune_old_backups(bin_dir: str, keep: str) -> None:
    """เก็บแค่ backup ล่าสุด (keep) — ลบตัวเก่ากว่านั้นทิ้ง"""
    for old in glob.glob(f"{bin_dir}.bak-*"):
        if old != keep and os.path.isdir(old):
            shutil.rmtree(old, ignore_errors=True)


def upgrade_llamacpp(
    *,
    bin_dir: str | None = None,
    base_url: str = _UPGRADE_BASE_URL,
    client: httpx.Client | None = None,
) -> UpgradeResult:
    """ดาวน์โหลด+แตก llama-pack-gb10 ทับ bin_dir — backup ของเดิมไว้ก่อนเสมอ

    ล้มเหลวตรงไหนก็ตาม (ดาวน์โหลด/แตกไฟล์/รันไม่ได้) → ไม่ทิ้ง engine เดิมให้พัง:
    รันไม่ได้หลังแตกไฟล์ = rollback อัตโนมัติทันที (ดู docstring โมดูล/task)
    """
    bin_dir = os.path.expanduser(bin_dir) if bin_dir else engines.llamacpp_bin_dir()

    code = license_code()
    if not code:
        return UpgradeResult(
            ok=False,
            message="ไม่พบ license (~/.aiserver2/license หรือ ~/.aiserver/license) — ติดตั้ง AI Server ให้เสร็จก่อนจึงอัป engine ได้",
        )

    if _is_dev_checkout(bin_dir):
        return UpgradeResult(
            ok=False,
            message="เครื่องนี้ build llama.cpp เองจาก source (พบ .git) — ไม่อัปทับให้ กันทับ build ที่ทำเอง",
        )

    old_info = _detect_llamacpp_at(bin_dir)
    old_build = old_info.build

    mid = machine_id()
    url = f"{base_url}?f=llama-pack-gb10.tar.gz&code={code}&mid={mid}"

    own_client = client is None
    http_client = client or httpx.Client()
    try:
        resp = http_client.get(url, follow_redirects=True)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        return UpgradeResult(ok=False, message=f"ดาวน์โหลด engine ใหม่ไม่สำเร็จ: {e}", old_build=old_build)
    finally:
        if own_client:
            http_client.close()

    fd, tmp_path = tempfile.mkstemp(suffix=".tar.gz")
    backup_dir: str | None = None
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(resp.content)

        if os.path.isdir(bin_dir):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            backup_dir = f"{bin_dir}.bak-{timestamp}"
            shutil.copytree(bin_dir, backup_dir)

        os.makedirs(bin_dir, exist_ok=True)
        try:
            with tarfile.open(tmp_path, "r:gz") as tar:
                tar.extractall(bin_dir, filter="data")
        except (tarfile.TarError, OSError) as e:
            if backup_dir:
                rollback_llamacpp(bin_dir=bin_dir)
            return UpgradeResult(
                ok=False, message=f"แตกไฟล์ pack ไม่สำเร็จ: {e}", old_build=old_build, backup_dir=backup_dir,
            )

        new_info = _detect_llamacpp_at(bin_dir)
        if not new_info.installed:
            if backup_dir:
                rollback_llamacpp(bin_dir=bin_dir)
                message = f"engine ใหม่รันไม่ได้บนเครื่องนี้ ({new_info.detail}) — ย้อนกลับเป็นตัวเดิมให้แล้ว"
            else:
                message = f"engine ใหม่รันไม่ได้บนเครื่องนี้ ({new_info.detail})"
            return UpgradeResult(ok=False, message=message, old_build=old_build, backup_dir=backup_dir)

        if backup_dir:
            _prune_old_backups(bin_dir, keep=backup_dir)

        return UpgradeResult(
            ok=True,
            message=f"อัป llama-server สำเร็จ: build {old_build} → {new_info.build}",
            old_build=old_build, new_build=new_info.build, backup_dir=backup_dir,
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def rollback_llamacpp(*, bin_dir: str | None = None) -> UpgradeResult:
    """เอา backup ล่าสุดกลับมาแทน bin_dir — ใช้เองได้ (ปุ่ม rollback) หรือถูกเรียกอัตโนมัติจาก upgrade_llamacpp"""
    bin_dir = os.path.expanduser(bin_dir) if bin_dir else engines.llamacpp_bin_dir()

    backup_dir = _find_latest_backup(bin_dir)
    if backup_dir is None:
        return UpgradeResult(ok=False, message=f"ไม่พบ backup ให้ย้อนกลับ ({bin_dir}.bak-*)")

    broken_info = _detect_llamacpp_at(bin_dir)

    if os.path.isdir(bin_dir):
        shutil.rmtree(bin_dir)
    shutil.move(backup_dir, bin_dir)

    # backup อื่นที่เก่ากว่า (ถ้ามี) ก็ไม่มีความหมายแล้วหลัง rollback — เก็บของที่เพิ่งย้อนกลับไว้แทน
    restored_info = _detect_llamacpp_at(bin_dir)
    if not restored_info.installed:
        return UpgradeResult(
            ok=False,
            message=f"ย้อนกลับแล้วแต่ยังรันไม่ได้ ({restored_info.detail}) — เครื่องนี้อาจต้องตรวจสอบเอง",
            old_build=broken_info.build,
        )

    return UpgradeResult(
        ok=True,
        message=f"ย้อนกลับ llama-server สำเร็จ (build {restored_info.build})",
        old_build=broken_info.build, new_build=restored_info.build,
    )
