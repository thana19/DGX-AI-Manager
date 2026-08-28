"""server/engines.py — ตอบคำถามหัวใจของทั้งระบบ: "engine ในเครื่องนี้รันโมเดล arch นี้ได้ไหม"

แหล่งความจริงหลักคือ **ถาม engine ที่ติดตั้งอยู่จริง** ไม่ใช่ตารางที่คนจด
(บทเรียนจริง: `note:` ของ v1 จดว่า arch คือ `qwen4_exp` แต่ของจริงในไฟล์คือ `qwen4exp`
— ตารางที่คนดูแลผิดได้ ดู CONTEXT.md ข้อค้นพบข้อ 3-4)

`llama-server` เป็น wrapper บาง ๆ arch list ไม่ได้อยู่ในนั้น อยู่ใน `libllama.so`
เช็คได้ตรง ๆ ด้วย `strings libllama.so.x.y.z | grep -x <arch>`

`engine-compat.yaml` เป็นแค่ **ตัวสำรอง** ใช้ตอบสิ่งที่ arch list ตอบไม่ได้:
  1. engine ที่ไม่มี arch list ให้อ่าน (vllm ใน docker, ds4)
  2. "ถ้า engine ปัจจุบันไม่รองรับ แล้วต้องอัปเป็นตัวไหน" (upgrade_hint)
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum

import yaml

from server import paths

# token ที่หน้าตาเป็นชื่อ arch: ตัวพิมพ์เล็ก/เลข/-/_ ยาว 3-32 ตัว ไม่มีจุด ไม่มี /
_ARCH_TOKEN_RE = re.compile(r"^[a-z0-9_-]{3,32}$")

# ดึง run ของ ASCII ที่พิมพ์ได้ (0x20-0x7e) ยาวอย่างน้อย 3 ตัว — ใช้แทน `strings` เมื่อไม่มีในเครื่อง
_PRINTABLE_RUN_RE = re.compile(rb"[\x20-\x7e]{3,}")

# แยกเลข build จากข้อความ version ของ llama-server เช่น
# "version: 0.3.0-dev (build 10696, commit 1f0a36a35)"
_BUILD_RE = re.compile(r"build\s+(\d+)")

# แยก arch จาก error ของ llama-server เช่น
# "unknown model architecture: 'qwen4exp'" หรือ 'unknown model architecture: "glm5next"'
_UNKNOWN_ARCH_RE = re.compile(r"unknown model architecture:\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)


@dataclass
class EngineInfo:
    """สถานะของ engine ตัวหนึ่งที่ตรวจจากเครื่องจริง"""

    name: str  # "llamacpp" | "vllm" | "ds4"
    installed: bool
    version: str | None  # ข้อความเวอร์ชันดิบ
    build: int | None  # เลข build ของ llama-server (10696) — None ถ้าไม่ใช่ llamacpp
    archs: frozenset[str]  # arch ที่รองรับ (ว่างถ้าอ่านไม่ได้ เช่น vllm)
    detail: str  # ข้อความอธิบายสถานะให้ผู้ใช้อ่าน


class Compat(str, Enum):
    OK = "ok"  # รันได้เลย
    NEEDS_UPGRADE = "needs_upgrade"  # ต้องอัป engine ก่อน
    UNKNOWN = "unknown"  # ตอบไม่ได้ (อ่าน arch list ไม่ได้ / ไม่รู้ arch)
    NO_ENGINE = "no_engine"  # ยังไม่ได้ติดตั้ง engine นี้เลย


@dataclass
class CompatResult:
    status: Compat
    reason: str  # ภาษาไทย อธิบายให้ผู้ใช้เข้าใจว่าทำไม
    action: str | None  # สิ่งที่ระบบทำให้ได้ เช่น "upgrade_llamacpp" — None ถ้าไม่ต้องทำอะไร


# ---------------------------------------------------------------------------
# llama.cpp — หา binary/lib แล้วถามตรง ๆ ว่ารองรับ arch อะไรบ้าง
# ---------------------------------------------------------------------------


def llamacpp_bin_dir() -> str:
    """~/llama.cpp/build/bin — override ด้วย env LLAMA_BIN_DIR ตอน test/เครื่องอื่น"""
    return os.environ.get("LLAMA_BIN_DIR") or os.path.expanduser("~/llama.cpp/build/bin")


def _newest_libllama(bin_dir: str) -> str | None:
    """หา libllama.so* ที่ใหม่สุดใน bin_dir (มีหลายเวอร์ชันวางคู่กันได้ เช่น
    libllama.so, libllama.so.0, libllama.so.0.3.0) — ตัดสินด้วยเวลาแก้ไขไฟล์ล่าสุด
    เพราะชื่อไฟล์ไม่ได้เรียงเป็น semantic version ที่เทียบกันตรง ๆ ได้เสมอ
    """
    candidates = glob.glob(os.path.join(bin_dir, "libllama.so*"))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _extract_ascii_tokens(data: bytes) -> list[str]:
    """ดึง token ASCII ที่พิมพ์ได้ทั้งก้อน (fallback เมื่อไม่มีคำสั่ง `strings` ในเครื่อง)

    แยกด้วย byte ที่พิมพ์ไม่ได้ (null/control) เป็นตัวคั่น เหมือนพฤติกรรมของ `strings`
    """
    return [m.decode("ascii") for m in _PRINTABLE_RUN_RE.findall(data)]


def _run_strings(path: str) -> list[str] | None:
    """เรียกคำสั่ง `strings -n 3 <path>` ถ้ามีในเครื่อง — คืน None ถ้าไม่มี/รันไม่สำเร็จ"""
    if not shutil.which("strings"):
        return None
    try:
        result = subprocess.run(
            ["strings", "-n", "3", path], capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.splitlines()


def read_supported_archs(bin_dir: str) -> frozenset[str]:
    """ดึงรายชื่อ arch จาก libllama.so ที่ใหม่ที่สุดในโฟลเดอร์

    ใช้ `strings` ถ้ามี ไม่มีก็อ่านไฟล์เองแล้วดึง ASCII run (ไม่พึ่ง binary ภายนอกอย่างเดียว)
    กรองให้เหลือเฉพาะ token ที่หน้าตาเป็นชื่อ arch เท่านั้น — libllama.so มีสัญลักษณ์อื่น
    ปนอยู่มหาศาล (ชื่อฟังก์ชัน/lib อื่น ๆ) การกรองรูปแบบชื่อช่วยตัดขยะพวกนี้ออกไปมาก
    """
    lib_path = _newest_libllama(bin_dir)
    if lib_path is None:
        return frozenset()

    tokens = _run_strings(lib_path)
    if tokens is None:
        try:
            with open(lib_path, "rb") as f:
                data = f.read()
        except OSError:
            return frozenset()
        tokens = _extract_ascii_tokens(data)

    return frozenset(t for t in tokens if _ARCH_TOKEN_RE.match(t))


def _llama_server_bin(bin_dir: str) -> str | None:
    """หา path ของ llama-server ใน bin_dir — คืน None ถ้าไม่พบหรือรันไม่ได้"""
    candidate = os.path.join(bin_dir, "llama-server")
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    return None


def _run_llama_version(bin_path: str, bin_dir: str) -> str | None:
    """รัน `<bin_path> --version` — คืน None ถ้ารันไม่สำเร็จ

    binary ที่ก๊อปข้ามเครื่องมี RUNPATH ฝัง path เครื่องต้นทาง ต้องตั้ง LD_LIBRARY_PATH
    ไปที่โฟลเดอร์ของ binary เองก่อนเสมอ (บทเรียนเดียวกับที่ engines/llamacpp.sh ทำ)
    llama-server พิมพ์ข้อความ version ไป stderr ⇒ รวม stdout+stderr เข้าด้วยกัน
    """
    env = dict(os.environ)
    ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = bin_dir + (f":{ld}" if ld else "")
    try:
        result = subprocess.run(
            [bin_path, "--version"], capture_output=True, text=True, timeout=10, env=env,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (result.stdout or "") + (result.stderr or "")


def _parse_build(version_text: str) -> int | None:
    m = _BUILD_RE.search(version_text)
    return int(m.group(1)) if m else None


def _detect_llamacpp() -> EngineInfo:
    bin_dir = llamacpp_bin_dir()
    bin_path = _llama_server_bin(bin_dir)
    if bin_path is None:
        return EngineInfo(
            name="llamacpp", installed=False, version=None, build=None, archs=frozenset(),
            detail=f"ไม่พบ llama-server ใน {bin_dir}",
        )

    version_text = _run_llama_version(bin_path, bin_dir)
    if version_text is None:
        return EngineInfo(
            name="llamacpp", installed=False, version=None, build=None, archs=frozenset(),
            detail=f"รัน {bin_path} --version ไม่สำเร็จ",
        )

    build = _parse_build(version_text)
    archs = read_supported_archs(bin_dir)
    detail = f"llama-server build {build}" if build is not None else "llama-server (ไม่พบเลข build ใน --version)"
    if not archs:
        detail += " — อ่าน arch list จาก libllama.so ไม่ได้"

    return EngineInfo(
        name="llamacpp", installed=True, version=version_text.strip(), build=build,
        archs=archs, detail=detail,
    )


# ---------------------------------------------------------------------------
# vLLM — รันใน docker · ไม่มี arch list ให้อ่าน ต้องพึ่ง requires: ใน catalog
# ---------------------------------------------------------------------------


def _docker_images() -> list[str]:
    """คืนรายชื่อ image:tag ที่มีในเครื่อง — คืน [] ถ้าไม่มี docker หรือรันไม่สำเร็จ"""
    try:
        result = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _detect_vllm() -> EngineInfo:
    images = [img for img in _docker_images() if img.split(":", 1)[0] == "aiserver-vllm"]
    if not images:
        return EngineInfo(
            name="vllm", installed=False, version=None, build=None, archs=frozenset(),
            detail="ไม่พบ docker image aiserver-vllm บนเครื่องนี้",
        )
    version = images[0]
    return EngineInfo(
        name="vllm", installed=True, version=version, build=None, archs=frozenset(),
        detail=f"พบ image {version} — vLLM ไม่มี arch list ให้อ่าน ต้องพึ่ง requires: ใน catalog",
    )


# ---------------------------------------------------------------------------
# ds4 — เฟส 1 ยังไม่มี engines/ds4.sh ในโปรเจกต์ ตรวจแบบขั้นต่ำไว้ก่อน
# ---------------------------------------------------------------------------


def _detect_ds4() -> EngineInfo:
    bin_path = os.environ.get("DS4_BIN") or shutil.which("ds4")
    if not bin_path:
        return EngineInfo(
            name="ds4", installed=False, version=None, build=None, archs=frozenset(),
            detail="ไม่พบ ds4 บนเครื่องนี้ (เฟส 1 ยังไม่มีสคริปต์ engines/ds4.sh)",
        )
    return EngineInfo(
        name="ds4", installed=True, version=None, build=None, archs=frozenset(),
        detail=f"พบ {bin_path} — ds4 ไม่มี arch list ให้อ่าน ต้องพึ่ง requires: ใน catalog",
    )


def detect(name: str) -> EngineInfo:
    if name == "llamacpp":
        return _detect_llamacpp()
    if name == "vllm":
        return _detect_vllm()
    if name == "ds4":
        return _detect_ds4()
    raise ValueError(f"ไม่รู้จัก engine: {name}")


def detect_all() -> list[EngineInfo]:
    return [detect(name) for name in ("llamacpp", "vllm", "ds4")]


# ---------------------------------------------------------------------------
# engine-compat.yaml — ตัวสำรอง (ดู docstring หัวไฟล์)
# ---------------------------------------------------------------------------


def _load_compat_yaml(compat_path: str | None) -> dict:
    path = compat_path or paths.repo("engine-compat.yaml")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return {}


def _upgrade_hint(compat: dict, arch: str) -> str | None:
    hint_map = ((compat.get("llamacpp") or {}).get("upgrade_hint")) or {}
    return hint_map.get(arch) or None


# ---------------------------------------------------------------------------
# เรียนรู้จาก error ตอนรันจริง — ความจริงจากสนามชนะ arch list เสมอ
# ---------------------------------------------------------------------------


def learn_from_log(log_text: str) -> str | None:
    """ดึงชื่อ arch จาก error ของ llama-server เช่น
    "unknown model architecture: 'xyz'" → "xyz" — คืน None ถ้าไม่เจอ error แบบนี้
    """
    m = _UNKNOWN_ARCH_RE.search(log_text)
    return m.group(1) if m else None


def _compat_learned_path(path: str | None) -> str:
    return path or paths.state("compat-learned.json")


def _read_learned_file(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except (OSError, json.JSONDecodeError):
        return {}


def record_unsupported(arch: str, build: int | None, path: str | None = None) -> None:
    """จด arch ที่รันไม่ได้จริงลง ~/.aiserver2/compat-learned.json (atomic write)

    เก็บแยกตาม build เพราะ build ใหม่อาจรองรับ arch นี้แล้วก็ได้ — ไม่งั้นจดค้างไว้ผิด ๆ ตลอดไป
    """
    file_path = _compat_learned_path(path)
    data = _read_learned_file(file_path)
    key = str(build) if build is not None else "unknown"
    archs = set(data.get(key, []))
    archs.add(arch)
    data[key] = sorted(archs)

    dir_name = os.path.dirname(file_path) or "."
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, prefix=".compat-learned-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, file_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def learned_unsupported(build: int | None, path: str | None = None) -> set[str]:
    file_path = _compat_learned_path(path)
    data = _read_learned_file(file_path)
    key = str(build) if build is not None else "unknown"
    return set(data.get(key, []))


# ---------------------------------------------------------------------------
# check_arch — จุดตัดสินใจหลักที่ทุกส่วนของระบบเรียกใช้
# ---------------------------------------------------------------------------


def _as_build(value: object) -> int | None:
    """แปลงค่า llama_build จาก catalog เป็น int — รับได้ทั้ง 10353, `10353` และ `b10353`"""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    m = re.search(r"\d+", str(value))
    return int(m.group()) if m else None


def check_arch(
    arch: str | None,
    engine: str,
    *,
    info: EngineInfo | None = None,
    requires: dict | None = None,
    compat_path: str | None = None,
) -> CompatResult:
    """ตัดสินว่า engine ในเครื่องนี้รันโมเดล arch นี้ได้ไหม

    ลำดับกติกา (หยุดที่ข้อแรกที่ตรง):
      1. engine ไม่ได้ติดตั้ง → NO_ENGINE
      2. เคยลองรันจริงแล้วไม่ได้ (learned_unsupported) → NEEDS_UPGRADE (ความจริงจากสนามชนะ arch list เสมอ)
      3. requires.llama_build สูงกว่า build ปัจจุบัน → NEEDS_UPGRADE
      4. requires.vllm_image ไม่ตรงกับ image ที่มี → NEEDS_UPGRADE
      5. arch เป็น None (ยังไม่รู้) → UNKNOWN
      4b. engine ไม่มี arch list ให้อ่าน แต่ requires ผ่าน → OK
      6. อ่าน arch list ไม่ได้ (vllm/ds4 หรือ libllama หาไม่เจอ) → UNKNOWN
      7. arch อยู่ใน arch list → OK
      8. arch ไม่อยู่ใน arch list → NEEDS_UPGRADE + action="upgrade_llamacpp"
    """
    if info is None:
        info = detect(engine)
    requires = requires or {}

    # 1. engine ไม่ได้ติดตั้ง
    if not info.installed:
        return CompatResult(
            Compat.NO_ENGINE, f"ยังไม่ได้ติดตั้ง {engine} บนเครื่องนี้ — {info.detail}", None,
        )

    compat = _load_compat_yaml(compat_path)

    # 2. เคยจดไว้ว่า build นี้รันไม่ได้ (เฉพาะ llamacpp — build ผูกกับ llama-server เท่านั้น)
    if arch is not None and engine == "llamacpp" and arch in learned_unsupported(info.build):
        reason = f"เคยลองรันจริงแล้วไม่ได้ (build {info.build} ไม่รองรับ {arch})"
        hint = _upgrade_hint(compat, arch)
        if hint:
            reason += f" — แนะนำอัปเป็น {hint}"
        return CompatResult(Compat.NEEDS_UPGRADE, reason, "upgrade_llamacpp")

    # 3. requires ต้องการ llama-server build สูงกว่าที่มี
    # ค่าใน catalog.yaml เป็น string ("b10353") — YAML ไม่มีทางรู้ว่าเป็นตัวเลข ต้อง normalize เอง
    need_build = _as_build(requires.get("llama_build"))
    if need_build is not None and engine == "llamacpp" and (info.build is None or info.build < need_build):
        return CompatResult(
            Compat.NEEDS_UPGRADE,
            f"โมเดลนี้ต้องการ llama-server build >= {need_build} แต่เครื่องนี้มี build {info.build}",
            "upgrade_llamacpp",
        )

    # 4. requires ต้องการ vLLM image ที่ไม่ตรงกับที่มี
    need_image = requires.get("vllm_image")
    if need_image is not None and engine == "vllm" and info.version != need_image:
        return CompatResult(
            Compat.NEEDS_UPGRADE,
            f"โมเดลนี้ต้องการ vLLM image {need_image} แต่เครื่องนี้มี {info.version or 'ไม่มี'}",
            "upgrade_vllm_image",
        )

    # 4b. engine ที่ไม่มี arch list ให้อ่าน (vllm/ds4): requires คือสัญญาเดียวที่มี
    # ผ่านข้อ 3-4 มาได้ = ตรงตามที่โมเดลประกาศไว้แล้ว ⇒ ตอบ ok ไม่ใช่ปล่อยเป็น unknown
    if not info.archs and requires:
        detail = ", ".join(f"{k}={v}" for k, v in requires.items())
        return CompatResult(Compat.OK, f"{engine} ตรงตาม requires ที่โมเดลประกาศไว้ ({detail})", None)

    # 5. ยังไม่รู้ arch ของโมเดล
    if arch is None:
        return CompatResult(Compat.UNKNOWN, "ยังไม่รู้ arch ของโมเดลนี้ (อ่าน GGUF header ไม่สำเร็จ)", None)

    # 6. อ่าน arch list ของ engine นี้ไม่ได้
    if not info.archs:
        return CompatResult(
            Compat.UNKNOWN,
            f"อ่าน arch list ของ {engine} ไม่ได้ ({info.detail}) — ตอบไม่ได้ว่า {arch} รันได้ไหม",
            None,
        )

    # 7. รองรับ
    if arch in info.archs:
        return CompatResult(Compat.OK, f"{engine} รองรับ {arch} (build {info.build})", None)

    # 8. ไม่รองรับ — แนะนำ action ให้อัป พร้อม hint ถ้ามี
    reason = f"{engine} build {info.build} ยังไม่รองรับ {arch}"
    hint = _upgrade_hint(compat, arch)
    if hint:
        reason += f" — แนะนำอัปเป็น {hint}"
    return CompatResult(Compat.NEEDS_UPGRADE, reason, "upgrade_llamacpp")
