"""ทะเบียนโมเดลของระบบ — รวม catalog ของ vendor (catalog.yaml) เข้ากับโมเดลที่ผู้ใช้เพิ่มเอง
(~/.aiserver2/user-models.json) แล้วบอกสถานะไฟล์บนดิสก์

ดู docs/adr/0001-registry-split.md: 2 ไฟล์ schema เดียวกัน — vendor push catalog.yaml ได้อิสระ
โดยไม่ทับของผู้ใช้ · merge ตอนอ่าน id ชนกัน = ของผู้ใช้ชนะ
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict

from . import paths

# ไฟล์ที่ยังโหลดไม่จบจะมี "<ชื่อไฟล์>.aria2" วางคู่อยู่ (aria2c เขียน control file ระหว่างโหลด)
_ARIA2_SUFFIX = ".aria2"

# path แบบ "...-00001-of-00003.gguf" = sharded — จับกลุ่ม index/total ไว้ generate shard อื่น ๆ
_SHARD_RE = re.compile(r"^(?P<base>.+)-(?P<idx>\d+)-of-(?P<total>\d+)\.gguf$")

# หาไฟล์ .gguf ที่ args อ้างถึงเพิ่ม (draft/mmproj เช่น "-md ~/x/mtp.gguf --mmproj ~/x/y.gguf")
_ARGS_GGUF_RE = re.compile(r"\S+\.gguf")

# args ห้ามมี "-c" เป็น flag โดด ๆ (llamacpp.sh ใส่ -c จาก ctx ให้แล้ว จะทับค่าที่ผู้ใช้ตั้ง)
_DASH_C_RE = re.compile(r"(?:^|\s)-c(?:\s|$)")

_KNOWN_ENGINES = {"llamacpp", "vllm", "ds4"}


class CatalogError(Exception):
    """ไฟล์ catalog (vendor หรือ user) อ่าน/parse ไม่ได้ — โครงสร้างผิด ไม่ใช่แค่ field ผิด"""


class ModelEntry(BaseModel):
    """1 รายการโมเดลใน catalog — schema ตรงกับ field ใน catalog.yaml"""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    engine: str  # ไม่ใช้ Literal ตั้งใจ — ค่าที่ไม่รู้จักให้ lint() จับแทน ไม่ใช่ปฏิเสธตั้งแต่ parse
    path: str
    arch: str | None = None
    source_repo: str | None = None
    ctx: int | None = None
    ctx_train: int | None = None
    kv_kb_per_token: int | None = None
    need_gb: float | None = None
    args: str = ""
    requires: dict[str, str] | None = None
    dl: list[str] | None = None
    note: str | None = None
    # ไม่ได้อยู่ในไฟล์ — เติมให้ตอนโหลด (vendor/user) เพื่อบอก UI ว่ามาจากไหน
    source: Literal["vendor", "user"] = "vendor"


# ---------------------------------------------------------------------------
# โหลด / merge
# ---------------------------------------------------------------------------


def load_vendor(path: str | None = None) -> list[ModelEntry]:
    """อ่าน catalog.yaml ของ vendor (default: paths.repo('catalog.yaml'))"""
    p = path or paths.repo("catalog.yaml")
    try:
        with open(p, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise CatalogError(f"catalog.yaml parse ไม่ผ่าน ({p}): {e}") from e

    models = raw.get("models")
    if models is None:
        raise CatalogError(f"catalog.yaml ไม่มี key 'models' ({p})")

    return [ModelEntry(**m, source="vendor") for m in models]


def load_user(path: str | None = None) -> list[ModelEntry]:
    """อ่าน ~/.aiserver2/user-models.json — ไม่มีไฟล์ = list ว่าง (ไม่ใช่ error)"""
    p = path or paths.state("user-models.json")
    if not os.path.isfile(p):
        return []

    try:
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        raise CatalogError(f"user-models.json parse ไม่ผ่าน ({p}): {e}") from e

    if not isinstance(raw, list):
        raise CatalogError(f"user-models.json ต้องเป็น list ของ model entry ({p})")

    return [ModelEntry(**m, source="user") for m in raw]


def load_all(
    vendor_path: str | None = None,
    user_path: str | None = None,
) -> list[ModelEntry]:
    """merge vendor + user — id ชนกัน ของผู้ใช้ชนะ (ADR 0001) · ลำดับ: vendor ก่อน แล้วต่อด้วย user ที่เหลือ"""
    vendor = load_vendor(vendor_path)
    user = load_user(user_path)
    user_by_id = {e.id: e for e in user}

    merged = [user_by_id.pop(e.id, e) for e in vendor]
    merged.extend(user_by_id.values())  # user entry ที่ไม่ได้ชนกับ vendor เลย
    return merged


# ---------------------------------------------------------------------------
# แก้ไข user-models.json (เขียนได้เฉพาะไฟล์นี้ — ห้ามแตะ catalog.yaml ของ vendor)
# ---------------------------------------------------------------------------


def _user_path(path: str | None) -> str:
    return path or paths.state("user-models.json")


def _read_user_raw(path: str) -> list[dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise CatalogError(f"user-models.json ต้องเป็น list ของ model entry ({path})")
    return raw


def _write_user_raw(entries: list[dict[str, Any]], path: str) -> None:
    """เขียนแบบ atomic — tmp file ใน dir เดียวกันแล้ว os.replace กันไฟล์พังกลางทาง"""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def add_user_model(entry: ModelEntry | dict[str, Any], path: str | None = None) -> ModelEntry:
    """เพิ่ม/แทนที่ใน user-models.json — id ชนกับของเดิมในไฟล์ = แทนที่ · validate ก่อนเขียนเสมอ"""
    if not isinstance(entry, ModelEntry):
        entry = ModelEntry(**entry)
    # source ไม่เก็บลงไฟล์ (เติมตอนโหลด) — แต่ validate ผ่าน ModelEntry ให้ชัวร์ว่า field อื่นถูกต้อง
    entry = entry.model_copy(update={"source": "user"})

    p = _user_path(path)
    raw = _read_user_raw(p)
    dump = entry.model_dump(exclude={"source"}, exclude_none=True)

    raw = [m for m in raw if m.get("id") != entry.id]
    raw.append(dump)
    _write_user_raw(raw, p)
    return entry


def remove_user_model(model_id: str, path: str | None = None) -> bool:
    """ลบออกจาก user-models.json — คืน False ถ้าไม่มี · ไม่แตะ catalog.yaml ของ vendor เลย"""
    p = _user_path(path)
    raw = _read_user_raw(p)
    remaining = [m for m in raw if m.get("id") != model_id]
    if len(remaining) == len(raw):
        return False
    _write_user_raw(remaining, p)
    return True


# ---------------------------------------------------------------------------
# ตรรกะไฟล์บนดิสก์ — ยกมาจาก v1 (เช็คครบทุก shard, .aria2 = ยังไม่ ready)
# ---------------------------------------------------------------------------


def expand(entry: ModelEntry) -> str:
    return os.path.expanduser(entry.path)


def _is_folder_entry(entry: ModelEntry) -> bool:
    """vllm หรือ path ไม่ลงท้าย .gguf = โมเดลแบบโฟลเดอร์"""
    return entry.engine == "vllm" or not expand(entry).endswith(".gguf")


def shard_paths(entry: ModelEntry) -> list[str]:
    """ทุก shard ของโมเดล (ไฟล์เดียว = list 1 ตัว) — ไม่รวมไฟล์เพิ่มเติมจาก args"""
    full = expand(entry)
    if not full.endswith(".gguf"):
        return [full]

    directory = os.path.dirname(full)
    basename = os.path.basename(full)
    m = _SHARD_RE.match(basename)
    if not m:
        return [full]

    base, idx, total = m.group("base"), m.group("idx"), m.group("total")
    idx_width = len(idx)
    total_n = int(total)
    return [os.path.join(directory, f"{base}-{i:0{idx_width}d}-of-{total}.gguf") for i in range(1, total_n + 1)]


def _extra_gguf_from_args(args: str) -> list[str]:
    """ไฟล์ .gguf เพิ่มเติมที่ args อ้างถึง (draft/mmproj) — parse path ออกจาก args string"""
    return [os.path.expanduser(tok) for tok in _ARGS_GGUF_RE.findall(args or "")]


def is_ready(entry: ModelEntry) -> bool:
    """ไฟล์ครบพร้อมรันเลยไหม"""
    if _is_folder_entry(entry):
        d = expand(entry)
        return os.path.isdir(d) and len(os.listdir(d)) > 0

    files = shard_paths(entry) + _extra_gguf_from_args(entry.args)
    for f in files:
        if not os.path.isfile(f):
            return False
        if os.path.isfile(f + _ARIA2_SUFFIX):
            return False
    return True


def disk_bytes(entry: ModelEntry) -> int:
    """ขนาดจริงบนดิสก์ (รวมทุก shard + ไฟล์ .gguf ใน args ที่มีอยู่จริง — ไฟล์หายนับเป็น 0)"""
    if _is_folder_entry(entry):
        d = expand(entry)
        if not os.path.isdir(d):
            return 0
        total = 0
        for root, _dirs, files in os.walk(d):
            for fn in files:
                try:
                    total += os.path.getsize(os.path.join(root, fn))
                except OSError:
                    pass
        return total

    total = 0
    for f in shard_paths(entry) + _extra_gguf_from_args(entry.args):
        if os.path.isfile(f):
            total += os.path.getsize(f)
    return total


# ---------------------------------------------------------------------------
# lint — แทน tools/lint-models.py เดิมที่หายไป
# ---------------------------------------------------------------------------


def lint(entries: list[ModelEntry]) -> list[str]:
    """คืนรายการปัญหา (ว่าง = ผ่าน) — ข้อความอธิบายชัดว่าปัญหาคืออะไรและอยู่ที่ entry ไหน"""
    problems: list[str] = []

    seen_ids: set[str] = set()
    for entry in entries:
        if entry.id in seen_ids:
            problems.append(f"{entry.id}: id ซ้ำกับ entry อื่นใน catalog")
        seen_ids.add(entry.id)

        if entry.engine not in _KNOWN_ENGINES:
            problems.append(f"{entry.id}: engine '{entry.engine}' ไม่รู้จัก (ต้องเป็น llamacpp/vllm/ds4)")

        if _DASH_C_RE.search(entry.args or ""):
            problems.append(f"{entry.id}: args มี '-c' — ห้ามใส่ llamacpp.sh ใส่ -c จาก ctx ให้แล้ว จะทับค่าที่ตั้งไว้")

        if entry.ctx is not None and entry.ctx_train is not None and entry.ctx > entry.ctx_train:
            problems.append(f"{entry.id}: ctx ({entry.ctx}) มากกว่า ctx_train ({entry.ctx_train}) — ตั้งเกินที่โมเดลรับได้")

        if entry.dl and not _is_folder_entry(entry):
            expected = {os.path.basename(p) for p in shard_paths(entry)}
            expected |= {os.path.basename(p) for p in _extra_gguf_from_args(entry.args)}
            for url in entry.dl:
                basename = url.rsplit("/", 1)[-1]
                if basename not in expected:
                    problems.append(
                        f"{entry.id}: dl url '{url}' ชื่อไฟล์ '{basename}' ไม่ตรงกับไฟล์ที่ path/shard/args คาดหวัง"
                    )

    return problems
