"""ตรวจ arch / ctx_train ของโมเดลใน catalog — อ่านจากไฟล์บนดิสก์ก่อน (ไม่ใช้เน็ต) ไม่มีไฟล์ค่อยถาม HF

ผลลัพธ์ถูกเก็บใน overlay model-facts.json (catalog.save_fact) ไม่แตะ catalog.yaml ของ vendor
"""
from __future__ import annotations

import json
import os

from . import catalog, gguf, hf

# อ่าน header ทีละขั้น: 1 MiB ก่อน ถ้ายังไม่ครบค่อยขยายถึง 8 MiB
_READ_STEPS = (1 << 20, 8 << 20)


class InspectError(Exception):
    """ตรวจสอบไม่สำเร็จ — message เป็นภาษาไทยพร้อมแสดงผู้ใช้"""


def _inspect_gguf_file(path: str) -> tuple[str | None, int | None]:
    arch: str | None = None
    ctx: int | None = None
    try:
        with open(path, "rb") as f:
            for size in _READ_STEPS:
                f.seek(0)
                data = f.read(size)
                try:
                    info = gguf.parse_header(data)
                except gguf.NotGgufError as e:
                    raise InspectError(f"ไฟล์ไม่ใช่ GGUF: {os.path.basename(path)}") from e
                arch, ctx = info.arch, info.context_length
                if (arch and ctx) or len(data) < size:
                    break
    except OSError as e:
        raise InspectError(f"อ่านไฟล์ไม่ได้: {e}") from e
    return arch, ctx


def inspect_local(entry: catalog.ModelEntry) -> tuple[str | None, int | None]:
    """อ่าน (arch, ctx_train) จากไฟล์บนดิสก์ — raise InspectError ถ้าอ่านไม่ได้หรือไม่เจอข้อมูลเลย"""
    if catalog._is_folder_entry(entry):
        cfg_path = os.path.join(catalog.expand(entry), "config.json")
        try:
            with open(cfg_path, encoding="utf-8") as f:
                config = json.load(f)
        except (OSError, ValueError) as e:
            raise InspectError(f"อ่าน config.json ไม่ได้ ({cfg_path}): {e}") from e
        if not isinstance(config, dict):
            raise InspectError("config.json รูปแบบไม่ถูกต้อง")
        arch, ctx = hf.config_arch_ctx(config)
    else:
        arch, ctx = _inspect_gguf_file(catalog.shard_paths(entry)[0])

    if arch is None and ctx is None:
        raise InspectError("ไม่พบ arch/ctx ในไฟล์โมเดล")
    return arch, ctx


def inspect_remote(entry: catalog.ModelEntry, token: str | None = None) -> tuple[str | None, int | None]:
    """ถาม HF จาก source_repo — ใช้เมื่อไฟล์ยังไม่อยู่บนดิสก์"""
    repo = entry.source_repo
    if not repo:
        raise InspectError("ไม่มีไฟล์บนดิสก์และไม่มี source_repo ให้ตรวจ")
    token = token or hf.load_token()
    try:
        files = hf.list_files(hf.fetch_repo(repo, token=token))
        groups = hf.group_quants(files)
        if groups:
            smallest = min(groups, key=lambda g: g.total_bytes)
            info, _ = gguf.fetch_header(hf.resolve_url(repo, smallest.files[0].path), token=token)
            arch, ctx = info.arch, info.context_length
        elif hf.has_safetensors(files):
            arch, ctx = hf.config_arch_ctx(hf.fetch_config(repo, token=token))
        else:
            raise InspectError(f"repo {repo} ไม่มีไฟล์ .gguf หรือ .safetensors")
    except InspectError:
        raise
    except Exception as e:
        raise InspectError(f"ตรวจสอบจาก HF ไม่สำเร็จ ({repo}): {e}") from e
    if arch is None and ctx is None:
        raise InspectError("ไม่พบ arch/ctx จาก HF")
    return arch, ctx
