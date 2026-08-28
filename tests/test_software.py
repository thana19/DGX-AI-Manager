"""test สำหรับ server/software.py — เช็คสถานะ software + อัป/rollback llama.cpp engine

ห้ามยิงเน็ตจริง ห้ามแตะ ~/llama.cpp จริง — mock HTTP ด้วย httpx.MockTransport และใช้
tmp_path เป็น bin_dir เสมอ (ผ่าน LLAMA_BIN_DIR override ภายใน software.upgrade_llamacpp)
"""
from __future__ import annotations

import io
import os
import stat
import tarfile

import httpx
import pytest

from server import engines, software
from server.software import SoftwareItem, UpgradeResult, check_all, disk_free_gb, mem_total_gb


# ---------------------------------------------------------------------------
# helper: สร้าง "engine" ปลอมในโฟลเดอร์หนึ่ง ๆ — llama-server ที่รันได้จริง (bash script)
# + libllama.so ที่มี arch token ฝังอยู่ (ให้ read_supported_archs อ่านเจอ)
# ---------------------------------------------------------------------------


def _write_fake_engine(bin_dir: str, build: int, archs: list[str]) -> None:
    os.makedirs(bin_dir, exist_ok=True)
    server_path = os.path.join(bin_dir, "llama-server")
    with open(server_path, "w", encoding="utf-8") as f:
        f.write(f"#!/bin/bash\necho 'version: 0.3.0-dev (build {build}, commit abc123)'\n")
    os.chmod(server_path, os.stat(server_path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    lib_path = os.path.join(bin_dir, "libllama.so.0.3.0")
    blob = b"\x00" + b"\x00".join(a.encode("ascii") for a in archs) + b"\x00"
    with open(lib_path, "wb") as f:
        f.write(blob)


def _make_pack_tar(build: int, archs: list[str]) -> bytes:
    """สร้าง tar.gz ในหน่วยความจำ ที่แตกออกมาแล้วได้ engine ที่รันได้จริง (จำลอง llama-pack-gb10.tar.gz)"""
    src_dir_items = {}
    server_script = f"#!/bin/bash\necho 'version: 0.3.0-dev (build {build}, commit abc123)'\n"
    lib_blob = b"\x00" + b"\x00".join(a.encode("ascii") for a in archs) + b"\x00"
    src_dir_items["llama-server"] = server_script.encode("utf-8")
    src_dir_items["libllama.so.0.3.0"] = lib_blob

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in src_dir_items.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            if name == "llama-server":
                info.mode = 0o755
            tar.addfile(info, io.BytesIO(content))
    return buf.getvalue()


def _make_broken_pack_tar() -> bytes:
    """pack ที่ทับ llama-server เดิมด้วยไฟล์ที่รันไม่ได้จริง — จำลองเคส build ใหม่พังบนเครื่องนี้
    (ต้องทับ llama-server ของเดิม ไม่ใช่แค่เติมไฟล์ใหม่ — ไม่งั้นของเดิมที่ยังรันได้จะบังผลลัพธ์)
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        content = b"not a real binary\n"
        info = tarfile.TarInfo(name="llama-server")
        info.size = len(content)
        info.mode = 0o644  # ไม่ executable — รัน --version ไม่ได้แน่นอน
        tar.addfile(info, io.BytesIO(content))
    return buf.getvalue()


def _mock_client(tar_bytes: bytes, *, status_code: int = 200) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, content=tar_bytes)

    return httpx.Client(transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------------------
# check_all
# ---------------------------------------------------------------------------


def test_check_all_returns_item_per_software():
    items = check_all()
    assert len(items) == len(software.SOFTWARE)
    assert all(isinstance(i, SoftwareItem) for i in items)
    names = {i.name for i in items}
    assert "git" in names
    assert "llama-server" in names
    assert "thclaws" in names


def test_check_all_marks_unknown_command_not_installed(monkeypatch):
    monkeypatch.setattr(software, "SOFTWARE", [("totally-bogus-tool-xyz", "totally-bogus-tool-xyz --version", "base")])
    items = check_all()
    assert len(items) == 1
    assert items[0].installed is False
    assert items[0].version is None


def test_check_all_pipefail_catches_real_exit_code(monkeypatch):
    # คำสั่งจริงล้มเหลวแต่ต่อท้ายด้วย | head — ถ้าไม่มี set -o pipefail จะดูเหมือนสำเร็จ (บทเรียนจาก v1)
    monkeypatch.setattr(software, "SOFTWARE", [("fail-piped", "bash -c 'exit 3' | head -1", "base")])
    items = check_all()
    assert items[0].installed is False


# ---------------------------------------------------------------------------
# mem_total_gb / disk_free_gb
# ---------------------------------------------------------------------------


def test_mem_total_gb_reads_proc_meminfo(tmp_path, monkeypatch):
    fake_meminfo = tmp_path / "meminfo"
    fake_meminfo.write_text("MemTotal:       134217728 kB\nMemFree:        1000 kB\n")

    real_open = open

    def fake_open(path, *args, **kwargs):
        if path == "/proc/meminfo":
            return real_open(fake_meminfo, *args, **kwargs)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)
    assert mem_total_gb() == 128.0


def test_mem_total_gb_returns_none_when_no_proc_meminfo(monkeypatch):
    def fake_open(path, *args, **kwargs):
        if path == "/proc/meminfo":
            raise OSError("no such file")
        return open(path, *args, **kwargs)

    import builtins
    monkeypatch.setattr(builtins, "open", fake_open)
    assert mem_total_gb() is None


def test_disk_free_gb_existing_path(tmp_path):
    result = disk_free_gb(str(tmp_path))
    assert result is not None
    assert result >= 0


def test_disk_free_gb_missing_path_returns_none():
    assert disk_free_gb("/no/such/path/at/all/xyz") is None


# ---------------------------------------------------------------------------
# license_code / machine_id
# ---------------------------------------------------------------------------


def test_license_code_prefers_v2_over_v1(tmp_path, monkeypatch):
    monkeypatch.setenv("AISERVER2_STATE", str(tmp_path))
    (tmp_path / "license").write_text("V2CODE\n")

    v1_dir = tmp_path / "v1home"
    v1_dir.mkdir()
    monkeypatch.setattr(os.path, "expanduser", lambda p: p.replace("~/.aiserver/license", str(v1_dir / "license")))

    assert software.license_code() == "V2CODE"


def test_license_code_falls_back_to_v1(tmp_path, monkeypatch):
    monkeypatch.setenv("AISERVER2_STATE", str(tmp_path))  # ~/.aiserver2/license ไม่มีไฟล์

    v1_license = tmp_path / "v1-license"
    v1_license.write_text("V1CODE\n")
    monkeypatch.setattr(os.path, "expanduser", lambda p: str(v1_license) if p == "~/.aiserver/license" else p)

    assert software.license_code() == "V1CODE"


def test_license_code_none_when_neither_exists(tmp_path, monkeypatch):
    monkeypatch.setenv("AISERVER2_STATE", str(tmp_path))
    monkeypatch.setattr(os.path, "expanduser", lambda p: str(tmp_path / "nope") if p == "~/.aiserver/license" else p)
    assert software.license_code() is None


def test_machine_id_is_deterministic_16_hex(monkeypatch):
    def fake_run(cmd, **kwargs):
        class R:
            stdout = "gpu-uuid-abc\n"
        return R()

    monkeypatch.setattr(software.subprocess, "run", fake_run)
    mid1 = software.machine_id()
    mid2 = software.machine_id()
    assert mid1 == mid2
    assert len(mid1) == 16
    int(mid1, 16)  # ต้องเป็น hex ล้วน


# ---------------------------------------------------------------------------
# upgrade_llamacpp / rollback_llamacpp
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    monkeypatch.setenv("AISERVER2_STATE", str(tmp_path / "state"))
    monkeypatch.delenv("LLAMA_BIN_DIR", raising=False)


def _license_file(tmp_path, monkeypatch) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AISERVER2_STATE", str(state_dir))
    (state_dir / "license").write_text("TESTLICENSE\n")


def test_upgrade_rejects_when_no_license(tmp_path, monkeypatch):
    bin_dir = str(tmp_path / "bin")
    result = software.upgrade_llamacpp(bin_dir=bin_dir, client=_mock_client(b""))
    assert result.ok is False
    assert "license" in result.message.lower()


def test_upgrade_rejects_git_checkout(tmp_path, monkeypatch):
    _license_file(tmp_path, monkeypatch)
    root = tmp_path / "llama.cpp"
    bin_dir = root / "build" / "bin"
    _write_fake_engine(str(bin_dir), build=10353, archs=["qwen35"])
    (root / ".git").mkdir(parents=True)

    result = software.upgrade_llamacpp(bin_dir=str(bin_dir), client=_mock_client(b""))

    assert result.ok is False
    assert "git" in result.message.lower() or ".git" in result.message


def test_upgrade_success_creates_backup_and_new_build(tmp_path, monkeypatch):
    _license_file(tmp_path, monkeypatch)
    bin_dir = tmp_path / "llama.cpp" / "build" / "bin"
    _write_fake_engine(str(bin_dir), build=10353, archs=["qwen35"])

    tar_bytes = _make_pack_tar(build=10696, archs=["qwen35", "qwen4exp", "glm5next"])
    client = _mock_client(tar_bytes)

    result = software.upgrade_llamacpp(bin_dir=str(bin_dir), client=client)

    assert result.ok is True
    assert result.old_build == 10353
    assert result.new_build == 10696
    assert result.backup_dir is not None
    assert os.path.isdir(result.backup_dir)
    # backup มีของเดิม (build 10353) เก็บไว้จริง
    with open(os.path.join(result.backup_dir, "llama-server"), encoding="utf-8") as f:
        assert "10353" in f.read()
    # ของใหม่ใช้งานได้จริงที่ bin_dir เดิม
    info = software._detect_llamacpp_at(str(bin_dir))
    assert info.build == 10696
    assert "qwen4exp" in info.archs


def test_upgrade_new_binary_cant_run_rolls_back_automatically(tmp_path, monkeypatch):
    _license_file(tmp_path, monkeypatch)
    bin_dir = tmp_path / "llama.cpp" / "build" / "bin"
    _write_fake_engine(str(bin_dir), build=10353, archs=["qwen35"])

    broken_tar = _make_broken_pack_tar()
    client = _mock_client(broken_tar)

    result = software.upgrade_llamacpp(bin_dir=str(bin_dir), client=client)

    assert result.ok is False
    assert result.old_build == 10353
    # rollback อัตโนมัติ — bin_dir กลับมารันได้เหมือนเดิม (build 10353)
    info = software._detect_llamacpp_at(str(bin_dir))
    assert info.installed is True
    assert info.build == 10353


def test_upgrade_download_failure_returns_error(tmp_path, monkeypatch):
    _license_file(tmp_path, monkeypatch)
    bin_dir = tmp_path / "llama.cpp" / "build" / "bin"
    _write_fake_engine(str(bin_dir), build=10353, archs=["qwen35"])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"server error")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = software.upgrade_llamacpp(bin_dir=str(bin_dir), client=client)

    assert result.ok is False
    # ยังไม่ได้แตะ bin_dir เดิมเลย (ดาวน์โหลดพังตั้งแต่ก่อน backup/extract)
    info = software._detect_llamacpp_at(str(bin_dir))
    assert info.build == 10353


def test_rollback_manual_restores_backup(tmp_path, monkeypatch):
    _license_file(tmp_path, monkeypatch)
    bin_dir = tmp_path / "llama.cpp" / "build" / "bin"
    _write_fake_engine(str(bin_dir), build=10353, archs=["qwen35"])

    tar_bytes = _make_pack_tar(build=10696, archs=["qwen35", "qwen4exp"])
    upgrade_result = software.upgrade_llamacpp(bin_dir=str(bin_dir), client=_mock_client(tar_bytes))
    assert upgrade_result.ok is True

    rollback_result = software.rollback_llamacpp(bin_dir=str(bin_dir))

    assert rollback_result.ok is True
    assert rollback_result.new_build == 10353
    info = software._detect_llamacpp_at(str(bin_dir))
    assert info.build == 10353


def test_rollback_no_backup_returns_error(tmp_path):
    bin_dir = tmp_path / "llama.cpp" / "build" / "bin"
    _write_fake_engine(str(bin_dir), build=10353, archs=["qwen35"])

    result = software.rollback_llamacpp(bin_dir=str(bin_dir))

    assert result.ok is False
    assert "backup" in result.message.lower()
