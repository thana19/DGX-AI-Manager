"""test สำหรับ server/engines.py — "engine ในเครื่องนี้รันโมเดล arch นี้ได้ไหม"

ห้ามพึ่งเครื่อง DGX หรือไฟล์จริงในเครื่อง — ทุกอย่างจำลองด้วยไฟล์ปลอมใน tmp_path
และ monkeypatch การรันคำสั่งภายนอก (subprocess) เพื่อให้ผ่านบน Mac เปล่า ๆ ได้เสมอ
"""
from __future__ import annotations

import json
import os
import time

import pytest
import yaml

from server import engines, paths
from server.engines import (
    Compat,
    CompatResult,
    EngineInfo,
    check_arch,
    detect,
    learn_from_log,
    learned_unsupported,
    llamacpp_bin_dir,
    read_supported_archs,
    record_unsupported,
)


# ---------------------------------------------------------------------------
# read_supported_archs
# ---------------------------------------------------------------------------


def test_read_supported_archs_filters_noise(tmp_path):
    lib = tmp_path / "libllama.so.0.3.0"
    lib.write_bytes(b"\x00qwen35\x00glm5next\x00\x01\x02foo/bar\x00libggml.so\x00")

    archs = read_supported_archs(str(tmp_path))

    assert archs == frozenset({"qwen35", "glm5next"})


def test_read_supported_archs_picks_newest_file_by_mtime(tmp_path):
    old = tmp_path / "libllama.so.0.3.0"
    new = tmp_path / "libllama.so.0.0.10195"
    old.write_bytes(b"\x00oldarch\x00")
    new.write_bytes(b"\x00newarch\x00")

    now = time.time()
    os.utime(old, (now - 1000, now - 1000))
    os.utime(new, (now, now))

    archs = read_supported_archs(str(tmp_path))

    assert archs == frozenset({"newarch"})


def test_read_supported_archs_no_lib_file_returns_empty(tmp_path):
    assert read_supported_archs(str(tmp_path)) == frozenset()


def test_read_supported_archs_filters_short_and_long_tokens(tmp_path):
    lib = tmp_path / "libllama.so"
    too_short = b"ab"  # ต่ำกว่า 3 ตัว ไม่ควรถูกนับ (เว้นแต่ไปติดกับ noise ข้างเคียงจนยาวพอ)
    too_long = ("x" * 40).encode("ascii")  # เกิน 32 ตัว ไม่ควรถูกนับ
    lib.write_bytes(b"\x00" + too_short + b"\x00qwen35\x00" + too_long + b"\x00")

    archs = read_supported_archs(str(tmp_path))

    assert archs == frozenset({"qwen35"})


# ---------------------------------------------------------------------------
# llamacpp_bin_dir
# ---------------------------------------------------------------------------


def test_llamacpp_bin_dir_default(monkeypatch):
    monkeypatch.delenv("LLAMA_BIN_DIR", raising=False)
    assert llamacpp_bin_dir() == os.path.expanduser("~/llama.cpp/build/bin")


def test_llamacpp_bin_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("LLAMA_BIN_DIR", str(tmp_path))
    assert llamacpp_bin_dir() == str(tmp_path)


# ---------------------------------------------------------------------------
# detect("llamacpp")
# ---------------------------------------------------------------------------


def _make_fake_llama_server(bin_dir) -> None:
    exe = bin_dir / "llama-server"
    exe.write_text("#!/bin/sh\necho fake\n")
    os.chmod(exe, 0o755)


def test_detect_llamacpp_installed(monkeypatch, tmp_path):
    monkeypatch.setenv("LLAMA_BIN_DIR", str(tmp_path))
    _make_fake_llama_server(tmp_path)
    (tmp_path / "libllama.so.0.3.0").write_bytes(b"\x00qwen35\x00glm5next\x00")

    monkeypatch.setattr(
        engines,
        "_run_llama_version",
        lambda bin_path, bin_dir: "version: 0.3.0-dev (build 10696, commit 1f0a36a35)",
    )

    info = detect("llamacpp")

    assert isinstance(info, EngineInfo)
    assert info.name == "llamacpp"
    assert info.installed is True
    assert info.build == 10696
    assert info.archs == frozenset({"qwen35", "glm5next"})
    assert "10696" in info.detail


def test_detect_llamacpp_not_installed_no_binary(monkeypatch, tmp_path):
    monkeypatch.setenv("LLAMA_BIN_DIR", str(tmp_path))  # โฟลเดอร์ว่างเปล่า ไม่มี llama-server

    info = detect("llamacpp")

    assert info.installed is False
    assert info.build is None
    assert info.archs == frozenset()


def test_detect_llamacpp_binary_fails_to_run(monkeypatch, tmp_path):
    monkeypatch.setenv("LLAMA_BIN_DIR", str(tmp_path))
    _make_fake_llama_server(tmp_path)
    monkeypatch.setattr(engines, "_run_llama_version", lambda bin_path, bin_dir: None)

    info = detect("llamacpp")

    assert info.installed is False


# ---------------------------------------------------------------------------
# detect("vllm") / detect("ds4") — เช็คขั้นต่ำว่าไม่พังตอนไม่มี docker/binary
# ---------------------------------------------------------------------------


def test_detect_vllm_not_installed(monkeypatch):
    monkeypatch.setattr(engines, "_docker_images", lambda: [])

    info = detect("vllm")

    assert info.name == "vllm"
    assert info.installed is False
    assert info.archs == frozenset()


def test_detect_vllm_installed(monkeypatch):
    monkeypatch.setattr(engines, "_docker_images", lambda: ["aiserver-vllm:26.07", "nvcr.io/nvidia/vllm:26.07-py3"])

    info = detect("vllm")

    assert info.installed is True
    assert info.version == "aiserver-vllm:26.07"
    assert info.archs == frozenset()


def test_detect_ds4_not_installed(monkeypatch):
    monkeypatch.delenv("DS4_BIN", raising=False)
    monkeypatch.setattr(engines.shutil, "which", lambda name: None)

    info = detect("ds4")

    assert info.installed is False


def test_detect_unknown_engine_raises():
    with pytest.raises(ValueError):
        detect("no-such-engine")


def test_detect_all_returns_three_engines(monkeypatch, tmp_path):
    monkeypatch.setenv("LLAMA_BIN_DIR", str(tmp_path))
    monkeypatch.setattr(engines, "_docker_images", lambda: [])
    monkeypatch.delenv("DS4_BIN", raising=False)

    infos = engines.detect_all()

    assert [i.name for i in infos] == ["llamacpp", "vllm", "ds4"]


# ---------------------------------------------------------------------------
# check_arch — 8 กติกา
# ---------------------------------------------------------------------------


def _llamacpp_info(**overrides) -> EngineInfo:
    base = dict(
        name="llamacpp", installed=True, version="version: 0.3.0-dev (build 10696, ...)",
        build=10696, archs=frozenset({"qwen35", "glm5next"}), detail="llama-server build 10696",
    )
    base.update(overrides)
    return EngineInfo(**base)


def test_check_arch_rule1_no_engine():
    info = EngineInfo(
        name="llamacpp", installed=False, version=None, build=None, archs=frozenset(),
        detail="ไม่พบ llama-server ใน /fake/bin",
    )

    result = check_arch("qwen35", "llamacpp", info=info)

    assert result.status == Compat.NO_ENGINE
    assert result.action is None


def test_check_arch_rule2_learned_unsupported_wins_over_arch_list(monkeypatch, tmp_path):
    monkeypatch.setenv("AISERVER2_STATE", str(tmp_path))
    record_unsupported("qwen35", 10696)  # จดว่าลองจริงแล้วไม่ได้ ทั้งที่ arch อยู่ใน lib จริง
    info = _llamacpp_info(archs=frozenset({"qwen35", "glm5next"}))

    result = check_arch("qwen35", "llamacpp", info=info)

    assert result.status == Compat.NEEDS_UPGRADE
    assert result.action == "upgrade_llamacpp"
    assert "เคยลองรันจริง" in result.reason


def test_check_arch_rule3_requires_higher_llama_build():
    info = _llamacpp_info(build=10696)

    result = check_arch("qwen35", "llamacpp", info=info, requires={"llama_build": 10700})

    assert result.status == Compat.NEEDS_UPGRADE
    assert result.action == "upgrade_llamacpp"
    assert "10700" in result.reason


def test_check_arch_rule4_requires_vllm_image_mismatch():
    info = EngineInfo(
        name="vllm", installed=True, version="aiserver-vllm:26.07", build=None,
        archs=frozenset(), detail="พบ image aiserver-vllm:26.07",
    )

    result = check_arch("some-arch", "vllm", info=info, requires={"vllm_image": "aiserver-vllm:26.08"})

    assert result.status == Compat.NEEDS_UPGRADE
    assert result.action == "upgrade_vllm_image"
    assert "26.08" in result.reason


def test_check_arch_rule5_arch_unknown():
    info = _llamacpp_info()

    result = check_arch(None, "llamacpp", info=info)

    assert result.status == Compat.UNKNOWN
    assert result.action is None


def test_check_arch_rule5_empty_string_arch_treated_as_unknown():
    # entry ที่ UI เพิ่มไว้ก่อนอ่าน GGUF header สำเร็จ จะมี arch เป็น "" ไม่ใช่ None
    # ต้องได้ unknown เหมือน None ทุกประการ — ไม่ใช่หลุดไปโดนข้อ 8 (NEEDS_UPGRADE)
    info = _llamacpp_info()

    result = check_arch("", "llamacpp", info=info)

    assert result.status == Compat.UNKNOWN
    assert result.action is None


def test_check_arch_rule6_cannot_read_arch_list():
    info = EngineInfo(
        name="vllm", installed=True, version="aiserver-vllm:26.07", build=None,
        archs=frozenset(), detail="พบ image aiserver-vllm:26.07 — vLLM ไม่มี arch list ให้อ่าน",
    )

    result = check_arch("qwen35", "vllm", info=info)

    assert result.status == Compat.UNKNOWN
    assert result.action is None


def test_check_arch_rule7_ok():
    info = _llamacpp_info(archs=frozenset({"qwen35", "glm5next"}))

    result = check_arch("qwen35", "llamacpp", info=info)

    assert result.status == Compat.OK
    assert result.action is None


def test_check_arch_rule8_needs_upgrade_with_hint(tmp_path):
    compat_path = tmp_path / "engine-compat.yaml"
    compat_path.write_text(
        yaml.safe_dump({"llamacpp": {"upgrade_hint": {"glm5next": "อัปเป็น llama-pack build >= 10700"}}}),
        encoding="utf-8",
    )
    info = _llamacpp_info(archs=frozenset({"qwen35"}))

    result = check_arch("glm5next", "llamacpp", info=info, compat_path=str(compat_path))

    assert result.status == Compat.NEEDS_UPGRADE
    assert result.action == "upgrade_llamacpp"
    assert "llama-pack build >= 10700" in result.reason


def test_check_arch_rule8_needs_upgrade_without_hint(tmp_path):
    # ไฟล์ compat ไม่มี upgrade_hint สำหรับ arch นี้ (หรือไม่มีไฟล์เลย) — ต้องไม่พัง แค่ไม่มี hint แถม
    info = _llamacpp_info(archs=frozenset({"qwen35"}))

    result = check_arch("brand-new-arch", "llamacpp", info=info, compat_path=str(tmp_path / "missing.yaml"))

    assert result.status == Compat.NEEDS_UPGRADE
    assert result.action == "upgrade_llamacpp"


def test_check_arch_uses_detect_when_info_not_given(monkeypatch, tmp_path):
    monkeypatch.setenv("LLAMA_BIN_DIR", str(tmp_path))  # ไม่มี llama-server ⇒ ต้องได้ NO_ENGINE ผ่าน detect() จริง

    result = check_arch("qwen35", "llamacpp")

    assert result.status == Compat.NO_ENGINE


# ---------------------------------------------------------------------------
# learn_from_log
# ---------------------------------------------------------------------------


def test_learn_from_log_single_quotes():
    log = "main: error: unknown model architecture: 'qwen4exp'\n"
    assert learn_from_log(log) == "qwen4exp"


def test_learn_from_log_double_quotes_with_prefix():
    log = 'llama_model_load: error loading model: unknown model architecture: "glm5next"\n'
    assert learn_from_log(log) == "glm5next"


def test_learn_from_log_no_error_returns_none():
    log = "llama_model_load: loading model...\nmain: model loaded, listening on :8001\n"
    assert learn_from_log(log) is None


def test_learn_from_log_multiline_picks_first_match():
    log = "some noise\nunknown model architecture: 'deepseek4'\nmore noise\n"
    assert learn_from_log(log) == "deepseek4"


# ---------------------------------------------------------------------------
# record_unsupported / learned_unsupported
# ---------------------------------------------------------------------------


def test_record_and_read_back_unsupported(monkeypatch, tmp_path):
    monkeypatch.setenv("AISERVER2_STATE", str(tmp_path))

    record_unsupported("qwen4exp", 10696)

    assert learned_unsupported(10696) == {"qwen4exp"}
    assert learned_unsupported(10697) == set()  # build อื่นไม่ได้รับผลกระทบ


def test_record_unsupported_accumulates_multiple_archs(monkeypatch, tmp_path):
    monkeypatch.setenv("AISERVER2_STATE", str(tmp_path))

    record_unsupported("qwen4exp", 10696)
    record_unsupported("glm5next", 10696)

    assert learned_unsupported(10696) == {"qwen4exp", "glm5next"}


def test_record_unsupported_build_none_uses_unknown_bucket(monkeypatch, tmp_path):
    monkeypatch.setenv("AISERVER2_STATE", str(tmp_path))

    record_unsupported("mystery-arch", None)

    assert learned_unsupported(None) == {"mystery-arch"}
    assert learned_unsupported(10696) == set()


def test_learned_unsupported_missing_file_returns_empty(tmp_path):
    assert learned_unsupported(10696, path=str(tmp_path / "does-not-exist.json")) == set()


def test_record_unsupported_writes_valid_json_atomically(monkeypatch, tmp_path):
    monkeypatch.setenv("AISERVER2_STATE", str(tmp_path))

    record_unsupported("qwen4exp", 10696)

    state_file = tmp_path / "compat-learned.json"
    assert state_file.exists()
    with open(state_file, encoding="utf-8") as f:
        data = json.load(f)
    assert data == {"10696": ["qwen4exp"]}
    # ไม่ทิ้งไฟล์ temp ค้าง
    leftover = [p for p in os.listdir(tmp_path) if p.startswith(".compat-learned-")]
    assert leftover == []


def test_record_unsupported_explicit_path_does_not_touch_state_env(monkeypatch, tmp_path):
    monkeypatch.delenv("AISERVER2_STATE", raising=False)
    explicit_path = str(tmp_path / "custom-compat.json")

    record_unsupported("qwen4exp", 10696, path=explicit_path)

    assert os.path.exists(explicit_path)
    assert learned_unsupported(10696, path=explicit_path) == {"qwen4exp"}


def test_engine_ไม่มี_arch_list_แต่_requires_ตรง_ต้องตอบ_ok():
    """vllm/ds4 อ่าน arch list ไม่ได้ — requires คือสัญญาเดียวที่มี ผ่านแล้วต้องไม่ปล่อยเป็น unknown"""
    info = EngineInfo(name="vllm", installed=True, version="aiserver-vllm:26.07",
                      build=None, archs=frozenset(), detail="")
    r = check_arch(None, "vllm", info=info, requires={"vllm_image": "aiserver-vllm:26.07"})
    assert r.status is Compat.OK
    assert r.action is None


def test_engine_ไม่มี_arch_list_และไม่มี_requires_ยังเป็น_unknown():
    info = EngineInfo(name="vllm", installed=True, version="aiserver-vllm:26.07",
                      build=None, archs=frozenset(), detail="")
    r = check_arch(None, "vllm", info=info, requires=None)
    assert r.status is Compat.UNKNOWN
