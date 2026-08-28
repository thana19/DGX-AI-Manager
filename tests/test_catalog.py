"""test สำหรับ server/catalog.py — ทะเบียนโมเดลของระบบ (vendor catalog.yaml + user-models.json)

ดู docs/adr/0001-registry-split.md เรื่อง merge logic (id ชนกัน = user ชนะ)
"""
from __future__ import annotations

import json
import os

import pytest
from pydantic import ValidationError

from server import paths
from server.catalog import (
    CatalogError,
    ModelEntry,
    add_user_model,
    disk_bytes,
    expand,
    is_ready,
    lint,
    load_all,
    load_user,
    load_vendor,
    remove_user_model,
    shard_paths,
)


def _entry(**overrides) -> ModelEntry:
    """สร้าง ModelEntry ที่ valid ขั้นต่ำ แล้ว override เฉพาะ field ที่ต้องการทดสอบ"""
    data = {
        "id": "test-model",
        "name": "Test Model",
        "engine": "llamacpp",
        "path": "~/models/test/test.gguf",
        "args": "",
    }
    data.update(overrides)
    return ModelEntry(**data)


# ---------------------------------------------------------------------------
# load_vendor — catalog.yaml จริงของโปรเจกต์ ต้องได้ 9 entry และ lint ผ่าน
# ---------------------------------------------------------------------------


def test_load_vendor_real_catalog_has_9_entries():
    entries = load_vendor()
    assert len(entries) == 9
    assert all(e.source == "vendor" for e in entries)


def test_load_vendor_real_catalog_lints_clean():
    entries = load_vendor()
    problems = lint(entries)
    assert problems == [], f"catalog.yaml มีปัญหาจริง: {problems}"


def test_load_vendor_explicit_path(tmp_path):
    p = tmp_path / "vendor.yaml"
    p.write_text(
        """
models:
  - id: a
    name: "A"
    engine: llamacpp
    path: ~/models/a.gguf
    args: ""
""",
        encoding="utf-8",
    )
    entries = load_vendor(str(p))
    assert len(entries) == 1
    assert entries[0].id == "a"
    assert entries[0].source == "vendor"


def test_load_vendor_missing_models_key_raises_catalog_error(tmp_path):
    p = tmp_path / "vendor.yaml"
    p.write_text("not_models: []\n", encoding="utf-8")
    with pytest.raises(CatalogError):
        load_vendor(str(p))


# ---------------------------------------------------------------------------
# load_user — ~/.aiserver2/user-models.json (ไฟล์ไม่มี = list ว่าง ไม่ใช่ error)
# ---------------------------------------------------------------------------


def test_load_user_missing_file_returns_empty_list(tmp_path):
    p = tmp_path / "user-models.json"
    assert not p.exists()
    assert load_user(str(p)) == []


def test_load_user_parses_entries_and_tags_source(tmp_path):
    p = tmp_path / "user-models.json"
    p.write_text(
        json.dumps([{"id": "u1", "name": "User One", "engine": "llamacpp", "path": "~/x.gguf"}]),
        encoding="utf-8",
    )
    entries = load_user(str(p))
    assert len(entries) == 1
    assert entries[0].id == "u1"
    assert entries[0].source == "user"


# ---------------------------------------------------------------------------
# load_all — merge: id ชนกัน user ต้องชนะ (ADR 0001)
# ---------------------------------------------------------------------------


def test_load_all_merges_vendor_and_user(tmp_path):
    vendor_path = tmp_path / "vendor.yaml"
    vendor_path.write_text(
        """
models:
  - id: a
    name: "Vendor A"
    engine: llamacpp
    path: ~/models/a.gguf
    args: ""
  - id: b
    name: "Vendor B"
    engine: llamacpp
    path: ~/models/b.gguf
    args: ""
""",
        encoding="utf-8",
    )
    user_path = tmp_path / "user-models.json"
    user_path.write_text(json.dumps([{"id": "c", "name": "User C", "engine": "vllm", "path": "~/models/c"}]), encoding="utf-8")

    entries = load_all(vendor_path=str(vendor_path), user_path=str(user_path))
    ids = [e.id for e in entries]
    assert set(ids) == {"a", "b", "c"}


def test_load_all_user_id_collision_wins_over_vendor(tmp_path):
    vendor_path = tmp_path / "vendor.yaml"
    vendor_path.write_text(
        """
models:
  - id: a
    name: "Vendor A"
    engine: llamacpp
    path: ~/models/vendor-a.gguf
    args: ""
""",
        encoding="utf-8",
    )
    user_path = tmp_path / "user-models.json"
    user_path.write_text(
        json.dumps([{"id": "a", "name": "User A override", "engine": "llamacpp", "path": "~/models/user-a.gguf"}]),
        encoding="utf-8",
    )

    entries = load_all(vendor_path=str(vendor_path), user_path=str(user_path))
    assert len(entries) == 1
    winner = entries[0]
    assert winner.id == "a"
    assert winner.name == "User A override"
    assert winner.source == "user"


# ---------------------------------------------------------------------------
# add_user_model / remove_user_model — เขียนเฉพาะ user-models.json แบบ atomic
# ต้องไม่แตะ ~/.aiserver2 จริงระหว่างเทส ⇒ override ด้วย AISERVER2_STATE + tmp_path
# ---------------------------------------------------------------------------


def test_add_user_model_writes_file_via_state_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AISERVER2_STATE", str(tmp_path))

    entry = add_user_model({"id": "new1", "name": "New One", "engine": "llamacpp", "path": "~/models/new1.gguf"})

    assert entry.id == "new1"
    assert entry.source == "user"

    state_file = tmp_path / "user-models.json"
    assert state_file.exists()
    raw = json.loads(state_file.read_text(encoding="utf-8"))
    assert len(raw) == 1
    assert raw[0]["id"] == "new1"
    # source ไม่ควรถูกเขียนลงไฟล์ — เติมตอนโหลดเท่านั้น (ตาม docstring ของ ModelEntry.source)
    assert "source" not in raw[0]


def test_add_user_model_replaces_existing_id(tmp_path):
    path = str(tmp_path / "user-models.json")
    add_user_model({"id": "dup", "name": "First", "engine": "llamacpp", "path": "~/a.gguf"}, path=path)
    add_user_model({"id": "dup", "name": "Second", "engine": "llamacpp", "path": "~/b.gguf"}, path=path)

    entries = load_user(path)
    assert len(entries) == 1
    assert entries[0].name == "Second"


def test_add_user_model_accepts_model_entry_instance(tmp_path):
    path = str(tmp_path / "user-models.json")
    entry = _entry(id="via-instance")
    saved = add_user_model(entry, path=path)
    assert saved.id == "via-instance"
    assert load_user(path)[0].id == "via-instance"


def test_add_user_model_invalid_entry_raises_validation_error(tmp_path):
    path = str(tmp_path / "user-models.json")
    with pytest.raises(ValidationError):
        add_user_model({"name": "no id"}, path=path)


def test_add_user_model_is_atomic_no_tmp_file_left(tmp_path):
    path = str(tmp_path / "user-models.json")
    add_user_model({"id": "atomic1", "name": "A", "engine": "llamacpp", "path": "~/a.gguf"}, path=path)
    leftovers = [f for f in os.listdir(tmp_path) if f != "user-models.json"]
    assert leftovers == []


def test_remove_user_model_removes_existing(tmp_path):
    path = str(tmp_path / "user-models.json")
    add_user_model({"id": "rm1", "name": "A", "engine": "llamacpp", "path": "~/a.gguf"}, path=path)

    result = remove_user_model("rm1", path=path)
    assert result is True
    assert load_user(path) == []


def test_remove_user_model_missing_id_returns_false(tmp_path):
    path = str(tmp_path / "user-models.json")
    add_user_model({"id": "keep1", "name": "A", "engine": "llamacpp", "path": "~/a.gguf"}, path=path)

    result = remove_user_model("does-not-exist", path=path)
    assert result is False
    assert len(load_user(path)) == 1


def test_remove_user_model_never_touches_vendor_file(tmp_path):
    # ยืนยันว่า remove_user_model ไม่มีการอ้าง path ของ vendor เลย — เรียกโดยไม่มี vendor ในเทสต์นี้
    # ก็ยังทำงานได้ปกติ เพราะไม่มีการอ่าน/เขียน catalog.yaml
    path = str(tmp_path / "user-models.json")
    assert remove_user_model("anything", path=path) is False


# ---------------------------------------------------------------------------
# expand / shard_paths / is_ready / disk_bytes — ตรรกะไฟล์บนดิสก์
# ---------------------------------------------------------------------------


def test_expand_expands_user_home():
    entry = _entry(path="~/models/x.gguf")
    assert expand(entry) == os.path.expanduser("~/models/x.gguf")


def test_shard_paths_single_file_not_sharded(tmp_path):
    entry = _entry(path=str(tmp_path / "solo.gguf"))
    assert shard_paths(entry) == [str(tmp_path / "solo.gguf")]


def test_shard_paths_sharded_generates_all_shards(tmp_path):
    entry = _entry(path=str(tmp_path / "big-00001-of-00003.gguf"))
    result = shard_paths(entry)
    assert result == [
        str(tmp_path / "big-00001-of-00003.gguf"),
        str(tmp_path / "big-00002-of-00003.gguf"),
        str(tmp_path / "big-00003-of-00003.gguf"),
    ]


def _touch(p, size=10):
    p.write_bytes(b"x" * size)


def test_is_ready_sharded_all_present(tmp_path):
    for i in (1, 2, 3):
        _touch(tmp_path / f"big-{i:05d}-of-00003.gguf")
    entry = _entry(path=str(tmp_path / "big-00001-of-00003.gguf"))
    assert is_ready(entry) is True


def test_is_ready_sharded_missing_middle_shard(tmp_path):
    _touch(tmp_path / "big-00001-of-00003.gguf")
    _touch(tmp_path / "big-00003-of-00003.gguf")
    # 00002 หายไป
    entry = _entry(path=str(tmp_path / "big-00001-of-00003.gguf"))
    assert is_ready(entry) is False


def test_is_ready_aria2_marker_means_not_ready(tmp_path):
    for i in (1, 2, 3):
        _touch(tmp_path / f"big-{i:05d}-of-00003.gguf")
    # ไฟล์โหลดยังไม่จบ — มี .aria2 ค้างอยู่คู่กับ shard แรก
    (tmp_path / "big-00001-of-00003.gguf.aria2").write_bytes(b"")
    entry = _entry(path=str(tmp_path / "big-00001-of-00003.gguf"))
    assert is_ready(entry) is False


def test_is_ready_missing_args_referenced_file(tmp_path):
    _touch(tmp_path / "main.gguf")
    # args อ้างไฟล์ mtp ที่ยังไม่มีอยู่จริง
    entry = _entry(path=str(tmp_path / "main.gguf"), args=f"-md {tmp_path / 'mtp.gguf'} --spec-type draft-mtp")
    assert is_ready(entry) is False


def test_is_ready_true_when_main_and_args_files_present(tmp_path):
    _touch(tmp_path / "main.gguf")
    _touch(tmp_path / "mtp.gguf")
    entry = _entry(path=str(tmp_path / "main.gguf"), args=f"-md {tmp_path / 'mtp.gguf'} --spec-type draft-mtp")
    assert is_ready(entry) is True


def test_is_ready_folder_engine_with_files(tmp_path):
    d = tmp_path / "vllm-model"
    d.mkdir()
    _touch(d / "config.json")
    entry = _entry(engine="vllm", path=str(d), args="")
    assert is_ready(entry) is True


def test_is_ready_folder_engine_empty_dir(tmp_path):
    d = tmp_path / "vllm-model-empty"
    d.mkdir()
    entry = _entry(engine="vllm", path=str(d), args="")
    assert is_ready(entry) is False


def test_is_ready_folder_engine_missing_dir(tmp_path):
    d = tmp_path / "does-not-exist"
    entry = _entry(engine="vllm", path=str(d), args="")
    assert is_ready(entry) is False


def test_disk_bytes_sharded_sums_all_shards(tmp_path):
    for i in (1, 2, 3):
        _touch(tmp_path / f"big-{i:05d}-of-00003.gguf", size=100)
    entry = _entry(path=str(tmp_path / "big-00001-of-00003.gguf"))
    assert disk_bytes(entry) == 300


def test_disk_bytes_missing_shard_counts_only_existing(tmp_path):
    _touch(tmp_path / "big-00001-of-00003.gguf", size=100)
    _touch(tmp_path / "big-00003-of-00003.gguf", size=100)
    entry = _entry(path=str(tmp_path / "big-00001-of-00003.gguf"))
    assert disk_bytes(entry) == 200


def test_disk_bytes_includes_args_referenced_files(tmp_path):
    _touch(tmp_path / "main.gguf", size=100)
    _touch(tmp_path / "mtp.gguf", size=50)
    entry = _entry(path=str(tmp_path / "main.gguf"), args=f"-md {tmp_path / 'mtp.gguf'} --spec-type draft-mtp")
    assert disk_bytes(entry) == 150


def test_disk_bytes_folder_engine_recursive(tmp_path):
    d = tmp_path / "vllm-model"
    (d / "sub").mkdir(parents=True)
    _touch(d / "config.json", size=10)
    _touch(d / "sub" / "weights.bin", size=200)
    entry = _entry(engine="vllm", path=str(d), args="")
    assert disk_bytes(entry) == 210


def test_disk_bytes_missing_folder_is_zero(tmp_path):
    entry = _entry(engine="vllm", path=str(tmp_path / "nope"), args="")
    assert disk_bytes(entry) == 0


# ---------------------------------------------------------------------------
# lint — ต้องจับปัญหาจริงและคืนข้อความที่อธิบายได้ (ไม่ใช่แค่เช็คว่า list ไม่ว่าง)
# ---------------------------------------------------------------------------


def test_lint_clean_entries_return_empty_list():
    entries = [_entry(id="a"), _entry(id="b", path="~/models/b.gguf")]
    assert lint(entries) == []


def test_lint_catches_duplicate_id():
    entries = [_entry(id="dup"), _entry(id="dup", name="Different name")]
    problems = lint(entries)
    assert any("dup" in p for p in problems)


def test_lint_catches_unknown_engine():
    entries = [_entry(id="bad-engine", engine="foo")]
    problems = lint(entries)
    assert any("bad-engine" in p and "engine" in p for p in problems)


def test_lint_catches_forbidden_dash_c_in_args():
    entries = [_entry(id="has-c", args="-c 4096")]
    problems = lint(entries)
    assert any("has-c" in p and "-c" in p for p in problems)


def test_lint_does_not_false_positive_on_dash_c_substring():
    # "--spec-type" และ "-md" มี "c" อยู่ในตัวอักษรแต่ไม่ใช่ flag -c โดด ๆ ห้ามเตือน
    entries = [_entry(id="ok-args", args="-md ~/x/mtp.gguf --spec-type draft-mtp")]
    assert lint(entries) == []


def test_lint_catches_ctx_over_ctx_train():
    entries = [_entry(id="ctx-over", ctx=200000, ctx_train=100000)]
    problems = lint(entries)
    assert any("ctx-over" in p for p in problems)


def test_lint_ctx_equal_ctx_train_is_ok():
    entries = [_entry(id="ctx-eq", ctx=100000, ctx_train=100000)]
    assert lint(entries) == []


def test_lint_warns_on_dl_basename_mismatch():
    entries = [
        _entry(
            id="dl-mismatch",
            path="~/models/expected-00001-of-00002.gguf",
            dl=["https://huggingface.co/x/y/resolve/main/WRONG-NAME-00001-of-00002.gguf"],
        )
    ]
    problems = lint(entries)
    assert any("dl-mismatch" in p for p in problems)


def test_lint_dl_basename_matching_is_ok():
    entries = [
        _entry(
            id="dl-ok",
            path="~/models/expected-00001-of-00002.gguf",
            dl=[
                "https://huggingface.co/x/y/resolve/main/expected-00001-of-00002.gguf",
                "https://huggingface.co/x/y/resolve/main/expected-00002-of-00002.gguf",
            ],
        )
    ]
    assert lint(entries) == []


def test_lint_skips_dl_check_for_folder_engine():
    entries = [
        _entry(
            id="vllm-dl",
            engine="vllm",
            path="~/models/some-vllm-folder",
            dl=["https://huggingface.co/x/y/resolve/main/config.json"],
        )
    ]
    assert lint(entries) == []


# ---------------------------------------------------------------------------
# pydantic ต้อง reject entry ที่ field บังคับหาย
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing", ["id", "name", "engine", "path"])
def test_model_entry_requires_mandatory_fields(missing):
    data = {
        "id": "x",
        "name": "X",
        "engine": "llamacpp",
        "path": "~/x.gguf",
    }
    del data[missing]
    with pytest.raises(ValidationError):
        ModelEntry(**data)


def test_model_entry_defaults():
    entry = _entry()
    assert entry.args == ""
    assert entry.source == "vendor"
    assert entry.arch is None
    assert entry.dl is None
