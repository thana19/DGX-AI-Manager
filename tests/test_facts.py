"""test สำหรับ server/facts.py + overlay model-facts.json ใน catalog.load_all — ตรวจสอบ arch/ctx_train ของโมเดลใหม่"""
from __future__ import annotations

import json
import os
import shutil

import pytest
import yaml

from server import catalog, facts

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _write_vendor(tmp_path, models):
    p = tmp_path / "catalog.yaml"
    p.write_text(yaml.safe_dump({"models": models}), encoding="utf-8")
    return str(p)


def _vendor_entry(**kw):
    d = {"id": "m1", "name": "M1", "engine": "llamacpp", "path": "~/x/m1.gguf"}
    d.update(kw)
    return d


# --- overlay ใน load_all ---


def test_overlay_fills_only_none_fields(tmp_path):
    vendor = _write_vendor(tmp_path, [_vendor_entry(), _vendor_entry(id="m2", arch="keep", ctx_train=100)])
    facts_p = tmp_path / "facts.json"
    facts_p.write_text(json.dumps({
        "m1": {"arch": "qwen35", "ctx_train": 4096},
        "m2": {"arch": "OVERRIDE", "ctx_train": 999},
    }), encoding="utf-8")
    out = {e.id: e for e in catalog.load_all(vendor, str(tmp_path / "none.json"), str(facts_p))}
    assert out["m1"].arch == "qwen35" and out["m1"].ctx_train == 4096
    assert out["m1"].source == "vendor"
    assert out["m2"].arch == "keep" and out["m2"].ctx_train == 100


@pytest.mark.parametrize("content", ["{not json", "[1,2]", '{"m1": "x"}'])
def test_overlay_corrupt_ignored(tmp_path, content):
    vendor = _write_vendor(tmp_path, [_vendor_entry()])
    facts_p = tmp_path / "facts.json"
    facts_p.write_text(content, encoding="utf-8")
    out = catalog.load_all(vendor, str(tmp_path / "none.json"), str(facts_p))
    assert out[0].arch is None


def test_overlay_missing_file_ok(tmp_path):
    vendor = _write_vendor(tmp_path, [_vendor_entry()])
    out = catalog.load_all(vendor, str(tmp_path / "none.json"), str(tmp_path / "nofacts.json"))
    assert out[0].arch is None


def test_save_fact_merges(tmp_path):
    p = str(tmp_path / "facts.json")
    catalog.save_fact("a", {"arch": "x", "ctx_train": 1}, p)
    catalog.save_fact("b", {"arch": "y", "ctx_train": None}, p)
    data = json.load(open(p, encoding="utf-8"))
    assert data["a"] == {"arch": "x", "ctx_train": 1}
    assert data["b"]["arch"] == "y"


# --- inspect local ---


def test_inspect_gguf_local(tmp_path):
    f = tmp_path / "m.gguf"
    shutil.copy(os.path.join(FIXTURES, "gguf_head_qwen35_262144.bin"), f)
    e = catalog.ModelEntry(id="g", name="g", engine="llamacpp", path=str(f))
    assert facts.inspect_local(e) == ("qwen35", 262144)


def test_inspect_gguf_local_not_gguf(tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"x" * 100)
    e = catalog.ModelEntry(id="g", name="g", engine="llamacpp", path=str(f))
    with pytest.raises(facts.InspectError):
        facts.inspect_local(e)


def test_inspect_vllm_config_local(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    shutil.copy(os.path.join(FIXTURES, "config_qwen3_fp8.json"), d / "config.json")
    (d / "w.safetensors").write_bytes(b"0")
    e = catalog.ModelEntry(id="v", name="v", engine="vllm", path=str(d))
    arch, ctx = facts.inspect_local(e)
    assert arch and ctx


def test_inspect_vllm_missing_config(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    e = catalog.ModelEntry(id="v", name="v", engine="vllm", path=str(d))
    with pytest.raises(facts.InspectError):
        facts.inspect_local(e)
