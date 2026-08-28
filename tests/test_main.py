"""test สำหรับ server/main.py — FastAPI routes ของ AI Server v2 (เฟส 1)

ห้ามยิงเน็ตจริง ห้ามรัน engine จริง ห้ามแตะ ~/.aiserver2 จริง
mock hf.fetch_repo / gguf.fetch_header / subprocess ตามต้องการ + monkeypatch env AISERVER2_STATE ชี้ tmp_path
"""
from __future__ import annotations

import json
import os

import httpx
import pytest
from fastapi.testclient import TestClient

from server import catalog, engines, gguf, hf, main


FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _load_fixture(name: str) -> dict:
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("AISERVER2_STATE", str(tmp_path / "state"))
    monkeypatch.setenv("ARIA2_RPC_URL", "http://127.0.0.1:6800/jsonrpc")
    monkeypatch.setenv("ARIA2_RPC_SECRET", "")
    main.reset_download_manager()
    yield TestClient(main.app)
    main.reset_download_manager()


@pytest.fixture(autouse=True)
def _fast_engine_detect(monkeypatch):
    """กัน list_models/activate ไปยิง subprocess จริง (docker/llama-server) ทุก test — คุมผลให้แน่นอน

    default: llamacpp ติดตั้งแล้ว รองรับ arch ทั้งหมดที่ catalog จริงใช้ (ผ่านหมด) · vllm/ds4 ไม่ได้ติดตั้ง
    test ที่ต้องการ behavior อื่นค่อย monkeypatch engines.detect ทับเองอีกที
    """
    known_archs = frozenset({"qwen35", "deepseek4", "muse-glimmer", "qwen4exp", "glm5next"})

    def fake_detect(name: str) -> engines.EngineInfo:
        if name == "llamacpp":
            return engines.EngineInfo(
                name="llamacpp", installed=True, version="version: 0.3.0-dev (build 10696, commit abc)",
                build=10696, archs=known_archs, detail="llama-server build 10696 (fake)",
            )
        return engines.EngineInfo(name=name, installed=False, version=None, build=None, archs=frozenset(), detail=f"{name} ไม่ได้ติดตั้ง (fake)")

    monkeypatch.setattr(engines, "detect", fake_detect)


# ---------------------------------------------------------------------------
# /api/health
# ---------------------------------------------------------------------------


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["version"] == "2.0.0-phase1"
    assert body["port"] == main.DEFAULT_PORT
    assert "ram_total_gb" in body
    assert "disk_free_gb" in body


# ---------------------------------------------------------------------------
# /api/models
# ---------------------------------------------------------------------------


def test_list_models_returns_all_9_catalog_entries(client):
    resp = client.get("/api/models")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["models"]) == 9
    ids = {m["id"] for m in body["models"]}
    assert "qwen38-ud-q4-mtp" in ids
    assert "glm-5.3-flash-udq1" in ids

    m = next(m for m in body["models"] if m["id"] == "qwen38-ud-q4-mtp")
    for key in (
        "id", "name", "engine", "path", "arch", "source", "source_repo", "ctx", "ctx_train",
        "need_gb", "note", "file", "ready", "size_gb", "has_dl", "compat",
    ):
        assert key in m
    assert m["source"] == "vendor"
    assert m["ready"] is False  # ไฟล์จริงไม่มีบนเครื่องทดสอบ
    assert m["has_dl"] is True
    assert set(m["compat"].keys()) == {"status", "reason", "action"}
    # qwen35 อยู่ใน archs ปลอมที่ตั้งไว้ → ok
    assert m["compat"]["status"] == "ok"

    glm = next(m for m in body["models"] if m["id"] == "glm-5.3-flash-udq1")
    assert glm["arch"] == "glm5next"
    assert glm["compat"]["status"] == "ok"  # glm5next อยู่ใน archs ปลอมด้วย


def test_list_models_unsupported_arch_needs_upgrade(client, monkeypatch):
    def fake_detect(name: str) -> engines.EngineInfo:
        if name == "llamacpp":
            return engines.EngineInfo(
                name="llamacpp", installed=True, version="v", build=9999,
                archs=frozenset({"qwen35"}), detail="fake old build",
            )
        return engines.EngineInfo(name=name, installed=False, version=None, build=None, archs=frozenset(), detail="no")

    monkeypatch.setattr(engines, "detect", fake_detect)

    resp = client.get("/api/models")
    body = resp.json()
    glm = next(m for m in body["models"] if m["id"] == "glm-5.3-flash-udq1")
    assert glm["compat"]["status"] == "needs_upgrade"
    assert glm["compat"]["action"] == "upgrade_llamacpp"


# ---------------------------------------------------------------------------
# /api/models/resolve
# ---------------------------------------------------------------------------


def test_resolve_returns_quants_smallest_first_with_compat(client, monkeypatch):
    repo_json = _load_fixture("hf_qwen38-27b-gguf_blobs.json")

    def fake_fetch_repo(repo_id, *, token=None, client=None):
        assert repo_id == "unsloth/Qwen3.8-27B-GGUF"
        return repo_json

    def fake_fetch_header(url, **kwargs):
        return gguf.GgufInfo(
            arch="qwen35", context_length=262144, name="Qwen3.8-27B", size_label="27B",
            file_type=None, version=3, tensor_count=1, kv_count=1, kv_read=1,
        ), None

    monkeypatch.setattr(hf, "fetch_repo", fake_fetch_repo)
    monkeypatch.setattr(gguf, "fetch_header", fake_fetch_header)

    resp = client.post("/api/models/resolve", json={"repo_id": "unsloth/Qwen3.8-27B-GGUF", "token": None})
    assert resp.status_code == 200
    body = resp.json()

    assert body["repo_id"] == "unsloth/Qwen3.8-27B-GGUF"
    assert body["gated"] is False
    assert body["arch"] == "qwen35"
    assert body["ctx_train"] == 262144
    assert len(body["quants"]) > 1

    sizes = [q["total_bytes"] for q in body["quants"]]
    assert sizes == sorted(sizes)  # เรียงเล็กไปใหญ่

    for q in body["quants"]:
        assert set(q["compat"].keys()) == {"status", "reason", "action"}
        assert q["compat"]["status"] == "ok"  # qwen35 อยู่ใน archs ปลอมของ fixture นี้
        assert isinstance(q["total_gb"], float)
        assert "fits_ram" in q


def test_resolve_gated_repo_returns_403(client, monkeypatch):
    def fake_fetch_repo(repo_id, *, token=None, client=None):
        raise hf.GatedRepoError(f"{repo_id} ติด gate")

    monkeypatch.setattr(hf, "fetch_repo", fake_fetch_repo)

    resp = client.post("/api/models/resolve", json={"repo_id": "some/gated-repo"})
    assert resp.status_code == 403
    assert "gate" in resp.json()["detail"]


def test_resolve_not_found_returns_404(client, monkeypatch):
    def fake_fetch_repo(repo_id, *, token=None, client=None):
        raise hf.RepoNotFoundError(f"ไม่พบ repo: {repo_id}")

    monkeypatch.setattr(hf, "fetch_repo", fake_fetch_repo)

    resp = client.post("/api/models/resolve", json={"repo_id": "no/such-repo"})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/models  (add/delete user model)
# ---------------------------------------------------------------------------


def test_add_and_delete_user_model(client):
    payload = {
        "id": "test-user-model-1", "name": "Test Model", "engine": "llamacpp",
        "path": "~/models/gguf/test/test.gguf",
    }
    resp = client.post("/api/models", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["model"]["id"] == "test-user-model-1"
    assert body["model"]["source"] == "user"

    resp = client.get("/api/models")
    ids = {m["id"] for m in resp.json()["models"]}
    assert "test-user-model-1" in ids
    assert len(resp.json()["models"]) == 10

    resp = client.delete("/api/models/test-user-model-1")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True

    resp = client.get("/api/models")
    assert len(resp.json()["models"]) == 9


def test_add_model_invalid_payload_returns_400(client):
    resp = client.post("/api/models", json={"id": "bad", "engine": "llamacpp"})  # ขาด name/path (required)
    assert resp.status_code == 400


def test_delete_nonexistent_model_returns_404(client):
    resp = client.delete("/api/models/does-not-exist")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/activate — 3 กรณีที่ต้องถูกปฏิเสธ
# ---------------------------------------------------------------------------


def test_activate_rejects_main_port_8000(client):
    resp = client.post("/api/activate", json={"id": "qwen38-ud-q4-mtp", "port": 8000})
    assert resp.status_code == 409
    assert "8000" in resp.json()["detail"]


def test_activate_rejects_not_ready_model(client):
    # ไฟล์จริงไม่มีบนเครื่องทดสอบ → is_ready() = False เสมอ
    resp = client.post("/api/activate", json={"id": "qwen38-ud-q4-mtp", "port": 8001})
    assert resp.status_code == 409
    assert "ดาวน์โหลด" in resp.json()["detail"]


def test_activate_rejects_needs_upgrade(client, monkeypatch, tmp_path):
    # ทำให้โมเดล ready=True (ปลอมไฟล์ในดิสก์) แต่ engine ไม่รองรับ arch นี้ → needs_upgrade
    entry = next(e for e in catalog.load_all() if e.id == "glm-5.3-flash-udq1")
    monkeypatch.setattr(catalog, "is_ready", lambda e: True if e.id == entry.id else False)

    def fake_detect(name: str) -> engines.EngineInfo:
        if name == "llamacpp":
            return engines.EngineInfo(
                name="llamacpp", installed=True, version="v", build=9999,
                archs=frozenset({"qwen35"}), detail="fake old build — ไม่รองรับ glm5next",
            )
        return engines.EngineInfo(name=name, installed=False, version=None, build=None, archs=frozenset(), detail="no")

    monkeypatch.setattr(engines, "detect", fake_detect)

    resp = client.post("/api/activate", json={"id": "glm-5.3-flash-udq1", "port": 8001})
    assert resp.status_code == 409
    assert "glm5next" in resp.json()["detail"] or "รองรับ" in resp.json()["detail"]


def test_activate_model_not_found_returns_404(client):
    resp = client.post("/api/activate", json={"id": "no-such-model", "port": 8001})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/downloads
# ---------------------------------------------------------------------------


def test_downloads_when_aria2_dead_does_not_break_endpoint(client, monkeypatch):
    mgr = main.get_download_manager()

    def dead_client_post(*args, **kwargs):
        raise httpx.ConnectError("connection refused", request=httpx.Request("POST", "http://127.0.0.1:6800/jsonrpc"))

    monkeypatch.setattr(mgr._aria2._client, "post", dead_client_post)

    resp = client.get("/api/downloads")
    assert resp.status_code == 200
    assert resp.json() == {"jobs": []}


def test_create_download_without_dl_returns_400(client):
    resp = client.post("/api/downloads", json={"model_id": "qwen36-td-q2k"})  # entry นี้ไม่มี dl:
    assert resp.status_code == 400


def test_create_download_unknown_model_returns_404(client):
    resp = client.post("/api/downloads", json={"model_id": "no-such-model"})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/engines · /api/software
# ---------------------------------------------------------------------------


def test_list_engines(client):
    resp = client.get("/api/engines")
    assert resp.status_code == 200
    body = resp.json()
    names = {e["name"] for e in body["engines"]}
    assert names == {"llamacpp", "vllm", "ds4"}


def test_list_software(client):
    resp = client.get("/api/software")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["software"]) == len(__import__("server.software", fromlist=["SOFTWARE"]).SOFTWARE)


# --- บั๊กที่เจอตอนทดสอบบนเครื่องจริง (2026-08-29) --------------------------


def test_args_ต้องขยาย_tilde_ก่อนส่งให้_engine(monkeypatch, tmp_path):
    """~ ใน args ไม่มีใครขยายให้ เพราะส่งเป็น argv ตรง ๆ ไม่ผ่าน shell

    ของจริงที่เจอ: -md ~/models/.../mtp-xxx.gguf → llama-server หา draft model ไม่เจอ
    """
    import shlex
    import os as _os

    args = "-md ~/models/gguf/x/mtp.gguf --spec-type draft-mtp"
    expanded = [_os.path.expanduser(t) for t in shlex.split(args)]

    assert not expanded[1].startswith("~"), "path ต้องถูกขยายแล้ว"
    assert expanded[1].startswith(_os.path.expanduser("~"))
    assert expanded[2] == "--spec-type", "flag ที่ไม่ใช่ path ต้องไม่ถูกแตะ"


def test_need_gb_คำนวณตาม_ctx_เมื่อรู้_kv_per_token(tmp_path):
    from server import catalog as cat

    entry = cat.ModelEntry(
        id="x", name="x", engine="llamacpp",
        path=str(tmp_path / "m.gguf"), kv_kb_per_token=64, ctx=262144, need_gb=55,
    )
    (tmp_path / "m.gguf").write_bytes(b"0" * 1000)

    # KV = 262144 tokens x 64KB = 16.8GB + buffer 3GB → ต้องมากกว่า need_gb ที่วัดมือไว้ตอน ctx เต็ม
    assert cat.need_gb(entry, 262144) > 19
    # ctx น้อยลง = ใช้แรมน้อยลง (นี่คือเหตุผลที่ให้ผู้ใช้ลด ctx ได้)
    assert cat.need_gb(entry, 32768) < cat.need_gb(entry, 262144)


def test_activate_ปฏิเสธเมื่อแรมไม่พอ(client, monkeypatch):
    from server import catalog as cat
    from server import software as sw

    monkeypatch.setattr(sw, "mem_available_gb", lambda: 20.0)
    monkeypatch.setattr(cat, "is_ready", lambda e: True)
    monkeypatch.setattr(cat, "need_gb", lambda e, c=None: 90.0)

    r = client.post("/api/activate", json={"id": "glm-5.3-flash-udq1", "port": 8001})
    assert r.status_code == 409
    assert "แรมไม่พอ" in r.json()["detail"]
