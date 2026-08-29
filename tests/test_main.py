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

from server import catalog, engines, gguf, hf, instances, main, software


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
# /api/metrics — proxy ไปหา DGX Spark Monitor (:9100 ไม่เปิด CORS)
# ---------------------------------------------------------------------------


def _patch_metrics_client(monkeypatch, handler):
    """แทนที่ httpx.Client() ที่ main._fetch_metrics() สร้างเอง (ไม่มี client ส่งเข้ามา ตอนเรียกผ่าน route จริง)
    ด้วยตัวที่ผูก MockTransport ไว้ — กันยิงเน็ตจริงไป :9100
    """
    real_client_cls = httpx.Client

    def fake_client(*args, **kwargs):
        return real_client_cls(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(main.httpx, "Client", fake_client)


def test_metrics_ok(client, monkeypatch):
    def handler(request):
        return httpx.Response(200, json={
            "ok": True, "ts": 1787981764,
            "gauges": [
                {"label": "GPU อุณหภูมิ", "value": 55.3, "max": 90, "unit": "°C", "warn": 75},
                {"label": "RAM", "value": 42.1, "max": 121.7, "unit": "GB", "warn": 109.53},
            ],
            "services": {}, "model": None, "stale": False,
        })

    _patch_metrics_client(monkeypatch, handler)

    resp = client.get("/api/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["ts"] == 1787981764
    assert len(body["gauges"]) == 2
    assert body["gauges"][0]["label"] == "GPU อุณหภูมิ"
    assert body["gauges"][1]["unit"] == "GB"


def test_metrics_connect_error_returns_ok_false_ไม่_raise(client, monkeypatch):
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    _patch_metrics_client(monkeypatch, handler)

    resp = client.get("/api/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert body["gauges"] == []
    assert "9100" in body["detail"]


def test_metrics_ตอบ_json_เพี้ยน_returns_ok_false(client, monkeypatch):
    def handler(request):
        return httpx.Response(200, content=b"<html>not json</html>")

    _patch_metrics_client(monkeypatch, handler)

    resp = client.get("/api/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert body["gauges"] == []


def test_metrics_ตอบ_json_แต่ไม่ใช่_object_returns_ok_false(client, monkeypatch):
    def handler(request):
        return httpx.Response(200, json=[1, 2, 3])

    _patch_metrics_client(monkeypatch, handler)

    resp = client.get("/api/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert body["gauges"] == []


def test_metrics_timeout_returns_ok_false(client, monkeypatch):
    def handler(request):
        raise httpx.TimeoutException("timed out", request=request)

    _patch_metrics_client(monkeypatch, handler)

    resp = client.get("/api/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False


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
    # repo_id ตรงรูปแบบ org/repo แต่ fetch_repo ไม่เจอจริง ⇒ ตกไปค้นหาสำรอง
    # ถ้าค้นแล้วไม่เจอด้วย (suggestions ว่าง) ต้องตอบ 404 เหมือนเดิม — mock search_models ไว้ด้วย
    # กัน test นี้ยิงเน็ตจริงตอนตกไปเส้นทางค้นหา
    def fake_fetch_repo(repo_id, *, token=None, client=None):
        raise hf.RepoNotFoundError(f"ไม่พบ repo: {repo_id}")

    def fake_search_models(query, *, limit=12, token=None, client=None):
        return []

    monkeypatch.setattr(hf, "fetch_repo", fake_fetch_repo)
    monkeypatch.setattr(hf, "search_models", fake_search_models)

    resp = client.post("/api/models/resolve", json={"repo_id": "no/such-repo"})
    assert resp.status_code == 404


def test_resolve_github_link_returns_suggestions_instead_of_404(client, monkeypatch):
    """วางลิงก์ GitHub (ไม่ใช่ HF) ลงช่อง repo id → normalize_repo_id คืน repo_id เป็น None
    ⇒ ต้องตอบ 200 พร้อม suggestions จากการค้นหา ไม่ใช่ 404 (mock search_models กันยิงเน็ตจริง)
    """
    fake_hits = [
        hf.SearchHit(
            id="openai/gpt-oss-20b", downloads=1000, likes=50,
            gated=False, is_gguf=False, pipeline_tag="text-generation",
        ),
        hf.SearchHit(
            id="unsloth/gpt-oss-20b-GGUF", downloads=200, likes=10,
            gated=False, is_gguf=True, pipeline_tag=None,
        ),
    ]

    def fake_search_models(query, *, limit=12, token=None, client=None):
        assert query == "gpt-oss"
        return fake_hits

    monkeypatch.setattr(hf, "search_models", fake_search_models)

    resp = client.post("/api/models/resolve", json={"repo_id": "https://github.com/openai/gpt-oss"})
    assert resp.status_code == 200
    body = resp.json()

    assert body["repo_id"] == "https://github.com/openai/gpt-oss"
    assert body["resolved_repo_id"] is None
    assert body["quants"] == []
    assert body["query"] == "gpt-oss"
    assert len(body["suggestions"]) == 2
    assert body["suggestions"][0]["id"] == "openai/gpt-oss-20b"
    assert body["suggestions"][1]["is_gguf"] is True
    assert body["message"]


def test_resolve_full_hf_link_with_query_string_resolves_normally(client, monkeypatch):
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

    resp = client.post(
        "/api/models/resolve",
        json={"repo_id": "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF?utm_source=chatgpt.com"},
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["repo_id"] == "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF?utm_source=chatgpt.com"
    assert body["resolved_repo_id"] == "unsloth/Qwen3.8-27B-GGUF"
    assert body["suggestions"] == []
    assert body["message"] is None
    assert len(body["quants"]) > 0


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


# ---------------------------------------------------------------------------
# /api/instances
# ---------------------------------------------------------------------------


def test_list_instances(client, monkeypatch):
    fake = [
        instances.Instance(
            port=8001, engine="llamacpp", pid=123, model_file="m.gguf",
            model_path="/home/dgx/models/m.gguf", ctx=262144, rss_gb=2.5, up=True,
        ),
    ]
    monkeypatch.setattr(instances, "scan", lambda **kw: fake)

    resp = client.get("/api/instances")
    assert resp.status_code == 200
    body = resp.json()["instances"]
    assert len(body) == 1
    assert body[0]["port"] == 8001
    assert body[0]["engine"] == "llamacpp"
    assert body[0]["model_file"] == "m.gguf"
    assert body[0]["ctx"] == 262144
    assert body[0]["up"] is True


def test_list_instances_ไม่มี_instance_คืนลิสต์ว่าง(client, monkeypatch):
    monkeypatch.setattr(instances, "scan", lambda **kw: [])
    resp = client.get("/api/instances")
    assert resp.status_code == 200
    assert resp.json() == {"instances": []}


def test_stop_instance_port_8000_ไม่ส่ง_allow_main_port_ต้อง_409(client, monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(instances, "stop", lambda port: called.__setitem__("n", called["n"] + 1) or (True, "หยุดแล้ว"))

    resp = client.post("/api/instances/8000/stop", json={})
    assert resp.status_code == 409
    assert "8000" in resp.json()["detail"]
    assert called["n"] == 0  # ต้องปฏิเสธก่อนเรียก stop() จริง


def test_stop_instance_port_8000_ไม่ส่ง_body_เลยก็ต้อง_409(client, monkeypatch):
    monkeypatch.setattr(instances, "stop", lambda port: (True, "หยุดแล้ว"))
    resp = client.post("/api/instances/8000/stop")
    assert resp.status_code == 409


def test_stop_instance_port_8000_ส่ง_allow_main_port_ผ่านเข้าไปเรียก_stop(client, monkeypatch):
    calls = []
    monkeypatch.setattr(instances, "stop", lambda port: calls.append(port) or (True, "หยุดแล้ว"))
    monkeypatch.setattr(software, "mem_available_gb", lambda: None)

    resp = client.post("/api/instances/8000/stop", json={"allow_main_port": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert calls == [8000]


def test_stop_instance_พอร์ตนอกช่วง_8000_8009_คืน_400(client, monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(instances, "stop", lambda port: called.__setitem__("n", called["n"] + 1) or (True, "หยุดแล้ว"))

    resp = client.post("/api/instances/9999/stop", json={})
    assert resp.status_code == 400
    assert called["n"] == 0


def test_stop_instance_พอร์ตธรรมดา_สำเร็จ(client, monkeypatch):
    monkeypatch.setattr(instances, "stop", lambda port: (True, f"หยุด instance บนพอร์ต {port} แล้ว"))
    monkeypatch.setattr(software, "mem_available_gb", lambda: None)

    resp = client.post("/api/instances/8001/stop", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert "8001" in body["message"]


def test_stop_instance_ไม่มีอะไรบนพอร์ต_คืน_ok_false(client, monkeypatch):
    monkeypatch.setattr(instances, "stop", lambda port: (False, f"ไม่มี instance บนพอร์ต {port} อยู่แล้ว"))

    resp = client.post("/api/instances/8001/stop", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert "ไม่มี instance" in body["message"]


def test_stop_instance_แสดงแรมที่คืนได้(client, monkeypatch):
    monkeypatch.setattr(instances, "stop", lambda port: (True, "หยุด instance บนพอร์ต 8001 แล้ว"))
    values = iter([10.0, 25.0])  # ก่อน 10GB ว่าง หลัง 25GB ว่าง → คืนแรมมาราว 15GB
    monkeypatch.setattr(software, "mem_available_gb", lambda: next(values))

    resp = client.post("/api/instances/8001/stop", json={})
    assert resp.status_code == 200
    assert "15 GB" in resp.json()["message"]


# --- draft model โหลดไม่ขึ้น → ลองใหม่โดยถอด draft ออก (เจอจริง 2026-08-29) -----


def test_strip_draft_ตัด_md_และ_spec_type_แต่เก็บ_mmproj():
    from server.main import _strip_draft, _has_draft, _draft_failed

    args = "-md ~/m/FastMTP-32K.gguf --mmproj ~/m/mmproj.gguf --spec-type draft-mtp"
    out = _strip_draft(args)

    assert "-md" not in out
    assert "--spec-type" not in out
    assert out == "--mmproj ~/m/mmproj.gguf"
    assert _has_draft(args) and not _has_draft(out)


def test_draft_failed_จับข้อความจริงของ_llamacpp():
    from server.main import _draft_failed

    real_log = (
        "E llama_model_load: error loading model: check_tensor_dims: tensor "
        "'output.weight' has wrong shape; expected 5120, 248320, got 5120, 32768\n"
        "E srv load_model: failed to load draft model, '/home/dgx/models/.../FastMTP-32K.gguf'"
    )

    assert _draft_failed(real_log)
    assert not _draft_failed("I srv load_model: loaded multimodal model")


def test_has_draft_ไม่จับ_flag_อื่นที่ขึ้นต้นคล้ายกัน():
    from server.main import _has_draft

    assert not _has_draft("--mmproj ~/x.gguf")
    assert not _has_draft("--model-draft-something ~/x.gguf")
    assert _has_draft("-md ~/x.gguf")


# --- log ดิบดูเหมือน error ทั้งที่โหลดสำเร็จ (เจอจริง 2026-08-29) ---------------


_REAL_LOG = """0.01.019.478 W model has unused tensor blk.64.attn_output.weight (size = 33423360 bytes) -- ignoring
0.01.019.480 W model has unused tensor blk.64.attn_q_norm.weight (size = 1024 bytes) -- ignoring
0.05.685.637 W load_hparams: Qwen-VL models require at minimum 1024 image tokens to function correctly
0.05.851.919 I srv    load_model: loaded multimodal model, '/home/dgx/models/gguf/x/mmproj.gguf'
0.06.182.911 I srv    load_model: initializing, n_slots = 4, n_ctx_slot = 262144, kv_unified = 'true'
0.06.185.885 I srv  llama_server: model loaded
0.06.185.888 I srv  llama_server: listening on http://0.0.0.0:8001"""


def test_summary_ดึงค่าที่ผู้ใช้ต้องรู้จาก_log_จริง():
    from server.main import _activate_summary

    s = _activate_summary(_REAL_LOG)

    assert s["ctx"] == 262144
    assert s["slots"] == 4
    assert s["multimodal"] is True


def test_summary_ทิ้ง_unused_tensor_แต่เก็บ_warning_ที่สำคัญ():
    """unused tensor = MTP layer ที่ engine ไม่ได้ใช้ · เตือนสิบกว่าบรรทัดจนดูเหมือนพัง"""
    from server.main import _activate_summary

    warnings = _activate_summary(_REAL_LOG)["warnings"]

    assert not any("unused tensor" in w for w in warnings)
    assert any("image tokens" in w for w in warnings), "warning ที่มีประโยชน์ต้องไม่ถูกทิ้ง"


def test_summary_log_ที่ไม่มีอะไรเลย_ไม่พัง():
    from server.main import _activate_summary

    s = _activate_summary("")

    assert s == {"ctx": None, "slots": None, "multimodal": False, "warnings": []}


def test_metrics_ส่ง_field_เสริมของ_gauge_ต่อไปครบ(monkeypatch):
    """dgx-monitor ส่ง sub มาบางตัว (ดิสก์ = 'เหลือ 2.7 TB') — ห้ามกรองทิ้งระหว่างทาง"""
    import httpx
    from server import main as m

    payload = {"ok": True, "ts": 1, "gauges": [
        {"label": "ดิสก์", "value": 31.4, "max": 100, "unit": "%", "warn": 85, "sub": "เหลือ 2.7 TB"},
    ]}
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json=payload))
    monkeypatch.setattr(m.httpx, "Client", lambda **kw: httpx.Client(transport=transport, **kw))

    out = m._fetch_metrics()

    assert out["ok"] is True
    assert out["gauges"][0]["sub"] == "เหลือ 2.7 TB"
