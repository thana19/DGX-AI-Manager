"""test สำหรับ server/main.py — FastAPI routes ของ AI Server v2 (เฟส 1)

ห้ามยิงเน็ตจริง ห้ามรัน engine จริง ห้ามแตะ ~/.aiserver2 จริง
mock hf.fetch_repo / gguf.fetch_header / subprocess ตามต้องการ + monkeypatch env AISERVER2_STATE ชี้ tmp_path
"""
from __future__ import annotations

import json
import os
import subprocess

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


def test_resolve_gguf_response_shape_unchanged_plus_new_keys(client, monkeypatch):
    """เส้นทาง GGUF ต้องตอบเหมือนเดิมทุกประการ บวก engine/format/requires/engine_env ใหม่"""
    repo_json = _load_fixture("hf_qwen38-27b-gguf_blobs.json")

    def fake_fetch_repo(repo_id, *, token=None, client=None):
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

    assert body["format"] == "gguf"
    assert body["engine"] == "llamacpp"
    assert body["requires"] is None
    assert body["engine_env"] is None
    assert len(body["quants"]) > 1


def test_resolve_safetensors_fp8_returns_vllm_engine(client, monkeypatch):
    repo_json = _load_fixture("hf_qwen3-8b-fp8_blobs.json")
    config = _load_fixture("config_qwen3_fp8.json")

    def fake_fetch_repo(repo_id, *, token=None, client=None):
        assert repo_id == "Qwen/Qwen3-8B-FP8"
        return repo_json

    def fake_fetch_config(repo_id, *, token=None, client=None):
        assert repo_id == "Qwen/Qwen3-8B-FP8"
        return config

    def fake_detect(name: str) -> engines.EngineInfo:
        if name == "vllm":
            return engines.EngineInfo(
                name="vllm", installed=True, version="aiserver-vllm:26.07",
                build=None, archs=frozenset(), detail="fake vllm installed",
            )
        return engines.EngineInfo(name=name, installed=False, version=None, build=None, archs=frozenset(), detail="no")

    monkeypatch.setattr(hf, "fetch_repo", fake_fetch_repo)
    monkeypatch.setattr(hf, "fetch_config", fake_fetch_config)
    monkeypatch.setattr(engines, "detect", fake_detect)

    resp = client.post("/api/models/resolve", json={"repo_id": "Qwen/Qwen3-8B-FP8", "token": None})
    assert resp.status_code == 200
    body = resp.json()

    assert body["resolved_repo_id"] == "Qwen/Qwen3-8B-FP8"
    assert body["format"] == "safetensors"
    assert body["engine"] == "vllm"
    assert body["arch"] == "Qwen3ForCausalLM"
    assert body["ctx_train"] == 40960
    assert body["companions"] == []

    assert len(body["quants"]) == 1
    q = body["quants"][0]
    assert q["key"] == "FP8"
    assert q["compat"]["status"] == "ok"  # requires.vllm_image ตรงกับ info.version พอดี (PRD ข้อ 6)

    assert body["requires"] == {"vllm_image": "aiserver-vllm:26.07"}
    assert body["engine_env"]["VLLM_TOOL_PARSER"] == "qwen3_xml"
    assert body["engine_env"]["VLLM_REASONING_PARSER"] == "qwen3"
    assert body["engine_env"]["VLLM_SPECULATIVE"] == ""  # Qwen3-8B-FP8 ไม่มี model_mtp.safetensors


def test_resolve_gated_repo_with_token_saves_token_and_reads_header_with_it(client, monkeypatch):
    """repo gated (Navin-Models AD-4.27) + token ที่ผู้ใช้ใส่มา → gated: True, quants 2 ตัว
    (mainline/main — ดู task ส่วนที่ 1), fetch_header ถูกเรียกด้วย token นั้น, และ token ถูกจำไว้ใน state
    """
    repo_json = _load_fixture("hf_navin-qwen38-flash-next-ad427-gguf_blobs.json")
    seen_tokens = []

    def fake_fetch_repo(repo_id, *, token=None, client=None):
        assert token == "hf_mytoken"
        return repo_json

    def fake_fetch_header(url, **kwargs):
        seen_tokens.append(kwargs.get("token"))
        return gguf.GgufInfo(
            arch="qwen4exp", context_length=262144, name="Qwen3.8-Flash-Next", size_label=None,
            file_type=None, version=3, tensor_count=1, kv_count=1, kv_read=1,
        ), None

    monkeypatch.setattr(hf, "fetch_repo", fake_fetch_repo)
    monkeypatch.setattr(gguf, "fetch_header", fake_fetch_header)

    resp = client.post(
        "/api/models/resolve",
        json={"repo_id": "Navin-Models/Qwen3.8-Flash-Next-Uncensored-AD-4.27-GGUF", "token": "hf_mytoken"},
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["gated"] is True
    assert len(body["quants"]) == 2
    assert seen_tokens == ["hf_mytoken"]
    assert hf.load_token() == "hf_mytoken"


def test_resolve_gated_repo_without_token_and_header_fails_returns_message_not_500(client, monkeypatch):
    """repo gated ไม่มี token เลย (ไม่ได้ใส่มา และไม่เคยเซฟไว้ก่อน) แล้ว fetch_header ล้มเหลว (401)
    → arch/ctx_train เป็น None และมี message บอกให้ใส่ token — ห้าม 500
    """
    repo_json = _load_fixture("hf_navin-qwen38-flash-next-ad427-gguf_blobs.json")

    def fake_fetch_repo(repo_id, *, token=None, client=None):
        assert token is None
        return repo_json

    def fake_fetch_header(url, **kwargs):
        assert kwargs.get("token") is None
        raise httpx.HTTPStatusError("401", request=httpx.Request("GET", url), response=httpx.Response(401))

    monkeypatch.setattr(hf, "fetch_repo", fake_fetch_repo)
    monkeypatch.setattr(gguf, "fetch_header", fake_fetch_header)

    resp = client.post(
        "/api/models/resolve",
        json={"repo_id": "Navin-Models/Qwen3.8-Flash-Next-Uncensored-AD-4.27-GGUF"},
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["gated"] is True
    assert body["arch"] is None
    assert body["ctx_train"] is None
    assert body["message"]
    assert "token" in body["message"]


def test_resolve_safetensors_config_fetch_failure_degrades_gracefully(client, monkeypatch):
    """config.json อ่านไม่ได้ (network พัง/parse พัง) → arch/ctx เป็น None ไม่ใช่ 500"""
    repo_json = _load_fixture("hf_qwen3-8b-fp8_blobs.json")

    def fake_fetch_repo(repo_id, *, token=None, client=None):
        return repo_json

    def fake_fetch_config(repo_id, *, token=None, client=None):
        raise httpx.ConnectError("connection refused", request=httpx.Request("GET", "http://x"))

    def fake_detect(name: str) -> engines.EngineInfo:
        return engines.EngineInfo(name=name, installed=False, version=None, build=None, archs=frozenset(), detail="no")

    monkeypatch.setattr(hf, "fetch_repo", fake_fetch_repo)
    monkeypatch.setattr(hf, "fetch_config", fake_fetch_config)
    monkeypatch.setattr(engines, "detect", fake_detect)

    resp = client.post("/api/models/resolve", json={"repo_id": "Qwen/Qwen3-8B-FP8", "token": None})
    assert resp.status_code == 200
    body = resp.json()

    assert body["format"] == "safetensors"
    assert body["arch"] is None
    assert body["ctx_train"] is None
    assert len(body["quants"]) == 1
    assert body["quants"][0]["key"] == "safetensors"  # ไม่มี config เลย ⇒ fallback ของ quant_label({})


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


def test_activate_passes_engine_env_into_subprocess_env(client, monkeypatch, tmp_path):
    """engine_env ของ entry (parser/MTP ของ vLLM) ต้องถูกยัดเข้า env จริงตอนเรียก engine script"""
    model_dir = tmp_path / "vllm-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_bytes(b"{}")

    entry = catalog.ModelEntry(
        id="test-vllm-env", name="Test vLLM", engine="vllm", path=str(model_dir),
        engine_env={"VLLM_TOOL_PARSER": "qwen3_xml", "VLLM_REASONING_PARSER": "qwen3", "VLLM_SPECULATIVE": ""},
    )
    monkeypatch.setattr(catalog, "load_all", lambda *a, **kw: [entry])

    def fake_detect(name: str) -> engines.EngineInfo:
        if name == "vllm":
            return engines.EngineInfo(
                name="vllm", installed=True, version="aiserver-vllm:26.07",
                build=None, archs=frozenset(), detail="fake vllm installed",
            )
        return engines.EngineInfo(name=name, installed=False, version=None, build=None, archs=frozenset(), detail="no")

    monkeypatch.setattr(engines, "detect", fake_detect)

    captured: dict = {}

    def fake_run(argv, env=None, **kwargs):
        captured["argv"] = argv
        captured["env"] = env
        return subprocess.CompletedProcess(argv, 0, stdout="READY\n", stderr="")

    monkeypatch.setattr(main.subprocess, "run", fake_run)

    resp = client.post("/api/activate", json={"id": "test-vllm-env", "port": 8001})
    assert resp.status_code == 200

    env = captured["env"]
    assert env["VLLM_TOOL_PARSER"] == "qwen3_xml"
    assert env["VLLM_REASONING_PARSER"] == "qwen3"
    assert env["VLLM_SPECULATIVE"] == ""


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


def test_downloads_list_includes_created_at(client, monkeypatch):
    from server.downloads import DownloadJob, DownloadState

    mgr = main.get_download_manager()

    def dead_client_post(*args, **kwargs):
        raise httpx.ConnectError("connection refused", request=httpx.Request("POST", "http://127.0.0.1:6800/jsonrpc"))

    monkeypatch.setattr(mgr._aria2._client, "post", dead_client_post)

    job = DownloadJob(id="j1", model_id="m", files=[], created_at=123.0, state=DownloadState.DONE)
    mgr._jobs[job.id] = job

    resp = client.get("/api/downloads")
    assert resp.status_code == 200
    jobs = resp.json()["jobs"]
    assert len(jobs) == 1
    assert jobs[0]["created_at"] == 123.0


def test_create_download_without_dl_returns_400(client):
    resp = client.post("/api/downloads", json={"model_id": "qwen36-td-q2k"})  # entry นี้ไม่มี dl:
    assert resp.status_code == 400


def test_create_download_unknown_model_returns_404(client):
    resp = client.post("/api/downloads", json={"model_id": "no-such-model"})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/downloads — กัน submit repo ที่รู้อยู่แล้วว่าติด gate (hotfix)
# ---------------------------------------------------------------------------


def test_create_download_huggingface_no_token_gated_ตอบ_400_ไม่_submit(client, monkeypatch):
    from server.downloads import DownloadJob, DownloadState

    mgr = main.get_download_manager()
    calls = []
    monkeypatch.setattr(mgr, "submit", lambda *a, **k: calls.append((a, k)) or DownloadJob(
        id="j1", model_id="ds4-flash-unsloth-iq2xxs", files=[], created_at=0.0, state=DownloadState.QUEUED,
    ))

    seen = []

    def fake_probe(url: str, *, token: str | None = None, client: httpx.Client | None = None) -> str | None:
        seen.append((url, token))
        return "HF ปฏิเสธ (403)"

    monkeypatch.setattr(hf, "probe_download_access", fake_probe)

    # entry นี้ไม่มี token ใน state (client fixture ชี้ AISERVER2_STATE ไปที่ tmp ว่าง)
    resp = client.post("/api/downloads", json={"model_id": "ds4-flash-unsloth-iq2xxs"})

    assert resp.status_code == 400
    assert "token" in resp.json()["detail"]
    assert calls == []  # submit ต้องไม่ถูกเรียก
    assert seen == [
        (
            "https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF/resolve/main/UD-IQ2_XXS/DeepSeek-V4-Flash-0731-UD-IQ2_XXS-00001-of-00003.gguf",
            None,
        )
    ]  # ตรวจแค่ URL แรกพอ ไม่มี token


def test_create_download_huggingface_มี_token_probe_ปฏิเสธ_ตอบ_400_ไม่_submit(client, monkeypatch):
    from server.downloads import DownloadJob, DownloadState

    hf.save_token("hf_abc123")  # มี token แล้ว แต่ยังไม่ได้กด accept gate

    mgr = main.get_download_manager()
    calls = []
    monkeypatch.setattr(mgr, "submit", lambda *a, **k: calls.append((a, k)) or DownloadJob(
        id="j1", model_id="ds4-flash-unsloth-iq2xxs", files=[], created_at=0.0, state=DownloadState.QUEUED,
    ))

    seen = []

    def fake_probe(url: str, *, token: str | None = None, client: httpx.Client | None = None) -> str | None:
        seen.append((url, token))
        return "HF ปฏิเสธ (403): Access to model X is restricted and you are not in the authorized list."

    monkeypatch.setattr(hf, "probe_download_access", fake_probe)

    resp = client.post("/api/downloads", json={"model_id": "ds4-flash-unsloth-iq2xxs"})

    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "Agree" in detail
    assert "Access to model X is restricted" in detail
    assert calls == []  # submit ต้องไม่ถูกเรียก
    assert seen == [(seen[0][0], "hf_abc123")]  # probe ต้องถูกเรียกพร้อม token ที่จำไว้


def test_create_download_huggingface_มี_token_probe_ผ่าน_submit(client, monkeypatch):
    from server.downloads import DownloadJob, DownloadState

    hf.save_token("hf_abc123")  # มี token แล้ว และ probe ผ่าน (ไม่ติด gate/accept แล้ว)

    mgr = main.get_download_manager()
    calls = []
    monkeypatch.setattr(mgr, "submit", lambda *a, **k: calls.append((a, k)) or DownloadJob(
        id="j1", model_id="ds4-flash-unsloth-iq2xxs", files=[], created_at=0.0, state=DownloadState.QUEUED,
    ))

    probe_calls = []
    monkeypatch.setattr(
        hf, "probe_download_access",
        lambda *a, **k: probe_calls.append((a, k)) or None,
    )

    resp = client.post("/api/downloads", json={"model_id": "ds4-flash-unsloth-iq2xxs"})

    assert resp.status_code == 200
    assert len(probe_calls) == 1  # มี token แล้วก็ยังต้องเช็ค (อาจยังไม่ได้ accept gate)
    assert len(calls) == 1


def test_create_download_url_ไม่ใช่_huggingface_ไม่เรียก_probe(client, monkeypatch):
    from server.downloads import DownloadJob, DownloadState

    resp = client.post("/api/models", json={
        "id": "test-non-hf-model", "name": "Test", "engine": "llamacpp",
        "path": "~/models/gguf/test/test.gguf", "dl": ["https://example.com/model.gguf"],
    })
    assert resp.status_code == 200

    probe_calls = []
    monkeypatch.setattr(hf, "probe_download_access", lambda *a, **k: probe_calls.append((a, k)) or "gated")

    mgr = main.get_download_manager()
    monkeypatch.setattr(mgr, "submit", lambda *a, **k: DownloadJob(
        id="j1", model_id="test-non-hf-model", files=[], created_at=0.0, state=DownloadState.QUEUED,
    ))

    resp = client.post("/api/downloads", json={"model_id": "test-non-hf-model"})

    assert resp.status_code == 200
    assert probe_calls == []


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


# ---------------------------------------------------------------------------
# /api/usage — token สะสม (tok_in/tok_out) + busy ต่อ instance จาก Prometheus /metrics
# ---------------------------------------------------------------------------


def test_usage_parse_ชื่อ_metric_llamacpp(client, monkeypatch):
    fake = [
        instances.Instance(
            port=8001, engine="llamacpp", pid=123, model_file="m.gguf",
            model_path="/home/dgx/models/m.gguf", ctx=262144, rss_gb=2.5, up=True,
        ),
    ]
    monkeypatch.setattr(instances, "scan", lambda **kw: fake)

    metrics_text = (
        "# HELP llamacpp:prompt_tokens_total counter สะสม token ขาเข้า\n"
        "# TYPE llamacpp:prompt_tokens_total counter\n"
        "llamacpp:prompt_tokens_total 104179\n"
        "llamacpp:tokens_predicted_total 135300\n"
        "llamacpp:requests_processing 1\n"
    )

    def fake_get(url, timeout=None):
        assert url == "http://127.0.0.1:8001/metrics"
        return httpx.Response(200, text=metrics_text, request=httpx.Request("GET", url))

    monkeypatch.setattr(main.httpx, "get", fake_get)

    resp = client.get("/api/usage")
    assert resp.status_code == 200
    body = resp.json()
    assert body["usage"] == [{"port": 8001, "tok_in": 104179, "tok_out": 135300, "busy": 1}]
    assert "t" in body


def test_usage_parse_ชื่อ_metric_vllm_มี_label(client, monkeypatch):
    fake = [
        instances.Instance(
            port=8002, engine="vllm", pid=456, model_file="m2.safetensors",
            model_path="/home/dgx/models/m2", ctx=32768, rss_gb=10.0, up=True,
        ),
    ]
    monkeypatch.setattr(instances, "scan", lambda **kw: fake)

    metrics_text = (
        'vllm:prompt_tokens_total{model_name="m2"} 50000\n'
        'vllm:generation_tokens_total{model_name="m2"} 12345\n'
        'vllm:num_requests_running{model_name="m2"} 0\n'
    )

    def fake_get(url, timeout=None):
        return httpx.Response(200, text=metrics_text, request=httpx.Request("GET", url))

    monkeypatch.setattr(main.httpx, "get", fake_get)

    resp = client.get("/api/usage")
    assert resp.status_code == 200
    body = resp.json()
    assert body["usage"] == [{"port": 8002, "tok_in": 50000, "tok_out": 12345, "busy": 0}]


def test_usage_metrics_fetch_พัง_ยังคืน_port_ไม่_crash(client, monkeypatch):
    fake = [
        instances.Instance(
            port=8003, engine="llamacpp", pid=789, model_file="m3.gguf",
            model_path="/home/dgx/models/m3.gguf", ctx=8192, rss_gb=1.0, up=True,
        ),
    ]
    monkeypatch.setattr(instances, "scan", lambda **kw: fake)

    def fake_get(url, timeout=None):
        raise httpx.ConnectError("connection refused", request=httpx.Request("GET", url))

    monkeypatch.setattr(main.httpx, "get", fake_get)

    resp = client.get("/api/usage")
    assert resp.status_code == 200
    body = resp.json()
    assert body["usage"] == [{"port": 8003}]


# ---------------------------------------------------------------------------
# /api/endpoints — โค้ดตัวอย่างให้หน้าเว็บเอาไปสร้าง snippet ต่อกับโมเดลที่โหลดอยู่
# ---------------------------------------------------------------------------


def test_list_endpoints_instance_และ_gateway_ปกติ(client, monkeypatch):
    fake = [
        instances.Instance(
            port=8001, engine="llamacpp", pid=123,
            model_file="Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf",
            model_path="/home/dgx/models/gguf/x/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf",
            ctx=262144, rss_gb=2.5, up=True,
        ),
    ]
    monkeypatch.setattr(instances, "scan", lambda **kw: fake)

    def handler(request):
        if request.url.port == 4000:
            return httpx.Response(200, json={"data": [
                {"id": "qwen38"},
                {"id": "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P"},
            ]})
        if request.url.port == 8001:
            return httpx.Response(200, json={"data": [
                {"id": "/home/dgx/models/gguf/x/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf"},
            ]})
        raise AssertionError(f"unexpected port {request.url.port}")

    _patch_metrics_client(monkeypatch, handler)

    resp = client.get("/api/endpoints")
    assert resp.status_code == 200
    body = resp.json()

    assert body["gateway"]["port"] == 4000
    assert body["gateway"]["up"] is True
    assert body["gateway"]["models"] == [
        {"id": "qwen38", "live": False},
        {"id": "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P", "live": True},
    ]
    assert body["gateway"]["recommended_model"] == "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P"

    assert len(body["direct"]) == 1
    d = body["direct"][0]
    assert d["port"] == 8001
    assert d["engine"] == "llamacpp"
    assert d["model_file"] == "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf"
    assert d["served_model_id"] == "/home/dgx/models/gguf/x/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf"
    assert d["up"] is True


def test_list_endpoints_gateway_2_โมเดล_instance_รันตัวเดียว_match_case_insensitive_ตัด_gguf(client, monkeypatch):
    """เจอของจริงบนเครื่อง: gateway มีชื่อค้างที่ไม่มี instance รันจริงปนอยู่ — ต้องแยกได้ว่าตัวไหน "ยิงได้จริง" """
    fake = [
        instances.Instance(
            port=8001, engine="llamacpp", pid=123, model_file="qwen38-Live.GGUF",
            model_path="/home/dgx/models/gguf/x/qwen38-Live.GGUF",
            ctx=131072, rss_gb=2.0, up=True,
        ),
    ]
    monkeypatch.setattr(instances, "scan", lambda **kw: fake)

    def handler(request):
        if request.url.port == 4000:
            return httpx.Response(200, json={"data": [
                {"id": "QWEN38-LIVE"},  # ตัวพิมพ์ต่างจาก model_file แต่ต้อง match ได้
                {"id": "qwen38-ค้าง-ไม่มี-instance"},
            ]})
        if request.url.port == 8001:
            return httpx.Response(200, json={"data": [{"id": "/home/dgx/models/gguf/x/qwen38-Live.GGUF"}]})
        raise AssertionError(f"unexpected port {request.url.port}")

    _patch_metrics_client(monkeypatch, handler)

    resp = client.get("/api/endpoints")
    assert resp.status_code == 200
    body = resp.json()

    assert body["gateway"]["models"] == [
        {"id": "QWEN38-LIVE", "live": True},
        {"id": "qwen38-ค้าง-ไม่มี-instance", "live": False},
    ]
    assert body["gateway"]["recommended_model"] == "QWEN38-LIVE"


def test_list_endpoints_ไม่มี_instance_เลย(client, monkeypatch):
    monkeypatch.setattr(instances, "scan", lambda **kw: [])

    def handler(request):
        return httpx.Response(200, json={"data": [{"id": "qwen38"}]})

    _patch_metrics_client(monkeypatch, handler)

    resp = client.get("/api/endpoints")
    assert resp.status_code == 200
    body = resp.json()
    assert body["direct"] == []
    assert body["gateway"]["up"] is True
    assert body["gateway"]["models"] == [{"id": "qwen38", "live": False}]
    assert body["gateway"]["recommended_model"] is None


def test_list_endpoints_gateway_ต่อไม่ได้_ไม่_raise(client, monkeypatch):
    monkeypatch.setattr(instances, "scan", lambda **kw: [])

    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    _patch_metrics_client(monkeypatch, handler)

    resp = client.get("/api/endpoints")
    assert resp.status_code == 200
    body = resp.json()
    assert body["gateway"]["up"] is False
    assert body["gateway"]["models"] == []
    assert body["gateway"]["recommended_model"] is None


def test_list_endpoints_instance_v1_models_ตอบไม่ได้_served_model_id_เป็น_null(client, monkeypatch):
    fake = [
        instances.Instance(
            port=8002, engine="llamacpp", pid=456, model_file="m.gguf",
            model_path="/home/dgx/models/m.gguf", ctx=131072, rss_gb=1.0, up=True,
        ),
    ]
    monkeypatch.setattr(instances, "scan", lambda **kw: fake)

    def handler(request):
        if request.url.port == 4000:
            return httpx.Response(200, json={"data": []})
        raise httpx.ConnectError("connection refused", request=request)

    _patch_metrics_client(monkeypatch, handler)

    resp = client.get("/api/endpoints")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["direct"]) == 1
    assert body["direct"][0]["port"] == 8002
    assert body["direct"][0]["served_model_id"] is None


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


def test_metrics_ส่ง_field_เสริมของ_gauge_ต่อไปครบ(client, monkeypatch):
    """dgx-monitor ส่ง sub มาบางตัว (ดิสก์ = 'เหลือ 2.7 TB') — ห้ามกรองทิ้งระหว่างทาง"""
    def handler(request):
        return httpx.Response(200, json={"ok": True, "ts": 1, "gauges": [
            {"label": "ดิสก์", "value": 31.4, "max": 100, "unit": "%", "warn": 85,
             "sub": "เหลือ 2.7 TB"},
        ]})

    _patch_metrics_client(monkeypatch, handler)

    body = client.get("/api/metrics").json()

    assert body["ok"] is True
    assert body["gauges"][0]["sub"] == "เหลือ 2.7 TB"


def test_endpoints_พอร์ต_gateway_ตามที่ตั้งใน_env(monkeypatch):
    """snippet ที่ผู้ใช้ copy ต้องชี้พอร์ตจริง — hardcode 4000 ไว้จะผิดทันทีถ้ามีคน override"""
    from server import main as m

    monkeypatch.setattr(m, "_LITELLM_URL", "http://127.0.0.1:4444")
    assert m._litellm_port() == 4444

    monkeypatch.setattr(m, "_LITELLM_URL", "http://127.0.0.1")
    assert m._litellm_port() == 4000


# --- จับคู่ gateway ด้วยพอร์ต ไม่ใช่ชื่อ (เจอจริง 2026-08-29 18:47) ---------
# ชื่อที่ LiteLLM ตั้งไม่แน่นอน: บางตัวตามชื่อไฟล์ บางตัวตาม id ใน catalog
# ทำให้เทียบชื่อแล้ว live=false ทั้งที่โมเดลรันอยู่จริง → dropdown ในหน้าเว็บว่าง


def test_gateway_live_ตัดสินจากพอร์ตปลายทาง_ไม่ใช่ชื่อ(client, monkeypatch):
    from server import instances as ins
    from server import main as m

    # ชื่อใน gateway (qwen3.8-flash-next-udq2) ไม่ตรงกับชื่อไฟล์เลย แต่ route ไป :8001 ที่รันอยู่
    monkeypatch.setattr(ins, "scan", lambda **kw: [
        ins.Instance(port=8001, engine="llamacpp", pid=1,
                     model_file="Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf",
                     model_path="/m/x.gguf", ctx=32768, rss_gb=2.0, up=True),
    ])
    monkeypatch.setattr(m, "_fetch_v1_models",
                        lambda base, c: ["qwen3.8-flash-next-udq2", "qwen38"] if ":4000" in base else ["/m/x.gguf"])
    monkeypatch.setattr(m, "_fetch_gateway_routes",
                        lambda base, c: {"qwen3.8-flash-next-udq2": 8001, "qwen38": 8000})

    gw = client.get("/api/endpoints").json()["gateway"]

    assert {x["id"]: x["live"] for x in gw["models"]} == {
        "qwen3.8-flash-next-udq2": True,   # route ไป :8001 ที่รันอยู่
        "qwen38": False,                   # route ไป :8000 ที่ว่าง
    }
    assert gw["recommended_model"] == "qwen3.8-flash-next-udq2"


def test_gateway_ถอยไปเทียบชื่อเมื่อ_model_info_ใช้ไม่ได้(client, monkeypatch):
    from server import instances as ins
    from server import main as m

    monkeypatch.setattr(ins, "scan", lambda **kw: [
        ins.Instance(port=8001, engine="llamacpp", pid=1, model_file="MyModel-Q8_0.gguf",
                     model_path="/m/MyModel-Q8_0.gguf", ctx=4096, rss_gb=1.0, up=True),
    ])
    monkeypatch.setattr(m, "_fetch_v1_models",
                        lambda base, c: ["MyModel-Q8_0"] if ":4000" in base else ["/m/MyModel-Q8_0.gguf"])
    monkeypatch.setattr(m, "_fetch_gateway_routes", lambda base, c: None)  # LiteLLM เก่า/ตอบไม่ได้

    gw = client.get("/api/endpoints").json()["gateway"]

    assert gw["recommended_model"] == "MyModel-Q8_0", "fallback เทียบชื่อต้องยังทำงาน"
