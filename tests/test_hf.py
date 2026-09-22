"""test สำหรับ server/hf.py — แปลง HF repo id เป็นรายการ quant ให้ผู้ใช้เลือกก่อนดาวน์โหลด

ใช้ fixture จริงที่ดึงมาจาก HF API (?blobs=true) ของ tests/fixtures/hf_*.json
ไม่ยิงเน็ตจริงในเทสของ fetch_repo — ใช้ httpx.MockTransport ป้อนข้อมูลแทน
"""
from __future__ import annotations

import json
import os

import httpx
import pytest

from server import paths
from server.hf import (
    GatedRepoError,
    HfFile,
    QuantGroup,
    RepoNotFoundError,
    SearchHit,
    companion_kind,
    config_arch_ctx,
    fetch_config,
    fetch_repo,
    group_quants,
    group_weights,
    has_safetensors,
    is_gated,
    list_files,
    load_token,
    normalize_repo_id,
    probe_download_access,
    probe_gated_url,
    quant_label,
    resolve_url,
    save_token,
    search_models,
    suggest_args,
    vllm_engine_env,
    vllm_parsers,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _load_fixture(name: str) -> dict:
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as f:
        return json.load(f)


def _gb(n: int) -> float:
    # ปัดเป็น GB ทศนิยม 1 ตำแหน่ง แบบเดียวกับที่ระบุใน spec (หาร 1e9)
    return round(n / 1e9, 1)


# ---------------------------------------------------------------------------
# list_files
# ---------------------------------------------------------------------------


def test_list_files_returns_all_siblings():
    repo_json = _load_fixture("hf_glm53-flash-gguf_blobs.json")
    files = list_files(repo_json)

    assert len(files) == len(repo_json["siblings"])
    assert all(isinstance(f, HfFile) for f in files)


def test_list_files_carries_size_and_sha256():
    repo_json = _load_fixture("hf_glm53-flash-gguf_blobs.json")
    files = list_files(repo_json)

    by_path = {f.path: f for f in files}
    f = by_path["UD-IQ1_M/GLM-5.3-Flash-UD-IQ1_M-00002-of-00003.gguf"]
    assert f.size == 49996246592
    assert f.sha256 == "6bcfc2210cced11cede6cd6ce4ac3368c0b1ab52d6fc3e40f2a3f310f0614001"

    # ไฟล์ที่ไม่ใช่ lfs (เช่น .gitattributes) ต้องไม่มี sha256
    gitattr = by_path[".gitattributes"]
    assert gitattr.sha256 is None


def test_hf_file_url_path_matches_repo_path():
    f = HfFile(path="UD-IQ1_S/GLM-5.3-Flash-UD-IQ1_S-00001-of-00003.gguf", size=1, sha256=None)
    assert f.url_path == "UD-IQ1_S/GLM-5.3-Flash-UD-IQ1_S-00001-of-00003.gguf"


# ---------------------------------------------------------------------------
# group_quants — GLM-5.3-Flash: 7 quant group ตัวเลขต้องตรงเป๊ะตาม spec
# ---------------------------------------------------------------------------


def test_group_quants_glm53_matches_expected_sizes_and_order():
    repo_json = _load_fixture("hf_glm53-flash-gguf_blobs.json")
    files = list_files(repo_json)
    groups = group_quants(files)

    expected = [
        ("UD-IQ1_S", 3, 93.1),
        ("UD-IQ1_M", 3, 97.6),
        ("UD-Q2_K_XL", 4, 108.7),
        ("UD-IQ3_XXS", 4, 120.4),
        ("UD-Q3_K_XL", 4, 147.5),
        ("UD-IQ4_XS", 5, 156.8),
        ("UD-Q4_K_XL", 6, 199.7),
    ]

    assert [g.key for g in groups] == [e[0] for e in expected]
    assert len(groups) == 7

    for g, (key, n_files, gb) in zip(groups, expected):
        assert g.key == key
        assert len(g.files) == n_files
        assert g.shard_count == n_files
        assert _gb(g.total_bytes) == gb

    # เรียงจากน้อยไปมากตาม total_bytes (ข้อ 7)
    assert [g.total_bytes for g in groups] == sorted(g.total_bytes for g in groups)


def test_group_quants_glm53_shards_ordered_by_index():
    repo_json = _load_fixture("hf_glm53-flash-gguf_blobs.json")
    files = list_files(repo_json)
    groups = group_quants(files)

    ud_iq1_s = next(g for g in groups if g.key == "UD-IQ1_S")
    paths = [f.path for f in ud_iq1_s.files]
    assert paths == [
        "UD-IQ1_S/GLM-5.3-Flash-UD-IQ1_S-00001-of-00003.gguf",
        "UD-IQ1_S/GLM-5.3-Flash-UD-IQ1_S-00002-of-00003.gguf",
        "UD-IQ1_S/GLM-5.3-Flash-UD-IQ1_S-00003-of-00003.gguf",
    ]


def test_group_quants_glm53_companions_are_mmproj_not_a_group():
    repo_json = _load_fixture("hf_glm53-flash-gguf_blobs.json")
    files = list_files(repo_json)
    groups = group_quants(files)

    # mmproj-* ต้องไม่โผล่เป็น key ของ quant group ใด ๆ
    assert all(not g.key.startswith("mmproj") for g in groups)

    companion_paths = {f.path for g in groups for f in g.companions}
    assert "mmproj-BF16.gguf" in companion_paths
    assert "mmproj-F16.gguf" in companion_paths

    # total_bytes ต้องไม่รวม companion (ข้อ 6)
    for g in groups:
        assert sum(f.size for f in g.files) == g.total_bytes


# ---------------------------------------------------------------------------
# group_quants — Qwen3.8-27B: quant ที่ root, companion ใน MTP/, shard ใน BF16/
# ---------------------------------------------------------------------------


def test_group_quants_qwen38_root_quant_present():
    repo_json = _load_fixture("hf_qwen38-27b-gguf_blobs.json")
    files = list_files(repo_json)
    groups = group_quants(files)

    by_key = {g.key: g for g in groups}
    assert "Q8_0" in by_key
    q8 = by_key["Q8_0"]
    assert q8.shard_count == 1
    assert len(q8.files) == 1
    assert q8.files[0].path == "Qwen3.8-27B-Q8_0.gguf"
    assert q8.total_bytes == 29047086048


def test_group_quants_qwen38_root_key_extraction_various():
    repo_json = _load_fixture("hf_qwen38-27b-gguf_blobs.json")
    files = list_files(repo_json)
    groups = group_quants(files)

    by_key = {g.key: g for g in groups}
    expected_keys = {
        "Q4_0",
        "Q4_1",
        "Q8_0",
        "UD-IQ1_M",
        "UD-IQ1_S",
        "UD-IQ2_S",
        "UD-IQ2_XXS",
        "UD-IQ3_S",
        "UD-IQ3_XXS",
        "UD-IQ4_XS",
        "UD-Q2_K_XL",
        "UD-Q3_K_XL",
        "UD-Q4_K_M",
        "UD-Q4_K_S",
        "UD-Q4_K_XL",
        "UD-Q5_K_M",
        "UD-Q5_K_S",
        "UD-Q5_K_XL",
        "UD-Q6_K",
        "UD-Q6_K_L",
        "UD-Q6_K_M",
        "UD-Q6_K_XL",
        "UD-Q8_K_L",
        "UD-Q8_K_XL",
        "BF16",
    }
    assert expected_keys <= set(by_key.keys())
    for key in expected_keys:
        assert len(by_key[key].files) >= 1


def test_group_quants_qwen38_mtp_is_companion_not_group():
    repo_json = _load_fixture("hf_qwen38-27b-gguf_blobs.json")
    files = list_files(repo_json)
    groups = group_quants(files)

    assert all(g.key != "MTP" for g in groups)

    companion_paths = {f.path for g in groups for f in g.companions}
    assert "MTP/mtp-Qwen3.8-27B-Q4_0.gguf" in companion_paths


def test_group_quants_qwen38_bf16_shard_group():
    repo_json = _load_fixture("hf_qwen38-27b-gguf_blobs.json")
    files = list_files(repo_json)
    groups = group_quants(files)

    bf16 = next(g for g in groups if g.key == "BF16")
    assert bf16.shard_count == 2
    assert len(bf16.files) == 2
    assert [f.path for f in bf16.files] == [
        "BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf",
        "BF16/Qwen3.8-27B-BF16-00002-of-00002.gguf",
    ]
    assert bf16.total_bytes == 49986159616 + 4671576000


def test_group_quants_qwen38_ignores_unrecognized_gguf():
    # imatrix_unsloth.gguf ไม่ตรง pattern quant ใด ๆ และไม่ใช่ companion → ต้องไม่ถูกนับที่ไหนเลย
    repo_json = _load_fixture("hf_qwen38-27b-gguf_blobs.json")
    files = list_files(repo_json)
    groups = group_quants(files)

    for g in groups:
        assert all("imatrix" not in f.path for f in g.files)
        assert all("imatrix" not in f.path for f in g.companions)


# ---------------------------------------------------------------------------
# group_quants — Navin-Models/Qwen3.8-Flash-Next-...-AD-4.27-GGUF: ไฟล์ root ที่ไม่มี
# quant token ท้ายชื่อเลย (ลงท้ายด้วยชื่อ variant "mainline"/"main" แทน) → ต้อง fallback
# เป็น stem ทั้งก้อนแทนการข้ามไปเงียบ ๆ (ดู task ส่วนที่ 1)
# ---------------------------------------------------------------------------


def test_group_quants_navin_qwen38_falls_back_to_stem_when_no_quant_token():
    repo_json = _load_fixture("hf_navin-qwen38-flash-next-ad427-gguf_blobs.json")
    files = list_files(repo_json)
    groups = group_quants(files)

    assert len(groups) == 2

    by_key = {g.key: g for g in groups}
    mainline_key = "Qwen3.8-Flash-Next-Uncensored-AD-4.27-mainline"
    main_key = "Qwen3.8-Flash-Next-Uncensored-AD-4.27-main"
    assert set(by_key.keys()) == {mainline_key, main_key}

    mainline = by_key[mainline_key]
    main_ = by_key[main_key]

    assert mainline.shard_count == 33
    assert len(mainline.files) == 33
    assert mainline.total_bytes == 94525395584

    assert main_.shard_count == 34
    assert len(main_.files) == 34
    assert main_.total_bytes == 97301019488

    # เรียงเล็กไปใหญ่ (ข้อ 7 เดิม) — mainline (33 shard) เล็กกว่า main (34 shard) → ต้องมาก่อน
    assert [g.key for g in groups] == [mainline_key, main_key]


def test_group_quants_navin_qwen38_mmproj_is_vision_companion_not_a_group():
    repo_json = _load_fixture("hf_navin-qwen38-flash-next-ad427-gguf_blobs.json")
    files = list_files(repo_json)
    groups = group_quants(files)

    assert all("mmproj" not in g.key.lower() for g in groups)

    vision_paths = {f.path for g in groups for f in g.vision_files}
    assert "mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf" in vision_paths


# ---------------------------------------------------------------------------
# is_gated
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "repo_json,expected",
    [
        ({"gated": "auto"}, True),
        ({"gated": "manual"}, True),
        ({"gated": True}, True),
        ({"gated": False}, False),
        ({}, False),
    ],
)
def test_is_gated(repo_json, expected):
    assert is_gated(repo_json) is expected


# ---------------------------------------------------------------------------
# probe_gated_url — เช็คเร็ว ๆ ก่อน submit ดาวน์โหลดว่า URL นี้จะ 401/403 แน่ไหม (hotfix)
# HEAD ไม่ตาม redirect · เน็ตพัง/timeout → False (ปล่อยผ่านให้ aria2 ไปเจอเองตอนโหลดจริง)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [401, 403])
def test_probe_gated_url_401_403_returns_true(status):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "HEAD"
        return httpx.Response(status)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert probe_gated_url("https://huggingface.co/some/gated-repo/resolve/main/f.gguf", client=client) is True


@pytest.mark.parametrize("status", [200, 302])
def test_probe_gated_url_200_302_returns_false(status):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert probe_gated_url("https://huggingface.co/some/open-repo/resolve/main/f.gguf", client=client) is False


def test_probe_gated_url_exception_returns_false():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert probe_gated_url("https://huggingface.co/some/repo/resolve/main/f.gguf", client=client) is False


# ---------------------------------------------------------------------------
# probe_download_access — เหมือน probe_gated_url แต่คืนข้อความเหตุผลจาก HF ด้วย
# (hotfix: token จำไว้แล้วแต่ยังไม่ได้กด accept gate ⇒ probe ต้องรู้เรื่องนี้ด้วย)
# ---------------------------------------------------------------------------


def test_probe_download_access_403_with_x_error_message_header_included_in_message():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            headers={
                "x-error-code": "GatedRepo",
                "x-error-message": (
                    "Access to model X is restricted and you are not in the authorized list. "
                    "Visit https://huggingface.co/X to ask for access."
                ),
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    message = probe_download_access("https://huggingface.co/some/gated-repo/resolve/main/f.gguf", client=client)

    assert message is not None
    assert "403" in message
    assert "Access to model X is restricted" in message


def test_probe_download_access_403_without_header_returns_generic_message():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    message = probe_download_access("https://huggingface.co/some/gated-repo/resolve/main/f.gguf", client=client)

    assert message is not None
    assert "403" in message


@pytest.mark.parametrize("status", [200, 307])
def test_probe_download_access_200_307_returns_none(status):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert probe_download_access("https://huggingface.co/some/open-repo/resolve/main/f.gguf", client=client) is None


def test_probe_download_access_exception_returns_none():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert probe_download_access("https://huggingface.co/some/repo/resolve/main/f.gguf", client=client) is None


def test_probe_download_access_sends_bearer_token_when_given():
    seen_headers = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.update(request.headers)
        return httpx.Response(403)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    probe_download_access(
        "https://huggingface.co/some/gated-repo/resolve/main/f.gguf", token="hf_abc123", client=client
    )

    assert seen_headers.get("authorization") == "Bearer hf_abc123"


def test_probe_download_access_no_auth_header_when_no_token():
    seen_headers = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.update(request.headers)
        return httpx.Response(403)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    probe_download_access("https://huggingface.co/some/gated-repo/resolve/main/f.gguf", client=client)

    assert "authorization" not in seen_headers


# ---------------------------------------------------------------------------
# save_token / load_token — เก็บ HF token ไว้ใช้ตอนดาวน์โหลด (ดู task ส่วนที่ 2)
# ---------------------------------------------------------------------------


def test_load_token_returns_none_when_no_file(tmp_path, monkeypatch):
    monkeypatch.setenv("AISERVER2_STATE", str(tmp_path))
    assert load_token() is None


def test_save_and_load_token_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("AISERVER2_STATE", str(tmp_path))
    save_token("hf_abc123")
    assert load_token() == "hf_abc123"


def test_save_token_strips_whitespace(tmp_path, monkeypatch):
    monkeypatch.setenv("AISERVER2_STATE", str(tmp_path))
    save_token("  hf_abc123  \n")
    assert load_token() == "hf_abc123"


def test_save_token_writes_file_mode_0600(tmp_path, monkeypatch):
    monkeypatch.setenv("AISERVER2_STATE", str(tmp_path))
    save_token("hf_abc123")
    mode = os.stat(paths.state("hf_token")).st_mode & 0o777
    assert mode == 0o600


def test_save_token_empty_does_not_write_file(tmp_path, monkeypatch):
    monkeypatch.setenv("AISERVER2_STATE", str(tmp_path))
    save_token("")
    assert not os.path.exists(paths.state("hf_token"))
    assert load_token() is None


def test_save_token_whitespace_only_does_not_write_file(tmp_path, monkeypatch):
    monkeypatch.setenv("AISERVER2_STATE", str(tmp_path))
    save_token("   ")
    assert not os.path.exists(paths.state("hf_token"))


# ---------------------------------------------------------------------------
# fetch_repo
# ---------------------------------------------------------------------------


def test_fetch_repo_200_returns_dict():
    payload = {"id": "unsloth/Qwen3.8-27B-GGUF", "gated": False, "siblings": []}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/models/unsloth/Qwen3.8-27B-GGUF"
        assert request.url.params.get("blobs") == "true"
        return httpx.Response(200, json=payload)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = fetch_repo("unsloth/Qwen3.8-27B-GGUF", client=client)

    assert result == payload


@pytest.mark.parametrize("status", [401, 403])
def test_fetch_repo_gated_raises(status: int):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": "gated"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(GatedRepoError):
        fetch_repo("some/gated-repo", client=client)


def test_fetch_repo_404_raises_not_found():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "not found"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(RepoNotFoundError):
        fetch_repo("nope/does-not-exist", client=client)


def test_fetch_repo_sends_bearer_token_when_given():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"id": "x", "gated": False, "siblings": []})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    fetch_repo("some/repo", token="hf_secrettoken", client=client)

    assert seen["auth"] == "Bearer hf_secrettoken"


def test_fetch_repo_no_auth_header_when_no_token():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"id": "x", "gated": False, "siblings": []})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    fetch_repo("some/repo", client=client)

    assert seen["auth"] is None


# ---------------------------------------------------------------------------
# resolve_url
# ---------------------------------------------------------------------------


def test_resolve_url_root_file():
    url = resolve_url("unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-Q8_0.gguf")
    assert url == "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-Q8_0.gguf"


def test_resolve_url_subfolder_file():
    url = resolve_url(
        "unsloth/GLM-5.3-Flash-GGUF",
        "UD-IQ1_S/GLM-5.3-Flash-UD-IQ1_S-00001-of-00003.gguf",
    )
    assert url == (
        "https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/resolve/main/"
        "UD-IQ1_S/GLM-5.3-Flash-UD-IQ1_S-00001-of-00003.gguf"
    )


# ---------------------------------------------------------------------------
# normalize_repo_id
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("unsloth/Xxx-GGUF", ("unsloth/Xxx-GGUF", "Xxx-GGUF")),
        ("https://huggingface.co/unsloth/Xxx?a=1", ("unsloth/Xxx", "Xxx")),
        ("https://huggingface.co/unsloth/Xxx/tree/main", ("unsloth/Xxx", "Xxx")),
        ("https://github.com/openai/gpt-oss?utm_source=chatgpt.com", (None, "gpt-oss")),
        ("gpt oss 20b", (None, "gpt oss 20b")),
        ("", (None, "")),
        ("  org/repo/  ", ("org/repo", "repo")),
    ],
)
def test_normalize_repo_id_cases(text, expected):
    assert normalize_repo_id(text) == expected


# ---------------------------------------------------------------------------
# search_models
# ---------------------------------------------------------------------------


def test_search_models_ranks_gguf_first_even_with_fewer_downloads():
    payload = [
        {"id": "some-org/Big-Model", "downloads": 999999, "likes": 10, "gated": False, "tags": []},
        {"id": "unsloth/Small-Model-GGUF", "downloads": 5, "likes": 1, "gated": False, "tags": ["gguf"]},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    hits = search_models("model", client=client)

    assert [h.id for h in hits] == ["unsloth/Small-Model-GGUF", "some-org/Big-Model"]
    assert all(isinstance(h, SearchHit) for h in hits)
    assert hits[0].is_gguf is True
    assert hits[1].is_gguf is False


def test_search_models_empty_query_returns_empty_without_http():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=[])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    hits = search_models("", client=client)

    assert hits == []
    assert calls == []  # ห้ามยิง HTTP request เลยเมื่อ query ว่าง


def test_search_models_sends_bearer_token_when_given():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=[])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    search_models("gpt oss", token="hf_secrettoken", client=client)

    assert seen["auth"] == "Bearer hf_secrettoken"


# --- บั๊กจากของจริง: gpt-oss (2026-08-29) ---------------------------------
# เคยถูก revert ไปครั้งหนึ่ง — test นี้ล็อกไว้ไม่ให้หายอีก


def _load(name):
    import json, pathlib
    p = pathlib.Path(__file__).parent / "fixtures" / name
    return list_files(json.loads(p.read_text()))


def test_รู้จัก_quant_แบบ_MXFP4_ไม่งั้นโมเดลจริงหายทั้งตัว():
    """gpt-oss-120b-MXFP4.gguf = 63.4GB คือตัวโมเดลจริง ถ้า regex ไม่รู้จักจะหายไปเงียบ ๆ"""
    groups = group_quants(_load("hf_gpt-oss-120b-gguf_blobs.json"))
    keys = {g.key for g in groups}

    assert "MXFP4" in keys, f"MXFP4 หายไป เหลือแค่ {keys}"
    mxfp4 = next(g for g in groups if g.key == "MXFP4")
    assert mxfp4.total_bytes > 60e9


def test_eagle3_เป็น_draft_head_ไม่ใช่_quant():
    """ไม่กันไว้ = ผู้ใช้เลือก 'Q8_0 0.8GB' แล้วได้ draft head แทนโมเดลจริง"""
    groups = group_quants(_load("hf_gpt-oss-120b-gguf_blobs.json"))

    assert {g.key for g in groups} == {"MXFP4"}, "eagle3-* ต้องไม่กลายเป็น quant group"
    companions = {c.path for c in groups[0].companions}
    assert any(c.startswith("eagle3-") for c in companions)


def test_quant_ที่_root_ของ_gpt_oss_20b_ครบ():
    groups = group_quants(_load("hf_gpt-oss-20b-gguf_blobs.json"))
    keys = {g.key for g in groups}

    assert {"Q4_K_M", "Q8_0", "F16", "UD-Q4_K_XL"} <= keys
    assert len(groups) == 16


# ---------------------------------------------------------------------------
# บั๊กจากของจริง: HauhauCS/Qwen3.8-27B-...-MTP-GGUF (2026-08-29)
# ไฟล์ draft ชื่อ "...-FastMTP-32K.gguf" ไม่ขึ้นต้นด้วย prefix ไหนเลยใน _COMPANION_PREFIXES
# ⇒ เคยหายไปเงียบ ๆ ทั้งจาก quant list และ companion list
# ---------------------------------------------------------------------------

_HAUHAUCS_FIXTURE = {
    "id": "HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF",
    "gated": False,
    "siblings": [
        {
            "rfilename": "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-FastMTP-32K.gguf",
            "size": 900_000_000,
            "lfs": {"sha256": "a" * 64},
        },
        {
            "rfilename": "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf",
            "size": 10_300_000_000,
            "lfs": {"sha256": "b" * 64},
        },
        {
            "rfilename": "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf",
            "size": 31_500_000_000,
            "lfs": {"sha256": "c" * 64},
        },
        {
            "rfilename": "mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf",
            "size": 900_000_000,
            "lfs": {"sha256": "d" * 64},
        },
    ],
}


def test_group_quants_hauhaucs_fastmtp_ไม่โผล่เป็น_quant_group():
    files = list_files(_HAUHAUCS_FIXTURE)
    groups = group_quants(files)

    keys = {g.key for g in groups}
    assert "FastMTP-32K" not in keys
    assert not any("mtp" in k.lower() for k in keys)


def test_group_quants_hauhaucs_fastmtp_อยู่ใน_draft_files():
    files = list_files(_HAUHAUCS_FIXTURE)
    groups = group_quants(files)

    draft_paths = {f.path for g in groups for f in g.draft_files}
    assert "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-FastMTP-32K.gguf" in draft_paths


def test_group_quants_hauhaucs_mmproj_อยู่ใน_vision_files():
    files = list_files(_HAUHAUCS_FIXTURE)
    groups = group_quants(files)

    vision_paths = {f.path for g in groups for f in g.vision_files}
    assert "mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf" in vision_paths


def test_group_quants_hauhaucs_quant_ปกติยังถูกจับครบและไม่รวม_companion():
    files = list_files(_HAUHAUCS_FIXTURE)
    groups = group_quants(files)

    by_key = {g.key: g for g in groups}
    assert set(by_key.keys()) == {"IQ2_M", "Q8_K_P"}

    assert by_key["IQ2_M"].total_bytes == 10_300_000_000
    assert by_key["Q8_K_P"].total_bytes == 31_500_000_000

    # total_bytes ต้องไม่รวม companion (draft/vision) เข้าไปด้วย
    for g in groups:
        assert sum(f.size for f in g.files) == g.total_bytes


def test_companion_kind_จับได้ทุก_kind():
    assert companion_kind("mtp-foo.gguf") == "draft"
    assert companion_kind("Qwen-Aggressive-FastMTP-32K.gguf") == "draft"  # substring กลางชื่อไฟล์
    assert companion_kind("dflash-foo.gguf") == "draft"
    assert companion_kind("eagle-foo.gguf") == "draft"
    assert companion_kind("eagle3-foo.gguf") == "draft"
    assert companion_kind("draft-foo.gguf") == "draft"
    assert companion_kind("mmproj-BF16.gguf") == "vision"
    assert companion_kind("MMPROJ-bf16.gguf") == "vision"  # case-insensitive

    # quant ปกติ — ต้องไม่ถูกจับผิด
    assert companion_kind("Q8_K_P.gguf") is None
    assert companion_kind("Q4_K_M.gguf") is None
    assert companion_kind("UD-IQ1_S-00001-of-00003.gguf") is None


def test_companion_kind_เช็คแค่_basename_ไม่ใช่_path_เต็ม():
    """กับดัก: โฟลเดอร์ชื่อมีคำว่า MTP แต่ไฟล์ข้างในเป็น quant ปกติ ต้องไม่ถูกจับเป็น companion"""
    assert companion_kind("Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF/Model-Q4_K_M.gguf") is None
    # แต่ไฟล์ที่ชื่อจริง ๆ มีคำว่า mtp ต้องยังถูกจับได้ตามปกติ แม้อยู่ในโฟลเดอร์แบบนี้
    assert companion_kind("Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF/mtp-Model.gguf") == "draft"


# ---------------------------------------------------------------------------
# suggest_args
# ---------------------------------------------------------------------------


def test_suggest_args_ไม่มี_companion_คืนค่าว่าง():
    group = QuantGroup(key="Q8_0", files=[], total_bytes=0, shard_count=1, companions=[])
    assert suggest_args(group, "/models/foo") == ""


def test_suggest_args_มี_draft_อย่างเดียว():
    draft = HfFile(path="Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-FastMTP-32K.gguf", size=900_000_000, sha256=None)
    group = QuantGroup(key="Q8_K_P", files=[], total_bytes=0, shard_count=1, companions=[draft])

    args = suggest_args(group, "/models/HauhauCS")
    assert args == "-md /models/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-FastMTP-32K.gguf"
    assert " -c " not in args and not args.startswith("-c ")


def test_suggest_args_มี_vision_อย่างเดียว():
    mmproj = HfFile(path="mmproj-BF16.gguf", size=900_000_000, sha256=None)
    group = QuantGroup(key="Q8_0", files=[], total_bytes=0, shard_count=1, companions=[mmproj])

    args = suggest_args(group, "/models/foo")
    assert args == "--mmproj /models/foo/mmproj-BF16.gguf"


def test_suggest_args_มีทั้งสอง_draft_มาก่อน():
    draft = HfFile(path="mtp-foo.gguf", size=300_000_000, sha256=None)
    mmproj = HfFile(path="mmproj-BF16.gguf", size=900_000_000, sha256=None)
    group = QuantGroup(key="Q8_0", files=[], total_bytes=0, shard_count=1, companions=[mmproj, draft])

    args = suggest_args(group, "/models/foo")
    assert args == "-md /models/foo/mtp-foo.gguf --mmproj /models/foo/mmproj-BF16.gguf"


def test_suggest_args_มี_draft_หลายตัวเลือกตัวเล็กสุด():
    big = HfFile(path="dflash-big.gguf", size=900_000_000, sha256=None)
    small = HfFile(path="mtp-small.gguf", size=300_000_000, sha256=None)
    group = QuantGroup(key="Q8_0", files=[], total_bytes=0, shard_count=1, companions=[big, small])

    args = suggest_args(group, "/models/foo")
    assert args == "-md /models/foo/mtp-small.gguf"


def test_suggest_args_ห้ามมี_flag_c():
    draft = HfFile(path="mtp-foo.gguf", size=300_000_000, sha256=None)
    mmproj = HfFile(path="mmproj-BF16.gguf", size=900_000_000, sha256=None)
    group = QuantGroup(key="Q8_0", files=[], total_bytes=0, shard_count=1, companions=[draft, mmproj])

    args = suggest_args(group, "/models/foo")
    assert "-c" not in args.split()


# ---------------------------------------------------------------------------
# fetch_config
# ---------------------------------------------------------------------------


def test_fetch_config_follows_redirect_and_parses_json():
    payload = {"architectures": ["Qwen3ForCausalLM"], "max_position_embeddings": 40960}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/Qwen/Qwen3-8B-FP8/resolve/main/config.json"
        return httpx.Response(200, json=payload)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = fetch_config("Qwen/Qwen3-8B-FP8", client=client)

    assert result == payload


@pytest.mark.parametrize("status", [401, 403])
def test_fetch_config_gated_raises(status: int):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": "gated"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(GatedRepoError):
        fetch_config("some/gated-repo", client=client)


def test_fetch_config_404_raises_not_found():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "not found"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(RepoNotFoundError):
        fetch_config("nope/does-not-exist", client=client)


def test_fetch_config_sends_bearer_token_when_given():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    fetch_config("some/repo", token="hf_secrettoken", client=client)

    assert seen["auth"] == "Bearer hf_secrettoken"


# ---------------------------------------------------------------------------
# config_arch_ctx — arch/ctx ของ safetensors อ่านจาก config.json (top-level และ text_config)
# ---------------------------------------------------------------------------


def test_config_arch_ctx_top_level():
    config = _load_fixture("config_qwen3_fp8.json")
    arch, ctx = config_arch_ctx(config)
    assert arch == "Qwen3ForCausalLM"
    assert ctx == 40960


def test_config_arch_ctx_nested_text_config():
    """ข้อเท็จจริง verify แล้ว: โมเดล NVFP4 จริงมี max_position_embeddings อยู่ใน text_config เท่านั้น"""
    config = _load_fixture("config_nested_text_config.json")
    arch, ctx = config_arch_ctx(config)
    assert arch == "Qwen3ForCausalLM"
    assert ctx == 262144


def test_config_arch_ctx_top_level_wins_over_nested():
    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "max_position_embeddings": 40960,
        "text_config": {"max_position_embeddings": 262144},
    }
    arch, ctx = config_arch_ctx(config)
    assert ctx == 40960


def test_config_arch_ctx_missing_returns_none():
    assert config_arch_ctx({}) == (None, None)


# ---------------------------------------------------------------------------
# quant_label
# ---------------------------------------------------------------------------


def test_quant_label_fp8():
    config = _load_fixture("config_qwen3_fp8.json")
    assert quant_label(config) == "FP8"


def test_quant_label_modelopt_and_nvfp4_map_to_nvfp4():
    assert quant_label({"quantization_config": {"quant_method": "modelopt"}}) == "NVFP4"
    assert quant_label({"quantization_config": {"quant_method": "nvfp4"}}) == "NVFP4"


def test_quant_label_no_quant_config_uses_dtype():
    assert quant_label({"torch_dtype": "bfloat16"}) == "BF16"
    assert quant_label({"torch_dtype": "float16"}) == "F16"


def test_quant_label_fallback_safetensors():
    assert quant_label({}) == "safetensors"


def test_quant_label_compressed_tensors_nvfp4_regression():
    """Regression: unsloth/Qwen3.8-27B-NVFP4 มี quant_method == "compressed-tensors" (ชื่อ format
    ไม่ใช่ชื่อ quant) ⇒ ต้องอ่าน config_groups แล้วเลือกกลุ่มที่ num_bits ต่ำสุด (group_1: 4-bit float,
    group_size 16) ไม่ใช่ uppercase("compressed-tensors") หรือกลุ่มแรกที่เจอ (group_0: 8-bit → "FP8" ผิด)
    """
    config = _load_fixture("config_qwen38_nvfp4.json")
    assert quant_label(config) == "NVFP4"


def test_quant_label_compressed_tensors_single_fp8_group():
    config = {
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "config_groups": {
                "group_0": {"weights": {"type": "float", "num_bits": 8, "group_size": None, "strategy": "channel"}},
            },
        }
    }
    assert quant_label(config) == "FP8"


def test_quant_label_compressed_tensors_fp4_without_group_size_16():
    config = {
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "config_groups": {
                "group_0": {"weights": {"type": "float", "num_bits": 4, "group_size": None, "strategy": "tensor"}},
            },
        }
    }
    assert quant_label(config) == "FP4"

    config_other_group_size = {
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "config_groups": {
                "group_0": {"weights": {"type": "float", "num_bits": 4, "group_size": 32, "strategy": "tensor"}},
            },
        }
    }
    assert quant_label(config_other_group_size) == "FP4"


def test_quant_label_compressed_tensors_int_group():
    config = {
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "config_groups": {
                "group_0": {"weights": {"type": "int", "num_bits": 8, "group_size": 128, "strategy": "group"}},
            },
        }
    }
    assert quant_label(config) == "INT8"


def test_quant_label_compressed_tensors_missing_config_groups_falls_back():
    assert quant_label({"quantization_config": {"quant_method": "compressed-tensors"}}) == "COMPRESSED-TENSORS"
    assert (
        quant_label({"quantization_config": {"quant_method": "compressed-tensors", "config_groups": {}}})
        == "COMPRESSED-TENSORS"
    )


# ---------------------------------------------------------------------------
# has_safetensors / group_weights
# ---------------------------------------------------------------------------


def test_has_safetensors_true_for_fp8_repo():
    files = _load("hf_qwen3-8b-fp8_blobs.json")
    assert has_safetensors(files) is True


def test_has_safetensors_false_for_gguf_repo():
    files = _load("hf_qwen38-27b-gguf_blobs.json")
    assert has_safetensors(files) is False


def test_group_weights_no_safetensors_returns_empty():
    files = _load("hf_qwen38-27b-gguf_blobs.json")
    assert group_weights(files, {}) == []


def test_group_weights_nvfp4_matches_catalog_yaml_dl_list():
    """ตัวไม่แปรผันที่ verify แล้ว: กติกาคัดไฟล์ต้องได้ผลลัพธ์เดียวกับที่ vendor curate ไว้เองใน catalog.yaml:106-117"""
    import pathlib

    import yaml

    catalog_path = pathlib.Path(__file__).parent.parent / "catalog.yaml"
    catalog_raw = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    entry = next(m for m in catalog_raw["models"] if m["id"] == "qwen38-nvfp4-vllm")
    expected_basenames = {url.rsplit("/", 1)[-1] for url in entry["dl"]}

    files = _load("hf_qwen38-27b-nvfp4_blobs.json")
    groups = group_weights(files, {})
    assert len(groups) == 1

    got_basenames = {f.path.rsplit("/", 1)[-1] for f in groups[0].files}
    assert got_basenames == expected_basenames


def test_group_weights_nvfp4_mtp_stays_in_files_not_companions():
    files = _load("hf_qwen38-27b-nvfp4_blobs.json")
    groups = group_weights(files, {})
    assert len(groups) == 1
    g = groups[0]

    assert any(f.path == "model_mtp.safetensors" for f in g.files)
    assert g.companions == []


def test_group_weights_key_from_quant_label():
    files = _load("hf_qwen3-8b-fp8_blobs.json")
    config = _load_fixture("config_qwen3_fp8.json")
    groups = group_weights(files, config)

    assert len(groups) == 1
    assert groups[0].key == "FP8"
    assert groups[0].format == "safetensors"


def test_group_weights_shard_count_counts_safetensors_files_only():
    files = _load("hf_qwen3-8b-fp8_blobs.json")
    config = _load_fixture("config_qwen3_fp8.json")
    groups = group_weights(files, config)

    assert groups[0].shard_count == 2  # model-00001-of-00002 / model-00002-of-00002
    assert groups[0].total_bytes == sum(f.size for f in groups[0].files)


def test_group_weights_excludes_readme_gitattributes_license():
    files = _load("hf_qwen3-8b-fp8_blobs.json")
    groups = group_weights(files, {})
    basenames = {f.path for f in groups[0].files}

    assert "README.md" not in basenames
    assert ".gitattributes" not in basenames
    assert "LICENSE" not in basenames


# ---------------------------------------------------------------------------
# GGUF regression: group_quants ต้องไม่เปลี่ยนพฤติกรรม + format ต้องเป็น "gguf"
# ---------------------------------------------------------------------------


def test_group_quants_format_is_gguf():
    repo_json = _load_fixture("hf_qwen38-27b-gguf_blobs.json")
    files = list_files(repo_json)
    groups = group_quants(files)

    assert len(groups) > 0
    assert all(g.format == "gguf" for g in groups)


# ---------------------------------------------------------------------------
# vllm_parsers / vllm_engine_env
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_type, expected",
    [
        ("qwen3", {"VLLM_TOOL_PARSER": "qwen3_xml", "VLLM_REASONING_PARSER": "qwen3"}),
        ("qwen3_5", {"VLLM_TOOL_PARSER": "qwen3_xml", "VLLM_REASONING_PARSER": "qwen3"}),
        ("qwen3_moe", {"VLLM_TOOL_PARSER": "qwen3_xml", "VLLM_REASONING_PARSER": "qwen3"}),
        ("llama", {"VLLM_TOOL_PARSER": "llama3_json", "VLLM_REASONING_PARSER": ""}),
        ("mistral", {"VLLM_TOOL_PARSER": "mistral", "VLLM_REASONING_PARSER": ""}),
        ("deepseek_v3", {"VLLM_TOOL_PARSER": "deepseek_v3", "VLLM_REASONING_PARSER": "deepseek_r1"}),
        ("some-unknown-family", {"VLLM_TOOL_PARSER": "", "VLLM_REASONING_PARSER": ""}),
        (None, {"VLLM_TOOL_PARSER": "", "VLLM_REASONING_PARSER": ""}),
    ],
)
def test_vllm_parsers_by_family(model_type, expected):
    assert vllm_parsers(model_type) == expected


def test_vllm_engine_env_disables_speculative_when_no_mtp_file():
    """Qwen/Qwen3-8B-FP8 ไม่มี model_mtp.safetensors → ต้องปิด MTP ชัดเจน"""
    files = _load("hf_qwen3-8b-fp8_blobs.json")
    config = _load_fixture("config_qwen3_fp8.json")
    env = vllm_engine_env(files, config)

    assert env["VLLM_SPECULATIVE"] == ""
    assert env["VLLM_TOOL_PARSER"] == "qwen3_xml"
    assert env["VLLM_REASONING_PARSER"] == "qwen3"


def test_vllm_engine_env_omits_speculative_key_when_mtp_present():
    """unsloth/Qwen3.8-27B-NVFP4 มี model_mtp.safetensors → ปล่อยให้ default (เปิด) ของ vllm.sh ทำงาน"""
    files = _load("hf_qwen38-27b-nvfp4_blobs.json")
    env = vllm_engine_env(files, {"model_type": "qwen3"})

    assert "VLLM_SPECULATIVE" not in env
