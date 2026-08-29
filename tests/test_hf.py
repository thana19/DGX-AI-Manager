"""test สำหรับ server/hf.py — แปลง HF repo id เป็นรายการ quant ให้ผู้ใช้เลือกก่อนดาวน์โหลด

ใช้ fixture จริงที่ดึงมาจาก HF API (?blobs=true) ของ tests/fixtures/hf_*.json
ไม่ยิงเน็ตจริงในเทสของ fetch_repo — ใช้ httpx.MockTransport ป้อนข้อมูลแทน
"""
from __future__ import annotations

import json
import os

import httpx
import pytest

from server.hf import (
    GatedRepoError,
    HfFile,
    QuantGroup,
    RepoNotFoundError,
    SearchHit,
    fetch_repo,
    group_quants,
    list_files,
    normalize_repo_id,
    resolve_url,
    search_models,
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
