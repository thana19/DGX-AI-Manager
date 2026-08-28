"""test สำหรับ server/gguf.py — อ่าน arch/context_length จาก GGUF header

ใช้ fixture จริงที่ตัดมาจากไฟล์โมเดลจริง (tests/fixtures/gguf_head_*.bin)
ไม่ยิงเน็ตจริงในเทสของ fetch_header — ใช้ httpx.MockTransport ป้อนข้อมูลแทน
"""
from __future__ import annotations

import os

import httpx
import pytest

from server.gguf import GgufInfo, NotGgufError, fetch_header, parse_header

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _read_fixture(name: str) -> bytes:
    with open(os.path.join(FIXTURES, name), "rb") as f:
        return f.read()


# ---------------------------------------------------------------------------
# parse_header
# ---------------------------------------------------------------------------


def test_parse_header_qwen35():
    data = _read_fixture("gguf_head_qwen35_262144.bin")
    info = parse_header(data)

    assert isinstance(info, GgufInfo)
    assert info.arch == "qwen35"
    assert info.context_length == 262144
    assert info.name == "Qwen3.8-27B"
    assert info.size_label == "27B"
    assert info.version == 3
    assert info.tensor_count == 866
    assert info.kv_count == 50
    # หยุดอ่านทันทีที่ได้ arch + context_length ครบ → อ่านไม่ครบ kv_count แน่นอน
    assert info.kv_read < 50
    assert info.kv_read > 0


def test_parse_header_glm53():
    data = _read_fixture("gguf_head_glm53_1mb.bin")
    info = parse_header(data)

    assert info.arch == "glm5next"
    assert info.context_length == 1048576
    assert info.version == 3
    assert info.kv_count == 72
    assert info.kv_read < 72
    assert info.kv_read > 0


def test_parse_header_truncated_before_any_kv_does_not_raise():
    # ตัดแค่ตรง header (magic/version/tensor_count/kv_count) — ยังไม่ทันอ่าน key ตัวแรกเลย
    data = _read_fixture("gguf_head_qwen35_262144.bin")[:24]
    info = parse_header(data)

    assert isinstance(info, GgufInfo)
    assert info.arch is None
    assert info.context_length is None
    assert info.version == 3
    assert info.tensor_count == 866
    assert info.kv_count == 50
    assert info.kv_read == 0


def test_parse_header_truncated_mid_kv_does_not_raise():
    # ตัดกลาง KV ตัวที่สอง (general.type) — อ่าน general.architecture จบแล้วพอดี แต่ตัวถัดไปยังไม่จบ
    data = _read_fixture("gguf_head_qwen35_262144.bin")[:100]
    info = parse_header(data)

    assert isinstance(info, GgufInfo)
    assert info.arch == "qwen35"  # อ่านจบตัวแรกแล้ว
    assert info.context_length is None  # ยังไปไม่ถึง
    assert info.version == 3
    assert info.tensor_count == 866
    assert info.kv_count == 50
    assert info.kv_read == 1


def test_parse_header_mid_kv_truncation_does_not_raise():
    # ตัดกลาง KV entry (ไม่ใช่ตัดตรงขอบ) — ต้องไม่ raise เช่นกัน
    data = _read_fixture("gguf_head_qwen35_262144.bin")[:300]
    info = parse_header(data)

    assert isinstance(info, GgufInfo)
    # ที่ offset 300 ได้ผ่าน general.architecture ไปแล้วอย่างน้อย
    assert info.arch == "qwen35"
    assert info.kv_read < info.kv_count


def test_parse_header_not_gguf_raises():
    with pytest.raises(NotGgufError):
        parse_header(b"NOTGGUF" + b"\x00" * 100)


def test_parse_header_empty_bytes_raises_not_gguf():
    with pytest.raises(NotGgufError):
        parse_header(b"")


# ---------------------------------------------------------------------------
# fetch_header
# ---------------------------------------------------------------------------


def test_fetch_header_first_range_is_enough():
    full = _read_fixture("gguf_head_qwen35_262144.bin")
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        rng = request.headers["range"]
        assert rng == "bytes=0-262143"
        body = full[:262144]
        return httpx.Response(
            206,
            content=body,
            headers={
                "content-range": f"bytes 0-{len(body) - 1}/9828981664",
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    info, total_size = fetch_header(
        "https://huggingface.co/foo/bar/resolve/main/model.gguf",
        client=client,
        first_bytes=262144,
        max_bytes=1048576,
    )

    assert request_count == 1
    assert info.arch == "qwen35"
    assert info.context_length == 262144
    assert total_size == 9828981664


def test_fetch_header_expands_when_first_range_insufficient():
    full = _read_fixture("gguf_head_qwen35_262144.bin")
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        rng = request.headers["range"]
        if request_count == 1:
            # รอบแรกให้แค่ 16 ไบต์แรก (header เปล่า ไม่มี KV เลย) — ไม่พอแน่นอน
            assert rng == "bytes=0-15"
            body = full[:16]
        else:
            assert rng == "bytes=0-1048575"
            body = full  # fixture มีแค่ 262144 ไบต์ mock ก็ตอบเท่าที่มี
        return httpx.Response(
            206,
            content=body,
            headers={"content-range": f"bytes 0-{len(body) - 1}/9828981664"},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    info, total_size = fetch_header(
        "https://huggingface.co/foo/bar/resolve/main/model.gguf",
        client=client,
        first_bytes=16,
        max_bytes=1048576,
    )

    assert request_count == 2
    assert info.arch == "qwen35"
    assert info.context_length == 262144
    assert total_size == 9828981664


def test_fetch_header_no_content_range_returns_none_size():
    full = _read_fixture("gguf_head_qwen35_262144.bin")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(206, content=full[:262144])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    info, total_size = fetch_header(
        "https://huggingface.co/foo/bar/resolve/main/model.gguf",
        client=client,
    )

    assert total_size is None
    assert info.arch == "qwen35"


def test_fetch_header_follows_redirect():
    full = _read_fixture("gguf_head_qwen35_262144.bin")

    def handler(request: httpx.Request) -> httpx.Response:
        if "cdn-lfs" not in str(request.url):
            # HF จริง ๆ 302 ไป CDN เสมอ
            return httpx.Response(
                302,
                headers={"location": "https://cdn-lfs.example.com/model.gguf"},
            )
        body = full[:262144]
        return httpx.Response(
            206,
            content=body,
            headers={"content-range": f"bytes 0-{len(body) - 1}/9828981664"},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    info, total_size = fetch_header(
        "https://huggingface.co/foo/bar/resolve/main/model.gguf",
        client=client,
    )

    assert info.arch == "qwen35"
    assert total_size == 9828981664


def test_fetch_header_creates_own_client_when_not_given(monkeypatch):
    full = _read_fixture("gguf_head_qwen35_262144.bin")

    def handler(request: httpx.Request) -> httpx.Response:
        body = full[:262144]
        return httpx.Response(
            206,
            content=body,
            headers={"content-range": f"bytes 0-{len(body) - 1}/9828981664"},
        )

    real_client_cls = httpx.Client

    def fake_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client_cls(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", fake_client)

    info, total_size = fetch_header("https://huggingface.co/foo/bar/resolve/main/model.gguf")

    assert info.arch == "qwen35"
    assert total_size == 9828981664
