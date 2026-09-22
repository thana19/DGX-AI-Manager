"""test สำหรับ server/downloads.py — แยก "ดาวน์โหลด" ออกจาก "โหลดขึ้นแรม"

ห้ามรัน aria2 จริง ห้ามยิงเน็ต — ใช้ httpx.MockTransport จำลอง aria2 JSON-RPC ทั้งหมด
(ดู FakeAria2Server ด้านล่าง — จำลองพฤติกรรมจริงที่ verify แล้วบนเครื่อง ไม่ใช่การเดา)
"""
from __future__ import annotations

import json
import os

import httpx
import pytest

from server.downloads import (
    Aria2Client,
    Aria2Error,
    DownloadJob,
    DownloadManager,
    DownloadState,
    FileProgress,
    summarize_state,
)


# ---------------------------------------------------------------------------
# FakeAria2Server — จำลอง aria2 JSON-RPC ในหน่วยความจำ (ไม่มี network จริง)
# ---------------------------------------------------------------------------


class FakeAria2Server:
    """จำลอง daemon aria2 ตัวเดียว — เก็บ state ของแต่ละ gid ในตัวเอง

    ให้ test เรียก .complete(gid) / .fail(gid, msg) / .set_progress(...) เพื่อบังคับ
    ให้ tellStatus คืนค่าตามที่ต้องการ จำลองความคืบหน้าของการโหลดจริง
    """

    def __init__(self, secret: str = ""):
        self.secret = secret
        self._next_gid = 1
        self._jobs: dict[str, dict] = {}
        self.dead = False  # จำลอง daemon ตาย — ทุก call จะพัง
        self.calls: list[tuple[str, list]] = []

    def _check_token(self, params: list) -> list:
        if self.secret:
            assert params and params[0] == f"token:{self.secret}"
            return params[1:]
        return params

    def complete(self, gid: str):
        self._jobs[gid]["status"] = "complete"
        self._jobs[gid]["completedLength"] = self._jobs[gid]["totalLength"]

    def fail(self, gid: str, message: str, code: str = "1"):
        self._jobs[gid]["status"] = "error"
        self._jobs[gid]["errorCode"] = code
        self._jobs[gid]["errorMessage"] = message

    def set_progress(self, gid: str, *, done: int, total: int, speed: int = 0):
        j = self._jobs[gid]
        j["completedLength"] = done
        j["totalLength"] = total
        j["downloadSpeed"] = speed

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.calls.append((body["method"], body["params"]))

        if self.dead:
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "error": {"code": -1, "message": "connection refused"}})

        method = body["method"]
        params = self._check_token(body["params"])

        if method == "aria2.addUri":
            urls, opts = params
            gid = f"gid{self._next_gid:08x}"
            self._next_gid += 1
            dest = os.path.join(opts.get("dir", ""), opts.get("out", ""))
            self._jobs[gid] = {
                "status": "active",
                "totalLength": 0,
                "completedLength": 0,
                "downloadSpeed": 0,
                "errorCode": "0",
                "errorMessage": "",
                "files": [{"path": dest}],
                "url": urls[0],
            }
            result = gid
        elif method == "aria2.tellStatus":
            gid, keys = params
            j = self._jobs[gid]
            result = {k: str(j.get(k, "")) if k != "files" else j["files"] for k in keys}
        elif method == "aria2.pause":
            gid = params[0]
            self._jobs[gid]["status"] = "paused"
            result = gid
        elif method == "aria2.unpause":
            gid = params[0]
            self._jobs[gid]["status"] = "active"
            result = gid
        elif method == "aria2.remove":
            gid = params[0]
            self._jobs[gid]["status"] = "removed"
            result = gid
        else:
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "error": {"code": -32601, "message": f"method not found: {method}"}})

        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})


def _client(server: FakeAria2Server, secret: str = "") -> Aria2Client:
    transport = httpx.MockTransport(server.handler)
    http_client = httpx.Client(transport=transport)
    return Aria2Client("http://127.0.0.1:6800/jsonrpc", secret=secret, client=http_client)


# ---------------------------------------------------------------------------
# Aria2Client
# ---------------------------------------------------------------------------


def test_aria2client_inserts_token_when_secret_set():
    server = FakeAria2Server(secret="s3cr3t")
    client = _client(server, secret="s3cr3t")

    gid = client.add_uri("http://example.com/f.gguf", "/tmp/dest", "f.gguf")

    assert gid == "gid00000001"
    method, params = server.calls[0]
    assert params[0] == "token:s3cr3t"


def test_aria2client_no_token_when_no_secret():
    server = FakeAria2Server(secret="")
    client = _client(server, secret="")

    client.add_uri("http://example.com/f.gguf", "/tmp/dest", "f.gguf")

    method, params = server.calls[0]
    # ไม่มี secret → params ตัวแรกต้องเป็น list ของ url ไม่ใช่ token
    assert params[0] == ["http://example.com/f.gguf"]


def test_aria2client_add_uri_includes_header_option_when_given():
    server = FakeAria2Server()
    client = _client(server)

    client.add_uri("http://example.com/f.gguf", "/tmp/dest", "f.gguf", headers=["Authorization: Bearer tok"])

    method, params = server.calls[0]
    urls, opts = params
    assert opts["header"] == ["Authorization: Bearer tok"]


def test_aria2client_add_uri_omits_header_option_when_not_given():
    server = FakeAria2Server()
    client = _client(server)

    client.add_uri("http://example.com/f.gguf", "/tmp/dest", "f.gguf")

    method, params = server.calls[0]
    urls, opts = params
    assert "header" not in opts


def test_aria2client_raises_on_rpc_error():
    server = FakeAria2Server()
    server.dead = True
    client = _client(server)

    with pytest.raises(Aria2Error):
        client.add_uri("http://example.com/f.gguf", "/tmp/dest", "f.gguf")


def test_aria2client_tell_status_converts_string_numbers_to_int():
    server = FakeAria2Server()
    client = _client(server)
    gid = client.add_uri("http://example.com/f.gguf", "/tmp/dest", "f.gguf")
    server.set_progress(gid, done=1024, total=2048, speed=512)

    status = client.tell_status(gid)

    assert status["totalLength"] == 2048
    assert status["completedLength"] == 1024
    assert status["downloadSpeed"] == 512
    assert isinstance(status["totalLength"], int)


def test_aria2client_pause_unpause_remove_call_correct_methods():
    server = FakeAria2Server()
    client = _client(server)
    gid = client.add_uri("http://example.com/f.gguf", "/tmp/dest", "f.gguf")

    client.pause(gid)
    assert server._jobs[gid]["status"] == "paused"
    client.unpause(gid)
    assert server._jobs[gid]["status"] == "active"
    client.remove(gid)
    assert server._jobs[gid]["status"] == "removed"


# ---------------------------------------------------------------------------
# summarize_state — ครบ 6 กรณี (กติกาข้อ 1)
# ---------------------------------------------------------------------------


def _fp(state: DownloadState, error=None) -> FileProgress:
    return FileProgress(url="u", dest="d", gid=None, total_bytes=100, done_bytes=0, state=state, error=error)


def test_summarize_state_error_if_any_error():
    files = [_fp(DownloadState.DONE), _fp(DownloadState.ERROR, "boom")]
    assert summarize_state(files) == DownloadState.ERROR


def test_summarize_state_done_if_all_done():
    files = [_fp(DownloadState.DONE), _fp(DownloadState.DONE)]
    assert summarize_state(files) == DownloadState.DONE


def test_summarize_state_active_if_any_active():
    files = [_fp(DownloadState.DONE), _fp(DownloadState.ACTIVE)]
    assert summarize_state(files) == DownloadState.ACTIVE


def test_summarize_state_paused_if_all_paused():
    files = [_fp(DownloadState.PAUSED), _fp(DownloadState.PAUSED)]
    assert summarize_state(files) == DownloadState.PAUSED


def test_summarize_state_cancelled_if_any_cancelled():
    files = [_fp(DownloadState.QUEUED), _fp(DownloadState.CANCELLED)]
    assert summarize_state(files) == DownloadState.CANCELLED


def test_summarize_state_queued_otherwise():
    files = [_fp(DownloadState.QUEUED), _fp(DownloadState.QUEUED)]
    assert summarize_state(files) == DownloadState.QUEUED


# ---------------------------------------------------------------------------
# DownloadJob properties — percent / eta_seconds
# ---------------------------------------------------------------------------


def test_job_percent_zero_when_total_zero():
    job = DownloadJob(id="j1", model_id="m", files=[], created_at=0.0, state=DownloadState.QUEUED)
    assert job.percent == 0.0


def test_job_percent_and_speed_and_eta():
    fp = FileProgress(url="u", dest="d", gid="g1", total_bytes=1000, done_bytes=250, state=DownloadState.ACTIVE, error=None)
    job = DownloadJob(id="j1", model_id="m", files=[fp], created_at=0.0, state=DownloadState.ACTIVE)
    job._current_speed_bps = 250

    assert job.total_bytes == 1000
    assert job.done_bytes == 250
    assert job.percent == 25.0
    assert job.speed_bps == 250
    assert job.eta_seconds == 3  # (1000-250)/250 = 3


def test_job_eta_none_when_speed_zero():
    fp = FileProgress(url="u", dest="d", gid="g1", total_bytes=1000, done_bytes=250, state=DownloadState.ACTIVE, error=None)
    job = DownloadJob(id="j1", model_id="m", files=[fp], created_at=0.0, state=DownloadState.ACTIVE)
    job._current_speed_bps = 0

    assert job.eta_seconds is None


# ---------------------------------------------------------------------------
# DownloadManager._start_job — token_loader: ส่ง Authorization ให้ aria2 เฉพาะ host huggingface.co
# (ดู task ส่วนที่ 3) — token อ่านตอน _start_job ไม่ใช่ตอน submit
# ---------------------------------------------------------------------------


def test_start_job_attaches_authorization_header_for_huggingface_host(tmp_path):
    server = FakeAria2Server()
    client = _client(server)
    mgr = DownloadManager(client, state_path=str(tmp_path / "downloads.json"), token_loader=lambda: "hf_tok123")

    mgr.submit("model-a", [("https://huggingface.co/org/repo/resolve/main/model.gguf", str(tmp_path / "model.gguf"))])

    method, params = server.calls[0]
    urls, opts = params
    assert opts["header"] == ["Authorization: Bearer hf_tok123"]


def test_start_job_no_header_for_non_huggingface_host(tmp_path):
    server = FakeAria2Server()
    client = _client(server)
    mgr = DownloadManager(client, state_path=str(tmp_path / "downloads.json"), token_loader=lambda: "hf_tok123")

    mgr.submit("model-a", [("https://example.com/model.gguf", str(tmp_path / "model.gguf"))])

    method, params = server.calls[0]
    urls, opts = params
    assert "header" not in opts


def test_start_job_no_header_when_token_loader_returns_none(tmp_path):
    server = FakeAria2Server()
    client = _client(server)
    mgr = DownloadManager(client, state_path=str(tmp_path / "downloads.json"), token_loader=lambda: None)

    mgr.submit("model-a", [("https://huggingface.co/org/repo/resolve/main/model.gguf", str(tmp_path / "model.gguf"))])

    method, params = server.calls[0]
    urls, opts = params
    assert "header" not in opts


def test_start_job_no_header_when_no_token_loader_given(tmp_path):
    server = FakeAria2Server()
    client = _client(server)
    mgr = DownloadManager(client, state_path=str(tmp_path / "downloads.json"))

    mgr.submit("model-a", [("https://huggingface.co/org/repo/resolve/main/model.gguf", str(tmp_path / "model.gguf"))])

    method, params = server.calls[0]
    urls, opts = params
    assert "header" not in opts


def test_start_job_attaches_header_for_huggingface_subdomain_host(tmp_path):
    server = FakeAria2Server()
    client = _client(server)
    mgr = DownloadManager(client, state_path=str(tmp_path / "downloads.json"), token_loader=lambda: "hf_tok123")

    mgr.submit("model-a", [("https://cdn-lfs.huggingface.co/org/repo/model.gguf", str(tmp_path / "model.gguf"))])

    method, params = server.calls[0]
    urls, opts = params
    assert opts["header"] == ["Authorization: Bearer hf_tok123"]


# ---------------------------------------------------------------------------
# DownloadManager.submit — ข้ามไฟล์ที่มีครบแล้ว
# ---------------------------------------------------------------------------


def test_submit_marks_existing_complete_file_as_done_without_calling_aria2(tmp_path):
    dest = tmp_path / "model.gguf"
    dest.write_bytes(b"x" * 1024)

    server = FakeAria2Server()
    client = _client(server)
    mgr = DownloadManager(client, state_path=str(tmp_path / "downloads.json"))

    job = mgr.submit("model-a", [("http://example.com/model.gguf", str(dest))])

    assert job.files[0].state == DownloadState.DONE
    assert job.files[0].total_bytes == 1024
    assert job.state == DownloadState.DONE
    assert server.calls == []  # ไม่ส่งให้ aria2 เลย


def test_submit_sends_file_with_dangling_aria2_control_file_to_aria2(tmp_path):
    dest = tmp_path / "model.gguf"
    dest.write_bytes(b"x" * 512)  # โหลดค้างไว้ครึ่งหนึ่ง
    (tmp_path / "model.gguf.aria2").write_bytes(b"control")

    server = FakeAria2Server()
    client = _client(server)
    mgr = DownloadManager(client, state_path=str(tmp_path / "downloads.json"))

    job = mgr.submit("model-a", [("http://example.com/model.gguf", str(dest))])

    assert job.files[0].state == DownloadState.ACTIVE
    assert job.files[0].gid is not None
    assert len(server.calls) == 1
    assert server.calls[0][0] == "aria2.addUri"


def test_submit_creates_dest_directory(tmp_path):
    dest = tmp_path / "sub" / "dir" / "model.gguf"
    server = FakeAria2Server()
    client = _client(server)
    mgr = DownloadManager(client, state_path=str(tmp_path / "downloads.json"))

    mgr.submit("model-a", [("http://example.com/model.gguf", str(dest))])

    assert (tmp_path / "sub" / "dir").is_dir()


# ---------------------------------------------------------------------------
# คิว — 1 job ต่อครั้ง
# ---------------------------------------------------------------------------


def test_second_job_queued_until_first_finishes(tmp_path):
    server = FakeAria2Server()
    client = _client(server)
    mgr = DownloadManager(client, state_path=str(tmp_path / "downloads.json"))

    job1 = mgr.submit("model-a", [("http://example.com/a.gguf", str(tmp_path / "a.gguf"))])
    job2 = mgr.submit("model-b", [("http://example.com/b.gguf", str(tmp_path / "b.gguf"))])

    assert job1.state == DownloadState.ACTIVE
    assert job2.state == DownloadState.QUEUED
    assert job2.files[0].gid is None

    gid1 = job1.files[0].gid
    server.complete(gid1)
    mgr.refresh()

    job1 = mgr.get(job1.id)
    job2 = mgr.get(job2.id)
    assert job1.state == DownloadState.DONE
    assert job2.state == DownloadState.ACTIVE
    assert job2.files[0].gid is not None


# ---------------------------------------------------------------------------
# refresh — แปลงสถานะจาก aria2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "aria2_status,expected",
    [
        ("active", DownloadState.ACTIVE),
        ("complete", DownloadState.DONE),
        ("paused", DownloadState.PAUSED),
        ("error", DownloadState.ERROR),
    ],
)
def test_refresh_maps_aria2_status_to_download_state(tmp_path, aria2_status, expected):
    server = FakeAria2Server()
    client = _client(server)
    mgr = DownloadManager(client, state_path=str(tmp_path / "downloads.json"))

    job = mgr.submit("model-a", [("http://example.com/a.gguf", str(tmp_path / "a.gguf"))])
    gid = job.files[0].gid

    if aria2_status == "complete":
        server.complete(gid)
    elif aria2_status == "error":
        server.fail(gid, "network unreachable")
    else:
        server._jobs[gid]["status"] = aria2_status

    mgr.refresh()

    job = mgr.get(job.id)
    assert job.files[0].state == expected
    assert job.state == expected


def test_refresh_updates_progress_and_speed(tmp_path):
    server = FakeAria2Server()
    client = _client(server)
    mgr = DownloadManager(client, state_path=str(tmp_path / "downloads.json"))

    job = mgr.submit("model-a", [("http://example.com/a.gguf", str(tmp_path / "a.gguf"))])
    gid = job.files[0].gid
    server.set_progress(gid, done=500, total=2000, speed=100)

    mgr.refresh()

    job = mgr.get(job.id)
    assert job.done_bytes == 500
    assert job.total_bytes == 2000
    assert job.speed_bps == 100
    assert job.eta_seconds == 15


# ---------------------------------------------------------------------------
# pause / resume / cancel
# ---------------------------------------------------------------------------


def test_pause_and_resume_job(tmp_path):
    server = FakeAria2Server()
    client = _client(server)
    mgr = DownloadManager(client, state_path=str(tmp_path / "downloads.json"))

    job = mgr.submit("model-a", [("http://example.com/a.gguf", str(tmp_path / "a.gguf"))])

    assert mgr.pause(job.id) is True
    job = mgr.get(job.id)
    assert job.state == DownloadState.PAUSED

    assert mgr.resume(job.id) is True
    job = mgr.get(job.id)
    assert job.state == DownloadState.ACTIVE


def test_pause_unknown_job_returns_false(tmp_path):
    server = FakeAria2Server()
    client = _client(server)
    mgr = DownloadManager(client, state_path=str(tmp_path / "downloads.json"))

    assert mgr.pause("does-not-exist") is False


def test_cancel_removes_gid_and_dangling_control_file_but_keeps_downloaded_file(tmp_path):
    dest = tmp_path / "a.gguf"
    dest.write_bytes(b"partial")
    (tmp_path / "a.gguf.aria2").write_bytes(b"control")

    server = FakeAria2Server()
    client = _client(server)
    mgr = DownloadManager(client, state_path=str(tmp_path / "downloads.json"))

    job = mgr.submit("model-a", [("http://example.com/a.gguf", str(dest))])
    gid = job.files[0].gid

    assert mgr.cancel(job.id) is True

    job = mgr.get(job.id)
    assert job.state == DownloadState.CANCELLED
    assert server._jobs[gid]["status"] == "removed"
    assert dest.exists()  # ไฟล์ที่โหลดได้แล้วต้องไม่ถูกลบ
    assert not (tmp_path / "a.gguf.aria2").exists()  # ไฟล์ control ต้องถูกลบ


def test_cancel_starts_next_queued_job(tmp_path):
    server = FakeAria2Server()
    client = _client(server)
    mgr = DownloadManager(client, state_path=str(tmp_path / "downloads.json"))

    job1 = mgr.submit("model-a", [("http://example.com/a.gguf", str(tmp_path / "a.gguf"))])
    job2 = mgr.submit("model-b", [("http://example.com/b.gguf", str(tmp_path / "b.gguf"))])

    mgr.cancel(job1.id)

    job2 = mgr.get(job2.id)
    assert job2.state == DownloadState.ACTIVE
    assert job2.files[0].gid is not None


# ---------------------------------------------------------------------------
# state รอด restart
# ---------------------------------------------------------------------------


def test_state_survives_restart(tmp_path, monkeypatch):
    state_file = tmp_path / "downloads.json"
    monkeypatch.setenv("AISERVER2_STATE", str(tmp_path))

    server = FakeAria2Server()
    client = _client(server)
    mgr = DownloadManager(client, state_path=str(state_file))
    job = mgr.submit("model-a", [("http://example.com/a.gguf", str(tmp_path / "a.gguf"))])

    assert state_file.exists()

    mgr2 = DownloadManager(client, state_path=str(state_file))
    job2 = mgr2.get(job.id)

    assert job2 is not None
    assert job2.model_id == "model-a"
    assert job2.files[0].url == "http://example.com/a.gguf"
    assert job2.state == job.state


def test_default_state_path_uses_paths_state(tmp_path, monkeypatch):
    monkeypatch.setenv("AISERVER2_STATE", str(tmp_path))
    server = FakeAria2Server()
    client = _client(server)

    mgr = DownloadManager(client)
    mgr.submit("model-a", [("http://example.com/a.gguf", str(tmp_path / "a.gguf"))])

    assert (tmp_path / "downloads.json").exists()


# ---------------------------------------------------------------------------
# daemon ตายตอน refresh
# ---------------------------------------------------------------------------


def test_refresh_marks_job_error_when_daemon_dead_without_raising(tmp_path):
    server = FakeAria2Server()
    client = _client(server)
    mgr = DownloadManager(client, state_path=str(tmp_path / "downloads.json"))

    job = mgr.submit("model-a", [("http://example.com/a.gguf", str(tmp_path / "a.gguf"))])

    server.dead = True
    mgr.refresh()  # ต้องไม่ raise ออกมา

    job = mgr.get(job.id)
    assert job.state == DownloadState.ERROR
    assert job.files[0].error  # มีข้อความไทยบอกวิธีแก้


# --- เทียบขนาดกับที่ HF บอก (กันไฟล์ค้างที่ไม่มี .aria2 ถูกนับว่าครบ) ---------


def _mgr(tmp_path):
    server = FakeAria2Server()
    return server, DownloadManager(_client(server), state_path=str(tmp_path / "downloads.json"))


def test_submit_ไฟล์ขนาดไม่ตรงที่คาด_ต้องโหลดต่อไม่ใช่ข้าม(tmp_path):
    dest = tmp_path / "model.gguf"
    dest.write_bytes(b"x" * 1000)  # ค้างไว้ 1000 byte แต่ของจริง 999999 · ไม่มี .aria2 ให้เห็น

    server, mgr = _mgr(tmp_path)
    job = mgr.submit("model-a", [("http://example.com/model.gguf", str(dest), 999999)])

    assert job.files[0].state is not DownloadState.DONE
    assert job.files[0].gid is not None, "ต้องถูกส่งให้ aria2 โหลดต่อ"


def test_submit_ไฟล์ขนาดตรงที่คาด_ข้ามได้(tmp_path):
    dest = tmp_path / "model.gguf"
    dest.write_bytes(b"x" * 5000)

    server, mgr = _mgr(tmp_path)
    job = mgr.submit("model-a", [("http://example.com/model.gguf", str(dest), 5000)])

    assert job.files[0].state == DownloadState.DONE
    assert job.files[0].gid is None, "ไฟล์ครบแล้วห้ามส่งให้ aria2"
    assert server.calls == []


def test_submit_ขนาดต่างในระยะผ่อนผัน_ยังถือว่าครบ(tmp_path):
    dest = tmp_path / "model.gguf"
    dest.write_bytes(b"x" * 5000)

    server, mgr = _mgr(tmp_path)
    job = mgr.submit("model-a", [("http://example.com/model.gguf", str(dest), 5100)])

    assert job.files[0].state == DownloadState.DONE
