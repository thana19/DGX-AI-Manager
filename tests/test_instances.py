"""test สำหรับ server/instances.py — สแกน instance ที่รันอยู่จริง + หยุด

ห้ามรัน ps/docker/fuser จริง ห้ามยิงเน็ต — ป้อน ps_output และ monkeypatch subprocess/helper ภายในโมดูล
"""
from __future__ import annotations

from server import instances

PS_LINE_REAL = (
    "2804047 2621440 /home/dgx/llama.cpp/build/bin/llama-server -m "
    "/home/dgx/models/gguf/X/Y-Q8_K_P.gguf --host 0.0.0.0 --port 8001 "
    "-c 262144 -ngl 99 --jinja --reasoning-format auto --metrics"
)


def _no_docker(monkeypatch):
    """ปิด path ของ docker ไปเลย — เทสที่ไม่สนใจ vLLM ใช้ helper นี้กันไม่ให้ subprocess จริงรัน"""
    monkeypatch.setattr(instances, "_docker_ps_names", lambda: set())


# ---------------------------------------------------------------------------
# scan() — แยก port/-m/-c จาก ps output
# ---------------------------------------------------------------------------


def test_scan_แยก_port_model_ctx_จาก_ps_line_จริง(monkeypatch):
    _no_docker(monkeypatch)
    result = instances.scan(ps_output=PS_LINE_REAL, probe=False)

    assert len(result) == 1
    inst = result[0]
    assert inst.port == 8001
    assert inst.engine == "llamacpp"
    assert inst.pid == 2804047
    assert inst.model_file == "Y-Q8_K_P.gguf"
    assert inst.model_path == "/home/dgx/models/gguf/X/Y-Q8_K_P.gguf"
    assert inst.ctx == 262144
    assert inst.rss_gb == 2.5  # 2621440 KiB / 1048576 = 2.5 GB
    assert inst.up is False  # probe=False → ไม่เช็ค


def test_scan_แยก_ds4_server_ได้ด้วย(monkeypatch):
    _no_docker(monkeypatch)
    line = "111 1024 /home/dgx/ds4/ds4-server -m /home/dgx/models/x.bin --port 8002"
    result = instances.scan(ps_output=line, probe=False)

    assert len(result) == 1
    assert result[0].engine == "ds4"
    assert result[0].port == 8002


def test_scan_ps_ที่ไม่มี_port_ถูกข้าม(monkeypatch):
    _no_docker(monkeypatch)
    line = "999 512 /home/dgx/llama.cpp/build/bin/llama-server -m /home/dgx/models/x.gguf -c 4096"
    result = instances.scan(ps_output=line, probe=False)
    assert result == []


def test_scan_ไม่มี_process_เลย_คืนลิสต์ว่าง(monkeypatch):
    _no_docker(monkeypatch)
    assert instances.scan(ps_output="", probe=False) == []


def test_scan_ข้ามแถวหัวตารางกับแถวไม่เกี่ยว(monkeypatch):
    _no_docker(monkeypatch)
    text = "\n".join([
        "  PID   RSS COMMAND",
        "42 2048 /usr/bin/python3 -m http.server --port 9999",  # ไม่ใช่ llama-server/ds4-server
        PS_LINE_REAL,
    ])
    result = instances.scan(ps_output=text, probe=False)
    assert len(result) == 1
    assert result[0].port == 8001


def test_scan_probe_true_เรียก_probe_up(monkeypatch):
    _no_docker(monkeypatch)
    monkeypatch.setattr(instances, "_probe_up", lambda port: port == 8001)
    result = instances.scan(ps_output=PS_LINE_REAL, probe=True)
    assert result[0].up is True


def test_scan_เจอ_vllm_container_บนพอร์ต_8000(monkeypatch):
    monkeypatch.setattr(instances, "_docker_ps_names", lambda: {"aiserver-vllm"})
    monkeypatch.setattr(instances, "_vllm_model_info", lambda: ("/models/foo", "foo"))

    result = instances.scan(ps_output="", probe=False)
    assert len(result) == 1
    inst = result[0]
    assert inst.port == 8000
    assert inst.engine == "vllm"
    assert inst.pid is None
    assert inst.model_file == "foo"


# ---------------------------------------------------------------------------
# stop()
# ---------------------------------------------------------------------------


def test_stop_ไม่มีอะไรบนพอร์ต_คืน_false(monkeypatch):
    monkeypatch.setattr(instances, "scan", lambda **kw: [])
    ok, message = instances.stop(8001)
    assert ok is False
    assert "8001" in message


def test_stop_เรียก_fuser_kill_และ_รอจนพอร์ตว่าง(monkeypatch):
    calls = {"fuser": 0, "docker_rm": 0}
    monkeypatch.setattr(instances, "_fuser_kill", lambda port: calls.__setitem__("fuser", calls["fuser"] + 1))
    monkeypatch.setattr(instances, "_docker_rm_vllm", lambda: calls.__setitem__("docker_rm", calls["docker_rm"] + 1))

    states = [
        [instances.Instance(port=8001, engine="llamacpp", pid=1, model_file="m", model_path="/m", ctx=None, rss_gb=1.0, up=False)],
        [],  # รอบถัดไป: พอร์ตว่างแล้ว
    ]

    def fake_scan(**kw):
        return states.pop(0) if states else []

    monkeypatch.setattr(instances, "scan", fake_scan)

    ok, message = instances.stop(8001)
    assert ok is True
    assert "8001" in message
    assert calls["fuser"] == 1
    assert calls["docker_rm"] == 0  # ไม่ใช่พอร์ต 8000 → ไม่ควรยุ่งกับ docker


def test_stop_port_8000_ลอง_docker_rm_ด้วยเสมอ(monkeypatch):
    calls = {"docker_rm": 0}
    monkeypatch.setattr(instances, "_fuser_kill", lambda port: None)
    monkeypatch.setattr(instances, "_docker_rm_vllm", lambda: calls.__setitem__("docker_rm", calls["docker_rm"] + 1))

    states = [
        [instances.Instance(port=8000, engine="vllm", pid=None, model_file="foo", model_path="/models/foo", ctx=None, rss_gb=0.0, up=False)],
        [],
    ]

    def fake_scan(**kw):
        return states.pop(0) if states else []

    monkeypatch.setattr(instances, "scan", fake_scan)

    ok, _message = instances.stop(8000)
    assert ok is True
    assert calls["docker_rm"] == 1
