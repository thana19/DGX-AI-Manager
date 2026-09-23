"""แยก "ดาวน์โหลด" ออกจาก "โหลดขึ้นแรม" — ผู้ใช้ต้องเห็นความคืบหน้า หยุด/ต่อ/ยกเลิกได้

โมเดลใหญ่ 80-200GB ใช้เวลาหลายชั่วโมง ⇒ ต้องมีคิวงาน + progress + resume ข้ามการ
restart ของ v2 เอง (v1 กดปุ่มเดียวแล้วรอเงียบ ๆ ไม่มีของพวกนี้เลย)

หุ้มการดาวน์โหลดจริงด้วย aria2 (JSON-RPC) — เลือก aria2 เพราะรองรับ pause/resume ผ่าน
ไฟล์ .aria2 control file ในตัวอยู่แล้ว ไม่ต้องเขียน range-download เอง

คิวงาน: รันทีละ 1 job เท่านั้น (เน็ต+ดิสก์เป็นคอขวดของเครื่องนี้ รันขนานไม่ได้เร็วขึ้น
แถมแย่งกันเปลืองเปล่า ๆ) — job ถัดไปเริ่มเองอัตโนมัติเมื่อ job ปัจจุบันจบ (DONE/ERROR/CANCELLED)
"""
from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from urllib.parse import urlsplit

import httpx

from . import paths

# key ของ aria2.tellStatus ที่ค่าตัวเลขส่งกลับมาเป็น string เสมอ (ดู task) — ต้องแปลงเป็น int
_NUMERIC_STATUS_KEYS = {"totalLength", "completedLength", "downloadSpeed", "errorCode"}

_STATUS_KEYS = [
    "gid",
    "status",
    "totalLength",
    "completedLength",
    "downloadSpeed",
    "errorCode",
    "errorMessage",
    "files",
]


# ยอมให้ขนาดไฟล์ต่างจากที่ HF บอกได้เท่านี้ (ค่าที่ v1 ใช้จริงมาแล้ว)
_SIZE_TOLERANCE = 4096


class DownloadState(str, Enum):
    QUEUED = "queued"
    ACTIVE = "active"
    PAUSED = "paused"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


class Aria2Error(Exception):
    """aria2 ตอบ error กลับมา (เช่น daemon ตาย, gid ไม่รู้จัก)"""


@dataclass
class FileProgress:
    """ความคืบหน้าของไฟล์ 1 ไฟล์ในดาวน์โหลด job หนึ่ง"""

    url: str
    dest: str
    gid: str | None
    total_bytes: int
    done_bytes: int
    state: DownloadState
    error: str | None


@dataclass
class DownloadJob:
    """งานดาวน์โหลดโมเดล 1 ชุด — อาจมีหลายไฟล์ (shard)"""

    id: str
    model_id: str
    files: list[FileProgress]
    created_at: float
    state: DownloadState
    # ความเร็วรวม ณ ปัจจุบัน — อัปเดตจาก refresh() เท่านั้น ไม่ persist ข้าม restart
    # (ความเร็วเก่าไม่มีความหมายหลัง restart จนกว่าจะ refresh ใหม่)
    _current_speed_bps: int = field(default=0, repr=False)

    @property
    def total_bytes(self) -> int:
        return sum(f.total_bytes for f in self.files)

    @property
    def done_bytes(self) -> int:
        return sum(f.done_bytes for f in self.files)

    @property
    def percent(self) -> float:
        total = self.total_bytes
        if total <= 0:
            return 0.0
        return self.done_bytes / total * 100

    @property
    def speed_bps(self) -> int:
        return self._current_speed_bps

    @property
    def eta_seconds(self) -> int | None:
        if self.speed_bps <= 0:
            return None
        remaining = self.total_bytes - self.done_bytes
        if remaining <= 0:
            return 0
        return int(remaining / self.speed_bps)


def summarize_state(files: list[FileProgress]) -> DownloadState:
    """สรุป state รวมของ job จาก state ของแต่ละไฟล์ — ลำดับความสำคัญตามกติกาข้อ 1

    1. มีไฟล์ ERROR แม้ตัวเดียว → ERROR
    2. ทุกไฟล์ DONE → DONE
    3. มีไฟล์ ACTIVE → ACTIVE
    4. ทุกไฟล์ PAUSED → PAUSED
    5. มีไฟล์ CANCELLED → CANCELLED
    6. ที่เหลือ → QUEUED
    """
    if not files:
        return DownloadState.QUEUED
    states = [f.state for f in files]
    if any(s == DownloadState.ERROR for s in states):
        return DownloadState.ERROR
    if all(s == DownloadState.DONE for s in states):
        return DownloadState.DONE
    if any(s == DownloadState.ACTIVE for s in states):
        return DownloadState.ACTIVE
    if all(s == DownloadState.PAUSED for s in states):
        return DownloadState.PAUSED
    if any(s == DownloadState.CANCELLED for s in states):
        return DownloadState.CANCELLED
    return DownloadState.QUEUED


class Aria2Client:
    """หุ้ม aria2 JSON-RPC — รับ http client เข้ามาได้เพื่อให้ test mock ได้โดยไม่ต้องมี aria2 จริง"""

    def __init__(self, url: str, secret: str = "", client: httpx.Client | None = None):
        self.url = url
        self.secret = secret
        self._client = client or httpx.Client()

    def call(self, method: str, params: list) -> Any:
        """ยิง JSON-RPC ไปหา aria2 — แปลง error ทุกแบบให้เป็น Aria2Error เดียว ไม่มี exception ชนิดอื่นหลุดออกไป

        ⚠️ ของจริงบน DGX: aria2 ตอบ error กลับมาพร้อม HTTP 400 (ไม่ใช่ 200) — ต้องอ่าน body
        หา `"error"` **ก่อน** เช็ค status code เพราะเดิมเรียก raise_for_status() ก่อนอ่าน body
        ทำให้ได้ httpx.HTTPStatusError หลุดออกจาก call() แทน ผู้เรียก (refresh) ที่ดัก `except Aria2Error`
        อย่างเดียวเลยพลาด exception ทั้งก้อน
        """
        rpc_params = ([f"token:{self.secret}"] + list(params)) if self.secret else list(params)
        payload = {"jsonrpc": "2.0", "id": "1", "method": method, "params": rpc_params}
        try:
            resp = self._client.post(self.url, json=payload)
        except httpx.HTTPError as e:  # connect/timeout ก่อนได้ response กลับมาเลยด้วยซ้ำ
            raise Aria2Error(str(e)) from e
        try:
            data = resp.json()
        except ValueError:
            data = None
        if isinstance(data, dict) and "error" in data:
            err = data["error"]
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            raise Aria2Error(msg)
        if not (200 <= resp.status_code < 300):
            raise Aria2Error(f"aria2 ตอบ HTTP {resp.status_code}")
        return data["result"]

    def add_uri(self, url: str, dir: str, out: str, headers: list[str] | None = None) -> str:
        opts: dict[str, Any] = {"dir": dir, "out": out}
        if headers:
            opts["header"] = headers  # aria2 RPC option "header" รับ list ของ "Name: value"
        return self.call("aria2.addUri", [[url], opts])

    def tell_status(self, gid: str) -> dict:
        raw = self.call("aria2.tellStatus", [gid, _STATUS_KEYS])
        result = dict(raw)
        for key in _NUMERIC_STATUS_KEYS:
            if key in result and result[key] != "":
                result[key] = int(result[key])
        return result

    def pause(self, gid: str) -> None:
        self.call("aria2.pause", [gid])

    def unpause(self, gid: str) -> None:
        self.call("aria2.unpause", [gid])

    def remove(self, gid: str) -> None:
        self.call("aria2.remove", [gid])

    def remove_download_result(self, gid: str) -> None:
        """ลบผลลัพธ์ของ gid ที่จบไปแล้ว (error/cancelled) ออกจากความจำของ aria2 daemon"""
        self.call("aria2.removeDownloadResult", [gid])


# ---------------------------------------------------------------------------
# แปลง state ไป/กลับ JSON สำหรับ persist ลง disk
# ---------------------------------------------------------------------------


def _file_to_dict(f: FileProgress) -> dict:
    return {
        "url": f.url,
        "dest": f.dest,
        "gid": f.gid,
        "total_bytes": f.total_bytes,
        "done_bytes": f.done_bytes,
        "state": f.state.value,
        "error": f.error,
    }


def _file_from_dict(d: dict) -> FileProgress:
    return FileProgress(
        url=d["url"],
        dest=d["dest"],
        gid=d["gid"],
        total_bytes=d["total_bytes"],
        done_bytes=d["done_bytes"],
        state=DownloadState(d["state"]),
        error=d["error"],
    )


def _job_to_dict(job: DownloadJob) -> dict:
    return {
        "id": job.id,
        "model_id": job.model_id,
        "files": [_file_to_dict(f) for f in job.files],
        "created_at": job.created_at,
        "state": job.state.value,
    }


def _job_from_dict(d: dict) -> DownloadJob:
    return DownloadJob(
        id=d["id"],
        model_id=d["model_id"],
        files=[_file_from_dict(f) for f in d["files"]],
        created_at=d["created_at"],
        state=DownloadState(d["state"]),
    )


class DownloadManager:
    """คิวดาวน์โหลด — รันทีละ 1 job เท่านั้น · state persist ลง disk ทุกครั้งที่เปลี่ยน"""

    def __init__(
        self,
        aria2: Aria2Client,
        *,
        state_path: str | None = None,
        token_loader: Callable[[], str | None] | None = None,
    ):
        self._aria2 = aria2
        self._state_path = state_path or paths.state("downloads.json")
        self._token_loader = token_loader
        self._jobs: dict[str, DownloadJob] = {}
        self._running_job_id: str | None = None
        self._load_state()

    # -- persistence ---------------------------------------------------

    def _load_state(self) -> None:
        if not os.path.exists(self._state_path):
            return
        try:
            with open(self._state_path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return
        for job_dict in data.get("jobs", []):
            job = _job_from_dict(job_dict)
            self._jobs[job.id] = job
        # job ที่ยังมี gid ค้างอยู่และยังไม่จบ = job ที่ครอบครองคิวอยู่ ณ ตอน restart ก่อนหน้า
        for job in self._jobs.values():
            if job.state in (DownloadState.ACTIVE, DownloadState.PAUSED):
                self._running_job_id = job.id
                break

    def _save_state(self) -> None:
        os.makedirs(os.path.dirname(self._state_path), exist_ok=True)
        data = {"jobs": [_job_to_dict(j) for j in self._jobs.values()]}
        tmp_path = f"{self._state_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self._state_path)

    # -- query -----------------------------------------------------------

    def jobs(self) -> list[DownloadJob]:
        return list(self._jobs.values())

    def get(self, job_id: str) -> DownloadJob | None:
        return self._jobs.get(job_id)

    # -- submit ------------------------------------------------------------

    def submit(self, model_id: str, files: Sequence[tuple[str, str] | tuple[str, str, int]]) -> DownloadJob:
        """files = [(url, dest_path)] หรือ [(url, dest_path, expected_bytes)]

        ใส่ expected_bytes ได้เมื่อรู้ขนาดจริงจาก HF API — ใช้กันเคสไฟล์ค้างที่ไม่มี .aria2
        ถ้ามี job ที่ยังไม่จบอยู่ → job ใหม่เข้าคิว (QUEUED)
        """
        file_progresses = [
            self._make_file_progress(f[0], f[1], f[2] if len(f) > 2 else 0) for f in files
        ]
        job = DownloadJob(
            id=uuid.uuid4().hex[:8],
            model_id=model_id,
            files=file_progresses,
            created_at=time.time(),
            state=summarize_state(file_progresses),
        )
        self._jobs[job.id] = job
        self._advance_queue()
        self._save_state()
        return job

    def _make_file_progress(self, url: str, dest: str, expected_bytes: int = 0) -> FileProgress:
        has_dangling_control = os.path.exists(f"{dest}.aria2")
        if os.path.exists(dest) and not has_dangling_control:
            size = os.path.getsize(dest)
            # ไม่มี .aria2 ไม่ได้แปลว่าไฟล์ครบเสมอไป — โดนฆ่าแรง ๆ หรือก๊อปมาไม่จบก็ไม่ทิ้ง control file ไว้
            # เทียบขนาดกับที่ HF บอกก่อน (เผื่อ 4096 byte แบบเดียวกับ v1 ที่ใช้จริงมาแล้ว)
            if expected_bytes and abs(size - expected_bytes) > _SIZE_TOLERANCE:
                return FileProgress(
                    url=url, dest=dest, gid=None,
                    total_bytes=expected_bytes, done_bytes=size,
                    state=DownloadState.QUEUED, error=None,
                )
            return FileProgress(
                url=url, dest=dest, gid=None, total_bytes=size, done_bytes=size,
                state=DownloadState.DONE, error=None,
            )
        return FileProgress(
            url=url, dest=dest, gid=None, total_bytes=0, done_bytes=0,
            state=DownloadState.QUEUED, error=None,
        )

    # -- queue runner ------------------------------------------------------

    def _job_occupies_slot(self, job: DownloadJob) -> bool:
        """job นี้ยังไม่จบ (ต้องการใช้ aria2) หรือไม่"""
        return job.state not in (DownloadState.DONE, DownloadState.ERROR, DownloadState.CANCELLED)

    def _advance_queue(self) -> None:
        """เริ่ม job ถัดไปในคิวถ้าไม่มี job ครอบครองคิวอยู่"""
        if self._running_job_id is not None:
            current = self._jobs.get(self._running_job_id)
            if current is not None and self._job_occupies_slot(current):
                return  # มี job กำลังทำงานอยู่แล้ว
            self._running_job_id = None

        for job in self._jobs.values():
            already_started = any(f.gid is not None for f in job.files)
            if already_started:
                continue
            if not self._job_occupies_slot(job):
                continue  # job นี้ทุกไฟล์ DONE ตั้งแต่ submit แล้ว ไม่ต้องเริ่ม
            self._start_job(job)
            self._running_job_id = job.id
            return

    def _auth_headers_for(self, url: str) -> list[str] | None:
        """คืน ["Authorization: Bearer <token>"] เฉพาะ URL ที่ host เป็น huggingface.co (หรือ subdomain)
        และมี token ให้ใช้ — host อื่นไม่ส่งเด็ดขาด กัน token รั่วไปที่อื่น (ดู task ส่วนที่ 3)
        """
        if self._token_loader is None:
            return None
        host = urlsplit(url).hostname or ""
        host = host.lower()
        if not (host == "huggingface.co" or host.endswith(".huggingface.co")):
            return None
        token = self._token_loader()
        if not token:
            return None
        return [f"Authorization: Bearer {token}"]

    def _start_job(self, job: DownloadJob) -> None:
        for f in job.files:
            if f.state == DownloadState.DONE:
                continue
            dest_dir = os.path.dirname(f.dest)
            os.makedirs(dest_dir, exist_ok=True)
            headers = self._auth_headers_for(f.url)
            gid = self._aria2.add_uri(f.url, dest_dir, os.path.basename(f.dest), headers=headers)
            f.gid = gid
            f.state = DownloadState.ACTIVE
        job.state = summarize_state(job.files)

    # -- refresh ------------------------------------------------------------

    def refresh(self) -> None:
        """ดึงสถานะจาก aria2 มาอัปเดตทุก job · จบ job ปัจจุบันแล้วเริ่มตัวถัดไปในคิวอัตโนมัติ

        job ที่จบไปแล้ว (DONE/ERROR/CANCELLED) ข้ามไปเลย ไม่ถาม aria2 อีก — ทั้งลด RPC เปล่า ๆ
        และตัดปัญหา gid เก่าที่หายไปจาก aria2 ตั้งแต่ restart (ดู hotfix 2026-09-22)
        """
        for job in self._jobs.values():
            if not self._job_occupies_slot(job):
                continue
            speed_total = 0
            for f in job.files:
                if f.gid is None:
                    continue
                try:
                    status = self._aria2.tell_status(f.gid)
                except Aria2Error as e:
                    f.state = DownloadState.ERROR
                    if "not found" in str(e).lower():
                        # aria2 ไม่รู้จัก gid นี้แล้ว (เช่น aria2 ถูก restart แล้วลืม state เดิม)
                        f.error = "aria2 ไม่รู้จักงานนี้แล้ว (aria2 อาจถูก restart) — กดดาวน์โหลดใหม่"
                    else:
                        f.error = f"เชื่อมต่อ aria2 ไม่ได้ ({e}) — เช็คว่า aria2c daemon ยังทำงานอยู่หรือไม่"
                    continue
                self._apply_status(f, status)
                if f.state == DownloadState.ACTIVE:
                    speed_total += status.get("downloadSpeed", 0)
            job.state = summarize_state(job.files)
            job._current_speed_bps = speed_total

        self._advance_queue()
        self._save_state()

    @staticmethod
    def _apply_status(f: FileProgress, status: dict) -> None:
        f.total_bytes = status.get("totalLength", f.total_bytes) or f.total_bytes
        f.done_bytes = status.get("completedLength", f.done_bytes)
        aria2_status = status.get("status")
        mapping = {
            "active": DownloadState.ACTIVE,
            "waiting": DownloadState.QUEUED,
            "paused": DownloadState.PAUSED,
            "error": DownloadState.ERROR,
            "complete": DownloadState.DONE,
            "removed": DownloadState.CANCELLED,
        }
        f.state = mapping.get(aria2_status, f.state)
        if f.state == DownloadState.ERROR:
            f.error = status.get("errorMessage") or "aria2 รายงาน error โดยไม่มีรายละเอียด"
        else:
            f.error = None

    # -- controls ------------------------------------------------------------

    def pause(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False
        touched = False
        for f in job.files:
            if f.gid is not None and f.state == DownloadState.ACTIVE:
                self._aria2.pause(f.gid)
                f.state = DownloadState.PAUSED
                touched = True
        if touched:
            job.state = summarize_state(job.files)
            self._save_state()
        return touched

    def resume(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False
        touched = False
        for f in job.files:
            if f.gid is not None and f.state == DownloadState.PAUSED:
                self._aria2.unpause(f.gid)
                f.state = DownloadState.ACTIVE
                touched = True
        if touched:
            job.state = summarize_state(job.files)
            self._save_state()
        return touched

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False
        for f in job.files:
            if f.state == DownloadState.DONE:
                continue  # ห้ามลบไฟล์ที่โหลดได้แล้ว
            if f.gid is not None:
                try:
                    self._aria2.remove(f.gid)
                except Aria2Error:
                    pass  # gid อาจจบไปแล้วหรือ daemon ตาย — ไม่ต้องให้ cancel ล้มเหลวเพราะเหตุนี้
            control_file = f"{f.dest}.aria2"
            if os.path.exists(control_file):
                os.remove(control_file)
            f.state = DownloadState.CANCELLED
        job.state = summarize_state(job.files)
        if self._running_job_id == job.id:
            self._running_job_id = None
        self._advance_queue()
        self._save_state()
        return True

    def clear_failed(self) -> int:
        """ลบ job ที่ ERROR/CANCELLED ออกจากลิสต์ — ไม่แตะไฟล์บนดิสก์ (ต่างจาก cancel)

        best-effort บอก aria2 ให้ลืม gid พวกนี้ด้วย — ไม่บังคับต้องสำเร็จ เพราะ gid อาจไม่รู้จัก
        แล้ว (aria2 restart) หรือ daemon ตายอยู่ ก็ไม่ควรทำให้ clear ล้มเหลวไปด้วย
        """
        to_remove = [
            job for job in self._jobs.values()
            if job.state in (DownloadState.ERROR, DownloadState.CANCELLED)
        ]
        for job in to_remove:
            for f in job.files:
                if f.gid is not None:
                    try:
                        self._aria2.remove_download_result(f.gid)
                    except Aria2Error:
                        pass  # gid ไม่รู้จักหรือ daemon ตาย — ไม่ต้องให้ clear ล้มเหลวเพราะเหตุนี้
            del self._jobs[job.id]
            if self._running_job_id == job.id:
                self._running_job_id = None
        self._advance_queue()
        self._save_state()
        return len(to_remove)
