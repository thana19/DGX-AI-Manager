"""server/gguf.py — อ่าน GGUF header (arch, context_length ฯลฯ) โดยไม่โหลดไฟล์ทั้งก้อน

ไฟล์โมเดลจริงใหญ่ 80-200GB ⇒ ห้ามโหลดทั้งไฟล์แค่เพื่อดูว่า arch อะไร
โมดูลนี้ parse ตาม spec ของ GGUF (little-endian) จาก bytes ก้อนเล็ก ๆ ที่ดึงมาด้วย
HTTP Range เท่านั้น (ดู CONTEXT.md หัวข้อ "ข้อค้นพบที่ระบบ v2 พึ่งพา" ข้อ 1-2)

รูปแบบไฟล์:
    magic 'GGUF'(4) · version u32 · tensor_count u64 · kv_count u64
    ตามด้วย kv_count คู่ key(string)/value_type(u32)/value
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import httpx

_MAGIC = b"GGUF"

# GGUF value type codes ตาม spec
_T_U8, _T_I8, _T_U16, _T_I16 = 0, 1, 2, 3
_T_U32, _T_I32, _T_F32, _T_BOOL = 4, 5, 6, 7
_T_STRING, _T_ARRAY = 8, 9
_T_U64, _T_I64, _T_F64 = 10, 11, 12

# struct format สำหรับชนิดค่าที่เป็นตัวเลขล้วน (bool เก็บเป็น 1 byte)
_SCALAR_FORMATS = {
    _T_U8: "<B", _T_I8: "<b", _T_U16: "<H", _T_I16: "<h",
    _T_U32: "<I", _T_I32: "<i", _T_F32: "<f", _T_BOOL: "<B",
    _T_U64: "<Q", _T_I64: "<q", _T_F64: "<d",
}


class NotGgufError(Exception):
    """ข้อมูลที่ให้มาไม่ใช่ไฟล์ GGUF (magic bytes ไม่ตรง)"""


class _BufferExhausted(Exception):  # noqa: N818 — ใช้ภายในโมดูลเท่านั้น ไม่ใช่ error ที่ผู้เรียกต้องจับ
    """buffer หมดกลางคัน — parse_header จับไว้แล้วคืนสิ่งที่อ่านได้ ไม่ raise ออกไป"""


@dataclass
class GgufInfo:
    """สรุปผลที่ดึงได้จาก GGUF header"""

    arch: str | None
    context_length: int | None
    name: str | None  # general.name
    size_label: str | None  # general.size_label
    file_type: int | None  # general.file_type
    version: int
    tensor_count: int
    kv_count: int  # จำนวน KV ที่ประกาศไว้ใน header
    kv_read: int  # อ่านได้จริงกี่ตัว (< kv_count = buffer ไม่พอ/หยุดก่อน เป็นเรื่องปกติ)


class _Reader:
    """อ่าน bytes ทีละส่วนจาก offset ปัจจุบัน — raise _BufferExhausted แทน IndexError เมื่อข้อมูลไม่พอ"""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def skip(self, n: int) -> bytes:
        end = self._pos + n
        if end > len(self._data):
            raise _BufferExhausted
        chunk = self._data[self._pos:end]
        self._pos = end
        return chunk

    def read_u32(self) -> int:
        return struct.unpack("<I", self.skip(4))[0]

    def read_u64(self) -> int:
        return struct.unpack("<Q", self.skip(8))[0]

    def read_string(self) -> str:
        length = self.read_u64()
        raw = self.skip(length)
        return raw.decode("utf-8", errors="replace")

    def read_value(self, value_type: int):
        """อ่านค่าตาม value_type — คืน None สำหรับ array

        array ถูก "ข้าม" ไม่ใช่ "อ่านเก็บ" โดยตั้งใจ: tokenizer.ggml.tokens มีเป็นแสนสมาชิก
        ถ้า materialize เป็น list จะกินแรมและเวลาฟรี ๆ ทั้งที่ไม่มีใครใช้ค่านั้น
        (field ที่โมดูลนี้สนใจเป็น string/ตัวเลขล้วนทั้งหมด)
        """
        if value_type == _T_STRING:
            return self.read_string()
        if value_type == _T_ARRAY:
            self.skip_array()
            return None
        fmt = _SCALAR_FORMATS.get(value_type)
        if fmt is None:
            # value_type ที่ไม่รู้จัก — ไม่มีทางรู้ขนาดเพื่อข้ามไปต่อได้ปลอดภัย ถือว่า buffer จบแค่นี้
            raise _BufferExhausted
        return struct.unpack(fmt, self.skip(struct.calcsize(fmt)))[0]

    def skip_array(self) -> None:
        """เลื่อนตำแหน่งข้าม array ทั้งก้อนโดยไม่สร้าง object

        array ของตัวเลขข้ามได้ทีเดียวเพราะขนาดคงที่ · array ของ string ต้องเดินทีละตัว
        เพราะแต่ละตัวมีความยาวไม่เท่ากัน (ถ้า buffer ไม่พอ skip() จะ raise ให้เอง)
        """
        elem_type = self.read_u32()
        count = self.read_u64()
        if elem_type == _T_STRING:
            for _ in range(count):
                self.read_string()
            return
        if elem_type == _T_ARRAY:
            for _ in range(count):
                self.skip_array()
            return
        fmt = _SCALAR_FORMATS.get(elem_type)
        if fmt is None:
            raise _BufferExhausted
        self.skip(struct.calcsize(fmt) * count)


def parse_header(data: bytes) -> GgufInfo:
    """parse GGUF header จาก bytes ที่มี (อาจเป็นแค่บางส่วนของไฟล์จริง)

    ทนต่อ buffer ที่อ่านไม่จบ (คืนสิ่งที่อ่านได้ ไม่ raise) แต่ raise NotGgufError
    ทันทีถ้า magic ไม่ใช่ 'GGUF' · หยุดอ่านทันทีที่ได้ทั้ง arch และ context_length
    เพราะ token list ท้าย ๆ ยาวมาก อ่านต่อไปก็เสียเวลาเปล่า
    """
    if data[:4] != _MAGIC:
        raise NotGgufError(f"ไม่ใช่ไฟล์ GGUF: magic bytes = {data[:4]!r}")

    reader = _Reader(data)
    reader.skip(4)  # ข้าม magic ที่เช็คไปแล้ว

    version = 0
    tensor_count = 0
    kv_count = 0
    kv_read = 0
    arch: str | None = None
    context_length: int | None = None
    name: str | None = None
    size_label: str | None = None
    file_type: int | None = None

    try:
        version = reader.read_u32()
        tensor_count = reader.read_u64()
        kv_count = reader.read_u64()

        for _ in range(kv_count):
            key = reader.read_string()
            value_type = reader.read_u32()
            value = reader.read_value(value_type)
            kv_read += 1

            if key == "general.architecture" and isinstance(value, str):
                arch = value
            elif key == "general.name" and isinstance(value, str):
                name = value
            elif key == "general.size_label" and isinstance(value, str):
                size_label = value
            elif key == "general.file_type" and isinstance(value, int):
                file_type = value
            elif key.endswith(".context_length") and isinstance(value, int):
                # key ขึ้นต้นด้วยชื่อ arch เสมอ (qwen35.context_length, glm5next.context_length, ...)
                # ห้าม hardcode ชื่อ arch — เช็คแค่ suffix
                context_length = value

            if arch is not None and context_length is not None:
                break  # ได้ของที่ต้องการครบแล้ว เลิกอ่านทันที
    except _BufferExhausted:
        pass  # buffer ไม่พอ — คืนสิ่งที่อ่านได้ ถือเป็นเรื่องปกติ ไม่ใช่ error

    return GgufInfo(
        arch=arch,
        context_length=context_length,
        name=name,
        size_label=size_label,
        file_type=file_type,
        version=version,
        tensor_count=tensor_count,
        kv_count=kv_count,
        kv_read=kv_read,
    )


def _content_range_total(header_value: str | None) -> int | None:
    """แยกขนาดไฟล์เต็มจาก header 'content-range: bytes 0-262143/9828981664' → 9828981664"""
    if not header_value:
        return None
    _, _, total = header_value.partition("/")
    if not total or total == "*":
        return None
    try:
        return int(total)
    except ValueError:
        return None


def _fetch_range(client: httpx.Client, url: str, n_bytes: int) -> tuple[GgufInfo, int | None]:
    response = client.get(url, headers={"Range": f"bytes=0-{n_bytes - 1}"})
    response.raise_for_status()
    info = parse_header(response.content)
    total_size = _content_range_total(response.headers.get("content-range"))
    return info, total_size


def fetch_header(
    url: str,
    *,
    client: httpx.Client | None = None,
    first_bytes: int = 262144,
    max_bytes: int = 1048576,
) -> tuple[GgufInfo, int | None]:
    """ดึง GGUF header ผ่าน HTTP Range แล้ว parse — คืน (GgufInfo, ขนาดไฟล์เต็มเป็น byte)

    ยิง Range แรก first_bytes ก่อน ถ้ายังไม่ได้ arch/context_length ครบ (buffer ไม่พอ)
    ค่อยขยายเป็น max_bytes — follow redirect เสมอ (HF 302 ไป CDN)
    """
    owns_client = client is None
    if client is None:
        client = httpx.Client(follow_redirects=True)

    try:
        info, total_size = _fetch_range(client, url, first_bytes)
        if (info.arch is None or info.context_length is None) and max_bytes > first_bytes:
            info, total_size = _fetch_range(client, url, max_bytes)
        return info, total_size
    finally:
        if owns_client:
            client.close()
