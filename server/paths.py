"""ที่อยู่ของไฟล์/โฟลเดอร์ทั้งหมดที่ v2 ใช้ — รวมไว้ที่เดียวเพื่อให้ test ชี้ไป tmp ได้

state ของ v2 อยู่ที่ ~/.aiserver2 แยกจาก ~/.aiserver ของ hub เดิมเด็ดขาด (ดู CONTEXT.md)
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def state_dir() -> str:
    """~/.aiserver2 — override ด้วย AISERVER2_STATE ตอน test"""
    d = os.environ.get("AISERVER2_STATE") or os.path.expanduser("~/.aiserver2")
    os.makedirs(d, exist_ok=True)
    return d


def state(*parts: str) -> str:
    return os.path.join(state_dir(), *parts)


def repo(*parts: str) -> str:
    return os.path.join(ROOT, *parts)


def log_dir() -> str:
    """log ของ engine — ใช้ร่วมกับ hub เดิมโดยตั้งใจ เพราะ engines/*.sh เขียนลงที่นี่"""
    d = os.environ.get("AISERVER2_LOGS") or os.path.expanduser("~/.aiserver/logs")
    return d
