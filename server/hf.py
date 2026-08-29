"""แปลง HF repo id เป็นรายการ quant ให้ผู้ใช้เลือกก่อนดาวน์โหลด

ใช้ `GET /api/models/{repo_id}?blobs=true` ของ Hugging Face — คืน siblings ทุกไฟล์ใน repo
พร้อมขนาดและ sha256 (จาก lfs) โดยไม่ต้องโหลดไฟล์จริง (ดู CONTEXT.md ข้อ 5)

ไฟล์ quant มักอยู่ในโฟลเดอร์ย่อย (UD-IQ1_S/, MTP/, BF16/) แต่บาง repo วางไว้ที่ root เลย
(ดู CONTEXT.md ข้อ 6) — โมดูลนี้จัดกลุ่มให้เป็นก้อนเดียวไม่ว่าจะอยู่รูปแบบไหน
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import quote, urlparse

import httpx

API_BASE = "https://huggingface.co/api/models"
RESOLVE_BASE = "https://huggingface.co"

# ไฟล์ที่ไปคู่กับ quant แต่ไม่ใช่ quant เอง — mmproj (vision), mtp/dflash (draft model)
# ใช้ร่วมกันได้หลาย quant ⇒ แยกออกจาก quant group เสมอ ไม่ว่าจะอยู่โฟลเดอร์ไหน
# eagle3- = speculative draft head ของ gpt-oss (เจอจริงใน ggml-org/gpt-oss-120b-GGUF)
# ไม่กันไว้ = ผู้ใช้เห็น "Q8_0 0.8GB" แล้วเลือกไปโหลด ได้ draft head แทนโมเดลจริง 63GB
_COMPANION_PREFIXES = ("mmproj-", "mtp-", "dflash-", "eagle3-", "eagle-", "draft-")

# shard suffix แบบ "...-00001-of-00003.gguf" — ใช้ตัดท้ายก่อนหาชื่อ quant และหา shard_count
_SHARD_RE = re.compile(r"^(?P<base>.+)-(?P<idx>\d+)-of-(?P<total>\d+)$")

# ชื่อ quant ที่อยู่ในชื่อไฟล์ root (ไม่มีโฟลเดอร์ย่อย) — เอาแค่ token ท้ายสุดของ stem
# รูปแบบที่พบจริง: UD-Q4_K_XL, UD-IQ1_M, IQ2_XXS, Q8_0, Q4_0, BF16, F16, F32
_QUANT_RE = re.compile(
    r"(?:(?:UD-)?(?:IQ|Q|TQ)\d+(?:_[A-Za-z0-9]+)*"   # UD-Q4_K_XL, IQ2_XXS, Q8_0, TQ1_0
    r"|(?:MX|NV)?FP\d+(?:_[A-Za-z0-9]+)*"            # MXFP4 (gpt-oss), NVFP4, FP8
    r"|BF16|F16|F32)$"
)


class GatedRepoError(Exception):
    """repo ติด gate — ต้องใช้ HF_TOKEN ที่มีสิทธิ์เข้าถึง"""


class RepoNotFoundError(Exception):
    """ไม่พบ repo (พิมพ์ผิด หรือถูกลบ)"""


@dataclass
class HfFile:
    """ไฟล์ 1 ไฟล์ใน HF repo"""

    path: str  # path เต็มใน repo เช่น "UD-IQ1_S/xxx-00001-of-00003.gguf"
    size: int  # byte
    sha256: str | None  # จาก lfs.sha256 — ใช้ verify หลังโหลดเสร็จ (ไฟล์ที่ไม่ใช่ lfs จะเป็น None)

    @property
    def url_path(self) -> str:
        """path สำหรับต่อท้าย resolve URL — encode ให้ปลอดภัยแต่คง "/" ของโฟลเดอร์ไว้"""
        return quote(self.path, safe="/")


@dataclass
class QuantGroup:
    """quant 1 ตัวที่ผู้ใช้เลือกโหลดได้ — อาจมีหลายไฟล์ถ้าเป็น shard"""

    key: str  # ชื่อโชว์ผู้ใช้ เช่น "UD-IQ1_S" หรือ "Q8_0"
    files: list[HfFile]  # ไฟล์หลัก (shard ครบชุด) เรียงตาม shard
    total_bytes: int  # ผลรวมของ files เท่านั้น (ไม่รวม companions)
    shard_count: int  # 1 = ไฟล์เดียว
    companions: list[HfFile] = field(default_factory=list)  # mmproj-/mtp-/dflash- ของ repo นี้


def fetch_repo(repo_id: str, *, token: str | None = None, client: httpx.Client | None = None) -> dict:
    """ยิง HF API ขอ metadata ของ repo พร้อม blobs (size/sha256 ทุกไฟล์)

    401/403 = ติด gate (HF คืนสองแบบนี้แล้วแต่กรณี) · 404 = ไม่พบ repo
    """
    own_client = client is None
    http_client = client or httpx.Client()
    try:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        resp = http_client.get(f"{API_BASE}/{repo_id}", params={"blobs": "true"}, headers=headers)
    finally:
        if own_client:
            http_client.close()

    if resp.status_code in (401, 403):
        raise GatedRepoError(f"{repo_id} ติด gate — ต้องใช้ HF_TOKEN ที่มีสิทธิ์เข้าถึง")
    if resp.status_code == 404:
        raise RepoNotFoundError(f"ไม่พบ repo: {repo_id}")
    resp.raise_for_status()
    return resp.json()


def list_files(repo_json: dict) -> list[HfFile]:
    """แปลง siblings จาก HF API response เป็น HfFile ทั้งหมด (ยังไม่กรอง .gguf)"""
    files = []
    for sib in repo_json.get("siblings", []):
        lfs = sib.get("lfs") or {}
        files.append(HfFile(path=sib["rfilename"], size=sib["size"], sha256=lfs.get("sha256")))
    return files


def _is_companion(basename: str) -> bool:
    return basename.startswith(_COMPANION_PREFIXES)


def _strip_shard(stem: str) -> tuple[str, int | None, int | None]:
    """ตัด "-00001-of-00003" ท้าย stem ออก คืน (base, shard_idx, shard_total)"""
    m = _SHARD_RE.match(stem)
    if not m:
        return stem, None, None
    return m.group("base"), int(m.group("idx")), int(m.group("total"))


def group_quants(files: list[HfFile]) -> list[QuantGroup]:
    """จัดกลุ่มไฟล์ .gguf เป็น quant ให้ผู้ใช้เลือก — ดูกติกาเต็มใน docstring โมดูลนี้/task"""
    companions: list[HfFile] = []
    # key -> list[(shard_idx, HfFile)] — shard_idx เป็น 0 ถ้าไม่ใช่ shard (ไฟล์เดียว)
    buckets: dict[str, list[tuple[int, HfFile]]] = {}
    shard_totals: dict[str, int] = {}

    for f in files:
        if not f.path.endswith(".gguf"):
            continue

        parts = f.path.split("/")
        basename = parts[-1]
        if _is_companion(basename):
            companions.append(f)
            continue

        stem = basename[: -len(".gguf")]
        base, shard_idx, shard_total = _strip_shard(stem)

        if len(parts) > 1:
            # อยู่ในโฟลเดอร์ย่อย → ชื่อโฟลเดอร์คือ key เสมอ (ข้อ 2)
            key: str | None = parts[0]
        else:
            # อยู่ที่ root → ดึงชื่อ quant จากท้ายชื่อไฟล์ (ข้อ 3)
            key = _QUANT_RE.search(base)
            key = key.group(0) if key else None

        if key is None:
            # ไม่ตรง pattern quant ที่รู้จัก (เช่น imatrix_*.gguf) — ข้ามไปเลย
            continue

        buckets.setdefault(key, []).append((shard_idx or 0, f))
        if shard_total is not None:
            shard_totals[key] = shard_total

    groups = []
    for key, items in buckets.items():
        items.sort(key=lambda t: t[0])
        group_files = [f for _, f in items]
        groups.append(
            QuantGroup(
                key=key,
                files=group_files,
                total_bytes=sum(f.size for f in group_files),
                shard_count=shard_totals.get(key, len(group_files)),
                companions=list(companions),
            )
        )

    groups.sort(key=lambda g: g.total_bytes)  # น้อยไปมาก (ข้อ 7) — ผู้ใช้มองหาตัวที่แรมพอก่อน
    return groups


def resolve_url(repo_id: str, path: str) -> str:
    """URL ดาวน์โหลดไฟล์จริงจาก path ใน repo (รวมกรณีมีโฟลเดอร์ย่อย)"""
    return f"{RESOLVE_BASE}/{repo_id}/resolve/main/{quote(path, safe='/')}"


# ---------------------------------------------------------------------------
# normalize_repo_id / search_models — ผู้ใช้วางลิงก์ GitHub หรือพิมพ์ชื่อโมเดลเปล่า ๆ
# มาแทน HF repo id จริง (เช่น "github.com/openai/gpt-oss" ทั้งที่ repo จริงชื่อ
# "openai/gpt-oss-20b") ⇒ เดา repo_id ตรง ๆ ไม่ได้ ต้องค้นหาแล้วให้ผู้ใช้เลือกเอง
# ---------------------------------------------------------------------------

_HF_HOSTS = {"huggingface.co", "www.huggingface.co"}

# เครื่องหมาย/ช่องว่างท้ายข้อความที่ตัดทิ้งได้เสมอ (เช่น "org/repo/  " ที่มี "/" ต่อท้าย)
_TRAILING_JUNK_RE = re.compile(r"[\s/.,;:!]+$")

# org/repo แบบข้อความล้วน (ไม่ใช่ URL) — ตัวอักษร/ตัวเลข/จุด/ขีด อย่างละ 1 กลุ่ม คั่นด้วย "/" เดียว
_PLAIN_REPO_ID_RE = re.compile(r"^[\w.\-]+/[\w.\-]+$")


@dataclass
class SearchHit:
    """ผลค้นหา 1 รายการจาก HF search API"""

    id: str
    downloads: int
    likes: int
    gated: bool
    is_gguf: bool
    pipeline_tag: str | None


def normalize_repo_id(text: str) -> tuple[str | None, str]:
    """แปลงสิ่งที่ผู้ใช้วางมาเป็น (repo_id ถ้าเป็น HF ได้, คำค้นสำรอง)

    รับได้:
      "unsloth/Xxx-GGUF"                              → ("unsloth/Xxx-GGUF", "Xxx-GGUF")
      "https://huggingface.co/unsloth/Xxx?a=1"        → ("unsloth/Xxx", "Xxx")
      "https://huggingface.co/unsloth/Xxx/tree/main"  → ("unsloth/Xxx", "Xxx")
      "https://github.com/openai/gpt-oss?utm_source=" → (None, "gpt-oss")
      "gpt oss 20b"                                   → (None, "gpt oss 20b")
      ""                                               → (None, "")
    กติกา: ตัด query string / fragment / เครื่องหมายท้าย / ช่องว่างหัวท้ายเสมอ
           host ที่ไม่ใช่ huggingface.co ⇒ ไม่ใช่ repo_id (คืน None) แต่เอาส่วนท้าย path เป็นคำค้น
           path ของ HF ที่ยาวเกิน org/repo (เช่น /tree/main, /blob/...) ให้ตัดเหลือ org/repo
           กรณี "  org/repo/  " (มี / ต่อท้าย + ช่องว่างหัวท้าย) ก็ต้องได้ ("org/repo", "repo")
    """
    raw = (text or "").strip()
    if not raw:
        return None, ""

    # ตัด query string / fragment ทิ้งก่อนเสมอ ไม่ว่าจะเป็น URL เต็มหรือ path เปล่า ๆ
    raw = re.split(r"[?#]", raw, maxsplit=1)[0].strip()
    if not raw:
        return None, ""

    if re.match(r"^https?://", raw, re.IGNORECASE):
        parsed = urlparse(raw)
        host = parsed.netloc.lower()
        parts = [p for p in parsed.path.split("/") if p]

        if host in _HF_HOSTS:
            if len(parts) >= 2:
                repo_id = f"{parts[0]}/{parts[1]}"
                return repo_id, parts[1]
            if len(parts) == 1:
                return None, parts[0]
            return None, ""

        # host อื่น (github.com ฯลฯ) ไม่ใช่ HF ⇒ เอา path ท้ายสุดมาเป็นคำค้นแทน
        query = parts[-1] if parts else host
        return None, query

    # ไม่ใช่ URL — ตัดเครื่องหมาย/ช่องว่างท้ายออกก่อนเช็คว่าเป็น org/repo หรือเปล่า
    candidate = _TRAILING_JUNK_RE.sub("", raw)
    if _PLAIN_REPO_ID_RE.match(candidate):
        return candidate, candidate.split("/")[-1]

    # คำค้นธรรมดา (พิมพ์ชื่อโมเดลมาเฉย ๆ) — คืนตามที่ผู้ใช้พิมพ์ (ตัด query/fragment ไปแล้ว)
    return None, raw


def search_models(
    query: str, *, limit: int = 12, token: str | None = None, client: httpx.Client | None = None
) -> list[SearchHit]:
    """ค้น HF: GET https://huggingface.co/api/models?search=<q>&sort=downloads&direction=-1&limit=<n>
    ถ้ามี token ใส่ header Authorization: Bearer <token>

    เรียงผลลัพธ์: GGUF ขึ้นก่อนเสมอ แล้วค่อยเรียงตาม downloads มากไปน้อย
    (เหตุผล: engine หลักของเครื่องนี้คือ llama.cpp ซึ่งกิน GGUF เท่านั้น)
    is_gguf: id ลงท้ายด้วย "-GGUF"/"-gguf" หรือมี "gguf" ใน tags ของผลลัพธ์
    query ว่าง → คืน [] ทันที (ห้ามยิง HTTP request เลย)
    ใช้ client ที่ส่งเข้ามาถ้ามี (สำหรับ test), ไม่งั้นสร้าง httpx.Client() เอง แล้วปิดให้เรียบร้อย
    """
    if not query:
        return []

    own_client = client is None
    http_client = client or httpx.Client()
    try:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        resp = http_client.get(
            API_BASE,
            params={"search": query, "sort": "downloads", "direction": "-1", "limit": limit},
            headers=headers,
        )
        resp.raise_for_status()
        results = resp.json()
    finally:
        if own_client:
            http_client.close()

    hits = []
    for item in results:
        repo_id = item.get("id") or item.get("modelId") or ""
        tags = item.get("tags") or []
        is_gguf = repo_id.lower().endswith("-gguf") or any("gguf" in str(t).lower() for t in tags)
        hits.append(
            SearchHit(
                id=repo_id,
                downloads=item.get("downloads") or 0,
                likes=item.get("likes") or 0,
                gated=bool(item.get("gated", False)),
                is_gguf=is_gguf,
                pipeline_tag=item.get("pipeline_tag"),
            )
        )

    hits.sort(key=lambda h: (not h.is_gguf, -h.downloads))
    return hits
