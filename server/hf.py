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

# บั๊กจริง (HauhauCS/Qwen3.8-27B-...-MTP-GGUF): ไฟล์ draft ชื่อ "...-FastMTP-32K.gguf" ไม่ได้ขึ้นต้น
# ด้วย prefix ไหนใน _COMPANION_PREFIXES เลย ⇒ ตกไปเงียบ ๆ ไม่ถูกจับเป็น quant ก็ไม่ถูกจับเป็น companion
# (หายไปจากรายการทั้งที่เป็นของดี ใส่แล้วได้ speculative decoding เร็วขึ้นชัดเจน)
# ⇒ ขยายให้จับคำเหล่านี้ "ที่ไหนก็ได้ในชื่อไฟล์" (case-insensitive) แทนการเช็คแค่ prefix
# "fastmtp" ไม่ต้องแยกเขียน เพราะมีคำว่า "mtp" เป็น substring อยู่แล้ว
_DRAFT_KEYWORDS = ("mtp", "dflash", "eagle", "draft")
_VISION_KEYWORDS = ("mmproj",)

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

    key: str  # ชื่อโชว์ผู้ใช้ เช่น "UD-IQ1_S" หรือ "Q8_0" (safetensors: quant_label ของทั้ง repo)
    files: list[HfFile]  # ไฟล์หลัก (shard ครบชุด) เรียงตาม shard
    total_bytes: int  # ผลรวมของ files เท่านั้น (ไม่รวม companions)
    shard_count: int  # 1 = ไฟล์เดียว
    companions: list[HfFile] = field(default_factory=list)  # mmproj-/mtp-/dflash- ของ repo นี้ (gguf เท่านั้น)
    format: str = "gguf"  # "gguf" (group_quants) หรือ "safetensors" (group_weights)

    @property
    def draft_files(self) -> list[HfFile]:
        """companion ชนิด draft (mtp/fastmtp/dflash/eagle/draft) ของ quant กลุ่มนี้"""
        return [f for f in self.companions if companion_kind(f.path) == "draft"]

    @property
    def vision_files(self) -> list[HfFile]:
        """companion ชนิด vision (mmproj) ของ quant กลุ่มนี้"""
        return [f for f in self.companions if companion_kind(f.path) == "vision"]


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


def fetch_config(repo_id: str, *, token: str | None = None, client: httpx.Client | None = None) -> dict:
    """ยิง config.json ตรง ๆ จาก resolve URL ของ repo (ไฟล์เล็กหลักร้อย byte)

    ใช้กับ repo safetensors ที่ HF API ปกติไม่คืน arch/ctx มาให้ (ต่างจาก GGUF ที่อ่านจาก header เอาเอง)
    HF จะ 302 ไป CDN เสมอ ⇒ ต้อง follow_redirects=True ไม่งั้นได้ 3xx เปล่า ๆ
    401/403/404 conventions เดียวกับ fetch_repo()
    """
    own_client = client is None
    http_client = client or httpx.Client()
    try:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        resp = http_client.get(
            f"{RESOLVE_BASE}/{repo_id}/resolve/main/config.json", headers=headers, follow_redirects=True,
        )
    finally:
        if own_client:
            http_client.close()

    if resp.status_code in (401, 403):
        raise GatedRepoError(f"{repo_id} ติด gate — ต้องใช้ HF_TOKEN ที่มีสิทธิ์เข้าถึง")
    if resp.status_code == 404:
        raise RepoNotFoundError(f"ไม่พบ config.json ใน repo: {repo_id}")
    resp.raise_for_status()
    return resp.json()


def list_files(repo_json: dict) -> list[HfFile]:
    """แปลง siblings จาก HF API response เป็น HfFile ทั้งหมด (ยังไม่กรอง .gguf)"""
    files = []
    for sib in repo_json.get("siblings", []):
        lfs = sib.get("lfs") or {}
        files.append(HfFile(path=sib["rfilename"], size=sib["size"], sha256=lfs.get("sha256")))
    return files


def companion_kind(path: str) -> str | None:
    """ชนิดของไฟล์คู่: "draft" (mtp/fastmtp/dflash/eagle/draft) · "vision" (mmproj) · None ถ้าไม่ใช่ companion

    ⚠️ เช็คจาก basename ของไฟล์เท่านั้น (ตัด path เต็มทิ้งก่อน) — กันกับดักที่ชื่อ repo/โฟลเดอร์
    มีคำว่า MTP อยู่ในนั้นเอง (เช่น โฟลเดอร์ "...-MTP-GGUF") แต่ไฟล์ quant ปกติข้างในกลับไม่ใช่ companion
    """
    basename = path.rsplit("/", 1)[-1].lower()
    if any(kw in basename for kw in _VISION_KEYWORDS):
        return "vision"
    if any(kw in basename for kw in _DRAFT_KEYWORDS):
        return "draft"
    return None


def _is_companion(basename: str) -> bool:
    return basename.startswith(_COMPANION_PREFIXES) or companion_kind(basename) is not None


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


# ---------------------------------------------------------------------------
# safetensors (vLLM) — 1 repo = 1 quant group เสมอ ไม่มีโฟลเดอร์ย่อย/หลาย quant ต่อ repo
# ⚠️ ห้ามใช้ _is_companion()/companion_kind() กับไฟล์กลุ่มนี้ — นั่นคือกติกาของ llama.cpp (-md)
# ไม่ใช่ของ vLLM ⇒ model_mtp.safetensors ต้องอยู่ใน files เสมอ ไม่ใช่ companions
# ---------------------------------------------------------------------------

# นามสกุลไฟล์ที่ไม่ใช่น้ำหนักโมเดล — ตัดออกจากกลุ่มเสมอ (ตรงกับ dl: ที่ vendor curate ไว้เองใน catalog.yaml)
_SAFETENSORS_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp")


def _is_excluded_from_weights(basename: str) -> bool:
    """True ถ้าไฟล์นี้ไม่ใช่น้ำหนักโมเดล/asset ที่ต้องดาวน์โหลด (เอกสาร/รูป/license)"""
    if basename == ".gitattributes":
        return True
    if basename.upper().startswith("LICENSE"):
        return True
    lower = basename.lower()
    if lower.endswith(".md"):  # ครอบ README.md ไปในตัว
        return True
    if lower.endswith(_SAFETENSORS_IMAGE_EXTS):
        return True
    return False


def has_safetensors(files: list[HfFile]) -> bool:
    """repo นี้เป็น safetensors (vLLM) ไหม — มีไฟล์ .safetensors อย่างน้อย 1 ไฟล์"""
    return any(f.path.endswith(".safetensors") for f in files)


def config_arch_ctx(config: dict) -> tuple[str | None, int | None]:
    """(arch, ctx_train) จาก config.json — ctx อาจซ้อนอยู่ใน text_config (ข้อเท็จจริงที่ verify แล้ว)

    ระดับบนสุดชนะเสมอถ้ามีทั้งสองที่ — text_config เป็นแค่ fallback
    """
    architectures = config.get("architectures")
    arch = architectures[0] if architectures else None

    ctx = config.get("max_position_embeddings")
    if ctx is None:
        text_config = config.get("text_config") or {}
        ctx = text_config.get("max_position_embeddings")

    return arch, ctx


def _quant_label_from_weights(weights: dict) -> str | None:
    """map weights dict ({"type", "num_bits", "group_size", ...}) ของ 1 config_group เป็น label

    None = ไม่มี num_bits ใช้งานได้ (ขาด/ไม่ใช่ตัวเลข) — เรียกไม่ควรใช้กลุ่มนี้
    """
    num_bits = weights.get("num_bits")
    if not isinstance(num_bits, (int, float)) or isinstance(num_bits, bool):
        return None
    num_bits = int(num_bits)

    w_type = str(weights.get("type") or "").lower()
    if w_type == "int":
        return f"INT{num_bits}"
    if w_type == "float":
        if num_bits == 4 and weights.get("group_size") == 16:
            return "NVFP4"
        if num_bits == 4:
            return "FP4"
        return f"FP{num_bits}"
    return None


def quant_label(config: dict) -> str:
    """ชื่อ quant ของทั้ง repo safetensors — จาก quantization_config ก่อน ไม่มีก็ dtype ไม่มีอีกก็ safetensors เฉย ๆ"""
    quant_cfg = config.get("quantization_config")
    if quant_cfg:
        method = quant_cfg.get("quant_method")
        if method:
            method_lower = str(method).lower()
            if method_lower in ("modelopt", "nvfp4"):
                return "NVFP4"
            if method_lower == "compressed-tensors":
                # "compressed-tensors" คือชื่อ *format ของไฟล์* ไม่ใช่ชื่อ quant — quant จริงอยู่ใน
                # config_groups (repo ผสมหลายความละเอียดในไฟล์เดียว เช่น unsloth/Qwen3.8-27B-NVFP4
                # มี group_0 = 8-bit float, group_1 = 4-bit float group_size 16) ⇒ ต้องเลือก num_bits
                # ต่ำสุด เพราะนั่นคือความละเอียดที่หยาบที่สุด/เป็นตัวที่โมเดลถูกตั้งชื่อ-การตลาดตาม
                # (อ่าน group_0 เฉย ๆ จะได้ "FP8" ทั้งที่โมเดลคือ NVFP4)
                config_groups = quant_cfg.get("config_groups")
                if isinstance(config_groups, dict) and config_groups:
                    best_bits: float | None = None
                    best_weights: dict | None = None
                    for group in config_groups.values():
                        if not isinstance(group, dict):
                            continue
                        weights = group.get("weights")
                        if not isinstance(weights, dict):
                            continue
                        num_bits = weights.get("num_bits")
                        if not isinstance(num_bits, (int, float)) or isinstance(num_bits, bool):
                            continue
                        if best_bits is None or num_bits < best_bits:
                            best_bits = num_bits
                            best_weights = weights
                    if best_weights is not None:
                        label = _quant_label_from_weights(best_weights)
                        if label is not None:
                            return label
                # config_groups ไม่มี/ว่าง/ไม่มี group ไหนใช้ num_bits ได้ ⇒ fallback แบบเดิม
                return str(method).upper()
            return str(method).upper()

    dtype = config.get("torch_dtype") or config.get("dtype")
    if dtype:
        dtype_lower = str(dtype).lower()
        if dtype_lower == "bfloat16":
            return "BF16"
        if dtype_lower == "float16":
            return "F16"
        return str(dtype).upper()

    return "safetensors"


def group_weights(files: list[HfFile], config: dict) -> list[QuantGroup]:
    """จัดกลุ่ม repo safetensors เป็น 1 quant group เดียว (ทั้ง repo) — ดู docstring ก้อนบนนี้

    files ที่รวม = ทุกไฟล์ ยกเว้น .gitattributes/README/*.md/LICENSE*/รูปภาพ (verify ตรงกับ
    catalog.yaml:106-117 ของ unsloth/Qwen3.8-27B-NVFP4 เป๊ะ) — model_mtp.safetensors ต้องติดไปด้วยเสมอ
    """
    if not has_safetensors(files):
        return []

    included = [f for f in files if not _is_excluded_from_weights(f.path.rsplit("/", 1)[-1])]
    shard_count = sum(1 for f in included if f.path.endswith(".safetensors"))

    return [
        QuantGroup(
            key=quant_label(config),
            files=included,
            total_bytes=sum(f.size for f in included),
            shard_count=shard_count,
            companions=[],
            format="safetensors",
        )
    ]


# ตระกูลโมเดล → parser ของ vLLM (--tool-call-parser / --reasoning-parser)
# ตระกูลที่ไม่รู้จัก = ไม่ใส่ parser เลย (เดาผิดอันตรายกว่าไม่ใส่ — ดู PRD ข้อ 10)
def vllm_parsers(model_type: str | None) -> dict[str, str]:
    """คืน engine_env สำหรับเลือก parser ของ vLLM ตาม model_type ใน config.json"""
    empty = {"VLLM_TOOL_PARSER": "", "VLLM_REASONING_PARSER": ""}
    if not model_type:
        return empty

    mt = model_type.lower()
    if mt.startswith("qwen3"):  # qwen3 / qwen3_5 / qwen3_moe ฯลฯ
        return {"VLLM_TOOL_PARSER": "qwen3_xml", "VLLM_REASONING_PARSER": "qwen3"}
    if mt == "llama":
        return {"VLLM_TOOL_PARSER": "llama3_json", "VLLM_REASONING_PARSER": ""}
    if mt == "mistral":
        return {"VLLM_TOOL_PARSER": "mistral", "VLLM_REASONING_PARSER": ""}
    if mt.startswith("deepseek_v3"):
        return {"VLLM_TOOL_PARSER": "deepseek_v3", "VLLM_REASONING_PARSER": "deepseek_r1"}
    return empty


def vllm_engine_env(files: list[HfFile], config: dict) -> dict[str, str]:
    """engine_env เต็มของ vLLM: parser ตาม model_type + ปิด MTP ถ้าไม่มีไฟล์ *mtp*.safetensors

    ⚠️ ไม่มี MTP ⇒ ต้องใส่ VLLM_SPECULATIVE="" ชัดเจน (ค่าว่าง = ปิด) เพราะ default ของ vllm.sh คือ "เปิด"
    (ใช้ ${VAR-default} ไม่ใช่ ${VAR:-default}) มี MTP ⇒ ไม่ใส่คีย์นี้เลย ปล่อยให้ default เปิดตามเดิม
    """
    env = dict(vllm_parsers(config.get("model_type")))

    has_mtp = any(
        f.path.rsplit("/", 1)[-1].lower().endswith(".safetensors") and "mtp" in f.path.rsplit("/", 1)[-1].lower()
        for f in files
    )
    if not has_mtp:
        env["VLLM_SPECULATIVE"] = ""

    return env


def suggest_args(group: QuantGroup, dest_dir: str) -> str:
    """สร้าง args ของ llama-server จาก companion ที่มี

    - มี draft → "-md <dest_dir>/<ชื่อไฟล์ draft>"
    - มี vision → "--mmproj <dest_dir>/<ชื่อไฟล์ mmproj>"
    - มีทั้งคู่ → ต่อกันด้วยช่องว่าง (draft ก่อน)
    - ไม่มีเลย → ""
    ⚠️ ห้ามใส่ -c เด็ดขาด (กติกาเหล็กใน CONTEXT.md — llamacpp.sh ใส่ -c จาก ctx ให้แล้ว)
    ⚠️ ถ้ามี draft หลายตัว ให้เลือกตัวที่ไฟล์เล็กที่สุด (draft ยิ่งเล็กยิ่งเร็ว)
    """
    dest = dest_dir.rstrip("/")
    parts = []

    drafts = group.draft_files
    if drafts:
        smallest = min(drafts, key=lambda f: f.size)
        basename = smallest.path.rsplit("/", 1)[-1]
        parts.append(f"-md {dest}/{basename}")

    visions = group.vision_files
    if visions:
        basename = visions[0].path.rsplit("/", 1)[-1]
        parts.append(f"--mmproj {dest}/{basename}")

    return " ".join(parts)


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
