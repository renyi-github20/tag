#!/usr/bin/env python3
"""
ESG报告字段抽取（纯 VL 大模型，不做文字匹配/规则抽取）。

流程：
1) 读取 data/requirement/requirement_1.json 中 “ESG报告” 的字段清单（以及字段 comment，用来做提示词）
2) 将 PDF 转成多页 PNG(base64)
3) 调用 config.yaml 中 vl 配置的视觉大模型，输出 JSON
4) 结果写入 JSONL（每行一个 PDF）

写入与断点续跑：
- 每条记录写完后立即 flush，中断或异常时已写入的记录不会丢失。
- 单份 PDF 失败会记录日志并跳过，继续处理下一份，不中断整批。
- 使用 --resume 时从 output 中读取已完成的 filename，只处理未完成的 PDF 并追加写入。

超过 max_pages（默认 50）的 PDF：
- 按批解析：每批最多 max_pages 页，多批分别调用 VL，再将目录/ESG 议题/奖项/编制依据等合并为一条记录。

英文报告：
- 提示词要求：若报告为英文，除「语言」填“英文”外，其余抽取内容均以中文输出（由 VL 在抽取时一并翻译）。
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Optional

import fitz  # PyMuPDF

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yaml


def load_config(config_path: Path) -> dict[str, Any]:
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def load_fields_and_comments(requirement_path: Path) -> tuple[list[str], dict[str, str]]:
    """
    从 requirement_1.json 读取 “ESG报告” 的字段列表与 comment。

    注意：该 section 内可能有重复 field 名（会导致 JSON key 冲突），这里做去重（保留首次出现顺序）。
    """
    data = json.loads(requirement_path.read_text(encoding="utf-8"))
    for section in data.get("sections", []):
        if section.get("name") != "ESG报告":
            continue

        fields: list[str] = []
        comments: dict[str, str] = {}
        seen: set[str] = set()

        for f in section.get("fields", []):
            if not f or not f.get("field"):
                continue
            name = str(f["field"]).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            fields.append(name)
            if f.get("comment"):
                comments[name] = str(f["comment"])

        if not fields:
            raise ValueError("requirement_1.json 的 ESG报告 fields 为空")
        return fields, comments

    raise ValueError("requirement_1.json 中未找到 name=ESG报告 的 section")


def _png_to_jpeg(png_bytes: bytes, *, quality: int = 80) -> bytes:
    """将 PNG 字节转为 JPEG 字节，大幅缩减体积（约 5-10 倍）。"""
    from PIL import Image

    img = Image.open(io.BytesIO(png_bytes))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def get_pdf_page_count(pdf_path: Path) -> int:
    with fitz.open(str(pdf_path)) as doc:
        return len(doc)


def pdf_to_images_base64(
    pdf_path: Path,
    *,
    dpi: int,
    max_pages: int,
    use_jpeg: bool = True,
    jpeg_quality: int = 80,
    page_start: int = 0,
) -> tuple[int, list[dict[str, Any]]]:
    """
    将 PDF 指定页范围转成 base64 图片。
    page_start: 从 0 开始的起始页；与 max_pages 共同决定本批页范围 [page_start, page_start+max_pages)。
    返回 (总页数, 本批图片列表)。
    """
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    page_end = min(page_start + max_pages, total_pages)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    images: list[dict[str, Any]] = []
    total_bytes = 0
    for page_idx in range(page_start, page_end):
        page = doc[page_idx]
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        if use_jpeg:
            img_bytes = _png_to_jpeg(img_bytes, quality=jpeg_quality)
            mime = "image/jpeg"
        else:
            mime = "image/png"
        total_bytes += len(img_bytes)
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        images.append({"page": page_idx + 1, "base64": b64, "mime": mime})

    doc.close()
    logger.info(
        "[%s] 本批页 %d-%d（共 %d 页）, 图片格式=%s, 体积=%.1f MB",
        pdf_path.name,
        page_start + 1,
        page_end,
        total_pages,
        "JPEG" if use_jpeg else "PNG",
        total_bytes / 1024 / 1024,
    )
    return total_pages, images


def pdf_text_char_count(pdf_path: Path) -> int:
    """
    统计“字数”：使用 PyMuPDF 抽取到的文本，去掉空白字符后的字符数。
    """
    with fitz.open(str(pdf_path)) as doc:
        text_parts: list[str] = []
        for page in doc:
            t = (page.get_text() or "").strip()
            if t:
                text_parts.append(t)
    full_text = "\n".join(text_parts)
    return len(re.sub(r"\s+", "", full_text))


def _merge_batch_extractions(
    batch_results: list[dict[str, Any]], fields: list[str]
) -> dict[str, Any]:
    """
    合并多批 VL 抽取结果。单值字段取首个非「未找到」；目录/议题/奖项/编制依据等合并拼接。
    """
    if not batch_results:
        return {f: "未找到" for f in fields}
    if len(batch_results) == 1:
        return dict(batch_results[0])

    # 需合并的字段（多段内容拼接）
    CONCAT_FIELDS = {"目录", "ESG奖项荣誉", "编制指引/编制依据"}
    # ESG议题：按分号拆开后取并集再拼接
    SET_FIELD = "ESG议题"
    # 单值字段：取第一个非「未找到」
    SINGLE_FIELDS = {"公司全称", "公司简称", "公司代码", "公告名称", "落款日期", "语言"}

    merged: dict[str, Any] = {}
    for f in fields:
        if f in SINGLE_FIELDS:
            for rec in batch_results:
                v = rec.get(f)
                if v and str(v).strip() and str(v) != "未找到":
                    merged[f] = v
                    break
            if f not in merged:
                merged[f] = batch_results[0].get(f, "未找到")
        elif f == SET_FIELD:
            seen: set[str] = set()
            parts: list[str] = []
            for rec in batch_results:
                v = (rec.get(f) or "").strip()
                if not v or v == "未找到":
                    continue
                for item in re.split(r"[;；\n]", v):
                    item = item.strip()
                    if item and item not in seen:
                        seen.add(item)
                        parts.append(item)
            merged[f] = ";".join(parts) if parts else "未找到"
        elif f in CONCAT_FIELDS:
            parts = []
            for rec in batch_results:
                v = (rec.get(f) or "").strip()
                if v and v != "未找到":
                    parts.append(v)
            merged[f] = "；".join(parts) if parts else "未找到"
        else:
            # 篇幅页码/字数等由外层用 PDF 整体重算，这里取第一个
            merged[f] = batch_results[0].get(f, "未找到")

    return merged


def infer_company_short_name(full_name: str) -> Optional[str]:
    """
    公司简称：若模型没给，但公司全称存在，则做确定性裁剪（不猜测）。
    """
    if not full_name or full_name == "未找到":
        return None
    short = full_name
    for suffix in ["股份有限公司", "有限责任公司", "有限公司", "集团股份有限公司", "集团有限公司", "集团", "公司"]:
        if short.endswith(suffix):
            short = short[: -len(suffix)]
            break
    return short or full_name


def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if "```json" in s:
        return s.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in s:
        return s.split("```", 1)[1].split("```", 1)[0].strip()
    return s


def _loads_json_relaxed(s: str) -> dict[str, Any]:
    """
    尽量把模型输出解析成 JSON 对象。
    """
    raw = _strip_code_fences(s)
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        obj = json.loads(raw[start : end + 1])
        if isinstance(obj, dict):
            return obj
    raise ValueError("无法解析模型输出为 JSON 对象")


def build_prompt(fields: list[str], comments: dict[str, str]) -> str:
    guidance_lines: list[str] = []
    for f in fields:
        c = comments.get(f)
        if c:
            guidance_lines.append(f"- {f}：{c}")

    fields_json = ",\n".join([f'  "{f}": "..."' for f in fields])

    parts = []
    parts.append("你是一位专业的信息抽取专家。请阅读这份ESG报告（多页图片），抽取指定字段。\n\n")
    parts.append("## 需要抽取的字段（必须全部输出）\n")
    parts.append("\n".join([f"- {f}" for f in fields]))
    parts.append("\n\n## 字段要求（来自需求说明，务必遵守）\n")
    parts.append("\n".join(guidance_lines) if guidance_lines else "无\n")
    parts.append("\n\n## ESG议题说明（21类枚举）\n")
    parts.append("环境：1应对气候变化 2污染物排放 3废弃物处理 4生态系统和生物多样性保护 5环境合规管理 6能源利用 7水资源利用 8循环经济\n")
    parts.append("社会：9乡村振兴 10社会贡献 11创新驱动 12科技伦理 13供应链安全 14平等对待中小企业 15产品和服务安全与质量 16数据安全与客户隐私保护 17员工\n")
    parts.append("可持续发展相关治理：18尽职调查 19利益相关方沟通 20反商业贿赂及反贪污 21反不正当竞争\n")
    parts.append('"ESG议题"只输出报告中涉及到的议题选项：用分号分隔的编号或名称（如 1应对气候变化；6能源利用）。不要写理由、页码、章节说明或内容摘要。\n')
    parts.append("**ESG议题必尽量判断**：根据目录、章节标题与正文内容，若涉及治理/环境/社会/气候/员工/供应链/合规/反腐败等任一维度，须从上述21类中勾选对应议题并输出，不要轻易填「未找到」；仅当文档确无任何ESG相关表述（如纯资本充足率、第三支柱披露）时可填未找到。\n\n")
    parts.append("## 输出要求\n")
    parts.append("- 只输出 JSON（不要输出任何解释性文字）\n")
    parts.append("- 字段名必须与上面完全一致\n")
    parts.append('- 若确实找不到，填"未找到"\n')
    parts.append("- **若报告为英文**：除「语言」填“英文”外，其余所有字段内容（公告名称、目录、ESG议题、ESG奖项荣誉、编制指引等）均需翻译成中文后输出。\n")
    parts.append("- 日期统一输出 YYYY-MM-DD（如 2024-03-15）\n")
    parts.append('- "目录"：必须完整提取，精确到二级标题，一级和二级均不遗漏；用分号 ; 分隔，支持关联跳转\n')
    parts.append('- "语言"只能是：中文 / 英文（表示报告原文语言）\n')
    parts.append('- "篇幅页码"输出数字（总页数）\n')
    parts.append('- "字数"输出数字（全文字数/字符数均可）\n')
    parts.append('- "ESG议题"：只填选项（编号或名称），用分号分隔；结合目录与章节主动判断，勿轻易未找到\n')
    parts.append('- "编制指引/编制依据"：必须全面提取报告中引用的所有编制依据，如联交所ESG报告指引、CASS-ESG5.0、GRI Standards、联合国可持续发展目标、TCFD、SASB等，不要遗漏\n')
    parts.append('- "ESG奖项荣誉"：必须全面提取正文中获得的ESG相关奖项、荣誉、认证，奖项名称需准确完整，不要遗漏或写错\n\n')
    parts.append("请按以下 JSON 结构输出：\n{\n")
    parts.append(fields_json)
    parts.append("\n}\n")
    return "".join(parts)


def _build_session(*, verify_ssl: bool = True) -> requests.Session:
    """创建带自动重试的 Session（针对连接错误 / SSL 错误自动重试）。"""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=2,  # 2s, 4s, 8s
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.verify = verify_ssl
    return session


def _call_vl(*, vl: dict[str, Any], content: list[dict[str, Any]]) -> dict[str, Any]:
    n_images = sum(1 for x in content if x.get("type") == "image_url")
    from scripts.vl_utils import get_vl_url
    url = get_vl_url(vl)
    logger.info("VL 请求: POST %s, 共 %d 条 content（其中 %d 张图）, timeout=600s", url, len(content), n_images)

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if vl.get("api_key"):
        headers["Authorization"] = f"Bearer {vl['api_key']}"

    payload = {
        "model": vl["model"],
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": int(vl.get("max_tokens", 4096)),
        "stream": False,
    }

    verify_ssl = vl.get("verify_ssl", True)
    if not verify_ssl:
        logger.warning("VL 请求已关闭 SSL 证书校验（verify_ssl=false），仅建议在受信环境使用")

    # 手动重试：urllib3 Retry 无法覆盖 SSL 连接错误，这里额外包一层
    max_attempts = 4
    for attempt in range(1, max_attempts + 1):
        try:
            session = _build_session(verify_ssl=verify_ssl)
            resp = session.post(
                url, headers=headers, json=payload, timeout=600
            )
            resp.raise_for_status()
            logger.info("VL 响应: 成功, 状态码 %s", resp.status_code)
            return resp.json()
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
            if attempt < max_attempts:
                wait = 5 * attempt
                logger.warning(
                    "VL 请求失败 (第 %d/%d 次): %s — %s 秒后重试...",
                    attempt, max_attempts, type(e).__name__, wait,
                )
                time.sleep(wait)
            else:
                logger.error("VL 请求连续 %d 次失败，放弃", max_attempts)
                raise


def _extract_one_batch(
    *,
    pdf_path: Path,
    vl: dict[str, Any],
    fields: list[str],
    comments: dict[str, str],
    dpi: int,
    max_pages: int,
    page_start: int,
    jpeg_quality: int = 80,
) -> dict[str, Any]:
    """对 PDF 的某一页范围调用一次 VL 并返回解析后的字段字典。"""
    total_pages, images = pdf_to_images_base64(
        pdf_path, dpi=dpi, max_pages=max_pages, page_start=page_start,
        jpeg_quality=jpeg_quality,
    )
    if not images:
        return {f: "未找到" for f in fields}

    prompt = build_prompt(fields, comments)
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for img in images:
        mime = img.get("mime", "image/png")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{img['base64']}"},
            }
        )

    data = _call_vl(vl=vl, content=content)
    raw = data["choices"][0]["message"]["content"]
    extracted = _loads_json_relaxed(raw)
    for f in fields:
        if f not in extracted or extracted[f] in (None, "", []):
            extracted[f] = "未找到"
    return {k: extracted.get(k) for k in fields}


def call_vl_extract(
    *,
    pdf_path: Path,
    fields: list[str],
    comments: dict[str, str],
    config: dict[str, Any],
    dpi: int,
    max_pages: int,
    parallel_workers: int = 1,
    jpeg_quality: int = 80,
) -> dict[str, Any]:
    if "vl" not in config:
        raise ValueError("config.yaml 缺少 vl 配置")
    vl = config["vl"]
    if not vl.get("enable", True):
        raise ValueError("config.yaml 中 vl.enable=false，无法使用 VL 抽取")

    logger.info("[%s] 开始抽取", pdf_path.name)
    total_pages = get_pdf_page_count(pdf_path)

    if total_pages <= max_pages:
        # 单批
        batch_results = [
            _extract_one_batch(
                pdf_path=pdf_path,
                vl=vl,
                fields=fields,
                comments=comments,
                dpi=dpi,
                max_pages=max_pages,
                page_start=0,
                jpeg_quality=jpeg_quality,
            )
        ]
        logger.info("[%s] PDF 总页数 %d，单批完成", pdf_path.name, total_pages)
    else:
        # 多批：可并行调用 VL，再按页序合并
        page_starts = list(range(0, total_pages, max_pages))
        if parallel_workers and parallel_workers > 1:
            batch_results = [None] * len(page_starts)
            with ThreadPoolExecutor(max_workers=min(parallel_workers, len(page_starts))) as ex:
                future_to_idx = {
                    ex.submit(
                        _extract_one_batch,
                        pdf_path=pdf_path,
                        vl=vl,
                        fields=fields,
                        comments=comments,
                        dpi=dpi,
                        max_pages=max_pages,
                        page_start=page_start,
                        jpeg_quality=jpeg_quality,
                    ): i
                    for i, page_start in enumerate(page_starts)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        batch_results[idx] = future.result()
                    except Exception as e:
                        logger.exception("[%s] 第 %d 批失败: %s", pdf_path.name, idx + 1, e)
                        batch_results[idx] = {f: "未找到" for f in fields}
            batch_results = [r for r in batch_results if r is not None]
        else:
            batch_results = []
            for page_start in page_starts:
                logger.info(
                    "[%s] 第 %d 批，页 %d-%d（总 %d 页）",
                    pdf_path.name,
                    len(batch_results) + 1,
                    page_start + 1,
                    min(page_start + max_pages, total_pages),
                    total_pages,
                )
                rec = _extract_one_batch(
                    pdf_path=pdf_path,
                    vl=vl,
                    fields=fields,
                    comments=comments,
                    dpi=dpi,
                    max_pages=max_pages,
                    page_start=page_start,
                    jpeg_quality=jpeg_quality,
                )
                batch_results.append(rec)
        logger.info("[%s] 共 %d 批 VL 调用完成，正在合并结果", pdf_path.name, len(batch_results))

    extracted = _merge_batch_extractions(batch_results, fields)
    record: dict[str, Any] = {"filename": pdf_path.name, **extracted}

    if "篇幅页码" in record:
        record["篇幅页码"] = str(total_pages)
    if "字数" in record:
        cnt = pdf_text_char_count(pdf_path)
        record["字数"] = str(cnt) if cnt > 0 else (record.get("字数") or "未找到")
    if "公司简称" in record and (record["公司简称"] in (None, "", "未找到")):
        short = infer_company_short_name(str(record.get("公司全称") or ""))
        if short:
            record["公司简称"] = short

    logger.info("[%s] 本份抽取完成", pdf_path.name)
    return record


def iter_pdfs(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() == ".pdf":
            yield input_path
        return
    for p in sorted(input_path.glob("**/*")):
        if p.is_file() and p.suffix.lower() == ".pdf":
            yield p


def load_done_filenames(output_path: Path) -> set[str]:
    """
    从已有 JSONL 中读取已完成的 filename 集合，用于断点续跑时跳过。
    解析失败或非 JSON 行（如含大量前导空格的脏行）则忽略该行。
    """
    done: set[str] = set()
    if not output_path.exists():
        return done
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "filename" in obj:
                    done.add(str(obj["filename"]))
            except Exception:
                continue
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description="ESG报告字段抽取（VL大模型）")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径（需要包含 vl 配置）")
    parser.add_argument("--requirement", default="data/requirement/requirement_1.json", help="requirement_1.json 路径")
    parser.add_argument("--input", default="data/report/ESG报告", help="ESG报告PDF目录（或单个PDF）")
    parser.add_argument("--output", default="result/esg_reports_extracted_vl.jsonl", help="输出 jsonl 路径")
    parser.add_argument("--limit", type=int, default=1, help="最多处理多少份PDF（0=不限制；默认1用于生成样例）")
    parser.add_argument("--skip", type=int, default=0, help="跳过前 N 份 PDF，与 --limit 配合可只处理中间或最后若干份（如 --skip 18 --limit 10 处理第19-28份）")
    parser.add_argument("--dpi", type=int, default=150, help="PDF转图片的dpi")
    parser.add_argument("--max-pages", type=int, default=50, help="单份PDF最多处理多少页（默认50）")
    parser.add_argument("--parallel-workers", type=int, default=0, help="多批时并行 VL 调用数（0=串行；2~4 可显著缩短多批耗时）")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="PDF 转 JPEG 质量 1-100，越低体积越小、速度越快（默认80）")
    parser.add_argument("--append", action="store_true", help="追加写入 output（默认覆盖）")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续跑：读取 output 中已有 filename，只处理未完成的 PDF，并追加写入",
    )
    parser.add_argument("--print-sample", action="store_true", help="打印第一条结果到stdout")
    parser.add_argument(
        "--no-verify-ssl",
        action="store_true",
        help="关闭 VL 请求的 SSL 证书校验（用于解决 SSLEOFError 等握手问题，仅建议在受信环境使用）",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    requirement_path = Path(args.requirement)
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("开始抽取ESG报告")
    config = load_config(config_path)
    if args.no_verify_ssl:
        config.setdefault("vl", {})["verify_ssl"] = False
    fields, comments = load_fields_and_comments(requirement_path)
    logger.info("已加载 config 与 requirement，共 %d 个字段", len(fields))

    pdfs = list(iter_pdfs(input_path))
    if args.skip > 0:
        pdfs = pdfs[args.skip :]
        logger.info("已跳过前 %d 份，剩余 %d 份", args.skip, len(pdfs))
    if args.limit and args.limit > 0:
        pdfs = pdfs[: args.limit]

    # 断点续跑：跳过 output 中已有 filename 的 PDF，只处理未完成的
    if args.resume and output_path.exists():
        done = load_done_filenames(output_path)
        pdfs = [p for p in pdfs if p.name not in done]
        logger.info("断点续跑: 已有 %d 份已完成，待处理 %d 份", len(done), len(pdfs))
    else:
        logger.info("待处理 PDF: %d 个", len(pdfs))

    mode = "a" if (args.append or args.resume) else "w"
    output_file = output_path.open(mode, encoding="utf-8")
    total = len(pdfs)
    success_count = 0

    try:
        for i, p in enumerate(pdfs, 1):
            logger.info("===== 第 %d/%d 个: %s =====", i, total, p.name)
            try:
                rec = call_vl_extract(
                    pdf_path=p,
                    fields=fields,
                    comments=comments,
                    config=config,
                    dpi=args.dpi,
                    max_pages=args.max_pages,
                    parallel_workers=args.parallel_workers,
                    jpeg_quality=args.jpeg_quality,
                )
                output_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                output_file.flush()
                success_count += 1
                logger.info("[%s] 已写入 %s", p.name, output_path)
            except Exception as e:
                logger.exception("[%s] 本份抽取失败，跳过，继续下一份: %s", p.name, e)
    finally:
        output_file.close()

    logger.info("本轮完成: 成功 %d 份，共待处理 %d 份", success_count, total)

    if args.print_sample:
        with output_path.open("r", encoding="utf-8") as f:
            first_line = f.readline()
            if first_line:
                print(json.dumps(json.loads(first_line), ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
