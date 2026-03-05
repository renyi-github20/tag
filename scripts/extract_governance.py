#!/usr/bin/env python3
"""
治理制度字段抽取（纯 VL 大模型，流程与定期报告类似）。

流程：
1) 读取 data/requirement/requirement_1.json 中「治理制度」的字段清单及 comment
2) 将 PDF 转成多页 PNG(base64)
3) 调用 config.yaml 中 vl 配置的视觉大模型，输出 JSON
4) 结果写入 JSONL（每行一个 PDF）
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable

import fitz  # PyMuPDF

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
import requests
import yaml


def load_config(config_path: Path) -> dict[str, Any]:
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def load_fields_and_comments(requirement_path: Path) -> tuple[list[str], dict[str, str]]:
    """
    从 requirement_1.json 读取「治理制度」的字段列表与 comment。
    字段去重（保留首次出现顺序）。
    """
    data = json.loads(requirement_path.read_text(encoding="utf-8"))
    for section in data.get("sections", []):
        if section.get("name") != "治理制度":
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
            raise ValueError("requirement_1.json 的 治理制度 fields 为空")
        return fields, comments

    raise ValueError("requirement_1.json 中未找到 name=治理制度 的 section")


def pdf_to_images_base64(
    pdf_path: Path,
    *,
    dpi: int,
    max_pages: int,
) -> tuple[int, list[dict[str, Any]]]:
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)

    use_pages = min(total_pages, max_pages)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    images: list[dict[str, Any]] = []
    for page_idx in range(use_pages):
        page = doc[page_idx]
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        images.append({"page": page_idx + 1, "base64": b64})

    doc.close()
    return total_pages, images


def pdf_text_char_count(pdf_path: Path) -> int:
    """统计字数：PDF 抽取文本去掉空白后的字符数。"""
    with fitz.open(str(pdf_path)) as doc:
        text_parts: list[str] = []
        for page in doc:
            t = (page.get_text() or "").strip()
            if t:
                text_parts.append(t)
    full_text = "\n".join(text_parts)
    return len(re.sub(r"\s+", "", full_text))


def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if "```json" in s:
        return s.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in s:
        return s.split("```", 1)[1].split("```", 1)[0].strip()
    return s


def _loads_json_relaxed(s: str) -> dict[str, Any]:
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
    parts.append("你是一位专业的信息抽取专家。请阅读这份治理制度文档（多页图片），抽取指定字段。\n\n")
    parts.append("## 需要抽取的字段（必须全部输出）\n")
    parts.append("\n".join([f"- {f}" for f in fields]))
    parts.append("\n\n## 字段要求（来自需求说明，务必遵守）\n")
    parts.append("\n".join(guidance_lines) if guidance_lines else "无\n")
    parts.append("\n\n## 输出要求\n")
    parts.append("- 只输出 JSON（不要输出任何解释性文字）\n")
    parts.append("- 字段名必须与上面完全一致\n")
    parts.append('- 若确实找不到，填"未找到"\n')
    parts.append("- 落款日期统一输出 YYYY-MM-DD（如 2024-03-15）\n")
    parts.append(
        '- "制度分类"：若已知所属公告的三级分类则与之一致；否则根据制度类型（如独立董事制度、对外担保管理制度等）从文档推断分类名称，或填"未找到"\n'
    )
    parts.append('- "法规依据"提取制度正文中涉及的法规名称，可多条用分号分隔\n')
    parts.append(
        '- "关联公告名称"：指同一家公司、同一天披露的与修订本制度相关的董事会/股东会决议公告（如《第X届董事会第X次会议决议公告》），非治理制度文档。若本文档中未出现该决议公告名称则填"未找到"\n'
    )
    parts.append(
        '- "决策机构"只能单选其一：股东会 / 董事会 / 监事会。以文档正文表述为准，根据实际审议或批准该制度的机构填写（文档写的是哪个就填哪个），不要默认优先某一项\n'
    )
    parts.append('- "篇幅页码"输出数字（总页数）\n')
    parts.append('- "字数"输出数字（全文字数/字符数均可）\n\n')
    parts.append("请按以下 JSON 结构输出：\n{\n")
    parts.append(fields_json)
    parts.append("\n}\n")
    return "".join(parts)


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
    resp = requests.post(url, headers=headers, json=payload, timeout=600, verify=verify_ssl)
    try:
        data = resp.json()
    except requests.exceptions.JSONDecodeError as e:
        preview = (resp.text or "")[:500]
        raise RuntimeError(
            f"VL 接口返回非 JSON（status={resp.status_code}）。"
            f"请确认 config 中 vl.api_url 为完整地址：.../v1/chat/completions。响应预览: {preview!r}"
        ) from e

    # 阿里云 DashScope 可能返回 200 但 body 里带 error（如模型不存在、参数错误等）
    if "error" in data:
        err = data["error"]
        msg = err.get("message") or err.get("code") or str(err)
        logger.error("VL 接口返回错误: %s", msg)
        raise RuntimeError(f"阿里云 VL 调用失败: {msg}")

    resp.raise_for_status()
    logger.info("VL 响应: 成功, 状态码 %s", resp.status_code)
    return data


def infer_company_short_name(full_name: str) -> str | None:
    """公司简称：若模型没给，用公司全称做确定性裁剪。"""
    if not full_name or full_name == "未找到":
        return None
    short = full_name
    for suffix in ["股份有限公司", "有限责任公司", "有限公司", "集团股份有限公司", "集团有限公司", "集团", "公司"]:
        if short.endswith(suffix):
            short = short[: -len(suffix)]
            break
    return short or full_name


def call_vl_extract(
    *,
    pdf_path: Path,
    fields: list[str],
    comments: dict[str, str],
    config: dict[str, Any],
    dpi: int,
    max_pages: int,
) -> dict[str, Any]:
    if "vl" not in config:
        raise ValueError("config.yaml 缺少 vl 配置")
    vl = config["vl"]
    if not vl.get("enable", True):
        raise ValueError("config.yaml 中 vl.enable=false，无法使用 VL 抽取")

    logger.info("[%s] 开始抽取", pdf_path.name)
    total_pages, images = pdf_to_images_base64(pdf_path, dpi=dpi, max_pages=max_pages)
    logger.info("[%s] PDF 已转成 %d 页图片（总页数 %d）", pdf_path.name, len(images), total_pages)
    prompt = build_prompt(fields, comments)

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for img in images:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img['base64']}"},
            }
        )

    logger.info("[%s] 正在调用 VL...", pdf_path.name)
    data = _call_vl(vl=vl, content=content)
    choice = (data.get("choices") or [None])[0]
    msg = choice.get("message") if choice else None
    raw = (msg.get("content") if msg else None) or ""
    if not raw:
        raise RuntimeError("阿里云 VL 返回内容为空，请检查模型是否支持视觉输入（如 qwen-vl-plus / qwen3-vl-plus）及 API 额度")
    extracted = _loads_json_relaxed(raw)
    logger.info("[%s] VL 返回成功，已解析 JSON", pdf_path.name)

    for f in fields:
        if f not in extracted or extracted[f] in (None, "", []):
            extracted[f] = "未找到"

    record: dict[str, Any] = {"filename": pdf_path.name, **{k: extracted.get(k) for k in fields}}

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


def parse_date_from_governance_filename(filename: str) -> str | None:
    """从治理制度 PDF 文件名解析日期，格式如 145986903_2025-10-10_xxx.PDF，返回 YYYY-MM-DD。"""
    if not filename:
        return None
    # 匹配 _YYYY-MM-DD_ 或 _YYYYMMDD_
    m = re.search(r"_(\d{4})-?(\d{2})-?(\d{2})_", filename)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def build_sibling_index(pdf_paths: list[Path]) -> list[tuple[str, str]]:
    """构建 (filename, date_from_filename) 列表，用于同目录下同公司同日期的关联公告匹配。"""
    index: list[tuple[str, str]] = []
    for p in pdf_paths:
        d = parse_date_from_governance_filename(p.name)
        if d:
            index.append((p.name, d))
    return index


def _is_governance_doc_filename(fname: str) -> bool:
    """
    判断文件名是否为治理制度文档（非决议公告）。
    治理制度文档通常含「制度」「规则」「细则」「办法」「章程」等，且不含「决议」。
    """
    if "决议" in fname:
        return False
    # 典型治理制度文档特征
    gov_patterns = ("制度", "规则", "细则", "办法", "章程", "规程")
    return any(p in fname for p in gov_patterns)


def _infer_decision_maker_from_filename(fname: str) -> str | None:
    """从决议公告文件名推断决策机构：股东会 / 董事会 / 监事会。"""
    if "股东" in fname and "决议" in fname:
        return "股东会"
    if "监事" in fname and "决议" in fname:
        return "监事会"
    if "董事会" in fname and "决议" in fname:
        return "董事会"
    if "决议" in fname:
        return "董事会"  # 决议公告多为董事会
    return None


def find_related_announcement_from_dir(
    record: dict[str, Any],
    sibling_index: list[tuple[str, str]],
) -> tuple[str | None, str | None]:
    """
    从同一治理制度目录下的文件名中找关联公告（仅决议公告，非治理制度文档）。
    反馈要求：关联公告应是董事会/股东会决议公告，不能是另一份制度文档。
    同公司、同日期的多个制度应绑定同日的决议公告；若没有决议公告则返回 None。
    返回 (关联公告文件名, 从文件名推断的决策机构)，若无则 (None, None)。
    """
    company_short = (record.get("公司简称") or "").strip()
    company_full = (record.get("公司全称") or "").strip()
    current = (record.get("filename") or "").strip()
    record_date = _normalize_date(str(record.get("落款日期") or ""))
    if not company_short and not company_full:
        return None, None
    # 去掉 *ST、ST 等前缀再匹配文件名，避免简称与文件名不一致
    company_short_core = re.sub(r"^\*?ST\s*", "", company_short) if company_short else ""
    resolution_files: list[str] = []
    for fname, fdate in sibling_index:
        if fname == current:
            continue
        # 只接受决议公告，排除治理制度文档（制度/规则/细则/办法/章程等）
        if "决议" not in fname:
            continue
        if _is_governance_doc_filename(fname):
            continue
        # 必须同日期（若无落款日期则不过滤日期）
        if record_date and fdate and record_date != fdate:
            continue
        # 公司匹配
        if company_short and (company_short in fname or (company_short_core and company_short_core in fname)):
            resolution_files.append(fname)
        elif company_full and company_full in fname:
            resolution_files.append(fname)
    if not resolution_files:
        return None, None
    chosen = resolution_files[0]
    return chosen, _infer_decision_maker_from_filename(chosen)


def _normalize_date(s: str) -> str | None:
    """将日期统一为 YYYY-MM-DD，无法解析则返回 None。"""
    if not s or s == "未找到":
        return None
    s = re.sub(r"\s+", "", str(s))
    # 匹配 YYYY-MM-DD 或 YYYYMMDD
    m = re.match(r"(\d{4})-?(\d{2})-?(\d{2})", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def load_announcements_jsonl(path: Path) -> list[dict[str, Any]]:
    """加载公告列表 JSONL，每行需含 date、title，以及 company_short 或 company_full。"""
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and ("title" in obj or "announcement_title" in obj) and "date" in obj:
                    rows.append(obj)
            except Exception:
                continue
    return rows


def find_related_announcement(
    record: dict[str, Any],
    announcements: list[dict[str, Any]],
    *,
    institution_name: str,
) -> tuple[str | None, dict[str, Any] | None]:
    """
    在公告列表中查找关联公告：同公司、同日，且标题为董事会/股东会决议类（含修订、制度等关键词）。
    返回 (公告标题, 公告对象)。公告对象可用于取 决策机构 等字段；若无匹配则 (None, None)。
    """
    date_val = _normalize_date(str(record.get("落款日期") or ""))
    company_short = (record.get("公司简称") or "").strip()
    company_full = (record.get("公司全称") or "").strip()
    if not date_val:
        return None
    # 公司匹配：公告中的公司名与 record 的简称/全称有包含关系即可
    def company_match(a: dict[str, Any]) -> bool:
        title = (a.get("title") or a.get("announcement_title") or "").strip()
        cs = (a.get("company_short") or "").strip()
        cf = (a.get("company_full") or "").strip()
        if company_short and (company_short in title or (cs and (company_short in cs or cs in company_short))):
            return True
        if company_full and (company_full in title or (cf and (company_full in cf or cf in company_full))):
            return True
        if title and (company_short in title or company_full in title):
            return True
        return False

    def date_match(a: dict[str, Any]) -> bool:
        d = _normalize_date(str(a.get("date") or a.get("披露日期") or ""))
        return d == date_val

    def is_resolution_about_institution(a: dict[str, Any]) -> bool:
        title = (a.get("title") or a.get("announcement_title") or "").strip()
        if "决议" not in title and "董事会" not in title and "股东" not in title:
            return False
        if "修订" in title or "制度" in title:
            return True
        if institution_name and institution_name in title:
            return True
        return "决议" in title  # 决议公告可作关联

    for a in announcements:
        if not date_match(a) or not company_match(a):
            continue
        if not is_resolution_about_institution(a):
            continue
        title = (a.get("title") or a.get("announcement_title") or "").strip()
        if title:
            return title, a
    return None, None


def load_classification_by_filename(path: Path) -> dict[str, str]:
    """加载 filename -> 三级分类 映射。支持 JSONL（每行 {filename, classification}）或 JSON 对象。"""
    mapping: dict[str, str] = {}
    if not path.exists():
        return mapping
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return mapping
    # 尝试 JSON 对象
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            for k, v in obj.items():
                if v and isinstance(v, str):
                    mapping[str(k).strip()] = v.strip()
            return mapping
    except Exception:
        pass
    # JSONL
    for line in text.split("\n"):
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                fn = row.get("filename") or row.get("file_name")
                cls = row.get("classification") or row.get("制度分类")
                if fn and cls:
                    mapping[str(fn).strip()] = str(cls).strip()
        except Exception:
            continue
    return mapping


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
    parser = argparse.ArgumentParser(description="治理制度字段抽取（VL大模型）")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径（需包含 vl 配置）")
    parser.add_argument("--requirement", default="data/requirement/requirement_1.json", help="requirement_1.json 路径")
    parser.add_argument("--input", default="data/report/治理制度", help="治理制度 PDF 目录（或单个 PDF）")
    parser.add_argument("--output", default="result/governance_extracted_vl.jsonl", help="输出 jsonl 路径")
    parser.add_argument("--limit", type=int, default=5, help="最多处理多少份 PDF（0=不限制；默认 5 用于样例）")
    parser.add_argument("--skip", type=int, default=0, help="跳过前 N 份 PDF，与 --limit 配合可只处理中间或最后若干份（如 --skip 18 --limit 10 处理第19-28份）")
    parser.add_argument("--dpi", type=int, default=150, help="PDF 转图片的 dpi")
    parser.add_argument("--max-pages", type=int, default=50, help="单份 PDF 最多处理多少页")
    parser.add_argument("--append", action="store_true", help="追加写入 output（默认覆盖）")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续跑：读取 output 中已有 filename，只处理未完成的 PDF，并追加写入",
    )
    parser.add_argument("--print-sample", action="store_true", help="打印第一条结果到 stdout")
    parser.add_argument(
        "--no-verify-ssl",
        action="store_true",
        help="关闭 VL 请求的 SSL 证书校验（用于解决 SSLEOFError 等握手问题，仅建议在受信环境使用）",
    )
    parser.add_argument(
        "--announcements",
        type=str,
        default="",
        help="可选：公告列表 JSONL 路径，用于从同公司同日公告中匹配「关联公告名称」。每行需含 date、title，及 company_short 或 company_full",
    )
    parser.add_argument(
        "--classification",
        type=str,
        default="",
        help="可选：filename -> 三级分类 映射文件（JSON 或 JSONL），用于覆盖「制度分类」。无三级分类时可省略",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    requirement_path = Path(args.requirement)
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    announcements_list: list[dict[str, Any]] = []
    if args.announcements:
        ap = Path(args.announcements)
        announcements_list = load_announcements_jsonl(ap)
        logger.info("已加载公告列表: %s，共 %d 条", ap, len(announcements_list))

    classification_map: dict[str, str] = {}
    if args.classification:
        cp = Path(args.classification)
        classification_map = load_classification_by_filename(cp)
        logger.info("已加载制度分类映射: %s，共 %d 条", cp, len(classification_map))

    logger.info("开始抽取治理制度")
    config = load_config(config_path)
    if args.no_verify_ssl:
        config.setdefault("vl", {})["verify_ssl"] = False
    fields, comments = load_fields_and_comments(requirement_path)
    logger.info("已加载 config 与 requirement，共 %d 个字段", len(fields))

    # 同目录下所有 PDF（用于按「同公司、同日期」匹配关联公告名称）
    input_dir = input_path if input_path.is_dir() else input_path.parent
    all_pdfs_in_dir = list(iter_pdfs(input_dir))
    sibling_index = build_sibling_index(all_pdfs_in_dir)
    if sibling_index:
        logger.info("已构建同目录文件名索引，共 %d 个 PDF，用于关联公告匹配", len(sibling_index))

    pdfs = list(iter_pdfs(input_path))
    if args.skip > 0:
        pdfs = pdfs[args.skip:]
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
                )
                # 从同目录下同公司、同日期的文件名匹配「关联公告名称」（仅决议公告，不关联制度文档）
                if sibling_index and "关联公告名称" in rec and (rec.get("关联公告名称") in (None, "", "未找到")):
                    related_fname, inferred_decision = find_related_announcement_from_dir(rec, sibling_index)
                    if related_fname:
                        rec["关联公告名称"] = related_fname
                        if inferred_decision and rec.get("决策机构") in (None, "", "未找到"):
                            rec["决策机构"] = inferred_decision
                            logger.info("[%s] 从决议公告文件名推断决策机构: %s", p.name, inferred_decision)
                        logger.info("[%s] 从同目录匹配到决议公告: %s", p.name, related_fname[:60])
                # 可选：若仍未找到，从外部公告库 JSONL 匹配
                if announcements_list and "关联公告名称" in rec and (rec.get("关联公告名称") in (None, "", "未找到")):
                    related_title, related_ann = find_related_announcement(
                        rec,
                        announcements_list,
                        institution_name=str(rec.get("制度名称") or ""),
                    )
                    if related_title:
                        rec["关联公告名称"] = related_title
                        # 依据关联公告数据对应决策机构（反馈要求）
                        ann_decision = (related_ann or {}).get("决策机构") or (related_ann or {}).get("decision_maker")
                        if ann_decision and ann_decision in ("股东会", "董事会", "监事会"):
                            rec["决策机构"] = ann_decision
                            logger.info("[%s] 从公告库匹配到关联公告，并采纳决策机构: %s", p.name, ann_decision)
                        else:
                            logger.info("[%s] 从公告库匹配到关联公告: %s", p.name, related_title[:50])
                # 可选：用外部映射覆盖「制度分类」（无三级分类时可省略）
                if classification_map and rec.get("filename") in classification_map:
                    rec["制度分类"] = classification_map[rec["filename"]]
                    logger.info("[%s] 已用映射覆盖制度分类: %s", p.name, rec["制度分类"])
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
