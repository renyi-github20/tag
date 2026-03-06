#!/usr/bin/env python3
"""
定期报告字段抽取（纯 VL 大模型，不做文字匹配/规则抽取）。

流程：
1) 读取 data/requirement/requirement_1.json 中 “定期报告” 的字段清单（以及字段 comment，用来做提示词）
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
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Optional

import fitz  # PyMuPDF

# 保证多线程下 from scripts.xxx 可导入（无论当前工作目录）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
import requests
import yaml


def _default_input_dir() -> str:
    """默认 kg_data/data：该目录下所有子文件夹里的「定期报告」中的 PDF 都会被扫描。"""
    for p in [
        Path("/home/azureuser/workspace/data/kg_data/data"),
        Path("/workspace/data/kg_data/data"),
        _PROJECT_ROOT.parent / "data" / "kg_data" / "data",
        _PROJECT_ROOT / "data" / "kg_data" / "data",
    ]:
        if p.exists():
            return str(p)
    return "data/kg_data/data"


def load_config(config_path: Path) -> dict[str, Any]:
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


# 主要会计数据和财务指标表中，每类指标需提取三值：本期数据、上期数据、同比变动比例（%）
TRIPLE_VALUE_METRICS: frozenset[str] = frozenset({
    "营业收入（单位：元）",
    "归属于上市公司股东的净利润\n（单位：元）",
    "归属于上市公司股东的扣除非经常性损益的净利润（单位：元）",
    "总资产/资产总额\n（单位：元）",
    "归属于上市公司股东的净资产\n（单位：元）",
    "经营活动产生的现金流量净额",
    "基本每股收益（元／股）",
    "加权平均净资产收益率（%）",
})


def load_fields_and_comments(requirement_path: Path) -> tuple[list[str], dict[str, str]]:
    """
    从 requirement_1.json 读取 “定期报告” 的字段列表与 comment。

    注意：该 section 内可能有重复 field 名（会导致 JSON key 冲突），这里做去重（保留首次出现顺序）。
    """
    data = json.loads(requirement_path.read_text(encoding="utf-8"))
    for section in data.get("sections", []):
        if section.get("name") != "定期报告":
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

            # 主要会计数据和财务指标：每类指标输出三值（本期数据、上期数据、同比变动比例%）
            if name in TRIPLE_VALUE_METRICS:
                comment_str = str(f["comment"]) if f.get("comment") else ""
                for suffix in ["-本期数据", "-上期数据", "-同比变动比例（%）"]:
                    key = name + suffix
                    fields.append(key)
                    comments[key] = comment_str or f"该指标在「主要会计数据和财务指标」表中对应{suffix}列"
            else:
                fields.append(name)
                if f.get("comment"):
                    comments[name] = str(f["comment"])

        if not fields:
            raise ValueError("requirement_1.json 的 定期报告 fields 为空")
        return fields, comments

    raise ValueError("requirement_1.json 中未找到 name=定期报告 的 section")


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
    """
    统计“字数”：使用 PyMuPDF 抽取到的文本，去掉空白字符后的字符数。
    注意：这是统计值，不用于字段匹配/抽取。
    """
    with fitz.open(str(pdf_path)) as doc:
        text_parts: list[str] = []
        for page in doc:
            t = (page.get_text() or "").strip()
            if t:
                text_parts.append(t)
    full_text = "\n".join(text_parts)
    return len(re.sub(r"\s+", "", full_text))


def pdf_find_pages_by_keywords(
    pdf_path: Path,
    *,
    keywords: list[str],
    max_scan_pages: int,
) -> list[int]:
    """
    用 PDF 可抽取文本做“页定位”（只定位页，不抽取值），用于把关键页发给 VL 复核。
    返回 1-based 页码列表。
    """
    if not keywords:
        return []
    keys = [k for k in (kw.strip() for kw in keywords) if k]
    if not keys:
        return []

    hits: list[int] = []
    with fitz.open(str(pdf_path)) as doc:
        scan_pages = min(len(doc), max_scan_pages)
        for i in range(scan_pages):
            t = (doc[i].get_text() or "")
            if not t:
                continue
            if any(k in t for k in keys):
                hits.append(i + 1)
    return hits


def pdf_pages_to_images_base64(
    pdf_path: Path,
    *,
    dpi: int,
    pages_1based: list[int],
) -> list[dict[str, Any]]:
    """
    将指定页码渲染为 PNG(base64)；pages_1based 为 1-based。
    """
    if not pages_1based:
        return []
    pages = sorted(set(p for p in pages_1based if p and p > 0))
    if not pages:
        return []

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    images: list[dict[str, Any]] = []
    for p in pages:
        if p > total_pages:
            continue
        page = doc[p - 1]
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        images.append({"page": p, "base64": b64})

    doc.close()
    return images


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


def _merge_extracted(ext1: dict[str, Any], ext2: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    """合并两批抽取结果：优先取非「未找到」的值。"""
    merged: dict[str, Any] = {}
    for f in fields:
        v1 = ext1.get(f)
        v2 = ext2.get(f)
        if v1 not in (None, "", "未找到", []):
            merged[f] = v1
        elif v2 not in (None, "", "未找到", []):
            merged[f] = v2
        else:
            merged[f] = "未找到"
    for k, v in {**ext1, **ext2}.items():
        if k not in merged and v not in (None, "", "未找到"):
            merged[k] = v
    return merged


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

    # 回退：截取第一个 { ... } 块
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

    # 额外的"同义理解"提示：只用于引导模型，不做任何本地硬匹配替换
    synonym_notes: list[str] = []
    if "利润分配预案/利润分配方案" in fields:
        synonym_notes.append(
            '- "利润分配预案/利润分配方案"在报告里可能写作：利润分配预案 / 利润分配方案 / 分红预案 / 现金分红方案 / 利润分配安排等；请以报告原文对应段落为准。'
        )
    if "董事会决议通过的本报告期利润分配预案或公积金转增股本预案" in fields:
        synonym_notes.append(
            '- 若目录/标题使用"预案"而非"方案"（或反之），视为同义；请抽取该标题下的正文内容，不要因标题措辞差异而填"未找到"。'
        )
    if "经本次董事会审议通过的利润分配预案为：" in fields or "年度利润分配方案如下：" in fields:
        synonym_notes.append(
            '- 报告可能出现多个等价引导句（如"经本次董事会审议通过的利润分配预案为：""年度利润分配方案如下："等）；请分别抽取其后对应段落内容。'
        )
    if "归属于上市公司股东的净利润\n（单位：元）" in fields:
        synonym_notes.append(
            '- "归属于上市公司股东的净利润（单位：元）"在表格中可能不写单位、或把指标名称拆成两行。即使拆行，这也是同一个指标。请仔细查看表格结构：如果看到"净利"和"润"分在两行，数值通常在"润"这一行的同一行（同一行的数值列），或者紧邻的下一行开头。'
        )
    if "归属于上市公司股东的净资产\n（单位：元）" in fields:
        synonym_notes.append(
            '- "归属于上市公司股东的净资产（单位：元）"在表格中可能不写单位、或把指标名称拆成两行。常见拆行方式：第一行是"归属于上市公司股东的净资"，第二行是"产"。即使拆行，这也是同一个指标。请仔细查看表格结构。'
        )
    if "总资产/资产总额\n（单位：元）" in fields:
        synonym_notes.append(
            '- "总资产/资产总额（单位：元）"中，"总资产"和"资产总额"是同义词，完全等价。在报告中可能写作"总资产"或"资产总额"，请按任意一种名称查找该指标的本期数值填写。通常在"主要会计数据和财务指标"表格中，可能不写单位；请仔细查看表格，找到该指标对应的数值。'
        )

    fields_json = ",\n".join([f'  "{f}": "..."' for f in fields])

    # 构建提示词
    parts = []
    parts.append("你是一位专业的信息抽取专家。请阅读这份定期报告（多页图片），抽取指定字段。\n\n")
    parts.append("## 需要抽取的字段（必须全部输出）\n")
    parts.append("\n".join([f"- {f}" for f in fields]))
    parts.append("\n\n## 字段要求（来自需求说明，务必遵守）\n")
    parts.append("\n".join(guidance_lines) if guidance_lines else "无\n")
    if synonym_notes:
        parts.append("\n\n## 同义字段理解（重要）\n")
        parts.append("\n".join(synonym_notes))
    parts.append("\n\n## 主要会计数据与财务指标（三值格式，必须遵守）\n")
    parts.append('在「主要会计数据和财务指标」表格中，每类指标需提取**三个值**：\n')
    parts.append('- **本期数据**：对应报告期/本年/本期列（如 2021年、本年）\n')
    parts.append('- **上期数据**：对应上年/上期列（如 2020年、上年）\n')
    parts.append('- **同比变动比例（%）**：对应「本年比上年增减(%)」或「同比变动」等列\n')
    parts.append('字段名中带「-本期数据」「-上期数据」「-同比变动比例（%）」的，请严格按表格列对应填写；若某列不存在或找不到则填「未找到」。\n\n')
    parts.append("## 表格拆行处理（非常重要）\n")
    parts.append('在定期报告的"主要会计数据和财务指标"表格中，由于列宽限制，长指标名称经常被拆成多行。例如：\n')
    parts.append('- "归属于上市公司股东的净利润"可能显示为：第一行"归属于上市公司股东的净利"，第二行"润"\n')
    parts.append('- "归属于上市公司股东的净资产"可能显示为：第一行"归属于上市公司股东的净资"，第二行"产"\n')
    parts.append('- "总资产"可能单独一行，也可能和其他文字在一起\n')
    parts.append("即使拆行，也是同一个指标！请识别这些拆行的指标，并找到对应的数值（通常在同一行或下一行的数值列）。\n\n")
    parts.append("## 输出要求\n")
    parts.append("- 只输出 JSON（不要输出任何解释性文字）\n")
    parts.append("- 字段名必须与上面完全一致\n")
    parts.append('- 若确实找不到，填"未找到"\n')
    parts.append("- 日期统一输出 YYYY-MM-DD（如 2024-03-15）\n")
    parts.append('- "目录"用分号 ; 分隔，精确到 3 级标题\n')
    parts.append('- "语言"只能是：中文 / 英文\n')
    parts.append('- "精排"只能是：是 / 否（不要输出其他值）\n')
    parts.append('- "业绩分类"只能是：亏损 / 扭亏为盈 / 同比增长 / 同比下降（不要输出其他值）\n')
    parts.append('- "资产负债率"输出百分比并带 %，如 74.86%\n')
    parts.append('- "篇幅页码"输出数字（总页数）\n')
    parts.append('- "字数"输出数字（全文字数/字符数均可，保持一致即可）\n\n')
    parts.append("请按以下 JSON 结构输出：\n{\n")
    parts.append(fields_json)
    parts.append("\n}\n")
    return "".join(parts)


def build_recheck_prompt(fields: list[str], current: dict[str, Any], missing: list[str]) -> str:
    """
    二次复核提示词：只针对 missing 字段重新查找，严禁猜测。
    仍要求输出完整 JSON，便于直接替换。
    """
    fields_json = ",\n".join([f'  "{f}": "..."' for f in fields])
    current_json = json.dumps({k: current.get(k) for k in fields}, ensure_ascii=False, indent=2)
    missing_list = "\n".join([f"- {m}" for m in missing])

    return (
        "你是一位严谨的信息抽取专家。现在需要你对同一份定期报告做“二次复核”。\n\n"
        "## 已有抽取结果（可能有缺失）\n"
        f"{current_json}\n\n"
        "## 需要你重点复核并尽量补全的字段（仅限这些字段）\n"
        f"{missing_list}\n\n"
        "## 严格要求（非常重要）\n"
        "- 只能在报告图片中明确看到/读到的信息才可填写\n"
        "- 绝对不要猜测、不要补常识、不要根据行业推断\n"
        "- 如果仍然找不到，就保持“未找到”\n"
        "- 只输出 JSON（不要输出解释）\n"
        "- 字段必须齐全，字段名必须与下方模板一致\n\n"
        "请按以下 JSON 结构输出：\n"
        "{\n"
        + fields_json
        + "\n}\n"
    )


def _call_vl(*, vl: dict[str, Any], content: list[dict[str, Any]], num_pages: int | None = None, endpoint: tuple[str, str, int] | None = None) -> dict[str, Any]:
    """用 api_urls 项内的 url 和 model（若该项有 model），否则用 vl 顶层配置。"""
    n_images = sum(1 for x in content if x.get("type") == "image_url")
    from scripts.vl_utils import get_vl_endpoint
    if endpoint:
        url, model, max_tokens = endpoint
    else:
        url, model, max_tokens = get_vl_endpoint(vl, num_pages=num_pages)
    logger.info("VL 请求: POST %s, model=%s, 共 %d 条 content（其中 %d 张图）, timeout=400s", url, model, len(content), n_images)

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if vl.get("api_key"):
        headers["Authorization"] = f"Bearer {vl['api_key']}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": max_tokens,
        "stream": False,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=400)
    if not resp.ok:
        err_preview = (resp.text or "")[:800]
        logger.error("VL 请求失败: %s %s", resp.status_code, resp.reason)
        logger.error("响应内容: %s", err_preview)
    resp.raise_for_status()
    logger.info("VL 响应: 成功, 状态码 %s", resp.status_code)
    return resp.json()


def _parse_float_maybe(s: Any) -> Optional[float]:
    """
    从字符串中尽量解析出数值（支持千分位、括号负号、中文单位忽略）。
    """
    if s is None:
        return None
    txt = str(s).strip()
    if not txt or txt == "未找到":
        return None

    # 处理括号表示负数，例如 (123.45)
    neg = False
    m = re.fullmatch(r"\(\s*([^)]+?)\s*\)", txt)
    if m:
        neg = True
        txt = m.group(1).strip()

    # 清理常见干扰符号/单位
    txt = txt.replace(",", "").replace("，", "").replace("元", "").replace("万元", "").replace("亿", "")
    txt = txt.replace("％", "%")

    # 抓取第一个数字（允许负号与小数）
    m2 = re.search(r"-?\d+(?:\.\d+)?", txt)
    if not m2:
        return None
    val = float(m2.group(0))
    return -val if neg else val


def _normalize_field_name(name: str) -> str:
    """
    规范化字段名：去除换行、单位、多余空白，便于模糊匹配。
    """
    s = name.replace("\n", "").replace("（单位：元）", "").replace("（单位:元）", "")
    s = s.replace("（", "(").replace("）", ")").replace("／", "/")
    s = re.sub(r"\s+", "", s)
    return s.lower()


def _fuzzy_match_field(
    target_field: str,
    extracted: dict[str, Any],
) -> Optional[Any]:
    """
    从 extracted 中模糊查找与 target_field 语义等价的字段值。
    支持三值字段（-本期数据/-上期数据/-同比变动比例（%））：按后缀匹配优先。
    返回找到的值，或 None。
    """
    suffix = None
    base_field = target_field
    for s in ("-本期数据", "-上期数据", "-同比变动比例（%）"):
        if target_field.endswith(s):
            suffix = s
            base_field = target_field[: -len(s)]
            break
    target_norm = _normalize_field_name(base_field)

    # 定义语义等价的字段名变体（均为“指标基名”，不含三值后缀）
    field_variants: dict[str, list[str]] = {
        "归属于上市公司股东的净利润": [
            "归属于上市公司股东的净利润",
            "归属于上市公司股东净利润",
            "归母净利润",
            "归属净利润",
            "净利润(归属于上市公司股东)",
            "净利润（归属于上市公司股东）",
        ],
        "总资产/资产总额": [
            "总资产",
            "资产总额",
            "资产总计",
            "总资产/资产总额",
        ],
        "归属于上市公司股东的净资产": [
            "归属于上市公司股东的净资产",
            "归属于上市公司股东净资产",
            "归母净资产",
            "归属净资产",
            "净资产(归属于上市公司股东)",
            "净资产（归属于上市公司股东）",
        ],
    }
    
    # 找到当前目标字段对应的变体列表
    variants_to_check: list[str] = []
    for base_name, variants in field_variants.items():
        if base_name in base_field or target_norm in _normalize_field_name(base_name):
            variants_to_check = variants
            break

    if not variants_to_check:
        return None

    # 在 extracted 中查找匹配的字段（三值字段时优先同后缀）
    for key, val in extracted.items():
        if val in (None, "", "未找到", []):
            continue
        key_base = key
        key_suffix = None
        for s in ("-本期数据", "-上期数据", "-同比变动比例（%）"):
            if key.endswith(s):
                key_suffix = s
                key_base = key[: -len(s)]
                break
        key_norm = _normalize_field_name(key_base)
        for variant in variants_to_check:
            variant_norm = _normalize_field_name(variant)
            if variant_norm in key_norm or key_norm in variant_norm:
                if suffix and key_suffix and suffix != key_suffix:
                    continue  # 三值字段：只复用同后缀
                return val
    return None


def _semantic_field_fallback(
    extracted: dict[str, Any],
    fields: list[str],
) -> dict[str, Any]:
    """
    对"未找到"的字段，尝试用语义匹配和推算逻辑补全。
    """
    result = dict(extracted)
    
    # 1. 模糊匹配：从 extracted 中找类似字段名
    for f in fields:
        if result.get(f) in (None, "", "未找到"):
            matched_val = _fuzzy_match_field(f, extracted)
            if matched_val:
                result[f] = matched_val
    
    # 2. 推算逻辑：利用等价关系（三值字段只补「本期数据」）
    # 总资产-本期 = 负债和所有者权益总计（本期数据）
    total_asset_current = "总资产/资产总额\n（单位：元）-本期数据"
    if total_asset_current in result and result.get(total_asset_current) in (None, "", "未找到"):
        equity_field = "负债和所有者权益（或股东权益）总计（本期数据）"
        if equity_field in extracted and extracted.get(equity_field) not in (None, "", "未找到"):
            result[total_asset_current] = extracted[equity_field]

    # 3. 归属于上市公司股东的净利润-本期：尝试从「净利润（本期数据）」等获取
    net_profit_current = "归属于上市公司股东的净利润\n（单位：元）-本期数据"
    if net_profit_current in result and result.get(net_profit_current) in (None, "", "未找到"):
        for key, val in extracted.items():
            if val in (None, "", "未找到"):
                continue
            key_lower = key.lower().replace(" ", "").replace("\n", "")
            if "归属" in key_lower and "净利润" in key_lower and "扣除" not in key_lower:
                if key.endswith("-本期数据") or key == "净利润（本期数据）" or "本期" in key:
                    result[net_profit_current] = val
                    break
        else:
            if "净利润（本期数据）" in extracted and extracted.get("净利润（本期数据）") not in (None, "", "未找到"):
                result[net_profit_current] = extracted["净利润（本期数据）"]

    return result


def infer_performance_category(
    *,
    current_net_profit: Any,
    previous_net_profit: Any,
) -> Optional[str]:
    """
    根据 requirement 的确定性规则，推断业绩分类。
    仅在可解析出数值时计算；否则返回 None。
    """
    cur = _parse_float_maybe(current_net_profit)
    prev = _parse_float_maybe(previous_net_profit)
    if cur is None:
        return None

    if cur < 0:
        return "亏损"
    if prev is None:
        return None
    if prev < 0 and cur > 0:
        return "扭亏为盈"
    if prev > 0 and cur > 0 and cur > prev:
        return "同比增长"
    if prev > 0 and cur > 0 and cur < prev:
        return "同比下降"
    return None


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

    from scripts.vl_utils import get_vl_endpoint, PAGE_THRESHOLD_8B_BATCH

    endpoint = get_vl_endpoint(vl, num_pages=len(images))
    url, model, max_tokens = endpoint
    n_imgs = len(images)

    if model == "qwen3-vl-8b" and n_imgs > PAGE_THRESHOLD_8B_BATCH:
        # 8b 每批最多 20 页，多批合并
        batch_size = PAGE_THRESHOLD_8B_BATCH
        all_extracted: list[dict[str, Any]] = []
        for i in range(0, n_imgs, batch_size):
            batch_imgs = images[i : i + batch_size]
            content_batch: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for img in batch_imgs:
                content_batch.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img['base64']}"}})
            batch_num = len(all_extracted) + 1
            logger.info("[%s] 8b 第 %d 批: %d 页", pdf_path.name, batch_num, len(batch_imgs))
            data = _call_vl(vl=vl, content=content_batch, endpoint=endpoint)
            ext = _loads_json_relaxed(data["choices"][0]["message"]["content"])
            all_extracted.append(ext)
        extracted = all_extracted[0]
        for ext in all_extracted[1:]:
            extracted = _merge_extracted(extracted, ext, fields)
        logger.info("[%s] VL %d 批合并完成", pdf_path.name, len(all_extracted))
    else:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in images:
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img['base64']}"}}
            )
        logger.info("[%s] 正在调用 VL...", pdf_path.name)
        data = _call_vl(vl=vl, content=content, endpoint=endpoint)
        raw = data["choices"][0]["message"]["content"]
        extracted = _loads_json_relaxed(raw)
        logger.info("[%s] VL 返回成功，已解析 JSON", pdf_path.name)

    # 语义匹配后处理：尝试从类似字段名或等价关系补全
    extracted = _semantic_field_fallback(extracted, fields)

    # 补齐缺失字段
    for f in fields:
        if f not in extracted or extracted[f] in (None, "", []):
            extracted[f] = "未找到"

    logger.info("[%s] 正在写成本地 record（页数/字数/业绩分类等）", pdf_path.name)
    record: dict[str, Any] = {"filename": pdf_path.name, **{k: extracted.get(k) for k in fields}}

    # 页数：以真实 PDF 页数覆盖
    if "篇幅页码" in record:
        record["篇幅页码"] = str(total_pages)

    # 字数：本地统计（不依赖模型输出）
    if "字数" in record:
        cnt = pdf_text_char_count(pdf_path)
        record["字数"] = str(cnt) if cnt > 0 else (record.get("字数") or "未找到")

    # 公司简称：确定性裁剪
    if "公司简称" in record and (record["公司简称"] in (None, "", "未找到")):
        short = infer_company_short_name(str(record.get("公司全称") or ""))
        if short:
            record["公司简称"] = short

    # 精排：按需求“图片数量>20”为是；这里用真实 PDF 页数做判定
    if "精排" in record:
        record["精排"] = "是" if total_pages > 20 else "否"

    # 业绩分类：按需求规则确定性推断（仅在可解析数值时覆盖）
    if "业绩分类" in record:
        inferred = infer_performance_category(
            current_net_profit=record.get("净利润（本期数据）"),
            previous_net_profit=record.get("净利润（上期数据）"),
        )
        if inferred:
            record["业绩分类"] = inferred

    # 资产负债率：优先依据负债合计和负债与所有者权益计算（反馈要求）
    if "资产负债率" in record:
        val_str = (record.get("资产负债率") or "").strip()
        if val_str and val_str != "未找到":
            if "%" not in val_str:
                num = _parse_float_maybe(val_str)
                if num is not None:
                    if 0 <= abs(num) <= 1:
                        record["资产负债率"] = f"{num * 100:.2f}%"
                    else:
                        record["资产负债率"] = f"{num:.2f}%"
        else:
            # 反馈：需依据负债合计、负债与所有者权益算出资产负债率
            liability = _parse_float_maybe(record.get("负债合计（本期数据）"))
            total_equity = _parse_float_maybe(record.get("负债和所有者权益（或股东权益）总计（本期数据）"))
            if liability is not None and total_equity is not None and total_equity > 0:
                rate = liability / total_equity * 100
                record["资产负债率"] = f"{rate:.2f}%"
                logger.info("[%s] 已依据负债合计/负债与所有者权益计算资产负债率: %s", pdf_path.name, record["资产负债率"])

    # 利润分配预案：依据三者内容拼凑输出（反馈要求）
    profit_dist_field = "利润分配预案/利润分配方案"
    if profit_dist_field in record and (record.get(profit_dist_field) in (None, "", "未找到")):
        parts: list[str] = []
        for f in (
            "董事会决议通过的本报告期利润分配预案或公积金转增股本预案",
            "经本次董事会审议通过的利润分配预案为：",
            "年度利润分配方案如下：",
        ):
            v = (record.get(f) or "").strip()
            if v and v != "未找到":
                parts.append(v)
        if parts:
            record[profit_dist_field] = "\n".join(parts)
            logger.info("[%s] 已依据三字段拼凑利润分配预案", pdf_path.name)

    logger.info("[%s] 本份抽取完成", pdf_path.name)
    return record


def _extract_date_from_path(p: Path) -> tuple[int, int]:
    """
    从路径中提取 YYYY-MM 格式的日期，用于按时间倒序排序。
    返回 (year, month)，若未找到则返回 (0, 0)。
    """
    match = re.search(r"(\d{4})-(\d{2})", str(p))
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return (0, 0)


def iter_pdfs(input_path: Path, *, reverse_date: bool = False) -> Iterable[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() == ".pdf":
            yield input_path
        return
    pdfs = [p for p in input_path.glob("**/*") if p.is_file() and p.suffix.lower() == ".pdf"]
    if reverse_date:
        pdfs.sort(key=lambda p: str(p))  # 同月内按路径升序
        pdfs.sort(key=lambda p: _extract_date_from_path(p), reverse=True)  # 时间倒序
    else:
        pdfs.sort()
    for p in pdfs:
        yield p


def load_done_filenames(output_path: Path) -> set[str]:
    """
    从已有 JSONL 中读取已完成的 filename 集合，用于断点续跑时跳过。
    解析失败或非 JSON 行则忽略该行。
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
    parser = argparse.ArgumentParser(description="定期报告字段抽取（VL大模型）")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径（需要包含 vl 配置）")
    parser.add_argument("--requirement", default="data/requirement/requirement_1.json", help="requirement_1.json 路径")
    parser.add_argument("--input", default=None, help="定期报告 PDF 目录（默认 kg_data/data，其下为 YYYY-MM/定期报告/）")
    parser.add_argument("--output", default="result/periodic_reports_extracted_vl.jsonl", help="输出 jsonl 路径")
    parser.add_argument("--limit", type=int, default=5, help="最多处理多少份PDF（0=不限制；默认5用于生成样例）")
    parser.add_argument("--skip", type=int, default=0, help="跳过前 N 份 PDF，与 --limit 配合可只处理中间或最后若干份")
    parser.add_argument("--dpi", type=int, default=150, help="PDF转图片的dpi")
    parser.add_argument("--max-pages", type=int, default=50, help="单份PDF最多处理多少页（默认50）")
    parser.add_argument("--append", action="store_true", help="追加写入 output（默认覆盖）")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续跑：读取 output 中已有 filename，只处理未完成的 PDF，并追加写入",
    )
    parser.add_argument("--print-sample", action="store_true", help="打印第一条结果到stdout")
    parser.add_argument(
        "--reverse-date",
        action="store_true",
        help="按时间倒序处理（从路径中解析 YYYY-MM，新的月份优先）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="并行处理的 PDF 数（0=用 config 中 vl.parallel_workers，1=串行）",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="只处理该年份（从路径 YYYY-MM 解析，如 2025 表示只处理 25 年）",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    requirement_path = Path(args.requirement)
    input_path = Path(args.input if args.input else _default_input_dir())
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("开始抽取定期报告")
    if not input_path.exists():
        logger.error("输入目录不存在: %s", input_path)
        return 1
    config = load_config(config_path)
    fields, comments = load_fields_and_comments(requirement_path)
    logger.info("已加载 config 与 requirement，共 %d 个字段", len(fields))
    logger.info("输入目录: %s", input_path)

    pdfs = list(iter_pdfs(input_path, reverse_date=args.reverse_date))
    if args.year is not None:
        pdfs = [p for p in pdfs if _extract_date_from_path(p)[0] == args.year]
        logger.info("只处理 %d 年: 共 %d 份 PDF", args.year, len(pdfs))
    if args.reverse_date:
        logger.info("已按时间倒序排列（新月份优先）")
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

    workers = args.workers
    if workers <= 0:
        workers = config.get("vl", {}).get("parallel_workers", 1)
    workers = max(1, workers)
    if workers > 1:
        logger.info("并行数: %d 个 PDF 同时处理", workers)

    # 打开输出文件（追加或覆盖模式）
    mode = "a" if (args.append or args.resume) else "w"
    output_file = output_path.open(mode, encoding="utf-8")
    file_lock = threading.Lock()
    total = len(pdfs)
    success_count = 0

    def process_one(p: Path):
        try:
            rec = call_vl_extract(
                pdf_path=p,
                fields=fields,
                comments=comments,
                config=config,
                dpi=args.dpi,
                max_pages=args.max_pages,
            )
            return (p, rec, None)
        except Exception as e:
            return (p, None, e)

    try:
        if workers <= 1:
            for i, p in enumerate(pdfs, 1):
                logger.info("===== 第 %d/%d 个: %s =====", i, total, p.name)
                _, rec, err = process_one(p)
                if err:
                    logger.exception("[%s] 本份抽取失败，跳过: %s", p.name, err)
                else:
                    output_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    output_file.flush()
                    success_count += 1
                    logger.info("[%s] 已写入 %s", p.name, output_path)
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                fut = {ex.submit(process_one, p): p for p in pdfs}
                for i, f in enumerate(as_completed(fut), 1):
                    p = fut[f]
                    try:
                        _, rec, err = f.result()
                        if err:
                            logger.exception("[%s] 本份抽取失败，跳过: %s", p.name, err)
                        else:
                            with file_lock:
                                output_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                output_file.flush()
                            success_count += 1
                            logger.info("[%d/%d] [%s] 已写入 %s", i, total, p.name, output_path)
                    except Exception as e:
                        logger.exception("[%s] 本份失败: %s", p.name, e)
    finally:
        output_file.close()

    logger.info("本轮完成: 成功 %d 份，共待处理 %d 份", success_count, total)

    # 如果需要打印样例，重新读取第一条
    if args.print_sample:
        with output_path.open("r", encoding="utf-8") as f:
            first_line = f.readline()
            if first_line:
                print(json.dumps(json.loads(first_line), ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

