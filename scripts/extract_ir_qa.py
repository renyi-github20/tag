#!/usr/bin/env python3
"""

数据格式与处理方式：
- .txt 文件：首行为标题，随后为 JSON（含 title、qaList 等），用 config.llm 纯文本 LLM 抽取。
- .pdf 文件：与 ESG 相同，将 PDF 转成多页图片（JPEG base64），用 config.vl 视觉大模型看图抽取，速度与稳定性更好。
"""

from __future__ import annotations

import argparse
from datetime import datetime as _dt
import base64
import io
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Iterable

import fitz  # PyMuPDF
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# 活动类别枚举（与 requirement 一致）
ACTIVITY_TYPE_ENUM = "业绩说明会、调研接待、分析师会议、路演活动、媒体采访、新闻发布会、其他"
ACTIVITY_KEYWORDS = [
    ("业绩说明会", ["业绩说明会", "业绩会"]),
    ("调研接待", ["调研接待", "调研"]),
    ("分析师会议", ["分析师会议"]),
    ("路演活动", ["路演", "路演问答", "路演活动"]),
    ("媒体采访", ["媒体采访"]),
    ("新闻发布会", ["新闻发布会", "发布会"]),
]


def load_config(config_path: Path) -> dict[str, Any]:
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def get_llm_config(config: dict[str, Any], profile: str = "default") -> dict[str, Any]:
    """从 config 取 llm 配置，profile 如 default / chat。"""
    llm = config.get("llm") or {}
    cfg = llm.get(profile) or llm.get("default") or {}
    if not cfg.get("api_base") or not cfg.get("model"):
        raise ValueError("config 中 llm.%s 需提供 api_base 与 model" % profile)
    return cfg


def get_vl_config(config: dict[str, Any]) -> dict[str, Any]:
    """从 config 取 vl 配置（与 ESG 相同，用于 PDF 看图抽取）。"""
    from scripts.vl_utils import has_vl_config
    vl = config.get("vl") or {}
    if not has_vl_config(vl) or not vl.get("model"):
        raise ValueError("config 中 vl 需提供 api_url/api_urls 与 model")
    return vl


def _png_to_jpeg(png_bytes: bytes, *, quality: int = 80) -> bytes:
    """将 PNG 转为 JPEG，缩小体积（与 ESG 一致）。"""
    from PIL import Image

    img = Image.open(io.BytesIO(png_bytes))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def pdf_to_images_base64(
    pdf_path: Path,
    *,
    dpi: int = 150,
    max_pages: int = 30,
    use_jpeg: bool = True,
    jpeg_quality: int = 80,
    page_start: int = 0,
) -> tuple[int, list[dict[str, Any]]]:
    """
    将 PDF 指定页范围转成 base64 图片（与 ESG 一致）。
    返回 (总页数, 本批图片列表 [{page, base64, mime}, ...])。
    """
    if not pdf_path.exists() or not pdf_path.is_file():
        logger.warning("[%s] PDF 文件不存在或不是文件，跳过: %s", pdf_path.name, pdf_path)
        return (0, [])
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    page_end = min(page_start + max_pages, total_pages)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    images: list[dict[str, Any]] = []
    for page_idx in range(page_start, page_end):
        page = doc[page_idx]
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        if use_jpeg:
            img_bytes = _png_to_jpeg(img_bytes, quality=jpeg_quality)
            mime = "image/jpeg"
        else:
            mime = "image/png"
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        images.append({"page": page_idx + 1, "base64": b64, "mime": mime})

    doc.close()
    logger.info(
        "[%s] PDF 转图: 第 %d-%d 页（共 %d 页）, %d 张",
        pdf_path.name, page_start + 1, page_end, total_pages, len(images),
    )
    return total_pages, images


def _build_session(*, verify_ssl: bool = True) -> requests.Session:
    """带重试的 Session（与 ESG 一致）。"""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=2,
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
    """调用 VL 接口（与 ESG 一致）。"""
    from scripts.vl_utils import get_vl_url
    url = get_vl_url(vl)
    n_images = sum(1 for x in content if x.get("type") == "image_url")
    logger.info("VL 请求: POST %s, 共 %d 条 content（其中 %d 张图）, timeout=600s", url, len(content), n_images)

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if vl.get("api_key"):
        headers["Authorization"] = f"Bearer {vl['api_key']}"

    payload = {
        "model": vl["model"],
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": int(vl.get("max_tokens", 8192)),
        "stream": False,
    }

    verify_ssl = vl.get("verify_ssl", True)
    max_attempts = 4
    for attempt in range(1, max_attempts + 1):
        try:
            session = _build_session(verify_ssl=verify_ssl)
            resp = session.post(url, headers=headers, json=payload, timeout=600)
            resp.raise_for_status()
            logger.info("VL 响应: 成功, 状态码 %s", resp.status_code)
            return resp.json()
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if attempt < max_attempts:
                wait = 5 * attempt
                logger.warning("VL 请求失败 (第 %d/%d 次): %s — %s 秒后重试...", attempt, max_attempts, type(e).__name__, wait)
                time.sleep(wait)
            else:
                logger.error("VL 请求连续 %d 次失败，放弃", max_attempts)
                raise
    return {}


def load_fields_and_comments(requirement_path: Path) -> tuple[list[str], dict[str, str]]:
    """从 requirement_1.json 读取「投关问答」的字段列表与 comment。"""
    data = json.loads(requirement_path.read_text(encoding="utf-8"))
    for section in data.get("sections", []):
        if section.get("name") != "投关问答":
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
            raise ValueError("requirement_1.json 的 投关问答 fields 为空")
        return fields, comments

    raise ValueError("requirement_1.json 中未找到 name=投关问答 的 section")


def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if "```json" in s:
        return s.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in s:
        return s.split("```", 1)[1].split("```", 1)[0].strip()
    return s


def _parse_llm_json_array(raw: str) -> list[dict[str, Any]]:
    """解析 LLM 输出为 JSON 数组；若为单个对象则包装成单元素数组。支持前后有说明文字。"""
    if not (raw or raw.strip()):
        return []
    raw = _strip_code_fences(raw)
    raw = raw.strip()
    # 优先找 [...] 完整数组
    start = raw.find("[")
    if start >= 0:
        depth = 0
        for i in range(start, len(raw)):
            if raw[i] == "[":
                depth += 1
            elif raw[i] == "]":
                depth -= 1
                if depth == 0:
                    try:
                        arr = json.loads(raw[start : i + 1])
                        if isinstance(arr, list):
                            return [x for x in arr if isinstance(x, dict)]
                        if isinstance(arr, dict):
                            return [arr]
                    except json.JSONDecodeError as e:
                        logger.debug("解析 JSON 数组失败: %s", e)
                    break
    # 退化为找单个 {...}
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(raw[start : end + 1])
            if isinstance(obj, dict):
                return [obj]
        except json.JSONDecodeError:
            pass
    return []


def parse_one_txt(txt_path: Path) -> tuple[str, str, list[dict[str, Any]]]:
    """
    解析单份 .txt：第一行为标题，后续为 JSON。
    返回 (title, raw_source, qa_list)。
    """
    text = txt_path.read_text(encoding="utf-8")
    first_line = (text.split("\n")[0] or "").strip()
    json_start = text.find("{")
    if json_start < 0:
        return first_line, "", []

    try:
        obj = json.loads(text[json_start:])
    except json.JSONDecodeError:
        return first_line, "", []

    qa_list = obj.get("qaList") or obj.get("qa_list") or []
    if not isinstance(qa_list, list):
        qa_list = []

    source = obj.get("source") or obj.get("subSourceDesc") or ""
    title = (obj.get("title") or first_line or "").strip()
    if not title:
        title = first_line
    return title, source, qa_list


def parse_one_pdf(pdf_path: Path) -> tuple[str, str, int, int]:
    """
    解析单份 PDF：提取全文文本、标题（从文件名推断）、页数、字数。
    返回 (full_text, title, total_pages, char_count)。仅 .txt 分支或 fallback 时用。
    """
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    text_parts: list[str] = []
    for page in doc:
        t = (page.get_text() or "").strip()
        if t:
            text_parts.append(t)
    doc.close()
    full_text = "\n\n".join(text_parts)
    char_count = len(re.sub(r"\s+", "", full_text))
    title = pdf_path.stem
    title = re.sub(r"^\d+_\d{4}-\d{2}-\d{2}_", "", title)
    return full_text, title, total_pages, char_count


def pdf_text_char_count(pdf_path: Path) -> int:
    """统计 PDF 字数（去空白字符数），用于 VL 抽取后的篇幅/字数填充。"""
    with fitz.open(str(pdf_path)) as doc:
        text_parts = [(page.get_text() or "").strip() for page in doc if (page.get_text() or "").strip()]
    full = "\n".join(text_parts)
    return len(re.sub(r"\s+", "", full))


def build_qa_input_text(
    title: str, source: str, qa_list: list[dict[str, Any]], max_qa: int = 0
) -> str:
    """将标题、来源与问答列表整理成一段可读文本，供 LLM 抽取。max_qa=0 表示不限制条数。"""
    use_list = qa_list[:max_qa] if max_qa and max_qa > 0 else qa_list
    lines = [f"文件名称/标题：{title}", f"来源：{source or '未注明'}", ""]
    for i, qa in enumerate(use_list, 1):
        asker = (qa.get("asker") or "").strip()
        question = (qa.get("question") or "").strip()
        responder = (qa.get("responder") or "").strip()
        answer = (qa.get("answer") or "").strip()
        answer_time = qa.get("answerTime") or ""
        lines.append(f"--- 问答 {i} ---")
        if asker:
            lines.append(f"提问人/机构：{asker}")
        if answer_time:
            lines.append(f"回复时间：{answer_time}")
        lines.append(f"提问：{question or '（无）'}")
        if responder:
            lines.append(f"回复人：{responder}")
        lines.append(f"回复：{answer or '（无）'}")
        lines.append("")
    return "\n".join(lines).strip()


def build_prompt(fields: list[str], comments: dict[str, str], input_text: str) -> str:
    """构建投关问答抽取的 LLM 提示词。"""
    guidance = []
    for f in fields:
        c = comments.get(f)
        if c:
            guidance.append(f"- {f}：{c}")
    # 构建字段示例，提问内容和回复内容可以是字符串或数组
    fields_json_lines = []
    for f in fields:
        if f in ("提问内容", "回复内容"):
            fields_json_lines.append(f'  "{f}": "..." 或 ["...", "..."]')
        else:
            fields_json_lines.append(f'  "{f}": "..."')
    fields_json = ",\n".join(fields_json_lines)
    return (
        "你是一位专业的信息抽取专家。请根据下面给出的「投关问答」文本，按条抽取每条问答的字段。\n\n"
        "## 需要抽取的字段（每条问答都需输出）\n"
        + "\n".join([f"- {f}" for f in fields])
        + "\n\n## 字段说明\n"
        + ("\n".join(guidance) if guidance else "无")
        + "\n\n## 输出要求\n"
        "- 输出一个 JSON 数组，数组的每个元素对应一条问答（一条提问+回复），包含上述全部字段。\n"
        "- **重要：跳过开场白和结束语**（如「各位嘉宾、各位投资者...欢迎大家提问」「本次活动到此结束，感谢参与」等无实质问答的条目），只抽取有实际提问和回复的问答对。\n"
        "- 若某字段无法从文本推断，填「未找到」。\n"
        '- 「活动类别」只能是以下之一：' + ACTIVITY_TYPE_ENUM + "\n"
        "- 日期格式统一为 YYYY-MM-DD。\n"
        "- 「落款日期」仅填写文中明确写出的日期；若文中未出现落款日期则填「未找到」，不要推测或自动生成。\n"
        "- 「关联公告」指与本场活动相关的公告（如年报、季报、业绩说明会公告等），不是本活动记录本身。若文档中明确提到相关公告则抽取，否则填「未找到」。\n"
        "- 「字数」填该条问答（概括后的提问+回复）的字符数。\n\n"
        "## 调研机构（即参会机构）\n"
        "- 「调研机构」即**参会机构/参与机构**。请从文档中抽取参与本次活动的机构名单（如证券公司、基金、保险、私募等）。\n"
        "- 文档中常见表述为「参会机构」「参与单位」「参与机构」「调研机构」等，多机构用顿号、逗号或分号分隔。\n"
        "- 整场活动通常只有一份参会机构名单，各条问答可填相同值；若文档中未列出则填「未找到」。\n\n"
        "## 提问内容与回复内容：必须概括（重要）\n"
        "- 「提问内容」：**必须概括**，保留核心问题要点（问什么、关注点），去掉客套话、重复。\n"
        "  - 可以是字符串（单个问题）或**字符串数组**（多个要点）；开场白/结束语用数组做全文概括，如 [\"活动名称与时间\", \"主办方与平台\", \"欢迎提问\"]。\n"
        "  - 例如：原文「公司今年乘用车座椅业务订单情况怎么样？后续有什么业务拓展计划？」可概括为字符串「乘用车座椅业务订单情况及后续拓展计划」，或数组 [\"乘用车座椅业务订单情况\", \"后续业务拓展计划\"]。\n"
        "- 「回复内容」：**必须概括**，保留核心信息与结论（数据、结论、态度），**必须完全去掉**以下套话：\n"
        "  - 「尊敬的投资者：您好！」\n"
        "  - 「感谢您的提问」\n"
        "  - 「十分感谢您的关注！」\n"
        "  - 「如还有其他问题，欢迎致电...」\n"
        "  - 「具体内容详见...报告」\n"
        "  - 其他客套话和重复表述\n"
        "  - 可以是字符串（单个回复）或字符串数组（多个回复要点）。\n"
        "  - 例如：原文长段回复可概括为字符串「股价波动属正常市场行为，公司专注主业，对未来发展有信心，将继续做好主营业务以回报投资者」，或数组[\"股价波动属正常市场行为\", \"公司专注主业，对未来发展有信心\", \"将继续做好主营业务以回报投资者\"]。\n"
        "- **严禁照抄原文长段**，必须精炼概括。字符串形式不超过150字；数组形式每个元素不超过50字，最多5个元素。\n"
        "- 仅当有实际问答但某侧确实缺失时，对应字段填「未找到」；开场白/结束语用全文概括的 list 表示，不要填「未找到」。\n\n"
        "## 投关问答原文\n"
        "```\n"
        + input_text
        + "\n```\n\n"
        "请直接输出 JSON 数组，不要其他解释。格式示例：\n[{\n"
        + fields_json
        + "\n}, ...]\n"
    )


def build_prompt_vl(fields: list[str], comments: dict[str, str]) -> str:
    """构建投关问答 VL 抽取提示词（看图，无原文嵌入；与 ESG 风格一致）。"""
    guidance = []
    for f in fields:
        c = comments.get(f)
        if c:
            guidance.append(f"- {f}：{c}")
    fields_json_lines = []
    for f in fields:
        if f in ("提问内容", "回复内容"):
            fields_json_lines.append(f'  "{f}": "..." 或 ["...", "..."]')
        else:
            fields_json_lines.append(f'  "{f}": "..."')
    fields_json = ",\n".join(fields_json_lines)

    return (
        "你是一位专业的信息抽取专家。以下是一份「投关问答」文档的多页图片，请逐页阅读，按条抽取每条问答的字段。\n\n"
        "## 需要抽取的字段（每条问答都需输出）\n"
        + "\n".join([f"- {f}" for f in fields])
        + "\n\n## 字段说明\n"
        + ("\n".join(guidance) if guidance else "无")
        + "\n\n## 输出要求\n"
        "- 输出一个 **JSON 数组**，数组的每个元素对应一条问答（一条提问+回复），包含上述全部字段。\n"
        "- 跳过开场白和结束语，只抽取有实际提问和回复的问答对。\n"
        "- 若某字段无法从文档推断，填「未找到」。\n"
        '- 「活动类别」只能是：' + ACTIVITY_TYPE_ENUM + "\n"
        "- 日期格式 YYYY-MM-DD；「关联公告」指与本场活动相关的公告（如年报、季报、业绩说明会公告），不是本活动记录本身。若文档中明确提到则抽取，否则填「未找到」。\n"
        "## 调研机构（即参会机构）\n"
        "- 「调研机构」即**参会机构/参与机构**。请从文档中抽取参与本次活动的机构名单（如证券公司、基金、保险、私募等）。\n"
        "- 文档中常见表述为「参会机构」「参与单位」「参与机构」「调研机构」等，多机构用顿号、逗号或分号分隔；整场活动通常只有一份名单，各条可填相同值；未列出则填「未找到」。\n\n"
        "- 「提问内容」「回复内容」必须概括，可字符串或字符串数组；严禁照抄长段。\n\n"
        "请直接输出 JSON 数组，不要其他解释。格式示例：\n[{\n"
        + fields_json
        + "\n}, ...]\n"
    )


def call_llm(llm_config: dict[str, Any], prompt: str, *, debug_path: Path | None = None) -> str:
    """调用 config 中的 LLM（OpenAI 兼容接口），返回 content。"""
    api_base = (llm_config.get("api_base") or "").rstrip("/")
    # 如果 api_base 已经包含 /chat/completions，直接使用
    if "/chat/completions" in api_base:
        url = api_base
    else:
        # 火山引擎 api_base 可能是 .../api/v3/responses，去掉尾部多余路径段
        # 保留到 /v1 或 /v3，再拼 /chat/completions
        import re as _re
        m = _re.search(r"(/v\d+)", api_base)
        if m:
            url = api_base[: m.end()] + "/chat/completions"
        else:
            url = api_base + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if llm_config.get("api_key"):
        headers["Authorization"] = "Bearer " + str(llm_config["api_key"])
    payload = {
        "model": llm_config["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": int(llm_config.get("max_tokens", 8192)),
        "stream": False,
    }
    logger.info("LLM 请求: POST %s, model=%s, prompt 约 %d 字", url, llm_config.get("model"), len(prompt))
    try:
        timeout_sec = int(llm_config.get("timeout", 120))
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        logger.error("LLM 请求失败: %s", e)
        if hasattr(e, "response") and e.response is not None and e.response.text:
            logger.error("响应内容: %s", e.response.text[:500])
        raise
    except json.JSONDecodeError as e:
        logger.error("LLM 响应非 JSON: %s", e)
        if getattr(resp, "text", None):
            logger.error("原始内容前 500 字: %s", resp.text[:500])
        raise

    raw = ""
    if data.get("choices"):
        msg = (data["choices"] or [{}])[0].get("message") or {}
        raw = msg.get("content") or msg.get("text") or ""
    if not raw and isinstance(data.get("output"), dict):
        raw = data["output"].get("text") or data["output"].get("content") or ""
    if not raw and data.get("result"):
        raw = data["result"] if isinstance(data["result"], str) else ""
    if not raw:
        logger.warning("LLM 响应中未找到 content，顶层键: %s", list(data.keys()))
        if debug_path:
            debug_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    if debug_path and raw:
        debug_path.write_text(raw, encoding="utf-8")
    logger.info("LLM 响应: 成功, 约 %d 字", len(raw))
    return raw


def text_char_count(s: str) -> int:
    """字数：去掉空白后的字符数。"""
    return len(re.sub(r"\s+", "", s or ""))


def _clean_reply_content(val: str | list[str]) -> str | list[str]:
    """清理回复内容中的套话，进行二次概括。支持字符串或字符串数组。"""
    if isinstance(val, list):
        # 列表形式：清理每个元素
        cleaned = []
        for item in val:
            if isinstance(item, str) and item.strip():
                cleaned_item = _clean_reply_content(item)
                if cleaned_item and cleaned_item != "未找到":
                    cleaned.append(cleaned_item)
        return cleaned if cleaned else "未找到"
    
    # 字符串形式
    if not val or val == "未找到":
        return val
    text = str(val)
    # 移除常见套话开头
    text = re.sub(r"^尊敬的投资者[：:]\s*", "", text)
    text = re.sub(r"^您好[！!]\s*", "", text)
    text = re.sub(r"感谢您的提问[。，,]\s*", "", text)
    # 移除常见套话结尾
    text = re.sub(r"十分感谢您的关注[！!]\s*$", "", text)
    text = re.sub(r"感谢您的关注[！!]\s*$", "", text)
    text = re.sub(r"如还有其他问题[，,]欢迎致电[^。]+[。]\s*$", "", text)
    text = re.sub(r"具体内容详见[^。]+[。]\s*$", "", text)
    text = re.sub(r"欢迎致电[^。]+[。]\s*$", "", text)
    # 移除重复的客套表述
    text = re.sub(r"公司[^。]*[，,]公司[^。]*[，,]", lambda m: m.group(0).split("，")[0] + "，", text)
    text = text.strip()
    # 如果清理后仍然很长（>150字），尝试提取关键信息
    if len(text) > 150:
        # 提取数字、关键结论等
        sentences = re.split(r"[。！!；;]", text)
        key_sentences = []
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            # 保留包含数字、结论性词汇的句子
            if re.search(r"\d+|[预计|截至|完成|达到|增长|下降|提升|改善]", s):
                key_sentences.append(s)
            elif len(key_sentences) < 2:  # 至少保留前两句
                key_sentences.append(s)
        if key_sentences:
            text = "；".join(key_sentences[:3])  # 最多3句
    return text.strip() or "未找到"


def _infer_activity_type(title: str, source: str) -> str:
    """根据标题或来源推断活动类别。"""
    raw = f"{title or ''} {source or ''}"
    for label, keywords in ACTIVITY_KEYWORDS:
        if any(kw in raw for kw in keywords):
            return label
    return "其他"


def _infer_company_from_asker(asker: str | None) -> tuple[str, str]:
    """从 asker 如「思进智能003025」得到 (公司全称, 公司代码)。"""
    if not asker or not re.search(r"[\u4e00-\u9fa5]+\d{6}", asker.strip()):
        return "未找到", "未找到"
    name = re.sub(r"\d{6}.*$", "", asker.strip()).strip()
    if not name or len(name) > 10:
        return "未找到", "未找到"
    m = re.search(r"\d{6}", asker)
    code = m.group(0) if m else "未找到"
    for suf in ["股份有限公司", "有限责任公司", "有限公司"]:
        if name.endswith(suf):
            return name, code
    return f"{name}股份有限公司", code


# 从公司名/标题中剥离的报告类型、时期后缀（避免将「半年度」「年」等误入公司名）
_REPORT_PERIOD_SUFFIXES = ("年半", "年", "半年度", "季度", "年度")


def _strip_report_period_suffix(s: str) -> str:
    """去掉末尾的报告期、报告类型后缀，避免误入公司全称/简称。"""
    if not s or not s.strip():
        return s
    t = s.strip()
    for suffix in _REPORT_PERIOD_SUFFIXES:
        if t.endswith(suffix):
            t = t[: -len(suffix)].strip()
    return t if t else s.strip()


def _infer_company_from_title(title: str) -> str:
    """从标题推断公司全称（会剥离报告期/类型后缀，避免「年半」等混入）。"""
    for suffix in ["年度", "季度", "半年度", "年度暨", "业绩说明会", "说明会"]:
        if suffix in title:
            part = re.sub(r"\d{4}", "", title.split(suffix)[0]).strip()
            part = _strip_report_period_suffix(part)
            if part and len(part) <= 12:
                for suf in ["股份有限公司", "有限责任公司", "有限公司"]:
                    if part.endswith(suf):
                        return part
                return f"{part}股份有限公司"
    return "未找到"


def _company_short_name(full: str) -> str:
    if not full or full == "未找到":
        return "未找到"
    short = full
    for suf in ["股份有限公司", "有限责任公司", "有限公司", "集团股份有限公司", "集团有限公司", "集团", "公司"]:
        if short.endswith(suf):
            short = short[: -len(suf)] or short
            break
    return _strip_report_period_suffix(short) or full


def fallback_extract_rule_based(txt_path: Path, fields: list[str]) -> list[dict[str, Any]]:
    """LLM 不可用或返回空时，用规则从 JSON 直接抽取字段。无 qaList 时仍返回一条空记录。"""
    title, source, qa_list = parse_one_txt(txt_path)
    if not qa_list:
        return [_build_empty_file_record(txt_path, title, source, fields)]
    activity = _infer_activity_type(title, source)
    file_asker = None
    for qa in qa_list:
        a = (qa.get("asker") or "").strip()
        if a and re.search(r"\d{6}", a):
            file_asker = a
            break
    company_full, company_code = _infer_company_from_asker(file_asker)
    if company_full == "未找到":
        company_full = _infer_company_from_title(title)
    company_short = _company_short_name(company_full)
    records = []
    for qa in qa_list:
        question = (qa.get("question") or "").strip()
        answer = (qa.get("answer") or "").strip()
        if not question and not answer:
            continue
        answer_time = qa.get("answerTime")
        date_str = (answer_time[:10] if isinstance(answer_time, str) and len(answer_time) >= 10 else None) or "未找到"
        asker = (qa.get("asker") or "").strip()
        survey_org = "未找到"
        if asker and ("机构" in asker or re.match(r"^[\u4e00-\u9fa5]+(证券|基金|投资|研究所|咨询)", asker)):
            survey_org = asker
        # 清理回复内容中的套话（支持列表格式）
        cleaned_answer = _clean_reply_content(answer) if answer else "未找到"
        rec = {
            "filename": txt_path.name,
            "公司全称": company_full,
            "公司简称": company_short,
            "公司代码": company_code,
            "文件名称": title,
            "落款日期": date_str,
            "提问内容": question or "未找到",
            "回复内容": cleaned_answer,
            "调研机构": survey_org,
            "活动类别": activity,
            "篇幅页码": "1",
            "字数": str(text_char_count(
                (" ".join(question) if isinstance(question, list) else str(question)) +
                (" ".join(cleaned_answer) if isinstance(cleaned_answer, list) else str(cleaned_answer))
            )),
            "关联公告": title,
        }
        ordered = {"filename": txt_path.name}
        for k in fields:
            ordered[k] = rec.get(k, "未找到")
        records.append(ordered)
    return records


def _build_empty_file_record(txt_path: Path, title: str, source: str, fields: list[str]) -> dict[str, Any]:
    """无 qaList 时仍输出一条记录，保留 filename、文件名称等元数据，提问/回复为空。"""
    rec: dict[str, Any] = {"filename": txt_path.name}
    for k in fields:
        rec[k] = title if k == "文件名称" and title else "未找到"
    rec["提问内容"] = "未找到"
    rec["回复内容"] = "未找到"
    return rec


def process_one_file(
    file_path: Path,
    fields: list[str],
    comments: dict[str, str],
    llm_config: dict[str, Any],
    *,
    vl_config: dict[str, Any] | None = None,
    dpi: int = 150,
    max_pages: int = 30,
    max_qa: int = 0,
    debug_path: Path | None = None,
) -> list[dict[str, Any]]:
    """
    处理单份文件（.txt 或 .pdf）：
    - .txt：解析 qaList 后交给 LLM 抽取。
    - .pdf：与 ESG 一致，PDF 转图后交给 VL 看图抽取（需 vl_config）。
    返回多条 record。无内容时仍返回一条空记录。
    """
    is_pdf = file_path.suffix.lower() in (".pdf",)

    if is_pdf:
        # PDF：VL 看图抽取（与 ESG 相同流程）
        title = file_path.stem
        title = re.sub(r"^\d+_\d{4}-\d{2}-\d{2}_", "", title)
        source = ""
        if not vl_config:
            logger.warning("[%s] 未提供 vl 配置，无法处理 PDF", file_path.name)
            return [_build_empty_file_record(file_path, title, source, fields)]

        total_pages, images = pdf_to_images_base64(file_path, dpi=dpi, max_pages=max_pages)
        if not images:
            return [_build_empty_file_record(file_path, title, source, fields)]

        prompt = build_prompt_vl(fields, comments)
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in images:
            mime = img.get("mime", "image/jpeg")
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img['base64']}"}})

        try:
            data = _call_vl(vl=vl_config, content=content)
        except Exception as e:
            logger.warning("[%s] VL 调用失败（%s）", file_path.name, e)
            return [_build_empty_file_record(file_path, title, source, fields)]

        raw = (data.get("choices") or [{}])[0].get("message") or {}
        raw = raw.get("content") or raw.get("text") or ""
        arr = _parse_llm_json_array(raw)
        if not arr:
            logger.warning("[%s] VL 未返回有效 JSON 数组", file_path.name)
            return [_build_empty_file_record(file_path, title, source, fields)]

        char_count = pdf_text_char_count(file_path)
    else:
        # .txt：LLM 文本抽取
        title, source, qa_list = parse_one_txt(file_path)
        total_pages = 1
        if not qa_list:
            logger.warning("[%s] 无 qaList，仍输出一条空记录", file_path.name)
            return [_build_empty_file_record(file_path, title, source, fields)]
        input_text = build_qa_input_text(title, source, qa_list, max_qa=max_qa)
        char_count = text_char_count(input_text)

        prompt = build_prompt(fields, comments, input_text)
        try:
            raw = call_llm(llm_config, prompt, debug_path=debug_path)
        except Exception as e:
            logger.warning("[%s] LLM 调用失败（%s）", file_path.name, e)
            return fallback_extract_rule_based(file_path, fields)

        arr = _parse_llm_json_array(raw)
        if not arr:
            logger.warning("[%s] LLM 未返回有效 JSON 数组", file_path.name)
            return fallback_extract_rule_based(file_path, fields)

    records: list[dict[str, Any]] = []
    for rec in arr:
        ordered: dict[str, Any] = {"filename": file_path.name}
        for k in fields:
            val = rec.get(k)
            if val is None:
                val = "未找到"
            elif isinstance(val, str) and not val.strip():
                val = "未找到"
            elif isinstance(val, list):
                if not val or all(not str(item).strip() for item in val):
                    val = "未找到"
            if k == "活动类别" and isinstance(val, str):
                allowed = [x.strip() for x in ACTIVITY_TYPE_ENUM.split("、")]
                if val.strip() not in allowed:
                    val = "其他"
            # 对提问内容和回复内容进行二次清理（去除套话），支持列表格式
            if k in ("提问内容", "回复内容"):
                if isinstance(val, list):
                    cleaned_list = []
                    for item in val:
                        if isinstance(item, str) and item.strip():
                            if k == "回复内容":
                                cleaned_item = _clean_reply_content(item)
                            else:
                                cleaned_item = item.strip()
                            if cleaned_item and cleaned_item != "未找到":
                                cleaned_list.append(cleaned_item)
                    val = cleaned_list if cleaned_list else "未找到"
                elif isinstance(val, str) and val != "未找到":
                    if k == "回复内容":
                        val = _clean_reply_content(val)
                    else:
                        val = val.strip() or "未找到"
            ordered[k] = val

        # 对 PDF 文件覆盖篇幅页码和字数
        if is_pdf:
            if "篇幅页码" in ordered:
                ordered["篇幅页码"] = str(total_pages)
            if "字数" in ordered:
                ordered["字数"] = str(char_count)

        # 过滤：仅跳过提问与回复均为空或未找到、且非列表概括的条目
        question = ordered.get("提问内容", "")
        answer = ordered.get("回复内容", "")
        question_str = " ".join(question) if isinstance(question, list) else str(question)
        answer_str = " ".join(answer) if isinstance(answer, list) else str(answer)
        question_str = (question_str or "").strip()
        answer_str = (answer_str or "").strip()
        is_list_summary = isinstance(question, list) or isinstance(answer, list)
        if not question_str or question_str == "未找到":
            if not answer_str or answer_str == "未找到":
                if not is_list_summary:
                    logger.debug("[%s] 跳过无实质问答条目", file_path.name)
                    continue
        records.append(ordered)
    if not records:
        return [_build_empty_file_record(file_path, title, source, fields)]
    return records


def _content_to_list(val: Any) -> list[str]:
    """将提问内容/回复内容统一为 list：已是 list 则返回，否则 [val] 或 []。"""
    if isinstance(val, list):
        return [str(item).strip() for item in val if str(item).strip()]
    if isinstance(val, str) and val.strip() and val != "未找到":
        return [val.strip()]
    return []


def merge_records_to_one(records: list[dict[str, Any]], fields: list[str]) -> dict[str, Any]:
    """一份文件多条 Q&A 合并为一条：提问内容/回复内容为二维 list（每条一问一答对应）。"""
    if not records:
        return {}
    first = records[0]
    # filename 放第一个字段，其余按 fields 顺序
    merged: dict[str, Any] = {"filename": first.get("filename", "")}
    for k in fields:
        merged[k] = first.get(k, "未找到")
    # 提问内容 = [ 第1条的提问list, 第2条的提问list, ... ]
    merged["提问内容"] = [_content_to_list(r.get("提问内容")) or ["未找到"] for r in records]
    merged["回复内容"] = [_content_to_list(r.get("回复内容")) or ["未找到"] for r in records]
    # 落款日期取第一条非「未找到」
    for r in records:
        d = r.get("落款日期") or ""
        if isinstance(d, str) and d.strip() and d != "未找到":
            merged["落款日期"] = d
            break
    # 字数为各条之和
    total_chars = 0
    for r in records:
        w = r.get("字数")
        if w is not None and str(w).isdigit():
            total_chars += int(w)
    if total_chars > 0:
        merged["字数"] = str(total_chars)
    return merged


def iter_input_files(input_path: Path) -> Iterable[Path]:
    """遍历输入路径下的 .txt 和 .pdf 文件。"""
    supported = {".txt", ".pdf"}
    if input_path.is_file():
        if input_path.suffix.lower() in supported:
            yield input_path
        return
    for p in sorted(input_path.glob("**/*")):
        if p.is_file() and p.suffix.lower() in supported:
            yield p


def _normalize_date(s: str) -> str | None:
    """将日期统一为 YYYY-MM-DD。"""
    if not s or s == "未找到":
        return None
    s = re.sub(r"\s+", "", str(s))
    m = re.match(r"(\d{4})-?(\d{2})-?(\d{2})", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def load_announcements_jsonl(path: Path) -> list[dict[str, Any]]:
    """加载公告列表 JSONL，每行需含 date、title，及 company_short 或 company_full。"""
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


def find_related_announcement_for_ir_qa(
    record: dict[str, Any],
    announcements: list[dict[str, Any]],
) -> str | None:
    """
    在公告列表中查找与投关活动相关的公告（年报、季报、业绩说明会公告等）。
    同公司、日期相近（±7天）或同日，标题含报告/业绩说明会等。
    """
    date_val = _normalize_date(str(record.get("落款日期") or ""))
    company_short = (record.get("公司简称") or "").strip()
    company_full = (record.get("公司全称") or "").strip()
    if not date_val:
        return None

    def company_match(a: dict[str, Any]) -> bool:
        title = (a.get("title") or a.get("announcement_title") or "").strip()
        cs = (a.get("company_short") or "").strip()
        cf = (a.get("company_full") or "").strip()
        if company_short and (company_short in title or (cs and (company_short in cs or cs in company_short))):
            return True
        if company_full and (company_full in title or (cf and (company_full in cf or cf in company_full))):
            return True
        return bool(title and (company_short in title or company_full in title))

    def is_related_type(a: dict[str, Any]) -> bool:
        title = (a.get("title") or a.get("announcement_title") or "").strip()
        return any(kw in title for kw in ("报告", "年报", "季报", "半年报", "业绩说明会", "投资者关系"))

    for a in announcements:
        if not company_match(a):
            continue
        d = _normalize_date(str(a.get("date") or a.get("披露日期") or ""))
        if not d:
            continue
        try:
            dd = _dt.strptime(d, "%Y-%m-%d")
            dv = _dt.strptime(date_val, "%Y-%m-%d")
            if abs((dd - dv).days) > 7:
                continue
        except Exception:
            if d != date_val:
                continue
            if is_related_type(a):
                title = (a.get("title") or a.get("announcement_title") or "").strip()
                if title:
                    return title
    return None


def load_done_filenames(output_path: Path) -> set[str]:
    """从已有 JSONL 中读取已完成的 filename 集合，用于断点续跑时跳过。"""
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
    parser = argparse.ArgumentParser(description="投关问答字段抽取（用 LLM 分析 .txt/.pdf，每条 Q&A 一行 JSONL）")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径（需包含 llm 配置）")
    parser.add_argument("--llm-profile", default="default", help="使用的 LLM 配置项，默认 default（config 中 llm.default，如火山引擎）")
    parser.add_argument("--requirement", default="data/requirement/requirement_1.json", help="requirement_1.json 路径")
    parser.add_argument("--input", default="data/report/投关问答", help="投关问答目录（或单个文件），支持 .txt 和 .pdf")
    parser.add_argument("--output", default="result/ir_qa_extracted.jsonl", help="输出 jsonl 路径")
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少份文件（0=不限制）；用于样例可设 1")
    parser.add_argument("--skip", type=int, default=0, help="跳过前 N 份文件，与 --limit 配合可分批处理")
    parser.add_argument("--max-qa", type=int, default=0, help="单文件最多发送多少条问答给 LLM（0=全部）；仅对 .txt 有效")
    parser.add_argument("--dpi", type=int, default=150, help="PDF 转图 DPI（仅 .pdf 使用 VL 时有效）")
    parser.add_argument("--max-pages", type=int, default=30, help="单份 PDF 最多转多少页给 VL（仅 .pdf 有效，默认 30）")
    parser.add_argument("--append", action="store_true", help="追加写入 output（默认覆盖）")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续跑：读取 output 中已有 filename，只处理未完成的文件，并追加写入",
    )
    parser.add_argument("--print-sample", action="store_true", help="打印第一条结果到 stdout")
    parser.add_argument("--debug", action="store_true", help="将 LLM 原始返回写入 result/ir_qa_llm_debug.txt 便于排查")
    parser.add_argument(
        "--announcements",
        type=str,
        default="",
        help="可选：公告列表 JSONL 路径，用于从同公司同日公告中匹配「关联公告」。每行需含 date、title，及 company_short 或 company_full",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    requirement_path = Path(args.requirement)
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("开始抽取投关问答")
    config = load_config(config_path)
    llm_config = get_llm_config(config, args.llm_profile)
    vl_config = None
    try:
        vl_config = get_vl_config(config)
        logger.info("PDF 将使用 VL 看图抽取（config.vl），.txt 使用 LLM（config.llm.%s）", args.llm_profile)
    except ValueError:
        logger.warning("config 中未配置 vl，仅 .txt 可用 LLM 抽取；.pdf 将跳过或输出空记录")
    fields, comments = load_fields_and_comments(requirement_path)
    logger.info("共 %d 个抽取字段", len(fields))

    announcements_list: list[dict[str, Any]] = []
    if args.announcements:
        ap = Path(args.announcements)
        announcements_list = load_announcements_jsonl(ap)
        logger.info("已加载公告列表: %s，共 %d 条（用于关联公告匹配）", ap, len(announcements_list))

    files = list(iter_input_files(input_path))
    if args.skip > 0:
        files = files[args.skip:]
        logger.info("已跳过前 %d 份，剩余 %d 份", args.skip, len(files))
    if args.limit and args.limit > 0:
        files = files[: args.limit]
    if args.resume and output_path.exists():
        done = load_done_filenames(output_path)
        files = [p for p in files if p.name not in done]
        logger.info("断点续跑: 已有 %d 份已完成，待处理: %d 个", len(done), len(files))
    else:
        logger.info("待处理文件: %d 个", len(files))

    if not files:
        logger.warning("未找到任何 .txt/.pdf 文件，请检查 --input 路径：%s", input_path)
        return 0

    debug_path = (output_path.parent / "ir_qa_llm_debug.txt") if args.debug else None
    mode = "a" if (args.append or args.resume) else "w"
    total_records = 0
    with output_path.open(mode, encoding="utf-8") as out:
        for i, p in enumerate(files, 1):
            logger.info("===== 第 %d/%d 个: %s =====", i, len(files), p.name)
            try:
                records = process_one_file(
                    p,
                    fields,
                    comments,
                    llm_config,
                    vl_config=vl_config,
                    dpi=args.dpi,
                    max_pages=args.max_pages,
                    max_qa=args.max_qa,
                    debug_path=debug_path,
                )
                # 一份文件一条记录：仅合并本文件内的多条问答，无 qa 时也输出一条（仅元数据）
                merged = merge_records_to_one(records, fields)
                # 反馈：关联公告应关联至公告，若仍未找到则从公告库匹配
                if announcements_list and merged.get("关联公告") in (None, "", "未找到"):
                    related = find_related_announcement_for_ir_qa(merged, announcements_list)
                    if related:
                        merged["关联公告"] = related
                        logger.info("[%s] 从公告库匹配到关联公告: %s", p.name, related[:50])
                # 公司全称未找到则过滤掉，不写入
                if merged.get("公司全称") in (None, "", "未找到"):
                    logger.info("[%s] 公司全称未找到，跳过不写入", p.name)
                else:
                    out.write(json.dumps(merged, ensure_ascii=False) + "\n")
                    total_records += 1
                    if len(records) == 1 and records[0].get("提问内容") == "未找到" and records[0].get("回复内容") == "未找到":
                        logger.info("[%s] 产出 1 条（无问答，仅元数据），已写入 %s", p.name, output_path)
                    else:
                        logger.info("[%s] 产出 1 条（本文件内合并 %d 条问答），已写入 %s", p.name, len(records), output_path)
                out.flush()
            except Exception as e:
                logger.exception("[%s] 本份抽取失败，跳过，继续下一份: %s", p.name, e)
    logger.info("全部完成，共 %d 条记录", total_records)

    if args.print_sample and total_records > 0:
        with output_path.open("r", encoding="utf-8") as f:
            first_line = f.readline()
            if first_line:
                print(json.dumps(json.loads(first_line), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
