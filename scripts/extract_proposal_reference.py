#!/usr/bin/env python3
"""
议案参考字段抽取（流程与定期报告类似，VL 大模型）。

流程：
1) 读取 data/requirement/requirement_1.json 中「议案参考」的字段清单及 comment
2) 将 PDF 转成多页 PNG(base64)
3) 调用 config.yaml 中 vl 配置的视觉大模型，输出 JSON（含文档级字段 + 议案列表）
4) 将公告中的议案名称及议案内容单独拆解：每条议案输出一行 JSONL

写入与断点续跑：
- 每条记录写完后立即 flush，中断或异常时已写入的记录不会丢失。
- 单份 PDF 失败会记录日志并跳过，继续处理下一份，不中断整批。
- 使用 --resume 时从 output 中读取已完成的 filename，只处理未完成的 PDF 并追加写入。
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

# 文档级字段（整份公告共用）
DOC_FIELDS = [
    "公司全称",
    "公司简称",
    "公司代码",
    "文件名称",
    "落款日期",
    "会议类型",
    "会议届次",
    "会议召开时间",
    "篇幅页码",
    "字数",
    "关联公告",
]
# 每条议案单独抽取的字段
MOTION_FIELDS = [
    "议案名称",
    "议案内容",
    "审议状态",
    "表决情况",
    "回避情况",
]


def load_config(config_path: Path) -> dict[str, Any]:
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def load_fields_and_comments(requirement_path: Path) -> tuple[list[str], dict[str, str]]:
    """
    从 requirement_1.json 读取「议案参考」的字段列表与 comment。
    返回 (所有字段列表, 字段注释)。议案相关字段会用于议案列表项。
    """
    data = json.loads(requirement_path.read_text(encoding="utf-8"))
    for section in data.get("sections", []):
        if section.get("name") != "议案参考":
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
            raise ValueError("requirement_1.json 的 议案参考 fields 为空")
        return fields, comments

    raise ValueError("requirement_1.json 中未找到 name=议案参考 的 section")


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


def build_prompt(comments: dict[str, str]) -> str:
    """构建议案参考抽取提示词：文档级字段 + 议案列表（每条含议案名称、议案内容等）。"""
    guidance_lines: list[str] = []
    for f in DOC_FIELDS + MOTION_FIELDS:
        c = comments.get(f)
        if c:
            guidance_lines.append(f"- {f}：{c}")

    # 输出结构：文档级字段 + 议案列表（数组，每项含 MOTION_FIELDS）
    doc_json = ",\n".join([f'  "{f}": "..."' for f in DOC_FIELDS])
    motion_item_json = ",\n".join([f'      "{f}": "..."' for f in MOTION_FIELDS])

    parts = []
    parts.append("你是一位专业的信息抽取专家。请阅读这份决议/会议公告（多页图片），按以下要求抽取信息。\n\n")
    parts.append("## 一、文档级信息（整份公告共用，只填一份）\n")
    parts.append("\n".join([f"- {f}" for f in DOC_FIELDS]))
    parts.append("\n\n## 二、议案信息（重要：将公告中的每个议案单独拆解，每个字段独立填写）\n")
    parts.append("请识别公告中的每一个议案，为每个议案分别抽取以下字段，**每个字段单独填写、不要混在一起**：\n")
    parts.append("\n".join([f"- {f}" for f in MOTION_FIELDS]))
    parts.append("\n- 议案名称、议案内容、审议状态、表决情况、回避情况 必须分开输出，不要将多个字段内容合并到同一字段中\n")
    parts.append("\n\n## 字段要求（来自需求说明）\n")
    parts.append("\n".join(guidance_lines) if guidance_lines else "无\n")
    parts.append("\n\n## 输出要求\n")
    parts.append("- 只输出一个 JSON 对象，不要输出任何解释性文字\n")
    parts.append("- 文档级字段在顶层；所有议案放在 \"议案列表\" 数组中，数组每项包含：议案名称、议案内容、审议状态、表决情况、回避情况\n")
    parts.append("- 若某字段确实找不到，填 \"未找到\"\n")
    parts.append("- 落款日期、会议召开时间统一用 YYYY-MM-DD 或 YYYY年MM月DD日 均可\n")
    parts.append("- 会议类型只能为枚举之一：股东会、董事会、监事会、审计委员会、战略委员会、提名与薪酬委员会、ESG委员会、独立董事专门会议、职工代表大会、其他\n")
    parts.append("- 审议状态示例：已经董事会审议通过，尚需提交股东会审议；或已经董事会审议通过，无需提交股东会审议\n")
    parts.append("- 表决情况示例：同意X票，反对X票，弃权X票\n")
    parts.append("- 议案名称、关联公告请使用书名号《》包裹，例如：《关于2025年度董事、监事、高级管理人员薪酬方案的议案》《关于将前次股东大会未获通过议案再次提交股东大会审议的说明公告》\n")
    parts.append("- **关联公告**：指议案内容中提到的、与本决议相关的其他公告（如「详见《关于XX的公告》」「具体内容见《XX公告》」），不是本决议公告本身。若议案中未提到其他公告则填「未找到」。\n")
    parts.append("- 若无议案或无法识别，议案列表可为空数组 []\n\n")
    parts.append("请严格按以下 JSON 结构输出：\n{\n")
    parts.append(doc_json)
    parts.append(',\n  "议案列表": [\n    {\n')
    parts.append(motion_item_json)
    parts.append("\n    }\n  ]\n}\n")
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

    resp = requests.post(url, headers=headers, json=payload, timeout=600)
    resp.raise_for_status()
    logger.info("VL 响应: 成功, 状态码 %s", resp.status_code)
    return resp.json()


def infer_company_short_name(full_name: str) -> str | None:
    """从公司全称推断简称（确定性裁剪）。"""
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
    comments: dict[str, str],
    config: dict[str, Any],
    dpi: int,
    max_pages: int,
) -> dict[str, Any]:
    """
    对一份议案参考 PDF 做 VL 抽取，返回一条 record（文档级字段 + 议案列表数组），一个文件的内容放在一起。
    """
    if "vl" not in config:
        raise ValueError("config.yaml 缺少 vl 配置")
    vl = config["vl"]
    if not vl.get("enable", True):
        raise ValueError("config.yaml 中 vl.enable=false，无法使用 VL 抽取")

    logger.info("[%s] 开始抽取", pdf_path.name)
    total_pages, images = pdf_to_images_base64(pdf_path, dpi=dpi, max_pages=max_pages)
    logger.info("[%s] PDF 已转成 %d 页图片（总页数 %d）", pdf_path.name, len(images), total_pages)
    prompt = build_prompt(comments)

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
    raw = data["choices"][0]["message"]["content"]
    extracted = _loads_json_relaxed(raw)
    logger.info("[%s] VL 返回成功，已解析 JSON", pdf_path.name)

    # 文档级字段
    doc_record: dict[str, Any] = {"filename": pdf_path.name}
    for f in DOC_FIELDS:
        val = extracted.get(f)
        if val in (None, "", []):
            val = "未找到"
        doc_record[f] = val

    # 篇幅页码、字数：用真实值覆盖
    doc_record["篇幅页码"] = str(total_pages)
    cnt = pdf_text_char_count(pdf_path)
    doc_record["字数"] = str(cnt) if cnt > 0 else (doc_record.get("字数") or "未找到")

    # 公司简称：若未找到则从公司全称推断
    if doc_record.get("公司简称") in (None, "", "未找到"):
        short = infer_company_short_name(str(doc_record.get("公司全称") or ""))
        if short:
            doc_record["公司简称"] = short

    # 关联公告：反馈要求关联议案中提到的公告，而非本决议自身
    rel_ann = doc_record.get("关联公告")
    self_title = (doc_record.get("文件名称") or pdf_path.stem or "").strip()

    def _is_self_resolution(s: str) -> bool:
        if not s or s == "未找到":
            return True
        s = str(s).strip()
        return "决议" in s or s == self_title or self_title in s

    motion_list = extracted.get("议案列表")
    if not isinstance(motion_list, list):
        motion_list = []

    # 若模型填的是本决议自身或未找到，尝试从议案内容中提取提到的公告
    if _is_self_resolution(rel_ann) and motion_list:
        found: list[str] = []
        for item in motion_list:
            if not isinstance(item, dict):
                continue
            content = str(item.get("议案内容") or item.get("议案名称") or "").strip()
            for m in re.finditer(r"《([^》]+(?:公告|说明|报告)[^》]*)》", content):
                cand = m.group(1).strip()
                if cand and "议案" not in cand and cand not in found:
                    found.append(cand)
        if found:
            rel_ann = f"《{found[0]}》" if len(found) == 1 else "；".join(f"《{x}》" for x in found[:3])
            logger.info("[%s] 从议案内容提取关联公告: %s", pdf_path.name, rel_ann[:60])

    # 若仍未找到有效关联公告，才用文件名称作为兜底
    if rel_ann in (None, "", "未找到") or _is_self_resolution(rel_ann):
        rel_ann = doc_record.get("文件名称") or pdf_path.stem
    if rel_ann and str(rel_ann).strip() and rel_ann != "未找到":
        s = str(rel_ann).strip()
        if not (s.startswith("《") and s.endswith("》")) and "；" not in s:
            rel_ann = f"《{s}》"
    doc_record["关联公告"] = rel_ann if (rel_ann and rel_ann != "未找到") else "未找到"

    # 规范化议案列表：每项为 { 议案名称, 议案内容, 审议状态, 表决情况, 回避情况 }
    # 议案名称统一为《》包裹格式，如《关于2025年度董事、监事、高级管理人员薪酬方案的议案》
    normalized_motions: list[dict[str, Any]] = []
    for item in motion_list:
        if not isinstance(item, dict):
            continue
        m: dict[str, Any] = {}
        for f in MOTION_FIELDS:
            val = item.get(f)
            if val in (None, "", []):
                val = "未找到"
            m[f] = val
        # 议案名称：若非「未找到」且未带《》，则用《》包裹
        name_val = m.get("议案名称") or ""
        if name_val and name_val != "未找到":
            name_stripped = name_val.strip()
            if name_stripped and not (name_stripped.startswith("《") and name_stripped.endswith("》")):
                m["议案名称"] = f"《{name_stripped}》"
        normalized_motions.append(m)

    doc_record["议案列表"] = normalized_motions
    logger.info("[%s] 识别到 %d 条议案，一个文件一条记录", pdf_path.name, len(normalized_motions))
    return doc_record


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
    parser = argparse.ArgumentParser(description="议案参考字段抽取（VL大模型），一个文件一条 JSONL，议案列表在记录内")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--requirement", default="data/requirement/requirement_1.json", help="requirement_1.json 路径")
    parser.add_argument("--input", default="data/report/议案参考", help="议案参考 PDF 目录或单个 PDF")
    parser.add_argument("--output", default="result/proposal_reference_extracted_vl.jsonl", help="输出 jsonl 路径")
    parser.add_argument("--limit", type=int, default=5, help="最多处理多少份 PDF（0=不限制）")
    parser.add_argument("--skip", type=int, default=0, help="跳过前 N 份 PDF，与 --limit 配合可只处理中间或最后若干份")
    parser.add_argument("--dpi", type=int, default=150, help="PDF 转图片 dpi")
    parser.add_argument("--max-pages", type=int, default=50, help="单份 PDF 最多处理页数")
    parser.add_argument("--append", action="store_true", help="追加写入 output")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续跑：读取 output 中已有 filename，只处理未完成的 PDF，并追加写入",
    )
    parser.add_argument("--print-sample", action="store_true", help="打印第一条结果到 stdout")
    args = parser.parse_args()

    config_path = Path(args.config)
    requirement_path = Path(args.requirement)
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("开始抽取议案参考")
    config = load_config(config_path)
    _, comments = load_fields_and_comments(requirement_path)
    logger.info("已加载 config 与 requirement（议案参考）")

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
                    comments=comments,
                    config=config,
                    dpi=args.dpi,
                    max_pages=args.max_pages,
                )
                output_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                output_file.flush()
                success_count += 1
                logger.info("[%s] 已写入 1 条记录（含 %d 个议案）到 %s", p.name, len(rec.get("议案列表", [])), output_path)
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
