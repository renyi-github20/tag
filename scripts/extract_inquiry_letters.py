#!/usr/bin/env python3
"""
问询函件字段抽取（纯 VL 大模型，不做文字匹配/规则抽取）。

流程：
1) 读取 data/requirement/requirement_1.json 中「闻讯函件」的字段清单（以及字段 comment，用来做提示词）
2) 将 PDF 转成多页 PNG(base64)
3) 调用 config.yaml 中 vl 配置的视觉大模型，输出 JSON
4) 结果写入 JSONL（每行一个 PDF）

处理过程与定期报告类似。

写入与断点续跑：
- 每条记录写完后立即 flush，中断或异常时已写入的记录不会丢失。
- 单份 PDF 失败会记录日志并跳过，继续处理下一份，不中断整批。
- 使用 --resume 时从 output 中读取已完成的 filename，只处理未完成的 PDF 并追加写入。

分类与关联：
- VL 输出文档类型（问询函/回复函/整改报告/其他），主结果中保留该字段便于下游筛选；不按文档类型跳过，所有 PDF 均写入主结果。
- 关联函件：主结果含公司全称、收函时间、最晚回函时间等，便于下游按公司+时间范围关联「收到监管部门函件」「收到监管部门函件的回复」等公告，理清事件脉络。
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import re
from datetime import datetime
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
import yaml


def load_config(config_path: Path) -> dict[str, Any]:
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def load_fields_and_comments(requirement_path: Path) -> tuple[list[str], dict[str, str]]:
    """
    从 requirement_1.json 读取「闻讯函件」的字段列表与 comment。

    注意：该 section 内可能有重复 field 名，这里做去重（保留首次出现顺序）。
    """
    data = json.loads(requirement_path.read_text(encoding="utf-8"))
    for section in data.get("sections", []):
        if section.get("name") != "闻讯函件":
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
            raise ValueError("requirement_1.json 的 闻讯函件 fields 为空")
        return fields, comments

    raise ValueError("requirement_1.json 中未找到 name=闻讯函件 的 section")


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
    """统计「字数」：使用 PyMuPDF 抽取到的文本，去掉空白字符后的字符数。"""
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
    """尽量把模型输出解析成 JSON 对象。"""
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
    fields_json += ',\n  "文档类型": "问询函|回复函|整改报告|其他"'

    parts = []
    parts.append("你是一位专业的信息抽取专家。请阅读这份问询函件相关公告/函件（多页图片），抽取指定字段。\n\n")
    parts.append("## 需要抽取的字段（必须全部输出）\n")
    parts.append("\n".join([f"- {f}" for f in fields]))
    parts.append("\n\n## 字段要求（来自需求说明，务必遵守）\n")
    parts.append("\n".join(guidance_lines) if guidance_lines else "无\n")
    parts.append("\n\n## 输出要求\n")
    parts.append("- 只输出 JSON（不要输出任何解释性文字）\n")
    parts.append("- 字段名必须与上面完全一致\n")
    parts.append('- 若确实找不到，填"未找到"\n')
    parts.append("- 日期统一输出 YYYY-MM-DD（如 2025-01-07）\n")
    parts.append('- "发函单位"从正文提取，优先使用需求中的枚举值（深圳证券交易所、上海证券交易所、北京证券交易所上市公司管理部等）\n')
    parts.append('- "函件类别"只能是需求中的枚举值之一：年度报告、半年度报告、季度报告、业绩预测、权益分派、关联交易、对外担保、资金占用、对外投资、收并购、募集资金、再融资、股权激励、重大资产重组、其他\n')
    parts.append('- "函件目录"：从问询函/回复函正文提取问题作为全文目录。要求：精确到二级分类，一级和二级标题均需完整列出、不遗漏；用分号 ; 分隔，支持跳转。\n')
    parts.append('- "回函时限"：仅对问询函/回复函有效。若正文有收函时间和最晚回函时间，可计算：最晚回函时间 - 收函时间 + 1 天，输出如 "7天"。整改报告等无此字段填"未找到"。\n')
    parts.append('- "篇幅页数"输出数字（总页数）\n')
    parts.append('- "字数"输出数字（全文字数/字符数）\n')
    parts.append('- "文档类型"：必填，枚举值之一：问询函、回复函、整改报告、其他。根据正文判断（如标题或正文含《监管关注函》的整改报告、整改报告等，则为整改报告；问询函的回复则为回复函）。\n')
    parts.append('- "关联函件"：若本件为回复函，必须填写对应的发函方问询函名称或文号（如《关于对XX公司年报的问询函》）；若为问询函填"未找到"。回复函应关联发函方的问询函，便于理清事件脉络。\n')
    parts.append("请按以下 JSON 结构输出（含文档类型）：\n{\n")
    parts.append(fields_json)
    parts.append("\n}\n")
    return "".join(parts)


def infer_company_short_name(full_name: str) -> Optional[str]:
    """公司简称：若模型没给，但公司全称存在，则做确定性裁剪。"""
    if not full_name or full_name == "未找到":
        return None
    short = full_name
    for suffix in ["股份有限公司", "有限责任公司", "有限公司", "集团股份有限公司", "集团有限公司", "集团", "公司"]:
        if short.endswith(suffix):
            short = short[: -len(suffix)]
            break
    return short or full_name


def parse_date_maybe(s: Any) -> Optional[datetime]:
    """从字符串解析日期，支持 YYYY-MM-DD、YYYY年M月D日 等。"""
    if s is None:
        return None
    txt = str(s).strip()
    if not txt or txt == "未找到":
        return None
    # YYYY-MM-DD
    m = re.match(r"(\d{4})-(\d{1,2})-(\d{1,2})", txt)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    # YYYY年M月D日
    m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", txt)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    return None


def compute_reply_deadline_days(receive_date_str: Any, deadline_str: Any) -> Optional[str]:
    """回函时限 = 最晚回函时间 - 收函时间 + 1，输出 X天。"""
    receive = parse_date_maybe(receive_date_str)
    deadline = parse_date_maybe(deadline_str)
    if receive is None or deadline is None:
        return None
    delta = (deadline - receive).days + 1
    if delta < 0:
        return None
    return f"{delta}天"


def _call_vl(*, vl: dict[str, Any], content: list[dict[str, Any]]) -> dict[str, Any]:
    n_images = sum(1 for x in content if x.get("type") == "image_url")
    from scripts.vl_utils import get_vl_endpoint
    url, model, max_tokens = get_vl_endpoint(vl)
    logger.info("VL 请求: POST %s, 共 %d 条 content（其中 %d 张图）, timeout=400s", url, len(content), n_images)

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
    resp.raise_for_status()
    logger.info("VL 响应: 成功, 状态码 %s", resp.status_code)
    return resp.json()


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
    raw = data["choices"][0]["message"]["content"]
    extracted = _loads_json_relaxed(raw)
    logger.info("[%s] VL 返回成功，已解析 JSON", pdf_path.name)

    for f in fields:
        if f not in extracted or extracted[f] in (None, "", []):
            extracted[f] = "未找到"

    record = {"filename": pdf_path.name, **{k: extracted.get(k) for k in fields}}
    # 文档类型用于区分问询函/回复函/整改报告，整改报告不入问询函件主结果
    doc_type = (extracted.get("文档类型") or "").strip()
    if doc_type not in ("问询函", "回复函", "整改报告", "其他"):
        doc_type = "其他"
    record["文档类型"] = doc_type

    if "篇幅页数" in record:
        record["篇幅页数"] = str(total_pages)

    if "字数" in record:
        cnt = pdf_text_char_count(pdf_path)
        record["字数"] = str(cnt) if cnt > 0 else (record.get("字数") or "未找到")

    if "公司简称" in record and (record["公司简称"] in (None, "", "未找到")):
        short = infer_company_short_name(str(record.get("公司全称") or ""))
        if short:
            record["公司简称"] = short

    if "回函时限" in record and record.get("回函时限") in (None, "", "未找到"):
        days = compute_reply_deadline_days(record.get("收函时间"), record.get("最晚回函时间"))
        if days:
            record["回函时限"] = days

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
    parser = argparse.ArgumentParser(description="问询函件字段抽取（VL大模型）")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径（需要包含 vl 配置）")
    parser.add_argument("--requirement", default="data/requirement/requirement_1.json", help="requirement_1.json 路径")
    parser.add_argument("--input", default="data/report/问询函件", help="问询函件PDF目录（或单个PDF）")
    parser.add_argument("--output", default="result/inquiry_letters_extracted_vl.jsonl", help="输出 jsonl 路径")
    parser.add_argument("--limit", type=int, default=5, help="最多处理多少份PDF（0=不限制；默认5用于生成样例）")
    parser.add_argument("--dpi", type=int, default=150, help="PDF转图片的dpi")
    parser.add_argument("--max-pages", type=int, default=50, help="单份PDF最多处理多少页（默认50）")
    parser.add_argument("--append", action="store_true", help="追加写入 output（默认覆盖）")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续跑：读取 output 中已有 filename，只处理未完成的 PDF，并追加写入",
    )
    parser.add_argument("--print-sample", action="store_true", help="打印第一条结果到stdout")
    args = parser.parse_args()

    config_path = Path(args.config)
    requirement_path = Path(args.requirement)
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("开始抽取问询函件")
    config = load_config(config_path)
    fields, comments = load_fields_and_comments(requirement_path)
    logger.info("已加载 config 与 requirement，共 %d 个字段", len(fields))

    pdfs = list(iter_pdfs(input_path))
    if args.limit and args.limit > 0:
        pdfs = pdfs[: args.limit]

    # 断点续跑：跳过 output 中已有 filename 的 PDF，只处理未完成的
    if args.resume and output_path.exists():
        done = load_done_filenames(output_path)
        pdfs = [p for p in pdfs if p.name not in done]
        logger.info("断点续跑: 已有 %d 份已完成，待处理 %d 份", len(done), len(pdfs))
    else:
        logger.info("待处理 PDF: %d 个", len(pdfs))

    if not pdfs:
        logger.warning("未找到任何 PDF，请将问询函件 PDF 放入 %s 或指定 --input", input_path)
        return 0

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
