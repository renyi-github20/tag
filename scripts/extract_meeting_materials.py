#!/usr/bin/env python3
"""
会议资料字段抽取（流程与定期报告类似，纯 VL 大模型）。

流程：
1) 读取 data/requirement/requirement_1.json 中「会议资料」的字段清单（及 comment 用于提示词）
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
import requests
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict[str, Any]:
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def load_fields_and_comments(requirement_path: Path) -> tuple[list[str], dict[str, str]]:
    """
    从 requirement_1.json 读取「会议资料」的字段列表与 comment。
    该 section 内重复 field 名会去重（保留首次出现顺序）。
    """
    data = json.loads(requirement_path.read_text(encoding="utf-8"))
    for section in data.get("sections", []):
        if section.get("name") != "会议资料":
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
            raise ValueError("requirement_1.json 的 会议资料 fields 为空")
        return fields, comments

    raise ValueError("requirement_1.json 中未找到 name=会议资料 的 section")


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
    """统计字数：PyMuPDF 抽取文本去掉空白后的字符数。"""
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
    parts = [
        "你是一位专业的信息抽取专家。请阅读这份会议资料（多页图片），抽取指定字段。\n\n",
        "## 需要抽取的字段（必须全部输出）\n",
        "\n".join([f"- {f}" for f in fields]),
        "\n\n## 字段要求（来自需求说明，务必遵守）\n",
        "\n".join(guidance_lines) if guidance_lines else "无\n",
        "\n\n## 输出要求\n",
        "- 只输出 JSON（不要输出任何解释性文字）\n",
        "- 字段名必须与上面完全一致\n",
        '- 若确实找不到，填"未找到"\n',
        "- 日期统一输出 YYYY-MM-DD（如 2025-12-26）\n",
        '- "会议类型"只能是：年度股东会 / 临时股东会\n',
        '- "文件目录"从正文提取，支持点击跳转；多级用分号 ; 分隔，子项可缩进或换行表示层级\n',
        '- "会议召开时间"格式如：2025年12月26日 或 YYYY-MM-DD\n',
        '- "篇幅页码"输出数字（总页数）\n',
        '- "字数"输出数字（全文字数/字符数）\n\n',
        "请按以下 JSON 结构输出：\n{\n",
        fields_json,
        "\n}\n",
    ]
    return "".join(parts)


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


def infer_company_short_name(full_name: str) -> str | None:
    """公司简称：若模型未给且公司全称存在，则做确定性裁剪。"""
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
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img['base64']}"}})

    logger.info("[%s] 正在调用 VL...", pdf_path.name)
    data = _call_vl(vl=vl, content=content)
    raw = data["choices"][0]["message"]["content"]
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
    parser = argparse.ArgumentParser(description="会议资料字段抽取（VL大模型）")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径（需包含 vl 配置）")
    parser.add_argument("--requirement", default="data/requirement/requirement_1.json", help="requirement_1.json 路径")
    parser.add_argument("--input", default="data/report/会议资料", help="会议资料 PDF 目录（或单个 PDF）")
    parser.add_argument("--output", default="result/meeting_materials_extracted_vl.jsonl", help="输出 jsonl 路径")
    parser.add_argument("--limit", type=int, default=5, help="最多处理多少份 PDF（0=不限制；默认 5 用于样例）")
    parser.add_argument("--skip", type=int, default=0, help="跳过前 N 份 PDF，与 --limit 配合可分批处理（如 --skip 50 --limit 50）")
    parser.add_argument("--dpi", type=int, default=150, help="PDF 转图片的 dpi")
    parser.add_argument("--max-pages", type=int, default=50, help="单份 PDF 最多处理页数（默认 50）")
    parser.add_argument("--append", action="store_true", help="追加写入 output（默认覆盖）")
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

    logger.info("开始抽取会议资料")
    config = load_config(config_path)
    fields, comments = load_fields_and_comments(requirement_path)
    logger.info("已加载 config 与 requirement，共 %d 个字段", len(fields))

    pdfs = list(iter_pdfs(input_path))
    if args.skip > 0:
        pdfs = pdfs[args.skip:]
        logger.info("已跳过前 %d 份，剩余 %d 份", args.skip, len(pdfs))
    if args.limit and args.limit > 0:
        pdfs = pdfs[: args.limit]

    # 断点续跑：跳过 output 中已有 filename 的 PDF
    if args.resume and output_path.exists():
        done = load_done_filenames(output_path)
        pdfs = [p for p in pdfs if p.name not in done]
        logger.info("断点续跑: 已有 %d 份已完成，待处理 %d 份", len(done), len(pdfs))
    else:
        logger.info("待处理 PDF: %d 个", len(pdfs))

    if not pdfs:
        logger.warning("未找到任何 PDF，或均已处理完成（--resume 时）。请检查 --input 或 output 中的 filename。")
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

    if args.print_sample and pdfs:
        with output_path.open("r", encoding="utf-8") as f:
            first_line = f.readline()
            if first_line:
                print(json.dumps(json.loads(first_line), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
