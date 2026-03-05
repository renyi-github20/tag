#!/usr/bin/env python3
"""
将投关问答 JSONL 中同一文件的多条记录合并为一条：
- 按 filename 分组
- 提问内容、回复内容 转为列表（每条一问一答对应）

用法:
    python scripts/merge_ir_qa_jsonl.py result/2025-10/投关问答/result_ir_qa.jsonl
    python scripts/merge_ir_qa_jsonl.py result/2025-10/投关问答/result_ir_qa.jsonl -o result/2025-10/投关问答/result_ir_qa_merged.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _content_to_list(val) -> list[str]:
    """将提问内容/回复内容统一为 list。"""
    if isinstance(val, list):
        return [str(item).strip() for item in val if str(item).strip()]
    if isinstance(val, str) and val.strip() and val != "未找到":
        return [val.strip()]
    return []


def merge_records_by_filename(records: list[dict], fields: list[str]) -> dict:
    """同一文件多条 Q&A 合并为一条，提问内容/回复内容为二维 list。"""
    if not records:
        return {}
    first = records[0]
    merged: dict = {"filename": first.get("filename", "")}
    for k in fields:
        merged[k] = first.get(k, "未找到")
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


def main() -> int:
    parser = argparse.ArgumentParser(description="合并投关问答 JSONL：同一文件多条记录合并为一条")
    parser.add_argument("input", type=Path, help="输入的 result_ir_qa.jsonl 路径")
    parser.add_argument("-o", "--output", type=Path, default=None, help="输出路径，默认覆盖原文件")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or input_path

    if not input_path.exists():
        print(f"文件不存在: {input_path}")
        return 1

    from scripts.extract_ir_qa import load_fields_and_comments
    req_path = _PROJECT_ROOT / "data" / "requirement" / "requirement_1.json"
    fields, _ = load_fields_and_comments(req_path)

    records: list[dict] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if isinstance(rec, dict) and "filename" in rec:
                    records.append(rec)
            except json.JSONDecodeError:
                continue

    # 按 filename 分组
    from collections import defaultdict
    by_file: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_file[rec.get("filename", "")].append(rec)

    merged_list: list[dict] = []
    for fn, recs in sorted(by_file.items()):
        merged = merge_records_by_filename(recs, fields)
        if merged:
            merged_list.append(merged)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for m in merged_list:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"合并完成: {len(records)} 条 -> {len(merged_list)} 条 -> {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
