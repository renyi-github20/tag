#!/usr/bin/env python3
"""
只从 kg_data/data 抽取「定期报告」，不处理其他类型（ESG、会议资料、投关问答等）。

数据源: {data_root}/{YYYY-MM}/定期报告/
输出:   result/{YYYY-MM}/定期报告/result_periodic_report.jsonl
       result/{YYYY-MM}/定期报告/result_periodic_report.csv

用法:
    python scripts/extract_kg_periodic.py
    python scripts/extract_kg_periodic.py --data-root /path/to/kg_data/data
    python scripts/extract_kg_periodic.py --limit 5
    python scripts/extract_kg_periodic.py --max-total 9000
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 默认 kg_data 路径：只抽取该目录下各月份的「定期报告」，不处理其他类型
KG_DATA_ROOT = Path("/home/azure/workspace/data/kg_data/data")
if not KG_DATA_ROOT.exists():
    KG_DATA_ROOT = Path("/workspace/data/kg_data/data")
if not KG_DATA_ROOT.exists():
    KG_DATA_ROOT = _PROJECT_ROOT.parent / "data" / "kg_data" / "data"
if not KG_DATA_ROOT.exists():
    KG_DATA_ROOT = Path("/home/azureuser/workspace/data/kg_data/data")
RESULT_ROOT = _PROJECT_ROOT / "result"
TYPE_KEY = "periodic_report"
FOLDER = "定期报告"


def jsonl_to_csv(jsonl_path: Path, csv_path: Path) -> int:
    """将 JSONL 转为 CSV，返回行数。"""
    if not jsonl_path.exists():
        return 0
    rows: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if isinstance(rec, dict) and "_error" not in rec:
                    rows.append(rec)
            except json.JSONDecodeError:
                continue
    if not rows:
        return 0
    all_keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen and k != "_error":
                seen.add(k)
                all_keys.append(k)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="从 kg_data 抽取 2025 年定期报告")
    parser.add_argument("--data-root", type=str, default=None, help=f"kg_data 根目录，默认 {KG_DATA_ROOT}")
    parser.add_argument("--months", nargs="+", default=None, help="月份列表，如 2025-01 2025-02，默认全部 2025 年")
    parser.add_argument("--limit", type=int, default=0, help="每类最多处理数量，0=不限制")
    parser.add_argument("--max-total", type=int, default=9000, help="总共最多处理文件数，0=不限制，默认 9000")
    parser.add_argument("--skip", type=int, default=0, help="每类跳过前 N 个")
    parser.add_argument("--no-resume", action="store_true", help="不使用断点续跑")
    parser.add_argument("--no-csv", action="store_true", help="不生成 CSV")
    parser.add_argument("--workers", type=int, default=0, help="并行处理的 PDF 数")
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else KG_DATA_ROOT
    if not data_root.exists():
        print(f"数据目录不存在: {data_root}")
        return 1

    months = args.months
    if not months:
        months = sorted(p.name for p in data_root.iterdir() if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}", p.name))
        months.reverse()  # 时间倒序：新月份优先
    if not months:
        print("未找到月份目录（YYYY-MM 格式）")
        return 1

    # 导入并复用 extract_batch_by_month 的抽取逻辑
    from scripts.extract_batch_by_month import run_extract_batch

    total_success = 0
    total_processed = 0
    max_total = args.max_total

    for month in months:
        if max_total > 0 and total_processed >= max_total:
            break
        month_data = data_root / month
        input_path = month_data / FOLDER
        if not input_path.exists():
            print(f"跳过（不存在）: {input_path}")
            continue

        remaining = max_total - total_processed if max_total > 0 else 0

        output_dir = RESULT_ROOT / month / FOLDER
        output_jsonl = output_dir / f"result_{TYPE_KEY}.jsonl"
        output_csv = output_dir / f"result_{TYPE_KEY}.csv"

        print(f"\n>>> {month} / {FOLDER}")
        try:
            n, num_processed = run_extract_batch(
                input_path,
                TYPE_KEY,
                output_jsonl,
                limit=args.limit,
                max_per_type=0,
                max_files=remaining if max_total > 0 else 0,
                skip=args.skip,
                resume=not args.no_resume,
                workers=args.workers,
            )
            total_success += n
            total_processed += num_processed
            print(f"    抽取完成: {n} 条 -> {output_jsonl}")

            if not args.no_csv and output_jsonl.exists():
                csv_rows = jsonl_to_csv(output_jsonl, output_csv)
                print(f"    CSV 生成: {csv_rows} 行 -> {output_csv}")
        except Exception as e:
            print(f"    错误: {e}")
            raise

    msg = f"\n总计成功: {total_success} 条"
    if max_total > 0:
        msg += f"，处理文件数: {total_processed}/{max_total}"
    print(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
