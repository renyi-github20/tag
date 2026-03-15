#!/usr/bin/env python3
"""
从 kg_data/data 按类型抽取，支持所有 7 种文档类型。

数据源: {data_root}/{YYYY-MM}/{类型文件夹}/
输出:   {result_root}/{YYYY-MM}/{类型文件夹}/result_{type_key}.jsonl
       {result_root}/{YYYY-MM}/{类型文件夹}/result_{type_key}.csv

用法:
    python scripts/extract_kg.py --type periodic_report
    python scripts/extract_kg.py --type esg_report --months 2025-04 2025-05
    python scripts/extract_kg.py --type esg_report --start 2025-04 --end 2025-08
    python scripts/extract_kg.py --type ir_qa --data-root /data/kg_data/data
    python scripts/extract_kg.py --type governance --limit 5
    python scripts/extract_kg.py --types periodic_report esg_report   # 多类型
    python scripts/extract_kg.py                                      # 默认: periodic_report

支持的类型:
    esg_report        ESG报告
    periodic_report   定期报告
    meeting_materials 会议资料
    ir_qa             投关问答
    inquiry_letters   问询函件
    proposal_reference 议案参考
    governance        治理制度
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

from scripts.extract_batch_by_month import TYPE_TO_FOLDER

_ALL_TYPES = list(TYPE_TO_FOLDER.keys())


def _resolve_kg_paths(config: dict | None = None) -> tuple[Path, Path]:
    """从 config.yaml 的 paths 段读取 kg_data_root / result_root，空值则回退到项目默认。"""
    paths_cfg = (config or {}).get("paths", {}) or {}
    kg = paths_cfg.get("kg_data_root") or ""
    result = paths_cfg.get("result_root") or ""
    return (
        Path(kg) if kg else _PROJECT_ROOT.parent / "data" / "kg_data" / "data",
        Path(result) if result else _PROJECT_ROOT / "result",
    )


def jsonl_to_csv(jsonl_path: Path, csv_path: Path) -> int:
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


def run_type(
    type_key: str,
    data_root: Path,
    result_root: Path,
    *,
    months: list[str],
    limit: int,
    max_total: int,
    skip: int,
    resume: bool,
    no_csv: bool,
    workers: int,
) -> tuple[int, int]:
    """对单个类型执行按月抽取，返回 (成功条数, 处理文件数)。"""
    from scripts.extract_batch_by_month import run_extract_batch

    folder = TYPE_TO_FOLDER[type_key]
    total_success = 0
    total_processed = 0

    for month in months:
        if max_total > 0 and total_processed >= max_total:
            break
        input_path = data_root / month / folder
        if not input_path.exists():
            print(f"  跳过（不存在）: {input_path}")
            continue

        remaining = max_total - total_processed if max_total > 0 else 0
        output_dir = result_root / month / folder
        output_jsonl = output_dir / f"result_{type_key}.jsonl"
        output_csv = output_dir / f"result_{type_key}.csv"

        print(f"\n>>> {month} / {folder}")
        try:
            n, num_processed = run_extract_batch(
                input_path,
                type_key,
                output_jsonl,
                limit=limit,
                max_per_type=0,
                max_files=remaining if max_total > 0 else 0,
                skip=skip,
                resume=resume,
                workers=workers,
            )
            total_success += n
            total_processed += num_processed
            print(f"    抽取完成: {n} 条 -> {output_jsonl}")

            if not no_csv and output_jsonl.exists():
                csv_rows = jsonl_to_csv(output_jsonl, output_csv)
                print(f"    CSV 生成: {csv_rows} 行 -> {output_csv}")
        except Exception as e:
            print(f"    错误: {e}")
            raise

    return total_success, total_processed


def main() -> int:
    type_help = ", ".join(f"{k}({v})" for k, v in TYPE_TO_FOLDER.items())
    parser = argparse.ArgumentParser(
        description="从 kg_data 按类型抽取",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--type", type=str, default=None, help=f"单个类型: {type_help}")
    parser.add_argument("--types", nargs="+", default=None, help="多个类型，空格分隔")
    parser.add_argument("--data-root", type=str, default=None, help="kg_data 根目录，覆盖 config.yaml paths.kg_data_root")
    parser.add_argument("--result-root", type=str, default=None, help="结果输出根目录，覆盖 config.yaml paths.result_root")
    parser.add_argument("--months", nargs="+", default=None, help="月份列表，如 2025-01 2025-02，默认全部")
    parser.add_argument("--start", type=str, default=None, help="起始月份（含），如 2025-04")
    parser.add_argument("--end", type=str, default=None, help="结束月份（含），如 2025-08")
    parser.add_argument("--limit", type=int, default=0, help="每类最多处理数量，0=不限制")
    parser.add_argument("--max-total", type=int, default=9000, help="总共最多处理文件数，0=不限制，默认 9000")
    parser.add_argument("--skip", type=int, default=0, help="每类跳过前 N 个")
    parser.add_argument("--no-resume", action="store_true", help="不使用断点续跑")
    parser.add_argument("--no-csv", action="store_true", help="不生成 CSV")
    parser.add_argument("--workers", type=int, default=0, help="并行处理的 PDF 数")
    args = parser.parse_args()

    # --type 和 --types 合并；都没指定则默认 periodic_report
    types: list[str] = []
    if args.types:
        types = args.types
    elif args.type:
        types = [args.type]
    else:
        types = ["periodic_report"]

    for t in types:
        if t not in TYPE_TO_FOLDER:
            print(f"未知类型: {t}")
            print(f"支持: {type_help}")
            return 1

    import yaml
    config_path = _PROJECT_ROOT / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    cfg_kg_root, cfg_result_root = _resolve_kg_paths(config)

    data_root = Path(args.data_root) if args.data_root else cfg_kg_root
    result_root = Path(args.result_root) if args.result_root else cfg_result_root
    if not data_root.exists():
        print(f"数据目录不存在: {data_root}")
        return 1

    months = args.months
    if not months:
        months = sorted(
            p.name for p in data_root.iterdir()
            if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}", p.name)
        )
        months.reverse()
    if args.start:
        months = [m for m in months if m >= args.start]
    if args.end:
        months = [m for m in months if m <= args.end]
    if not months:
        print("未找到月份目录（YYYY-MM 格式）")
        return 1

    grand_success = 0
    grand_processed = 0

    for type_key in types:
        print(f"\n{'='*60}")
        print(f"  类型: {type_key} ({TYPE_TO_FOLDER[type_key]})")
        print(f"{'='*60}")
        s, p = run_type(
            type_key,
            data_root,
            result_root,
            months=months,
            limit=args.limit,
            max_total=args.max_total - grand_processed if args.max_total > 0 else 0,
            skip=args.skip,
            resume=not args.no_resume,
            no_csv=args.no_csv,
            workers=args.workers,
        )
        grand_success += s
        grand_processed += p

    msg = f"\n总计成功: {grand_success} 条"
    if args.max_total > 0:
        msg += f"，处理文件数: {grand_processed}/{args.max_total}"
    print(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
