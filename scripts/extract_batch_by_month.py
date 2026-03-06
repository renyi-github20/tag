#!/usr/bin/env python3
"""
按月份和类型批量抽取，输出到 result/{YYYY-MM}/{类型}/ 下。

数据源: data/report_new/{YYYY-MM}/{类型}/
输出:   result/{YYYY-MM}/{类型}/result_{type_key}.jsonl
       result/{YYYY-MM}/{类型}/result_{type_key}.csv

用法:
    python scripts/extract_batch_by_month.py                    # 处理所有月份、所有类型
    python scripts/extract_batch_by_month.py --months 2025-08   # 仅 2025-08
    python scripts/extract_batch_by_month.py --types esg_report  # 仅 ESG 报告
    python scripts/extract_batch_by_month.py --limit 5           # 每类最多 5 个文件（测试用）
    python scripts/extract_batch_by_month.py --max-total 9000    # 总共最多处理 9000 个文件（默认）
    python scripts/extract_batch_by_month.py --workers 2         # 2 个 PDF 并行抽取（默认取 config vl.parallel_workers）
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

DATA_ROOT = _PROJECT_ROOT / "data" / "report_new"
RESULT_ROOT = _PROJECT_ROOT / "result"

# type_key -> 数据目录中的类型文件夹名
TYPE_TO_FOLDER: dict[str, str] = {
    "esg_report": "ESG报告",
    "periodic_report": "定期报告",
    "meeting_materials": "会议资料",
    "ir_qa": "投关问答",
    "inquiry_letters": "问询函件",
    "proposal_reference": "议案参考",
    "governance": "治理制度",
}


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
    # 合并所有键作为表头
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


def _extract_one(
    file_path: Path,
    extract_fn,
    returns_list: bool,
    config: dict,
    req_path: Path,
) -> tuple[Path, list[dict] | dict | None, Exception | None]:
    """抽取单个文件，返回 (file_path, result, error)。"""
    try:
        result = extract_fn(
            file_path,
            config,
            config.get("vl", {}).get("dpi", 150),
            config.get("vl", {}).get("max_pages", 50),
            requirement_path=req_path,
            max_qa=0,
            parallel_workers=config.get("vl", {}).get("parallel_workers", 0),
            jpeg_quality=config.get("vl", {}).get("jpeg_quality", 80),
        )
        if returns_list:
            recs = [r for r in result if isinstance(r, dict)]
            return file_path, recs, None
        return file_path, result if isinstance(result, dict) else None, None
    except Exception as e:
        return file_path, None, e


def _merge_ir_qa_records(records: list[dict], req_path: Path) -> dict | None:
    """投关问答：同一文件内多条 Q&A 合并为一条，提问内容/回复内容为列表。"""
    if not records:
        return None
    from scripts.extract_ir_qa import load_fields_and_comments, merge_records_to_one
    fields, _ = load_fields_and_comments(req_path)
    return merge_records_to_one(records, fields)


def run_extract_batch(
    input_path: Path,
    type_key: str,
    output_jsonl: Path,
    *,
    limit: int = 0,
    max_per_type: int = 0,
    max_files: int = 0,
    skip: int = 0,
    resume: bool = True,
    workers: int = 0,
) -> tuple[int, int]:
    """调用 serve 的批量抽取逻辑，返回 (成功数, 本批处理文件数)。workers>1 时并行处理多个 PDF。"""
    from serve.app import (
        _get_config,
        _get_extractor,
        _iter_input_files,
        _load_done_filenames,
    )

    config = _get_config()
    if "vl" not in config or not config.get("vl", {}).get("enable", True):
        raise RuntimeError("VL 未启用，请检查 config.yaml")

    files = list(_iter_input_files(input_path, type_key))
    if skip > 0:
        files = files[skip:]
    # limit>0 优先；否则 max_per_type>0 时作为上限；max_files>0 为全局配额，取更小值
    effective_limit = limit if limit > 0 else (max_per_type if max_per_type > 0 else 0)
    if max_files > 0:
        effective_limit = min(effective_limit, max_files) if effective_limit > 0 else max_files
    if effective_limit > 0:
        files = files[:effective_limit]

    if resume and output_jsonl.exists():
        done = _load_done_filenames(output_jsonl)
        files = [p for p in files if p.name not in done]

    num_to_process = len(files)
    if not files:
        return 0, 0

    extract_fn, returns_list = _get_extractor(type_key)
    req_path = Path(config.get("_requirement_path", _PROJECT_ROOT / "data" / "requirement" / "requirement_1.json"))
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    merge_ir_qa = type_key == "ir_qa" and returns_list
    mode = "a" if resume and output_jsonl.exists() else "w"

    if workers <= 0:
        vl = config.get("vl", {})
        urls = vl.get("api_urls")
        if urls and isinstance(urls, list) and len(urls) > 1:
            workers = len(urls)  # 多实例时，并行数 = 实例数
        else:
            workers = vl.get("parallel_workers", 1)
    workers = max(1, min(workers, len(files)))
    if workers > 1:
        print(f"    并行数: {workers} 个 PDF 同时处理")

    write_lock = threading.Lock()
    success = 0

    # 首次创建/清空输出文件
    with output_jsonl.open(mode, encoding="utf-8") as out:
        pass

    if workers == 1:
        for file_path in files:
            _, result, err = _extract_one(file_path, extract_fn, returns_list, config, req_path)
            if err:
                print(f"  [失败] {file_path.name}: {err}")
                continue
            with output_jsonl.open("a", encoding="utf-8") as out:
                if returns_list and isinstance(result, list):
                    if merge_ir_qa:
                        merged = _merge_ir_qa_records(result, req_path)
                        if merged:
                            out.write(json.dumps(merged, ensure_ascii=False) + "\n")
                            success += 1
                    else:
                        for rec in result:
                            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            success += 1
                elif isinstance(result, dict):
                    out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    success += 1
                out.flush()
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_extract_one, fp, extract_fn, returns_list, config, req_path): fp for fp in files}
            for future in as_completed(futures):
                file_path, result, err = future.result()
                if err:
                    print(f"  [失败] {file_path.name}: {err}")
                    continue
                if result is None:
                    continue
                with write_lock:
                    with output_jsonl.open("a", encoding="utf-8") as out:
                        if returns_list and isinstance(result, list):
                            if merge_ir_qa:
                                merged = _merge_ir_qa_records(result, req_path)
                                if merged:
                                    out.write(json.dumps(merged, ensure_ascii=False) + "\n")
                                    success += 1
                            else:
                                for rec in result:
                                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                    success += 1
                        elif isinstance(result, dict):
                            out.write(json.dumps(result, ensure_ascii=False) + "\n")
                            success += 1
                        out.flush()

    return success, num_to_process


def main() -> int:
    parser = argparse.ArgumentParser(description="按月份和类型批量抽取")
    parser.add_argument("--months", nargs="+", default=None, help="月份列表，如 2025-06 2025-07，默认全部")
    parser.add_argument("--types", nargs="+", default=None, help="类型列表，如 esg_report governance，默认全部")
    parser.add_argument("--limit", type=int, default=0, help="每类最多处理数量，0=不限制")
    parser.add_argument("--max-total", type=int, default=9000, help="总共最多处理文件数，0=不限制，默认 9000")
    parser.add_argument("--skip", type=int, default=0, help="每类跳过前 N 个")
    parser.add_argument("--no-resume", action="store_true", help="不使用断点续跑")
    parser.add_argument("--no-csv", action="store_true", help="不生成 CSV")
    parser.add_argument("--workers", type=int, default=0, help="并行处理的 PDF 数（0=取 config vl.parallel_workers，默认 2）")
    args = parser.parse_args()

    months = args.months
    if not months:
        months = sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.startswith("2025-"))
    if not months:
        print("未找到月份目录，请检查 data/report_new/")
        return 1

    types = args.types or list(TYPE_TO_FOLDER.keys())
    for t in types:
        if t not in TYPE_TO_FOLDER:
            print(f"未知类型: {t}，支持: {list(TYPE_TO_FOLDER.keys())}")
            return 1

    total_success = 0
    total_processed = 0
    max_total = args.max_total
    for month in months:
        if max_total > 0 and total_processed >= max_total:
            break
        month_data = DATA_ROOT / month
        if not month_data.exists():
            print(f"跳过（不存在）: {month_data}")
            continue
        for type_key in types:
            if max_total > 0 and total_processed >= max_total:
                break
            folder = TYPE_TO_FOLDER[type_key]
            input_path = month_data / folder
            if not input_path.exists():
                print(f"跳过（不存在）: {input_path}")
                continue

            remaining = max_total - total_processed if max_total > 0 else 0

            output_dir = RESULT_ROOT / month / folder
            output_jsonl = output_dir / f"result_{type_key}.jsonl"
            output_csv = output_dir / f"result_{type_key}.csv"

            print(f"\n>>> {month} / {folder}")
            try:
                n, num_processed = run_extract_batch(
                    input_path,
                    type_key,
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
