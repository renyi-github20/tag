#!/usr/bin/env python3
"""
按文件名中的日期，将 data/report_new 下的文件重新整理到 2025-08、2025-09、2025-10 等月份文件夹及对应类型子文件夹中。

文件名格式: {ID}_{YYYY-MM-DD}_{描述}.pdf
"""

import re
import shutil
from pathlib import Path
from typing import Optional

DATA_ROOT = Path(__file__).resolve().parent / "data" / "report_new"

# 类型文件夹（与 202508-202510 平级的分类）
CATEGORIES = [
    "ESG报告",
    "会议资料",
    "定期报告",
    "投关问答",
    "治理制度",
    "议案参考",
    "问询函件",
]

# 文件名中的日期正则: ID_YYYY-MM-DD_描述
DATE_PATTERN = re.compile(r"^\d+_(\d{4}-\d{2}-\d{2})_.*\.(pdf|PDF)$")


def extract_month_from_filename(filename: str) -> Optional[str]:
    """从文件名提取 YYYY-MM，如 2025-09。无法解析则返回 None。"""
    m = DATE_PATTERN.match(filename)
    if not m:
        return None
    date_str = m.group(1)  # YYYY-MM-DD
    return date_str[:7]  # YYYY-MM


def main() -> None:
    moved = 0
    skipped = 0
    errors = []

    for category in CATEGORIES:
        src_dir = DATA_ROOT / category / "202508-202510"
        if not src_dir.exists():
            print(f"跳过（目录不存在）: {src_dir}")
            continue

        for f in src_dir.iterdir():
            if not f.is_file():
                continue
            if f.suffix.upper() != ".PDF":
                continue

            month = extract_month_from_filename(f.name)
            if not month:
                errors.append(f"无法解析日期: {f}")
                skipped += 1
                continue

            dest_dir = DATA_ROOT / month / category
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / f.name

            if dest_file.exists() and dest_file.resolve() == f.resolve():
                continue
            if dest_file.exists():
                errors.append(f"目标已存在，跳过: {dest_file}")
                skipped += 1
                continue

            try:
                shutil.move(str(f), str(dest_file))
                moved += 1
                if moved % 500 == 0:
                    print(f"已移动 {moved} 个文件...")
            except Exception as e:
                errors.append(f"移动失败 {f}: {e}")
                skipped += 1

    print(f"\n完成: 移动 {moved} 个文件, 跳过 {skipped} 个")
    if errors:
        print("\n错误/跳过详情（前 20 条）:")
        for e in errors[:20]:
            print(f"  {e}")
        if len(errors) > 20:
            print(f"  ... 共 {len(errors)} 条")


if __name__ == "__main__":
    main()
