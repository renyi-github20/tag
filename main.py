#!/usr/bin/env python3
"""
公告抽取服务 - 统一入口。

支持多种公告类型的字段抽取，通过 type 参数区分：
  esg_report        ESG报告
  periodic_report   定期报告
  meeting_materials 会议资料
  ir_qa             投关问答
  inquiry_letters   闻讯函件
  proposal_reference 议案参考
  governance        治理制度

启动方式（在项目根目录下）:
    python serve/main.py
    python serve/main.py --port 8010 --host 0.0.0.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="公告抽取服务 - 支持 ESG报告、定期报告、会议资料、投关问答、闻讯函件、议案参考、治理制度"
    )
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8013, help="监听端口")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径（相对项目根）")
    parser.add_argument("--reload", action="store_true", help="开发模式热重载")
    parser.add_argument("--limit", type=int, default=0, help="批量抽取默认最多处理数量（0=不限制）")
    parser.add_argument("--skip", type=int, default=0, help="批量抽取默认跳过前 N 个文件")
    parser.add_argument("--dpi", type=int, default=150, help="默认 PDF 转图 DPI")
    parser.add_argument("--max-pages", type=int, default=50, help="默认单份 PDF 最大页数")
    parser.add_argument("--no-verify-ssl", action="store_true", help="关闭 VL 请求 SSL 证书校验")
    args = parser.parse_args()

    config_path = _PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"错误: 配置文件不存在 {config_path}", file=sys.stderr)
        sys.exit(1)

    # 将默认参数传入 app（批量抽取未指定时使用）
    import os
    os.environ["EXTRACT_DEFAULT_LIMIT"] = str(args.limit)
    os.environ["EXTRACT_DEFAULT_SKIP"] = str(args.skip)
    os.environ["EXTRACT_DEFAULT_DPI"] = str(args.dpi)
    os.environ["EXTRACT_DEFAULT_MAX_PAGES"] = str(args.max_pages)
    os.environ["EXTRACT_NO_VERIFY_SSL"] = "1" if args.no_verify_ssl else "0"

    import uvicorn
    uvicorn.run(
        "serve.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
