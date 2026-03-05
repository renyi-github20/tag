#!/bin/bash
# 在项目根目录下启动公告抽取服务
# 用法: ./serve/run.sh [端口]
# 默认端口: 8010

cd "$(dirname "$0")/.." || exit 1
PORT="${1:-8010}"
echo "启动公告抽取服务: http://0.0.0.0:$PORT"
echo "API 文档: http://localhost:$PORT/docs"
exec python main.py --port "$PORT"
