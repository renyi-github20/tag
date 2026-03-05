# 公告抽取服务 - 单容器镜像
# 构建: 在项目根目录执行 docker build -t tag-extract:latest -f docker/Dockerfile .

FROM python:3.11-slim

WORKDIR /app

# 系统依赖（PyMuPDF 等可能需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

# 项目文件（按需通过 .dockerignore 排除大文件/缓存）
COPY serve/requirements.txt serve/
COPY main.py config.yaml ./
COPY serve/ serve/
COPY scripts/ scripts/
# 需求文件为启动必需；大体积 data/report 等可通过 .dockerignore 排除后挂载
COPY data/ data/

RUN pip install --no-cache-dir -r serve/requirements.txt

# 默认端口与 main.py 一致
ENV PORT=8013
EXPOSE 8013

# 使用环境变量覆盖端口时需在 run 时传 -e PORT=8010 并 -p 8010:8010
CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8013"]
