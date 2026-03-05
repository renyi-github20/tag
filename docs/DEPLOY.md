# 公告抽取服务 - 部署方案

在另一台机器上部署本套代码时，可根据环境与运维习惯选择以下方案之一。

---

## 前置条件（所有方案通用）

- **Python**：3.10+（推荐 3.11）
- **配置文件**：将当前机器的 `config.yaml` 拷贝到新机器，并按新环境修改（尤其是 `vl.api_url`、数据库、ES 等地址）
- **VL 服务**：抽取依赖视觉语言模型接口，需确保 `config.yaml` 中 `vl.api_url` 在新机器可访问（同机部署或网络可达）

---

## 方案一：Docker 单容器（推荐，环境一致）

**适用**：新机器已装 Docker，希望环境一致、易迁移。

**步骤**：

1. 在项目根目录构建镜像：
   ```bash
   docker build -t tag-extract:latest -f docker/Dockerfile .
   ```

2. 运行容器（挂载配置与数据目录）：
   ```bash
   docker run -d --name tag-extract \
     -p 8013:8013 \
     -v /path/on/host/config.yaml:/app/config.yaml:ro \
     -v /path/on/host/data:/app/data \
     -v /path/on/host/result:/app/result \
     tag-extract:latest
   ```

3. 健康检查：`curl http://localhost:8013/health`

**优点**：环境隔离、与宿主机 Python 版本无关；**缺点**：需安装 Docker。

---

## 方案二：Docker Compose（多服务一起跑）

**适用**：新机器上同时部署 VL 推理服务、PostgreSQL、Elasticsearch 等，希望一键启停。

**步骤**：

1. 将 `config.yaml` 放到与 `docker-compose.yaml` 同目录（或按 compose 中 volume 路径放置）。
2. 在项目根目录执行：
   ```bash
   docker compose -f docker/docker-compose.yaml up -d
   ```
3. 仅启动抽取服务（不启动其他服务）时，可：
   ```bash
   docker compose -f docker/docker-compose.yaml up -d extract
   ```

**优点**：服务编排清晰，便于扩展；**缺点**：需维护 compose 与各服务配置。

---

## 方案三：Systemd + Python 虚拟环境（无 Docker）

**适用**：不想用 Docker，或机器已有 Python/venv 运维规范。

**步骤**：

1. 在新机器克隆代码到目标目录，例如 `/opt/tag`。
2. 创建虚拟环境并安装依赖：
   ```bash
   cd /opt/tag
   python3 -m venv .venv
   .venv/bin/pip install -r serve/requirements.txt
   ```
3. 将 `config.yaml` 放到 `/opt/tag/config.yaml` 并修改为当前环境。
4. 按需修改 `deploy/tag-extract.service` 中的 `User`、`Group` 和 `WorkingDirectory`（若未使用 `/opt/tag`），然后安装 systemd 单元：
   ```bash
   sudo cp deploy/tag-extract.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now tag-extract
   ```
5. 查看状态：`sudo systemctl status tag-extract`；日志：`journalctl -u tag-extract -f`。

**优点**：不依赖 Docker，与现有 systemd 体系一致；**缺点**：需自行保证 Python 版本与依赖一致。

---

## 方案四：手动 / 脚本部署（最轻量）

**适用**：临时环境或快速验证，不需要常驻服务。

**步骤**：

1. 拷贝整个项目到新机器（或 git clone）。
2. 准备 `config.yaml` 并修改 `vl.api_url` 等。
3. 安装依赖并启动：
   ```bash
   cd /path/to/tag
   python3 -m venv .venv && .venv/bin/pip install -r serve/requirements.txt
   .venv/bin/python main.py --port 8013 --host 0.0.0.0
   ```
4. 需后台运行时可用 `nohup` 或 `screen`/`tmux`：
   ```bash
   nohup .venv/bin/python main.py --port 8013 --host 0.0.0.0 >> /var/log/tag-extract.log 2>&1 &
   ```

**优点**：零额外组件；**缺点**：进程不随系统自启，需自行做日志与重启管理。

---

## 方案对比小结

| 方案           | 环境一致性 | 常驻/自启 | 多服务编排 | 适用场景           |
|----------------|------------|-----------|------------|--------------------|
| Docker 单容器  | 高         | 需自行配  | 否         | 单机、希望环境一致 |
| Docker Compose | 高         | 是        | 是         | 整栈一起部署       |
| Systemd + venv | 中         | 是        | 否         | 无 Docker、生产机 |
| 手动/脚本      | 低         | 否        | 否         | 临时/验证          |

**建议**：有 Docker 优先用 **方案一**；若新机器要同时跑 VL、数据库等，用 **方案二**；不能或不想用 Docker 时用 **方案三**。

---

## 部署后检查

- `GET http://<host>:8013/health` 返回正常。
- `GET http://<host>:8013/types` 返回支持的 type 列表。
- 使用小 PDF 调用 `POST /extract?type=esg_report` 做一次抽取验证。

修改端口时，请同时调整启动参数（如 `main.py --port 8010`）或 Docker/systemd 中的端口映射与配置。

---

## vLLM 视觉模型服务（Qwen3-VL-32B）

在 4 张 GPU 上部署 2 个 vLLM 实例，支持崩溃自动重启。

### 安装 vLLM

```bash
pip install -U vllm
pip install qwen-vl-utils==0.0.14
```

### 方式一：Systemd 托管（推荐，断线自动重启）

1. 确认模型路径：`/home/azureuser/models/Qwen3-VL-32B-Instruct-FP8`
2. 若 vLLM 在 venv 中，修改 service 文件中的 `ExecStart`，将 `vllm` 改为 venv 的绝对路径，例如：
   ```
   ExecStart=/home/azureuser/venv-vllm/bin/vllm serve ...
   ```
3. 安装并启用服务：
   ```bash
   sudo cp deploy/vllm-qwen-32b-fp8-1.service deploy/vllm-qwen-32b-fp8-2.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now vllm-qwen-32b-fp8-1 vllm-qwen-32b-fp8-2
   ```
4. 查看状态：`sudo systemctl status vllm-qwen-32b-fp8-1 vllm-qwen-32b-fp8-2`
5. 查看日志：`journalctl -u vllm-qwen-32b-fp8-1 -f`

**说明**：`Restart=on-failure` 会在进程异常退出时自动重启，`RestartSec=30` 为重启间隔。

### 方式二：脚本手动启动

```bash
./serve/vllm_qwen32b.sh
```

### 负载均衡

若需对外提供单一入口，可用 nginx 配置 upstream 指向 `127.0.0.1:8003` 和 `127.0.0.1:8004`。
