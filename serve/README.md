# 公告抽取服务

统一 HTTP 服务，按 `type` 参数调用不同公告类别的字段抽取。

## 启动

在项目根目录下执行：

```bash
# 方式一：使用 main.py（推荐）
python main.py

# 指定端口和地址
python main.py --port 8010 --host 0.0.0.0

# 带脚本参数（作为批量抽取默认值）
python main.py --limit 10 --skip 0 --dpi 150 --max-pages 50
python main.py --no-verify-ssl   # 关闭 VL SSL 校验

# 开发模式热重载
python main.py --reload

# 方式二：使用 run.sh
./serve/run.sh
./serve/run.sh 8011
```

## main.py 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| --host | 0.0.0.0 | 监听地址 |
| --port | 8013 | 监听端口 |
| --config | config.yaml | 配置文件路径 |
| --limit | 0 | 批量抽取默认最多处理数量（0=不限制） |
| --skip | 0 | 批量抽取默认跳过前 N 个文件 |
| --dpi | 150 | 默认 PDF 转图 DPI |
| --max-pages | 50 | 默认单份 PDF 最大页数 |
| --no-verify-ssl | false | 关闭 VL 请求 SSL 证书校验 |
| --reload | false | 开发模式热重载 |

## 支持的公告类型 (type)

| type | 说明 |
|------|------|
| esg_report | ESG报告 |
| periodic_report | 定期报告 |
| meeting_materials | 会议资料 |
| ir_qa | 投关问答 |
| inquiry_letters | 闻讯函件 |
| proposal_reference | 议案参考 |
| governance | 治理制度 |

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/types` | 列出支持的 type |
| POST | `/extract` | 上传 PDF，按 type 抽取 |
| POST | `/extract/path` | 按本地路径抽取 |
| POST | `/extract/batch` | 批量抽取目录 |

### 上传抽取 `/extract`

- **参数**: type（必填）, file（必填）, dpi, max_pages, max_qa（仅 ir_qa）, **parallel_workers**, **jpeg_quality**
- **parallel_workers**: 多批时并行 VL 数（0=串行，2~4 可显著缩短多页 PDF 耗时，仅 esg_report 等支持）
- **jpeg_quality**: PDF 转图质量 1-100，越低体积越小、越快（默认 80，可试 65~72）

```bash
curl -X POST "http://localhost:8010/extract?type=esg_report" \
  -F "file=@/path/to/esg_report.pdf"
```

### 路径抽取 `/extract/path`

```json
{
  "path": "data/report/ESG报告/xxx.pdf",
  "type": "esg_report",
  "dpi": 150,
  "max_pages": 50,
  "max_qa": 0
}
```

### 批量抽取 `/extract/batch`

```json
{
  "path": "data/report/ESG报告",
  "type": "esg_report",
  "limit": 10,
  "skip": 0,
  "output": "result/esg_extracted.jsonl",
  "append": false,
  "resume": false,
  "dpi": 150,
  "max_pages": 50,
  "max_qa": 0,
  "no_verify_ssl": false
}
```

- **limit**: 最多处理数量（0=不限制），不传则用 main.py --limit
- **skip**: 跳过前 N 个文件
- **output**: 输出 JSONL 路径，空则只返回不写入
- **append**: 追加写入
- **resume**: 断点续跑，跳过 output 中已有的 filename

```bash
curl -X POST "http://localhost:8010/extract/batch" \
  -H "Content-Type: application/json" \
  -d '{"path": "data/report/ESG报告", "type": "esg_report", "limit": 5, "output": "result/out.jsonl"}'
```

## 依赖

需确保 `config.yaml` 中 `vl` 配置正确，且 VL 模型服务可用。
