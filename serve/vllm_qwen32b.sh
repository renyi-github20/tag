#!/bin/bash
# 在 4 张 GPU 上启动 2 个 Qwen3-VL-32B-Instruct-FP8 vLLM 实例
# 实例 1: GPU 0,1 (tensor-parallel-size 2) -> 端口 8003
# 实例 2: GPU 2,3 (tensor-parallel-size 2) -> 端口 8004
#
# 用法:
#   前台启动两个实例: ./serve/vllm_qwen32b.sh
#   后台启动:         nohup ./serve/vllm_qwen32b.sh > vllm.log 2>&1 &
#
# 如需负载均衡，可在 nginx 等反向代理中配置 upstream 指向 8003 和 8004

set -e
MODEL_PATH="${MODEL_PATH:-/home/azureuser/models/Qwen3-VL-32B-Instruct-FP8}"
PORT1="${PORT1:-8003}"
PORT2="${PORT2:-8004}"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "错误: 模型路径不存在: $MODEL_PATH"
  exit 1
fi

# 多实例部署时建议限制 OMP 线程，减少 CPU 争用
export OMP_NUM_THREADS=1

# 实例 1: GPU 0,1
echo "启动实例 1 (GPU 0,1) -> 端口 $PORT1 ..."
CUDA_VISIBLE_DEVICES=0,1 vllm serve "$MODEL_PATH" \
  --tensor-parallel-size 2 \
  --port "$PORT1" \
  --host 0.0.0.0 \
  --limit-mm-per-prompt '{"video":0}' \
  --max-model-len 131072 \
  --served-model-name "Qwen/Qwen3-VL-32B-Instruct-FP8" \
  "$@" &
PID1=$!

# 实例 2: GPU 2,3
echo "启动实例 2 (GPU 2,3) -> 端口 $PORT2 ..."
CUDA_VISIBLE_DEVICES=2,3 vllm serve "$MODEL_PATH" \
  --tensor-parallel-size 2 \
  --port "$PORT2" \
  --host 0.0.0.0 \
  --limit-mm-per-prompt '{"video":0}' \
  --max-model-len 131072 \
  --served-model-name "Qwen/Qwen3-VL-32B-Instruct-FP8" \
  "$@" &
PID2=$!

echo "实例 1 PID: $PID1, 实例 2 PID: $PID2"
echo "API 地址: http://0.0.0.0:$PORT1/v1 和 http://0.0.0.0:$PORT2/v1"
echo "按 Ctrl+C 停止两个实例"

wait
