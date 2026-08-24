#!/usr/bin/env bash
# vLLM 冒烟服务端启动脚本（Hy3-FP8 × TP4 × 每副本一块本地盘）。
# 用法：source /data/home/ryeqiu/env-tutti.sh && bash scripts/vllm_smoke_server.sh
set -euo pipefail

CONNECTOR="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
MODEL=/data2/tencent/Hy3-FP8
PORT=8192

export PYTHONPATH="$CONNECTOR:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 几何为 per-rank 值（TP4 下每 rank 2 KV heads → 每 token 每层 1024B）：
# segment = 256 × 1024 = 256 KiB；chunk_kv = 80 × 256 KiB = 20 MiB；
# 池 512 chunk × 20 MiB = 10 GiB/盘。卡内 1MiB/80MiB 为 8-head 全量口径，
# 按 TP 切分实测修正（回执记录）。
# 部署形态（IO 路径实测后修正）：runtime 组装当前要求进程视角 0 号卡
# （gpu_id 非 0 组装必败，见 results/cpp-gap），全副本统一 gpu_id=0；
# 4 副本数据分片以 per-rank 子目录隔离（chunk key 全局同 key，同 root
# 会互相覆盖）；staging 为页锁定主存（注册不受 GPU 匹配校验限制）。
# 单盘（nvme0）承载 4 副本池：条带卷（striped）布局待 cpp-gap 修复后
# 切换——届时落盘分布由 tutti 条带机制决定。
exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name smoke \
    --tensor-parallel-size 4 \
    --block-size 64 \
    --enforce-eager \
    --max-model-len 32768 \
    --load-format phxsafetensors \
    --port "$PORT" \
    --kv-transfer-config '{
      "kv_connector": "TuttiConnectorV1",
      "kv_connector_module_path": "adapter.connector",
      "kv_role": "kv_both",
      "kv_connector_extra_config": {
        "chunk_tokens": 256,
        "chunk_kv_bytes": 20971520,
        "max_chunks_per_wave": 512,
        "num_layers": 80,
        "store": {
          "type": "tutti_nvme",
          "options": {
            "root": "/mnt/nvme0/tutti-kv-pool-r{LOCAL_RANK}",
            "num_chunks": 512,
            "segment_bytes": 262144,
            "io_stream": "auto",
            "preset": {
              "type": "local",
              "daemon_config": "/data/home/ryeqiu/Tutti/config/local/tutti_daemon.yaml",
              "device_id": 0,
              "gpu_id": 0
            }
          }
        }
      }
    }'
