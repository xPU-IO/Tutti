# tests/ — 按测试需求分类（唯一测试落点）

规则（ARCHITECTURE.md §0a-5）：新功能往既有类别里补用例；
**不建 per-任务 / per-round 测试目录**。

| 目录 | 测试需求 | 依赖 |
|------|----------|------|
| `contract/` | 读写接口契约：StorageBackend SPI 契约套件（所有 backend 跑同一套）+ engine 语义（计划态/环形窗口/背压/句柄） | 无盘无卡 |
| `unit/` | 纯逻辑组件（ChunkIndex 等） | 无盘无卡 |
| `adapter/` | vllm 适配层（connector/scheduler/worker 回调） | vllm |
| `kernels/` | gather/scatter GPU kernel 精度 | GPU |
| `bindings/` | tutti_runtime pybind 绑定（stub runtime） | tutti 构建树 |
| `perf/` | 性能基准（store/load 带宽、kernel 时延） | 按需 |
| `scale/` | 海量对象（十万级 chunk 索引/路径池、LRU 压力） | 按需 |
| `overlap/` | IO 与计算重叠模拟（事件链/波次调度） | 按需 |

运行：`source /data/home/ryeqiu/env-tutti.sh &&
cd integration/vllm-connector && python -m pytest tests/<类别> -v`
（perf/scale/overlap 不进默认快测，按需单独运行）。
