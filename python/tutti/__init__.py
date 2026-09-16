"""Tutti：GPU 直驱 NVMe 的 KV cache 卸载运行时。

包布局（`doc/` 的架构约定）：

- ``tutti.common`` / ``tutti.index`` / ``tutti.engine`` / ``tutti.storage``
  推理框架无关的核心：chunk 索引、读写计划编排、存储后端。
- ``tutti.integration.<framework>``
  框架专属适配层。目前只有 ``vllm``；新增框架只在此目录下扩展，核心层不动。

**本文件必须零副作用**：不得 import 任何子模块。调度进程（vLLM 的
scheduler）只用元数据侧，一旦这里导入数据面模块，就会把 CUDA/pybind
绑定与设备驱动依赖拉进调度进程——那是 I2 红线。子模块请显式按需导入，
例如 ``from tutti.engine.core import KVEngine``。
"""

__all__: list[str] = []
