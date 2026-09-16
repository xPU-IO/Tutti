"""``tutti.integration.vllm``：vLLM v1 KVConnector 适配层。

- ``connector``  ``TuttiConnectorV1``，调度侧与 worker 侧双角色壳
- ``worker``     worker 侧逐层读写编排
- ``worker_meta``worker → scheduler 的增量回传载荷
- ``geometry``   从 vLLM 配置推导 KV 几何与部署参数
- ``factory``    进程级实例装配

本文件不 import 子模块：``worker`` 会拉入数据面依赖，而调度进程只需
``connector``（其内部按角色惰性装配）。
"""
