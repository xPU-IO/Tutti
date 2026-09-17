"""``tutti.integration``：推理框架专属适配层。

每个框架一个子包（当前只有 ``vllm``）。新增框架只在此目录下扩展，
``tutti.{common,index,engine,storage}`` 保持框架无关。
"""
