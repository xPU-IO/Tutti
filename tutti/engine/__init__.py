"""``tutti.engine``：读写计划编排、传输后端与完成句柄（框架无关）。

原为顶层 ``engine`` 包，需靠本文件成为常规包以防被同名顶层模块抢占
（mooncake_transfer_engine 曾在 site-packages 顶层放置 ``engine.so``，
2026-09-09 随 mooncake 卸载）；现已收进 ``tutti`` 命名空间，顶层不再有
``engine`` 这一名字，该类抢占已结构性消除。

本文件不 import 子模块：``tutti.engine.core`` 会拉入数据面依赖，而调度
进程只用 ``tutti.engine.metadata``。
"""
