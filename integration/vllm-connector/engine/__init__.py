# 本文件保留作防御：mooncake_transfer_engine 曾在 site-packages 顶层放置
# 同名 C 扩展 engine.so（2026-09-09 已随 mooncake 卸载）。PEP 420 命名空间包
# （无 __init__.py 的目录）在 sys.path 扫描中优先级低于任何位置的常规模块，
# 无论 PYTHONPATH 顺序如何都会被顶层 engine 模块抢占，导致
# `from engine.core import KVEngine` 报 "'engine' is not a package"。保持本目录
# 为常规包可杜绝任何同名顶层模块（含未来重装的 mooncake）再次抢占。
# adapter/index/stores 无已知同名冲突，维持命名空间包、不需要 __init__.py。
