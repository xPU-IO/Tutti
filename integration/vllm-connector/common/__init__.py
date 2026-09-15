# 本文件保留作防御：与 engine/ 同理，PEP 420 命名空间包（无 __init__.py
# 的目录）在 sys.path 扫描中优先级低于任何位置的常规模块。`common` 是
# 极易与第三方库重名的名字，保持本目录为常规包可避免被同名顶层模块抢占
# （`from common.utils import ...` 报 "'common' is not a package"）。
