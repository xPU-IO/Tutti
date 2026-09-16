"""``tutti.storage``：KV 存储后端与插件注册表。

``registry`` 提供两套注册面：数据面（类对象，worker 进程）与调度侧仅元数据
（``"module:Class"`` 惰性字符串，避免调度进程导入数据面绑定）。
"""
