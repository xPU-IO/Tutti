# Tutti Agent Skills

`tutti-runtime/` 是一个 agent skill：把 Tutti 的架构约定、bring-up 流程、
不变量与排障方法打包成 agent 可加载的知识，使 agent 无需重新摸索即可在本项目
上工作。

## 安装

复制到 agent 的 skill 目录即可，无需注册或改配置：

```bash
# 用户级（跨所有工作区可用）
cp -r doc/skills/tutti-runtime ~/.codebuddy/skills/

# 或项目级（随仓库共享给协作者）
cp -r doc/skills/tutti-runtime .codebuddy/skills/
```

同一份内容对任何遵循「SKILL.md + YAML frontmatter」约定的 agent 都适用：
frontmatter 的 `name` / `description` 决定何时触发，正文与 `references/` 是纯
Markdown，不依赖任何特定 agent 的运行时。

## 内容

| 文件 | 作用 |
|---|---|
| `SKILL.md` | 仓库布局、5 条不可违背的不变量、bring-up/构建/测试命令、扩展新框架的入口 |
| `references/architecture.md` | 分层职责、单个请求的控制流、内存权威索引、KV 布局、对象池、惰性注册 |
| `references/diagnostics.md` | 排障方法论 + 6 类真实故障的证据链与修法 |
| `references/benchmarking.md` | 压测参数、容量/队列定容规则、运行隔离、从 nsys 导出量化 IO/compute 重叠 |

按渐进披露设计：`description` 常驻上下文，正文在触发时加载，`references/`
按需读取。

## 维护

内容随代码演进会过期，改动以下方面时应同步更新：

- 目录布局或包边界（`SKILL.md` 的 Repository Layout）
- 不变量（尤其调度进程不导入数据面、public include 命名空间）
- bring-up 步骤或 `scripts/tutti-env.sh` 的子命令
- 新增一类值得记录的故障模式（补进 `references/diagnostics.md`，连同证据链）

校验与打包（`skill-creator` 提供的脚本）：

```bash
scripts/package_skill.py doc/skills/tutti-runtime ./dist
```
