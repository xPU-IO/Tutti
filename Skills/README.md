# Tutti Agent Skills

`tutti-runtime/` 是一个 agent skill：把 Tutti 的编译、运行、测试流程与架构约定、
排障方法打包成 agent 可加载的知识。目标是**任何用户或 agent 拿到这个仓库，只依赖
这个 skill 就能从零编译、拉起环境、跑通 vLLM KV offload 测试**。

## 安装

复制到 agent 的 skill 目录即可，无需注册或改配置：

```bash
# 用户级（跨所有工作区可用）
cp -r Skills/tutti-runtime ~/.codebuddy/skills/

# 或项目级（随仓库共享给协作者）
mkdir -p .codebuddy/skills && cp -r Skills/tutti-runtime .codebuddy/skills/
```

同一份内容对任何遵循「`SKILL.md` + YAML frontmatter」约定的 agent 都适用：
frontmatter 的 `name` / `description` 决定何时触发，正文与 `references/` 是纯
Markdown，不依赖任何特定 agent 的运行时。

## 内容

| 文件 | 作用 |
|---|---|
| `SKILL.md` | `scripts/tutti-env.sh` 单一入口、从零 clone 到跑 benchmark、5 条不可违背的不变量、仓库布局、扩展新框架 |
| `references/architecture.md` | 分层职责、单请求控制流、内存权威索引、KV 布局、对象池、惰性注册 |
| `references/diagnostics.md` | 排障方法论 + 6 类真实故障的证据链与修法 |
| `references/benchmarking.md` | 压测参数、容量/队列定容规则、运行隔离、从 nsys 导出量化 IO/compute 重叠 |

按渐进披露组织：`description` 常驻上下文，正文在触发时加载，`references/`
按需读取。

## 与 scripts/tutti-env.sh 的关系

skill 不复制脚本的实现细节，只说明**何时用哪个子命令**以及**为什么顺序不能换**。
脚本自身是可执行的真相来源：

```bash
scripts/tutti-env.sh --help
```

因此脚本改了子命令时，`SKILL.md` 的 "One Entry Point" 一节需同步。

## 维护

内容随代码演进会过期。改动以下方面时应同步更新：

- 目录布局或包边界 → `SKILL.md` 的 Repository Layout
- 不变量（尤其"调度进程不导入数据面"、"public include 是命名空间"）
- `scripts/tutti-env.sh` 的子命令或选项
- 新增一类值得记录的故障模式 → 补进 `references/diagnostics.md`，**连同证据链**
  （只写结论的排障条目价值很低）

校验与打包（若安装了 skill-creator 的脚本）：

```bash
python <skill-creator>/scripts/package_skill.py Skills/tutti-runtime ./dist
```
