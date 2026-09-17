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
| `SKILL.md` | `scripts/tutti-env.sh` 单一入口、从零 clone 到跑 benchmark、特权/破坏性操作的确认边界、5 条不可违背的不变量、仓库布局、扩展新框架 |
| `references/hardware.md` | 软硬件协同前置（PCIe switch 配对、IOMMU、MDTS 由硬件报告）、`tutti_daemon.yaml` 为硬件事实唯一来源、模块↔用户态 ABI 握手、7 类破坏性操作的后果、新机器上机清单 |
| `references/architecture.md` | 分层职责、单请求控制流、内存权威索引、KV 布局、对象池、惰性注册 |
| `references/diagnostics.md` | 排障方法论 + 6 类真实故障的证据链与修法 |
| `references/benchmarking.md` | 压测参数、容量/队列定容规则、运行隔离、从 nsys 导出量化 IO/compute 重叠 |

按渐进披露组织：`description` 常驻上下文，正文在触发时加载，`references/`
按需读取。高危内容例外——破坏性操作的确认边界上提到 `SKILL.md` 正文，不能只藏在
`references/` 里，否则 agent 可能在没读 reference 的情况下就动了宿主环境。

## 与 scripts/tutti-env.sh 的关系

skill 不复制脚本的实现细节，只说明**何时用哪个子命令**以及**为什么顺序不能换**。
脚本自身是可执行的真相来源：

```bash
scripts/tutti-env.sh --help
```

因此脚本改了子命令时，`SKILL.md` 的 "One Entry Point" 一节需同步。

## 维护：经验往哪里沉，不是都沉进 skill

skill 是纯文本，**无法自证**。过期的 skill 比没有 skill 更危险——agent 会带着自信
按过期知识去 `rmmod`、去 `mount`、去写盘，而这些后果不可逆。所以经验的第一归宿不是
文档，而是能自己失败的东西：

| 经验形态 | 归宿 | 为什么优先 |
|---|---|---|
| 能变成一条检查 | `scripts/tutti-env.sh status` | 环境不对时当场红 |
| 能变成一条断言 | `tests/` | 回归时当场红 |
| 能变成运行时守卫 | fail-closed / 告警 | 线上自己暴露 |
| **只能靠人判断** | **skill** | 前三档吸收不了才写 |

写进 skill 的条目必须同时满足：症状与根因距离远（读代码推不出来）；带**判别方法**而非
只有结论；至今仍成立；无法被前三档吸收。

**明确不要写**：性能数字（会过期，还会诱导 agent 拿它当验收线）、commit hash、
机器专属路径（该进 `config/`）、已被代码修掉的问题全过程（价值已固化在代码里，
最多留一句"为什么代码长这样"）、能从 `--help` 或代码读到的事实（重复即漂移源）。

**必须淘汰，不能只增**：加新排障条目前，先跑一遍已有条目的判别命令，跑不通的直接删。
`references/diagnostics.md` 只增不减必然烂掉。

同步触发点：

- 目录布局或包边界 → `SKILL.md` 的 Repository Layout
- 不变量（尤其"调度进程不导入数据面"、"public include 是命名空间"）
- `scripts/tutti-env.sh` 的子命令或选项 → `SKILL.md` 的 One Entry Point
- 硬件拓扑约束、新的破坏性操作 → `references/hardware.md`，高危项同时上提 `SKILL.md`
- 新一类故障模式 → `references/diagnostics.md`，**连同证据链与判别命令**

校验与打包（若安装了 skill-creator 的脚本）：

```bash
python <skill-creator>/scripts/package_skill.py Skills/tutti-runtime ./dist
```
