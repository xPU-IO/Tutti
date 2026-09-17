# snvme Unified Kernel-Module Tree — Porting & Support Guide
# snvme 统一内核模块源码树 — 移植与支持指南（中英对照 / Bilingual）

> **EN** — This tree replaces the three per-kernel-version copies
> (`snvme-5.4.241-1-tlinux4-0017/`, `snvme-5.15.0-public/`,
> `snvme-6.8.0-public/`, all kept frozen as fallback) with a **single
> source tree**: one copy of the snvme-private code at the root, and
> per-kernel-lineage copies of the *upstream* NVMe driver under
> `baseline/<tag>/`.  For background on what snvme changes on top of
> upstream NVMe (queue groups, GPU P2P, char-device/ioctl surface), the
> pre-unification porting guide (function-level diff inventories, trap
> records) is preserved in git history (`git log --all -- '*kernel_modules/PORTING.md'`).
> Legacy source comments citing "PORTING.md §x" refer to that retired
> edition.
>
> **中文** — 本源码树用**单一源码树**取代原来三个按内核版本各存一份的副本
> （`snvme-5.4.241-1-tlinux4-0017/`、`snvme-5.15.0-public/`、
> `snvme-6.8.0-public/`，三者均冻结保留作为回退）：snvme 私有代码只保留
> 一份放在根目录，*上游* NVMe 驱动按内核血统存放在 `baseline/<tag>/`。
> 关于 snvme 在上游 NVMe 之上做了哪些修改（队列组、GPU P2P、字符设备/
> ioctl 接口）的背景（函数级 diff 清单、踩坑记录）保留在 git 历史中
> （`git log --all -- '*kernel_modules/PORTING.md'`）。源码注释里引用的
> "PORTING.md §x" 指向该已退役版本。

## 1. Supported kernels / verification matrix · 支持的内核 / 验证矩阵

| Baseline 基线 | Covers 覆盖内核 | Upstream source 上游来源 | Compile 编译 | Insmod + smoke 加载 + 冒烟 |
|---|---|---|---|---|
| `5.4-tlinux4` | 5.4.x Tencent tlinux4 / OpenCloudOS lineage · 腾讯 tlinux4 / OpenCloudOS 血统（如 `5.4.241-1-tlinux4-0017.7`） | [OpenCloudOS-Kernel `linux-5.4/lts/5.4.241-30.0017`](https://gitee.com/OpenCloudOS/OpenCloudOS-Kernel/tree/linux-5.4/lts/5.4.241-30.0017/drivers/nvme/host) — vanilla 5.4 also works (only ~447 lines of backports) · 主线 5.4 亦可用（backport 仅 ~447 行） | ✅ 2026-08-18, host `5.4.241-1-tlinux4-0017.7`, nvidia backend | ✅ 2026-08-18: insmod, 4 controllers bound + EXT4 mounted by daemon; UAPI smoke 8/8, qgroup smoke 25/25, addq verified through bind/cap/GET_DEV_INFO (full user-queue run blocked by the single-PRP ring limit at io_queue_depth=1024 — same limit as legacy trees; needs io_queue_depth<=64 insmod for addq/io) · 加载成功，daemon 绑定 4 控制器并挂载；addq 验证到 bind/cap/GET_DEV_INFO（完整用户队列受单 PRP 环上限拦截——与旧树相同限制，跑 addq/io 需 io_queue_depth≤64 重装） |
| `5.10` | 5.10.x public LTS · 公开版 LTS（Debian 11、openEuler 22.03 等） | [kernel.org v5.10 `drivers/nvme/host`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/drivers/nvme/host?h=v5.10) | ❌ not yet — **assigned: external tester · 已分配：外部测试者** (derived from the 5.15 baseline by 3-way merge against v5.10 upstream; static audit passed — no `init_ctrl_finish`/`nvme_check_ready`/trace leftovers, `snvme_set_queue_count` chain intact) | ❌ |
| `5.15` | 5.10 – 5.19 public · 公开版 | [kernel.org v5.15 `drivers/nvme/host`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/drivers/nvme/host?h=v5.15) | ❌ not yet — **assigned: external tester · 已分配：外部测试者** | ❌ |
| `6.8` | 6.x public · 公开版 | [kernel.org v6.8 `drivers/nvme/host`](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/drivers/nvme/host?h=v6.8) | ❌ not yet — **assigned: external tester · 已分配：外部测试者** | ❌ |

> **EN** Verification levels: **Compile** — `make` produces `snvme.ko` +
> `snvme-core.ko` against that kernel's headers.  **Insmod + smoke** —
> module loads (signed where required), `test/run_snvme_smoke.sh` passes,
> and `nm` symbol comparison against the corresponding legacy tree shows no
> functional delta (done for 5.4-tlinux4: `snvme-core.ko` symbol-identical;
> `snvme.ko` differs only in compiler clone suffixes and formerly-static
> symbols becoming cross-TU).  Update this table when a baseline is verified
> on a new host.
>
> **中文** 验证级别：**编译** — 在该内核头文件下 `make` 产出两个 .ko。
> **加载 + 冒烟** — 模块可加载（需签名的宿主上已签名）、smoke 通过、
> 且与对应旧树的 `nm` 符号对比无功能差异（5.4-tlinux4 已完成）。
> 在新宿主机上验证某基线后请更新此表。

## 2. Layout · 目录布局

```text
kernel_modules/
├── PORTING.md             this bilingual guide · 本中英对照指南
├── test/                  smoke suite (see §8) · 冒烟测试（见 §8）
│
├── snvme/                 ── the unified tree · 统一树 ──
│   ├── Makefile.in        kbuild template; picks baseline/ from kernel version
│   │                      kbuild 模板；按内核版本选择 baseline/
│   ├── Kconfig            documentation only (out-of-tree build) · 仅作文档
│   ├── snvme-rename.sed   nvme_* → snvme_* symbol rename script (core.c)
│   │                      符号重命名脚本（见 §3）
│   │   ── shared snvme-private code: ONE copy, built for every baseline ──
│   │   ── 共享的 snvme 私有代码：仅一份，对每个基线都编译 ──
│   ├── snvm_glue.h        global registries + PCI_DRIVER_NAME + shared init/exit API
│   │                      全局注册表（ctrl/host/device/device_queue）、共享 init/exit
│   ├── snvm_control.c     /dev/snvm_control, bind/unbind control plane,
│   │                      PCI driver registration (pointer-ized)
│   │                      控制字符设备、bind/unbind 控制面、PCI 驱动注册（指针化）
│   ├── snvm_dev_ioctl.{c,h}  /dev/ssnvme<N> per-fd ioctl surface + fops
│   │                      每-fd ioctl 接口（NVM_MAP_*、队列组、ADD_USER_QUEUE、GET_DEV_INFO）
│   ├── snvm_qgroup.{c,h}  per-fd queue groups, B3 user-QID pool,
│   │                      user-queue admin-command helpers (adapter_*)
│   │                      每-fd 队列组、用户 QID 池、管理命令辅助函数
│   ├── map.{c,h}          GPU/host DMA mapping registry (incl. Phoenix P2P)
│   │                      GPU/主机 DMA 映射注册表
│   ├── ctrl.{c,h}         per-controller char-device lifecycle · 每控制器字符设备生命周期
│   ├── list.{c,h}         intrusive doubly-linked list · 侵入式双向链表
│   ├── compat/            compat.{c,h} — the ONLY place allowed to test
│   │                      LINUX_VERSION_CODE · 全模块唯一允许版本宏判断的地方
│   ├── peer_memory/       vendor-neutral GPU P2P backends · 厂商中立 GPU P2P 后端
│   │   ── per-lineage upstream NVMe driver + snvme hooks ──
│   │   ── 按血统存放的上游 NVMe 驱动 + snvme hook ──
│   └── baseline/
│       ├── 5.4-tlinux4/   pci.c, core.c, nvme.h, fabrics*, tcp, rdma,
│       │                  multipath + snvm_ndev.h (struct nvme_dev)
│       ├── 5.10/          same file set as 5.4-tlinux4 + hwmon.c, zns.c
│       │                  (ioctl stays inside core.c until upstream 5.14)
│       ├── 5.15/          + ioctl.c, hwmon.c, zns.c
│       └── 6.8/           + sysfs.c, pr.c, auth.*, constants.c
│
└── snvme-5.4.241-1-tlinux4-0017/   ── DEPRECATED, frozen · 已废弃，冻结 ──
    snvme-5.15.0-public/               (each carries a DEPRECATED marker
    snvme-6.8.0-public/                file; kept only as fallback/diff
                                        reference until the matching
                                        baseline is verified; will be
                                        REMOVED — do not land changes)
                                        （各树带 DEPRECATED 标记；仅作
                                        回退/对比参考，验证通过后删除；
                                        不要在旧树提交新改动）
```

> **EN** What lives where: **Root** = snvme-private code, must compile
> against *every* supported kernel; version checks go through `compat/`.
> **`baseline/<tag>/`** = that lineage's upstream NVMe files (kept close to
> upstream; snvme hooks are the only edits) plus `snvm_ndev.h`, the
> extracted `struct nvme_dev` for shared-layer visibility.
> `baseline/*/pci.c` keeps upstream lifecycle functions with the hooks
> inline, the `struct pci_driver snvme_driver`, and a thin
> `nvme_init/nvme_exit` calling `snvm_global_init/exit()` +
> `snvm_pci_register()`.  The heavy private logic that used to duplicate in
> each tree's pci.c tail (≈2,000 lines × 3) now exists once, at the root.
>
> **中文** 代码放哪里：**根目录** = snvme 私有代码，必须在每个受支持内核上
> 都能编译，版本差异一律经 `compat/`。**`baseline/<tag>/`** = 该血统的
> 上游 NVMe 文件（保持贴近上游，snvme hook 是仅有的修改）+ `snvm_ndev.h`
> （抽出的 `struct nvme_dev`，供共享层访问其字段）。`baseline/*/pci.c`
> 保留上游生命周期函数及内联 hook、`snvme_driver` 和瘦身的
> `nvme_init/nvme_exit`。原先每棵树 pci.c 尾段的大块私有逻辑
> （约 2,000 行 × 3 份）现在只在根目录存在一份。

## 3. What `snvme-rename.sed` does · sed 脚本的作用

> **EN** The kernel simultaneously runs its own `nvme-core.ko` (module or
> built-in), which exports the *same* global symbols as snvme's copy of the
> upstream core (≈29, e.g. `nvme_submit_sync_cmd`, `nvme_setup_cmd`,
> `nvme_init_ctrl`).  Without renaming, loading both fails with
> "exports duplicate symbol" — or worse, in-tree callers get silently bound
> to snvme's implementations.  The script is a mechanical rename rule set
> (41 `s/nvme_xxx/snvme_xxx/g` rules) that moves every *conflicting
> exported* symbol into an independent `snvme_*` namespace.  Types and
> static helpers keep their upstream names — they are never exported, so
> they cannot collide, and keeping upstream names minimizes diffs for
> future kernel uplifts.  Re-run it after touching core.c or the symbol
> split breaks the build.  Current baselines store the *renamed* core.c;
> planned improvement: store near-upstream core.c and apply the sed at
> build time, shrinking the diff-vs-upstream from ±250 to ±20 lines.
>
> **中文** 内核同时跑着自带的 `nvme-core.ko`（模块或内建），与 snvme 的
> core.c 副本导出**同名**全局符号（约 29 个）。不改名的话两个模块同时
> 加载会被 "exports duplicate symbol" 拒绝，或者内核自带 nvme 的调用方被
> 静默绑定到 snvme 的实现上。脚本是一套机械重命名规则（41 条
> `s/nvme_xxx/snvme_xxx/g`），把**会冲突的导出符号**全部移入独立的
> `snvme_*` 命名空间。结构体类型和 static 内部函数刻意保持上游原名——
> 不导出、不可能冲突，且保持原名让与上游的 diff 最小、升级合并最容易。
> **改了 core.c 之后必须重跑本脚本**，否则符号分裂会破坏构建。当前基线
> 存放已重命名的 core.c；后续优化：改存贴近上游的文件、构建时自动跑
> sed，使与上游的 diff 从 ±250 行降到 ±20 行。

### 3.1 Version drift · 版本漂移（升级内核必读）

> **EN** The conflict set is NOT constant across kernels — it drifts with
> upstream evolution.  Measured on the current tree: the sed carries 41
> rules, but the per-baseline `snvme_*` export counts are 46 / 42 / 46 / 51
> (5.4 / 5.10 / 5.15 / 6.8).  Version-specific symbols that appeared later
> upstream were renamed **by hand in each baseline, never back-filled into
> the sed** — e.g. `snvme_init_ctrl_finish` (upstream 5.13, exists in
> 5.15/6.8 only), `snvme_alloc_request_qid` (5.15 only),
> `snvme_fail_nonready_command` / `__snvme_check_ready` (moved into core
> and exported in upstream 5.11).  So the *true* rename map today = the
> sed (5.4 set) + per-baseline hand patches.
>
> **中文** 冲突集不随内核恒定——随上游演进漂移。本树实测：sed 有 41 条
> 规则，但各基线 `snvme_*` 导出数分别为 46 / 42 / 46 / 51
> （5.4 / 5.10 / 5.15 / 6.8）。高版本特有的符号是**在基线里手工补的
> 改名、从未回填进 sed**——如 `snvme_init_ctrl_finish`（上游 5.13 出现，
> 仅 5.15/6.8 有）、`snvme_alloc_request_qid`（仅 5.15）、
> `snvme_fail_nonready_command` / `__snvme_check_ready`（上游 5.11 从
> fabrics 移入 core 并导出）。因此今天的重命名真值表 = sed（5.4 集）+
> 各基线手工补丁。

### 3.2 What conflicts actually means · 冲突的真实判定与三种处理

> **EN** Two modules having a *function* with the same name does NOT
> conflict — module-local symbols are invisible to each other.  A conflict
> exists only when the same name enters **both modules' export tables**
> (`EXPORT_SYMBOL[_GPL]` → ksymtab).  For each version-drifted symbol pick
> one of three treatments:
>
> **中文** 两个模块**函数同名并不冲突**——模块内部符号互不可见。只有
> 同名符号**同时进入两个模块的导出表**（`EXPORT_SYMBOL[_GPL]` →
> ksymtab）才冲突。对每个版本漂移符号三选一：

| Treatment 手段 | When 适用 | Examples 现有例子 |
|---|---|---|
| A. rename + export · 改名并导出（sed 加规则） | snvme's other units / future features call it · snvme 其他编译单元或后续功能需要调用 | `snvme_submit_sync_cmd` 等 |
| B. keep name, comment out the EXPORT · 保留原名，注释导出 | module-internal use only · 仅本模块内部使用 | `nvme_host_path_error`, `nvme_cancel_tagset` |
| C. delete the feature · 删除 | snvme does not need it · snvme 用不到 | trace support |

### 3.3 Audit command · 导出冲突审计命令

> **EN** Run this when porting any new kernel; the intersection must be
> empty before build.  The mechanism is fail-loud anyway: a missed symbol
> is rejected at `insmod` with `exports duplicate symbol nvme_xxx`.
>
> **中文** 移植任何新内核时执行；交集必须为空才可构建。机制本身
> fail-loud：漏掉的符号在 `insmod` 时被拒并报
> `exports duplicate symbol nvme_xxx`。

```bash
R=<dir with that kernel's pristine host files>   # 上游纯净参考目录
BL=snvme/baseline/<tag>
comm -12 \
  <(grep -rh '^EXPORT_SYMBOL' $R/*.c | grep -oP '\(\K\w+' | sort -u) \
  <(grep -rh '^EXPORT_SYMBOL' $BL/*.c | grep -oP '\(\Knvme_\w+' | sort -u)
# 非空交集 = 必须按 §3.2 处理 · non-empty = must treat per §3.2
```

## 4. baseline/: upstream originals? · baseline 是上游原文件吗？

> **EN** No — "upstream + two classes of edits", both quantified by
> per-file diffs against true upstream references:
>
> **中文** 不是——是“上游 + 两类修改”，均已相对上游参考文件逐文件量化：

| File 文件 | Edits relative to upstream · 相对上游的修改 | Size 量级 |
|---|---|---|
| `pci.c` | ① snvme hooks: ~50 in-function edit points (`nvme_probe` registers the ctrl, `nvme_irq` returns `IRQ_HANDLED` for GPU-consumed CQs, `s_nvme_setup_io_queues` kernel/user queue-budget negotiation, reset-path user-queue drain, …); ② the old ~2,000-line private tail was **extracted** into the shared root layer — pci.c keeps only calls into it · hook 打在上游函数体内；原尾段私有实现已抽到共享层，只剩调用 | ≈+200 lines of hooks per baseline |
| `core.c` | symbol renames (sed) + trace/sysfs param removal + 2 added functions (`__snvme_submit_sync_cmd`, `snvme_sec_submit`) · 符号重命名 + 删 trace/参数 + 新增 2 函数 | ±250 lines, renames dominate |
| `nvme.h`, `fabrics*`, `tcp`, `rdma`, `multipath`, `ioctl`, `hwmon`, `zns`, `sysfs`, `pr`, `auth.*`, `constants` | essentially symbol renames only (call sites follow the renamed exports) · 基本只有符号重命名 | ±2 ~ ±48 lines per file |
| `snvm_ndev.h` | **new file** (not upstream): extracted `struct nvme_dev` with the 4 snvme fields (`online_user_queues`, `user_start_qid`, `ctrl_max_io_queues`, `cap_kernel_ioq`) · 新增：抽出的结构体定义，含 4 个 snvme 私有字段 | new |

> **EN** One sentence: only `pci.c` carries substantive functional edits;
> everything else in baseline/ differs from upstream at the
> symbol-rename level; the real snvme logic lives once in the shared root.
> The per-baseline copies exist because the inter-version deltas are
> *upstream's own evolution* (upstream pci.c alone differs by 1,351 lines
> v5.15↔v6.8; 6.8 also split out sysfs/pr/auth/constants) — unifying them
> with version macros would mean hand-maintaining a kernel superset that
> does not exist.
>
> **中文** 一句话：只有 `pci.c` 有实质功能修改，baseline 其余文件相对上游
> 就是“改了个名”；真正的 snvme 功能逻辑只在根目录共享层存一份。每基线
> 一份的原因是版本间差异主体为**上游自身演进**（仅 pci.c 上游 v5.15↔v6.8
> 就差 1,351 行；6.8 还拆出 sysfs/pr/auth/constants）——用版本宏强行统一
> 等于手工维护一个不存在的“内核超集”。

## 5. Build & install · 构建与安装

```bash
# from the repository build (preferred) · 仓库构建入口（推荐）:
cmake --build build --target modules      # auto-selects baseline via uname -r
                                        # 通过 uname -r 自动选基线
# standalone, inside a configured output dir · 独立构建:
make TUTTI_P2P_BACKEND=nvidia             # or metax
make insmod                               # insmod snvme-core.ko then snvme.ko
```

Baseline auto-selection (`Makefile.in`; override with `make SNVME_BASELINE=<tag>`)
· 基线自动选择（可用 `SNVME_BASELINE=` 覆盖）:

| running kernel 运行内核 | baseline chosen 选择 |
|---|---|
| `5.4.*` (any 任意) | `5.4-tlinux4` |
| `5.10.x` | `5.10` |
| `5.11` … `5.19` | `5.15` |
| `6.x` / `7.x` | `6.8` |

> **EN** Runtime order is strict: insmod → `tutti_daemon` → mount;
> `/dev/ssnvme*` exists only after daemon bring-up.  Module signing
> (where a host requires it) is an internal deployment concern and is
> deliberately not covered here.
>
> **中文** 运行顺序严格：insmod → `tutti_daemon` → mount；`/dev/ssnvme*`
> 只在 daemon 起来后才存在。模块签名（如宿主机要求）属内部部署事项，
> 不在本文档讨论范围。

```bash
sudo insmod snvme-core.ko && sudo insmod snvme.ko
```

## 6. Differences vs the legacy trees · 与旧树的差异

- **EN** The 6.8 lineage's private code is the canonical cut; the 5.4 tree's
  older private code (1,155-line `snvm_dev_map_ioctl`,
  `snvme_disable_user_io_queues`, inline map/ctrl remnants) was not carried
  over.  The 5.4 baseline keeps only `snvme_disable_user_io_queues` (its
  reset-path hook) — it now calls the shared `adapter_delete_sq/cq`.
  **中文** 以 6.8 血统为权威切割；5.4 树的旧私有代码未带入。5.4 基线仅保留
  `snvme_disable_user_io_queues`（reset 路径 hook 需要），现在调用共享层实现。
- **EN** `snvm_rebind_driver` is the 5.4-lineage implementation
  (`driver_attach()` + bounded udev-race retry) — works on every supported
  kernel; the 6.8 `device_driver_attach()` variant was dropped.
  **中文** rebind 采用 5.4 血统实现（`driver_attach` + 防 udev 竞争重试），
  全内核通用；放弃了 6.8 的 `device_driver_attach` 变体。
- **EN** `snvm_user_qid_pool_init/alloc/free`, `snvm_queue_group_ida`,
  `adapter_*_user`, `find/destroy_qgroup_locked`,
  `snvm_ctrl_get_live_ndev` moved from static-in-pci.c to shared
  (declared in `snvm_qgroup.h`).
  **中文** 上述符号从 pci.c 内 static 改为共享，声明于 `snvm_qgroup.h`。
- **EN** Globals (`ctrl_list`, `host_list`, `device_list`,
  `device_queue_list`, …) live in `snvm_control.c`, declared in
  `snvm_glue.h`.  **中文** 全局注册表定义于 `snvm_control.c`，声明于
  `snvm_glue.h`。
- **EN** Version shims in `compat/compat.h`: `compat_get_user_pages`,
  `snvm_class_create` (owner arg dropped in v6.4), `SNVM_DEVNODE_ARGS`
  (devnode const in v6.1), `snvm_ns_lba_shift` (field moved ns→head in
  v6.7).  If you need another `#if LINUX_VERSION_CODE`, put it there —
  nowhere else.  **中文** 现有版本适配宏如上；再需要版本宏只能加在
  compat/，别处一律不允许。
- **EN** Per-baseline `core.c` stores the renamed version; planned: store
  near-upstream and apply `snvme-rename.sed` at build time.
  **中文** 各基线 core.c 目前存已重命名版本；后续改为构建时跑 sed。

## 7. Adding a new kernel lineage · 新增内核血统（完整 SOP）

> **EN** Battle-tested on the 5.10 port (2026-08-18).  Example below uses
> a hypothetical 6.12; replace versions accordingly.  Expected effort once
> familiar: 2–4 h + target-machine time when conflicts are routine (mostly
> renames-vs-upstream-tweaks); a working day when upstream restructured
> hook-bearing functions.
>
> **中文** 本流程在 5.10 移植中实战验证（2026-08-18）。下面以假想的 6.12
> 为例，按需替换版本号。熟练后预期耗时：冲突常规（多为"重命名 vs 上游
> 微调"）2–4 小时 + 测试机时；上游重构了带 hook 的函数则约一个工作日。

### Step 1 · Quantify and pick the derivation source · 量化选派生源

```bash
mkdir -p /data/home/ryeqiu/upstream-refs/v6.12
cd /data/home/ryeqiu/upstream-refs/v6.12
for f in pci.c core.c nvme.h fabrics.c fabrics.h multipath.c rdma.c \
         tcp.c zns.c hwmon.c ioctl.c sysfs.c pr.c auth.c auth.h constants.c; do
  curl -sfL "https://raw.githubusercontent.com/torvalds/linux/v6.12/drivers/nvme/host/$f" -o $f &
done; wait     # nonexistent files 404-skip silently · 不存在的文件自然跳过

# pick the baseline whose upstream is closest · 选上游差异最小的现有基线
for v in v5.10 v5.15 v6.8; do
  echo "$v: $(diff v6.12/pci.c $v/pci.c | grep -c '^[<>]') lines"
done          # e.g. 6.12 is closest to v6.8 → derive from baseline/6.8
```

### Step 2 · Generate via 3-way merge · 三方合并生成

> **EN** Semantics: base = derivation baseline's **pristine upstream**,
> ours = the derivation **baseline** (carrying the snvme edits), theirs =
> the **new upstream**.  Output = "new upstream + snvme edits".
>
> **中文** 语义：base = 派生源基线的**纯净上游**，ours = 派生源**基线**
> （带 snvme 修改），theirs = **新上游**。输出 = "新上游 + snvme 修改"。

```bash
mkdir -p snvme/baseline/6.12
cd snvme
R=/data/home/ryeqiu/upstream-refs
for f in <file list of the derivation baseline>; do
  cp baseline/6.8/$f /tmp/ours.c
  git merge-file -p --diff3 /tmp/ours.c $R/v6.8/$f $R/v6.12/$f \
      > baseline/6.12/$f 2>/dev/null
done
grep -c '^<<<<<<<' baseline/6.12/*.c    # conflict census · 冲突清点
```

> **EN** Auto-resolve first: apply the snvme rename map (the sed rules of
> §3 **plus** the hand-patch table — `snvme_init_ctrl_finish`,
> `snvme_alloc_request_qid`, `snvme_fail_nonready_command`,
> `__snvme_check_ready`, `s_nvme_wq`, `"nvme%dn%d"`→`"snvme%dn%d"`, …) to
> the theirs side; if it then equals ours, or ours is a pure rename of
> base, the block resolves mechanically.  Everything else is manual:
>
> **中文** 先自动解决：把 snvme 重命名映射（§3 的 sed 规则**加上**手工
> 补丁表）套到 theirs 侧；若结果等于 ours，或 ours 是 base 的纯重命名，
> 该块机械解决。其余人工：

| Conflict shape 冲突形态 | Decision 决策 | 5.10 precedent 5.10 实例 |
|---|---|---|
| pure rename vs upstream tweak · 纯重命名 vs 上游微调 | theirs + rename map (auto) · 取 theirs 套映射（自动） | `snvme_init_identify(ctrl)` |
| upstream-new API / function · 上游新 API/新函数 | theirs + rename map · 取 theirs 加映射 | 5.10-native APST heuristic |
| snvme hook overlaps upstream evolution · hook 与上游演进重叠 | **hand-fuse**: keep hook semantics on the new upstream skeleton · **手工融合**：保 hook 语义、用新上游骨架 | B3 cap-shrink fused into 5.10's trylock-free queue setup |
| upstream removed/reordered an snvme dependency · 上游删除/重排 snvme 依赖 | case-by-case · 逐例判断 | no `init_ctrl_finish` in 5.10 → use `snvme_init_identify` |

### Step 3 · Symbol-conflict audit · 符号冲突审计

Run the §3.3 `comm -12` audit against the new upstream; treat every
intersection per §3.2; back-fill any new rename rule into the sed under a
`# --- vX.Y+ ---` section so the next port inherits it.
· 执行 §3.3 审计，交集按 §3.2 处理；新增改名规则以 `# --- vX.Y+ ---`
分节回填 sed，让下次移植直接继承。

### Step 4 · Extract snvm_ndev.h · 抽取 snvm_ndev.h

Pull `struct nvme_dev` out of the merged pci.c into
`baseline/<tag>/snvm_ndev.h`; keep the 4 snvme fields
(`online_user_queues`, `user_start_qid`, `ctrl_max_io_queues`,
`cap_kernel_ioq`); leave a `/* moved to snvm_ndev.h */` comment in place.
· 从合并后的 pci.c 抽出 `struct nvme_dev` 到 `snvm_ndev.h`；保留 4 个
snvme 字段；原位置留注释。

### Step 5 · Wire the build · 接线构建

`Makefile.in`: add the version-mapping branch and adjust the file set
(`ioctl.c` exists ≥5.15; `sysfs/pr/auth/constants` exist ≥6.8 — a new
kernel may split out more files).
· `Makefile.in` 加版本映射分支并调整文件集（新内核可能拆出新文件）。

### Step 6 · Verification ladder · 验证阶梯（严格按序）

1. **Compile** against the target kernel headers:
   `make SNVME_BASELINE=<tag>` — zero warnings.
   在目标内核头下编译，零警告。
2. **Symbol diff**: `nm` compare with the derivation baseline's `.ko` —
   deltas must be upstream evolution + version-specific symbols only.
   与派生源基线产物做 `nm` 对比，差异应仅为上游演进+版本特有符号。
3. **Static audit**: zero conflict markers; zero leftovers of deleted
   APIs (`trace_*`, `init_ctrl_finish` on ≤5.12, `nvme_check_ready` on
   ≤5.10 …); `snvme_set_queue_count` call chain intact.
   零冲突标记；已删 API 零残留；`snvme_set_queue_count` 调用链完整。
4. **Load + smoke** on a real host (§8): UAPI 8/8 → qgroup 25/25 →
   addq/io on an idle disk.
   真机加载并跑冒烟（§8），addq/io 用空闲盘。
5. **Update the §1 matrix** with date / host / result.
   更新 §1 矩阵的日期/宿主/结果。

## 8. Smoke tests · 冒烟测试

> **EN** Unchanged: `test/` (`run_snvme_smoke.sh`; GPU smoke requires
> insmod → daemon → mount order).  Note the smoke binaries are
> **standalone UAPI-direct mode** (libc-only, ioctl `/dev/snvm_control`
> and `/dev/ssnvme*` directly — no daemon socket, not client mode).
> Always target an NVMe the daemon does NOT own and that is unmounted:
> `snvme_smoke_addq` / `snvme_smoke_io` BIND the target (destructive);
> recover a half-bound device with `snvme_ubind <BDF>`.  Recommended
> acceptance order for a new baseline: compile → `nm` symbol diff vs the
> legacy tree it replaces → smoke suite → GPU round-trip.
>
> **中文** 测试位于 `test/`（GPU 冒烟要求 insmod → daemon → mount
> 顺序）。注意 smoke 程序是**独立 UAPI 直连模式**（只链 libc，直接 ioctl
> 字符设备，不经 daemon，也不是 client 模式）。务必选 daemon 未管理且
> 未挂载的空闲盘：`snvme_smoke_addq` / `snvme_smoke_io` 会 bind 目标盘
> （破坏性）；半绑定状态可用 `snvme_ubind <BDF>` 恢复。新基线建议验收
> 顺序：编译 → 与被替换旧树的 `nm` 符号对比 → smoke 套件 → GPU 往返。
