#pragma once

namespace tutti::config {

// 2026-09-22：放置模型改为"一个对象 = 一个文件，slot 号在 N 块盘间轮转"
// 后，条带化不复存在，后端不再有可配置项。结构体保留作为 relation 的
// config 占位（variant 需要一个 distinct type 区分 ext4 单盘契约）。
struct StripedLocalNvmeBackendConfig {};

} // namespace tutti::config
