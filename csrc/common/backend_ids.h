#pragma once

// 各层契约标识常量的唯一来源（E2）。
//
// 这些标识**不是**"同一个后端的多种拼写"，而是各自描述所在层的契约：
//   * resolver 层（local-file）：文件 → extent。与文件系统、传输无关——
//     判据只有"常规文件 + st_dev 命中配置的块设备 + FIEMAP 完整覆盖"，
//     故同一个 resolver 可服务 ext4/xfs、本机盘或 NVMe-oF 命名空间；
//   * DataPath 层（local-nvme）：本机控制器直驱；载荷携带
//     controller_pci_addr，"local"这个约束落在这一层；
//   * contract（ext4-local-nvme）：上面两者的**合法配对**名——ext4 文件
//     落在本机 NVMe 命名空间上；它才是"一个后端组合"的标识；
//   * scheme / payload_id / resolver_id / datapath_key / store_scheme：
//     各注册面自己的键。
//
// 此前这些字面量散落在 config/spec、parser、resolver、payload、DataPath
// 工厂等处。集中到这里后：
//   * 新增一个组合只需在此处添加常量，并更新下列引用点；
//   * 各引用点不再出现裸字面量，"遗漏一处拼写"由编译器/测试暴露。
//
// 新增后端的引用点清单：
//   1. config/spec/tutti_runtime_spec.cpp  契约表（name/resolver_type/
//      resolver_scheme/datapath_type）与 type 分派
//   2. config/parser/tutti_runtime_config_parser.cpp  各 type 分派
//   3. payloads/<backend>/payload.h   kPayloadTypeId / kResolverTypeId /
//      kRecommendedDataPathKey 的取值
//   4. resolvers/<backend>/resolver.h  kScheme
//   5. data_paths/data_path_factory.cpp、resolvers/resolver_factory.cpp
//      的 (type, contract) 分派
//   6. storage_objects/store_factory.cpp  store scheme（同时同步
//      tutti/storage 下的 Python 常量）
//
// 命名约定（各层不同、此处仅集中取值，不强行改名）：
//   contract      配置面后端标识（kebab-case）
//   resolver_type ResolverSpec::type（kebab-case）
//   scheme        resolver URI scheme（单词）
//   datapath_type DataPathSpec::type（kebab-case）
//   payload_id    版本化 payload 契约 id（kebab-case + -vN）
//   resolver_id   版本化 resolver 契约 id
//   datapath_key  推荐的 DataPath 绑定键
//   store_scheme  对象层存储 scheme（snake_case）

#include <string_view>

namespace tutti::detail::backend_ids {

// ---- ext4 local NVMe（单盘文件后端）----
inline constexpr std::string_view kExt4Contract = "ext4-local-nvme";
inline constexpr std::string_view kExt4ResolverType = "local-file";
inline constexpr std::string_view kExt4Scheme = "file";
inline constexpr std::string_view kExt4DataPathType = "local-nvme";
inline constexpr std::string_view kExt4PayloadTypeId =
    "ext4-local-nvme-payload-v1";
inline constexpr std::string_view kExt4ResolverTypeId =
    "ext4-extent-resolver-v1";
inline constexpr std::string_view kExt4DataPathKey = "local-nvme-ext4";
inline constexpr std::string_view kExt4StoreScheme = "local_nvme_file";

// ---- rotating local NVMe（多盘文件后端）----
// 2026-09-22 起：放置模型改为"一个对象 = 一个文件，slot 号在 N 块盘间轮转"。
// resolver 复用 local-file（按挂载点前缀分派到对应设备的 LocalFileResolver
// 实例），payload 复用 ext4-local-nvme 的（文件即对象，无分片数学）。与
// ext4-local-nvme 的区别只在 DataPath：本契约走多设备 fused 提交路径。
// 历史名 "striped-*" 保留为配置面标识，避免一次改遍所有 YAML/Python 常量；
// 语义见 striped_data_path.h 头注。
inline constexpr std::string_view kStripedContract = "striped-local-nvme";
inline constexpr std::string_view kStripedDataPathType = "striped-local-nvme";
inline constexpr std::string_view kStripedDataPathKey = "striped-local-nvme";
inline constexpr std::string_view kStripedStoreScheme =
    "striped_local_nvme_file";

// ---- memfs（内存后端，示例/测试用）----
inline constexpr std::string_view kMemfsContract = "memfs";
inline constexpr std::string_view kMemfsResolverType = "memfs";
inline constexpr std::string_view kMemfsScheme = "memfs";
inline constexpr std::string_view kMemfsDataPathType = "memfs";
inline constexpr std::string_view kMemfsDataPathKey = "memfs";
inline constexpr std::string_view kMemfsPayloadTypeId = "memfs-payload-v1";
inline constexpr std::string_view kMemfsResolverTypeId = "memfs-resolver-v1";

} // namespace tutti::detail::backend_ids
