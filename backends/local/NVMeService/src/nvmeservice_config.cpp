#include "nvmeservice_config.h"

#include <yaml-cpp/yaml.h>
#include <fstream>
#include <set>
#include <sstream>

namespace nvmeservice {

namespace {

template <typename T>
T get_or(const YAML::Node& n, const std::string& key, T def) {
    if (!n || !n[key]) return def;
    return n[key].as<T>();
}

void parse_grpc(const YAML::Node& root, GrpcConfig& out) {
    if (!root["grpc"]) return;
    const auto& g = root["grpc"];
    out.endpoint = get_or<std::string>(g, "endpoint", out.endpoint);
}

void parse_gpus(const YAML::Node& root, std::vector<GpuEntry>& out) {
    if (!root["gpus"]) return;
    for (const auto& node : root["gpus"]) {
        GpuEntry e;
        e.id         = get_or<int>        (node, "id",         -1);
        e.mount_path = get_or<std::string>(node, "mount_path", "");
        out.push_back(std::move(e));
    }
}

void parse_nvmes(const YAML::Node& root, std::vector<NvmeEntry>& out) {
    if (!root["nvmes"]) return;
    for (const auto& node : root["nvmes"]) {
        NvmeEntry e;
        e.pci_addr       = get_or<std::string>(node, "pci_addr",       "");
        e.mount_path     = get_or<std::string>(node, "mount_path",     "");
        e.namespace_id   = get_or<uint32_t>   (node, "namespace_id",   1u);
        e.kernel_ioq_cap = get_or<uint32_t>   (node, "kernel_ioq_cap", 0u);

        if (node["allowed_gpus"]) {
            for (const auto& g : node["allowed_gpus"]) {
                e.allowed_gpus.push_back(g.as<int>());
            }
        }

        out.push_back(std::move(e));
    }
}

void parse_pool(const YAML::Node& root, QueuePoolConfig& out) {
    if (!root["queue_pool"]) return;
    const auto& q = root["queue_pool"];
    out.default_per_client = get_or<int>(q, "default_per_client", out.default_per_client);
    out.max_per_client     = get_or<int>(q, "max_per_client",     out.max_per_client);
}

void parse_lease(const YAML::Node& root, LeaseConfig& out) {
    if (!root["lease"]) return;
    const auto& l = root["lease"];
    out.heartbeat_interval_sec =
        get_or<uint32_t>(l, "heartbeat_interval_sec", out.heartbeat_interval_sec);
    out.timeout_sec =
        get_or<uint32_t>(l, "timeout_sec", out.timeout_sec);
}

} // namespace

std::optional<ServiceConfig> parse_config_file(const std::string& path,
                                                std::string* error) {
    try {
        YAML::Node root = YAML::LoadFile(path);
        ServiceConfig cfg;

        parse_grpc (root, cfg.grpc);
        parse_gpus (root, cfg.gpus);
        parse_nvmes(root, cfg.nvmes);
        parse_pool (root, cfg.queue_pool);
        parse_lease(root, cfg.lease);

        if (std::string verr; !validate_config(cfg, &verr)) {
            if (error) *error = "validation failed: " + verr;
            return std::nullopt;
        }
        return cfg;
    } catch (const YAML::Exception& e) {
        if (error) *error = std::string("YAML parse error: ") + e.what();
        return std::nullopt;
    } catch (const std::exception& e) {
        if (error) *error = std::string("parse error: ") + e.what();
        return std::nullopt;
    }
}

bool validate_config(const ServiceConfig& cfg, std::string* error) {
    auto emit = [&](const std::string& msg) {
        if (error) *error = msg;
        return false;
    };

    if (cfg.grpc.endpoint.empty()) {
        return emit("grpc.endpoint is empty");
    }
    if (cfg.gpus.empty()) {
        return emit("gpus list is empty");
    }
    if (cfg.nvmes.empty()) {
        return emit("nvmes list is empty");
    }

    std::set<int> gpu_ids;
    for (const auto& g : cfg.gpus) {
        if (g.id < 0) {
            return emit("gpus[].id must be >= 0");
        }
        if (!gpu_ids.insert(g.id).second) {
            std::ostringstream ss;
            ss << "duplicate gpus[].id: " << g.id;
            return emit(ss.str());
        }
        if (g.mount_path.empty()) {
            std::ostringstream ss;
            ss << "gpus[id=" << g.id << "].mount_path is empty";
            return emit(ss.str());
        }
    }

    std::set<std::string> pci_seen;
    std::set<std::string> mount_seen;
    for (const auto& n : cfg.nvmes) {
        if (n.pci_addr.empty()) {
            return emit("nvmes[].pci_addr is empty");
        }
        if (!pci_seen.insert(n.pci_addr).second) {
            return emit("duplicate nvmes[].pci_addr: " + n.pci_addr);
        }
        if (n.mount_path.empty()) {
            return emit("nvmes[pci=" + n.pci_addr + "].mount_path is empty");
        }
        if (!mount_seen.insert(n.mount_path).second) {
            return emit("duplicate nvmes[].mount_path: " + n.mount_path);
        }

        // allowed_gpus: every entry must reference a known gpus[].id.
        // Empty list means "any GPU"; that's valid.
        std::set<int> seen_in_acl;
        for (int gid : n.allowed_gpus) {
            if (gpu_ids.find(gid) == gpu_ids.end()) {
                std::ostringstream ss;
                ss << "nvmes[pci=" << n.pci_addr
                   << "].allowed_gpus references gpu_id=" << gid
                   << " which has no matching entry in gpus[]";
                return emit(ss.str());
            }
            if (!seen_in_acl.insert(gid).second) {
                std::ostringstream ss;
                ss << "nvmes[pci=" << n.pci_addr
                   << "].allowed_gpus has duplicate gpu_id=" << gid;
                return emit(ss.str());
            }
        }
    }

    if (cfg.queue_pool.default_per_client <= 0 ||
        cfg.queue_pool.max_per_client     <= 0 ||
        cfg.queue_pool.default_per_client >  cfg.queue_pool.max_per_client) {
        return emit("queue_pool: default_per_client must be in (0, max_per_client]");
    }

    if (cfg.lease.heartbeat_interval_sec == 0 ||
        cfg.lease.timeout_sec            == 0 ||
        cfg.lease.heartbeat_interval_sec >= cfg.lease.timeout_sec) {
        return emit("lease: heartbeat_interval_sec must be in (0, timeout_sec)");
    }

    return true;
}

} // namespace nvmeservice
