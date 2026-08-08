#ifndef TUTTI_TESTS_NVME_TEST_CLI_H_
#define TUTTI_TESTS_NVME_TEST_CLI_H_

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace tutti::test_support {

struct NvmeTestDevice {
    std::string ssnvme_path;
    std::string pci_bdf;
    std::string backing_device;
    std::string mount_path;
    std::uint32_t block_size = 4096;
    std::uint32_t bar0_size = 16384;
    std::uint32_t namespace_id = 1;
};

inline std::vector<NvmeTestDevice> default_nvme_test_devices() {
    return {
        {"/dev/ssnvme0", "0000:08:00.0", "/dev/snvme0n1", "/mnt/nvme0"},
        {"/dev/ssnvme1", "0000:4b:00.0", "/dev/snvme1n1", "/mnt/nvme1"},
        {"/dev/ssnvme2", "0000:57:00.0", "/dev/snvme2n1", "/mnt/nvme2"},
        {"/dev/ssnvme3", "0000:63:00.0", "/dev/snvme3n1", "/mnt/nvme3"},
    };
}

inline bool parse_u32(const std::string& text, std::uint32_t* value) {
    if (text.empty() || text.front() == '-') return false;
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text.c_str(), &end, 10);
    if (errno != 0 || end == text.c_str() || *end != '\0' ||
        parsed > std::numeric_limits<std::uint32_t>::max()) {
        return false;
    }
    *value = static_cast<std::uint32_t>(parsed);
    return true;
}

inline bool parse_nvme_test_device(const char* value,
                                   NvmeTestDevice* device,
                                   std::string* error) {
    std::vector<std::string> fields;
    const std::string input(value ? value : "");
    std::size_t begin = 0;
    while (begin <= input.size()) {
        const std::size_t comma = input.find(',', begin);
        fields.push_back(input.substr(begin, comma - begin));
        if (comma == std::string::npos) break;
        begin = comma + 1;
    }

    if (fields.size() < 4 || fields.size() > 7) {
        *error = "expected 4 to 7 comma-separated fields";
        return false;
    }
    for (const auto& field : fields) {
        if (field.empty()) {
            *error = "fields must not be empty";
            return false;
        }
    }

    NvmeTestDevice parsed;
    parsed.ssnvme_path = std::move(fields[0]);
    parsed.pci_bdf = std::move(fields[1]);
    parsed.backing_device = std::move(fields[2]);
    parsed.mount_path = std::move(fields[3]);

    if (fields.size() >= 5 &&
        (!parse_u32(fields[4], &parsed.block_size) ||
         parsed.block_size < 512 ||
         (parsed.block_size & (parsed.block_size - 1)) != 0)) {
        *error = "block_size must be a power of two of at least 512 bytes";
        return false;
    }
    if (fields.size() >= 6 &&
        (!parse_u32(fields[5], &parsed.bar0_size) || parsed.bar0_size == 0)) {
        *error = "bar0_size must be a positive integer";
        return false;
    }
    if (fields.size() >= 7 &&
        (!parse_u32(fields[6], &parsed.namespace_id) ||
         parsed.namespace_id == 0)) {
        *error = "namespace_id must be a positive integer";
        return false;
    }

    *device = std::move(parsed);
    return true;
}

inline std::string nvme_test_device_format() {
    return "ssnvme_path,pci_bdf,backing_device,mount_path"
           "[,block_size[,bar0_size[,namespace_id]]]";
}

}  // namespace tutti::test_support

#endif  // TUTTI_TESTS_NVME_TEST_CLI_H_
