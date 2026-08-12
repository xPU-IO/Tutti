#include <tutti/config/spec/backend/backend_spec.h>
#include <tutti/config/spec/backend/ext4_local_nvme_spec.h>
#include <tutti/config/spec/backend/memfs_spec.h>
#include <tutti/config/spec/backend/striped_local_nvme_spec.h>
#include <tutti/config/spec/datapath/datapath_spec.h>
#include <tutti/config/spec/datapath/local_nvme_spec.h>
#include <tutti/config/spec/datapath/memfs_spec.h>
#include <tutti/config/spec/datapath/nvme_datapath_spec.h>
#include <tutti/config/spec/datapath/striped_local_nvme_spec.h>
#include <tutti/config/spec/resolver/local_file_spec.h>
#include <tutti/config/spec/resolver/memfs_spec.h>
#include <tutti/config/spec/resolver/resolver_spec.h>
#include <tutti/config/spec/resolver/striped_file_spec.h>
#include <tutti/config/spec/resource/memory_spec.h>
#include <tutti/config/spec/resource/nvme_spec.h>
#include <tutti/config/spec/resource/resource_spec.h>
#include <tutti/config/tutti_runtime_spec.h>

#include <cstdint>
#include <type_traits>

static_assert(std::is_default_constructible_v<tutti::config::ResourceSpec>);
static_assert(std::is_default_constructible_v<tutti::config::ResolverSpec>);
static_assert(std::is_default_constructible_v<tutti::config::DataPathSpec>);
static_assert(std::is_default_constructible_v<tutti::config::BackendSpec>);
static_assert(std::is_default_constructible_v<tutti::config::TuttiRuntimeSpec>);
static_assert(tutti::config::kDefaultStripedStripeUnit ==
              std::uint64_t{512} * 1024);
static_assert(tutti::config::NvmeDataPathTuning::kDefaultThreadsPerBlock == 16);

int main() {
    tutti::config::StripedLocalNvmeBackendConfig striped;
    return striped.stripe_unit == tutti::config::kDefaultStripedStripeUnit
               ? 0
               : 1;
}
