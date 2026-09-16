// csrc/storage_objects/store_factory.cpp
//
// create_storage_object_store(): the SPI's entry point, selecting a backend by
// URI scheme.
//
// Deliberately NOT built on resolvers::create_resolver(): that path needs
// ResolverSpec/BackendSpec/Resource from the config system, which would make the
// storage object layer depend on configuration parsing. Instead the returned
// store constructs its own resolver during open(), from the devices declared in
// StoreConfig -- so this layer depends on the resolvers but not on how a
// deployment happens to describe them.

#include <memory>
#include <string>
#include <string_view>

#include <tutti/spi/storage_object_store.h>

#include "csrc/storage_objects/object_store_core.h"

namespace tutti {

Result<std::unique_ptr<StorageObjectStore>> create_storage_object_store(
    std::string_view scheme) {
    // Both supported schemes are served by the same core; which placement and
    // resolver get built is decided at open() by StoreConfig::stripe_unit
    // (0 selects the single-file layout) together with the device list.
    //
    // The scheme is therefore validated here rather than dispatched on: naming
    // a layout the store cannot provide should fail at creation, not silently
    // resolve to something else.
    if (scheme == "local_nvme_file" || scheme == "striped_local_nvme_file") {
        return Result<std::unique_ptr<StorageObjectStore>>::Success(
            std::make_unique<storage_objects::ObjectStoreCore>());
    }
    return Result<std::unique_ptr<StorageObjectStore>>::Failure(
        Status(StatusCode::UNSUPPORTED,
               std::string("unknown storage object store scheme: ") +
                   std::string(scheme)));
}

} // namespace tutti
