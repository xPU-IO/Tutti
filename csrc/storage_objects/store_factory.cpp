// csrc/storage_objects/store_factory.cpp
//
// create_storage_object_store(): the SPI's entry point, selecting a layout by
// scheme.
//
// This layer does not construct resolvers and does not depend on
// csrc/resolvers/ at all. Resolution belongs to StorageRuntime, which takes the
// slot URIs this store hands out. Keeping it that way means an object is
// resolved exactly once -- and resolution is open + fstat + fsync + FIEMAP plus
// a globally-serialised peer-memory DMA mapping, so doing it twice would be a
// multi-second cost on a cold pool, not a rounding error.

#include <memory>
#include <string>
#include <string_view>

#include "csrc/common/backend_ids.h"
#include <tutti/spi/storage_object_store.h>

#include "csrc/storage_objects/object_store_core.h"

namespace tutti {

Result<std::unique_ptr<StorageObjectStore>> create_storage_object_store(
    std::string_view scheme) {
    // Both supported schemes are served by the same core; which placement gets
    // built is decided at open() by StoreConfig::stripe_unit (0 selects the
    // single-file layout) together with the device list.
    //
    // The scheme is therefore validated here rather than dispatched on: naming
    // a layout the store cannot provide should fail at creation, not silently
    // resolve to something else.
    if (scheme == tutti::detail::backend_ids::kExt4StoreScheme ||
        scheme == tutti::detail::backend_ids::kStripedStoreScheme) {
        return Result<std::unique_ptr<StorageObjectStore>>::Success(
            std::make_unique<storage_objects::ObjectStoreCore>());
    }
    return Result<std::unique_ptr<StorageObjectStore>>::Failure(
        Status(StatusCode::UNSUPPORTED,
               std::string("unknown storage object store scheme: ") +
                   std::string(scheme)));
}

} // namespace tutti
