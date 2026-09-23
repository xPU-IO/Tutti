#pragma once

// csrc/storage_objects/slot_placement_policy.h -- slot number -> media location.
//
// IMPLEMENTATION DETAIL. Reached only through the SPI.
//
// This is the ONLY seam between the shared object-store core and the question
// of how a slot maps onto media. The core allocates slot numbers, encodes
// headers, maintains checkpoints and bitmaps, and recovers -- all without
// knowing which device a slot's bytes live on.
//
// Placement is a pure function of the slot number: a slot is ONE contiguous
// file on ONE device, and the slot number picks the device (see
// RotatingFilePlacement). Nothing is interleaved below this seam, so a logical
// offset in the object is a file offset is a device byte offset -- there is no
// stripe arithmetic to keep in sync between the layout, the resolver and the
// DataPath, and the bytes a device must be able to DMA are exactly the bytes
// of the slots it owns.
//
// Header placement: the object header lives at offset 0 of the slot's file,
// and the payload starts right after it.

#include <cstdint>
#include <string>
#include <vector>

#include <tutti/status.h>
#include <tutti/spi/storage_object_store.h>

namespace tutti::storage_objects {

class SlotPlacementPolicy {
public:
    virtual ~SlotPlacementPolicy() = default;

    // Physical files backing one slot. 1 for every layout in this file; the
    // plural form is kept because a different backend may split a slot.
    virtual std::uint32_t shard_count() const = 0;

    // Scheme to put in ResolveOptions. Distinct from the URI prefix only in
    // that it carries no "://" -- the resolvers check this field before parsing.
    virtual std::string resolver_scheme() const = 0;

    // Filesystem paths of the file(s) backing `slot`. Used for materialisation
    // (writing real zeros) and for header IO -- never handed to the caller, who
    // only ever sees an ObjectPlacement.
    //
    // MUST agree byte for byte with the path in uri_for_slot(), because the
    // resolver maps the URI's path, not this one: a mismatch would have
    // materialisation write one file while resolution maps another -- and DMA
    // would target unallocated extents.
    virtual Status paths_for_slot(std::uint64_t slot,
                                  std::vector<std::string>* out) const = 0;

    // Bytes the file(s) backing a slot of `slot_bytes` must be. Equal to
    // slot_bytes whenever a slot is one file.
    virtual std::uint64_t shard_file_bytes(std::uint64_t slot_bytes) const = 0;

    // URI identifying this slot, for the runtime to open. Its format is fixed by
    // the resolver that will parse it, not chosen here.
    virtual std::string uri_for_slot(std::uint64_t slot) const = 0;

    // Where the object header lives: which file, and at what offset in it.
    // Always file 0 at offset 0 in the current layouts, but expressed explicitly
    // so a backend with a different metadata region does not have to fight the
    // core.
    virtual std::uint32_t header_shard() const { return 0; }
    virtual std::uint64_t header_offset_in_shard() const { return 0; }

    // Offset of the payload within the object's logical address space. Equals
    // the reserved header prefix, so segment 0 starts here.
    virtual std::uint64_t payload_offset() const {
        return ObjectHeaderLayout::kHeaderBytes;
    }
};

// -------------------------------------------------------------------------
// SingleFilePlacement -- one file per slot.
//
// Layout of "<root>/slots/<slot>.obj":
//   [0, 4096)                 object header
//   [4096, 4096 + payload)    payload, segments back to back
//
// The local-file resolver takes the path verbatim from a "file://<abs path>"
// URI, so the path shape is this policy's choice.
// -------------------------------------------------------------------------
class SingleFilePlacement final : public SlotPlacementPolicy {
public:
    explicit SingleFilePlacement(std::string root);

    std::uint32_t shard_count() const override { return 1; }
    std::string resolver_scheme() const override { return "file"; }
    Status paths_for_slot(std::uint64_t slot,
                          std::vector<std::string>* out) const override;
    std::uint64_t shard_file_bytes(std::uint64_t slot_bytes) const override;
    std::string uri_for_slot(std::uint64_t slot) const override;

    const std::string& root() const { return root_; }

private:
    std::string slot_path(std::uint64_t slot) const;

    std::string root_;
};

// -------------------------------------------------------------------------
// RotatingFilePlacement -- one file per slot, rotating across devices.
//
// A slot is ONE contiguous file living entirely on ONE device, chosen by the
// slot number:
//     device = slot % N        path = <mount[device]>/<subdir>/<slot>.obj
//
// Consecutive slots therefore spread evenly over all N devices while a single
// slot's bytes never straddle a device boundary. Combined with the store
// handing out consecutive slot numbers for a prompt's consecutive chunks, a
// long prompt ends up with thousands of one-file IOs spread over all devices.
//
// Why this shape (vs. RAID-style interleaving): the memory a slot occupies is
// one contiguous GPU range, and this layout makes that range map 1:1 onto the
// file and onto the device byte address space. That removes the stripe
// arithmetic that otherwise has to agree between three components, and it makes
// the set of bytes a device needs DMA access to exactly the slots it owns.
//
// Layout of "<mount>/<subdir>/<slot>.obj":
//   [0, 4096)                 object header
//   [4096, 4096 + payload)    payload, segments back to back
// -------------------------------------------------------------------------
class RotatingFilePlacement final : public SlotPlacementPolicy {
public:
    // `mounts` is one directory per device; slots rotate over them in order.
    // `subdir` is created under each mount.
    explicit RotatingFilePlacement(std::vector<std::string> mounts,
                                   std::string subdir = "chunks");

    // One file per slot: the file *is* the slot, so there is no shard set.
    std::uint32_t shard_count() const override { return 1; }

    // Slots live on different devices but every one of them is a plain file at
    // an absolute path, so a single "file://" resolver serves all of them: it
    // picks the backing device from the path's mount, not from the scheme.
    std::string resolver_scheme() const override { return "file"; }

    Status paths_for_slot(std::uint64_t slot,
                          std::vector<std::string>* out) const override;
    std::uint64_t shard_file_bytes(std::uint64_t slot_bytes) const override {
        // The whole slot, header included, is one file.
        return slot_bytes;
    }
    std::string uri_for_slot(std::uint64_t slot) const override;

    const std::vector<std::string>& mounts() const { return mounts_; }
    const std::string& subdir() const { return subdir_; }

    std::uint64_t device_for_slot(std::uint64_t slot) const {
        return slot % mounts_.size();
    }

    // True when at least one mount is configured. Checked by the store at
    // open() so a bad configuration fails loudly instead of materialising into
    // a path the resolver cannot map to any device.
    bool geometry_valid() const { return !mounts_.empty(); }

private:
    std::string slot_path(std::uint64_t slot) const;

    std::vector<std::string> mounts_;
    std::string subdir_;
};

} // namespace tutti::storage_objects
