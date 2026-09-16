#pragma once

// tutti/data_paths/local_nvme/io/prp_builder.h
//
// Private PRP descriptor builder — ported from main:
//   memory/include/memory_subsystem.h:93-98  (AddressDescriptor)
//   memory/src/host_device_memory_subsystem.cu:828-893 (fill_address_descriptors)
//
// Computes SINGLE/DUAL/LIST PRP descriptors from nvm_dma_t::ioaddrs[].
// PRP-list backing is supplied by host-pinned PrpBufPool/PrpPageCache.

#include <nvm_types.h>

#include <cstdint>
#include <string>
#include <vector>

namespace tutti::data_paths::local_nvme {

// -------------------------------------------------------------------------
// AddressDescriptor — ported verbatim from main:memory/include/memory_subsystem.h
// -------------------------------------------------------------------------
struct AddressDescriptor {
    std::uint64_t prp1;         // first data page DMA IOVA
    std::uint64_t prp2;         // second page IOVA (DUAL) or PRP-list page IOVA (LIST) or 0 (SINGLE)
    std::uint64_t data_length;  // bytes in this sub-IO
};

// PRP kind classification.
enum class PrpKind {
    SINGLE,  // 1 page: prp1 = page, prp2 = 0
    DUAL,   // 2 pages: prp1 = page0, prp2 = page1
    LIST,   // >2 pages: prp1 = page0, prp2 = PRP-list page DMA IOVA
};

// Determine PRP kind from page count.
inline PrpKind classify_prp(std::uint32_t pages_in_io) {
    if (pages_in_io <= 1) return PrpKind::SINGLE;
    if (pages_in_io == 2) return PrpKind::DUAL;
    return PrpKind::LIST;
}

// -------------------------------------------------------------------------
// fill_prp_descriptor — ported from main:fill_address_descriptors.
//
// Fills a single AddressDescriptor for one sub-IO.
//
// Parameters:
//   data_dma        — nvm_dma_t from register_memory (has ioaddrs[])
//   start_page      — index into data_dma->ioaddrs[] for this sub-IO's first page
//   pages_in_io     — number of data pages in this sub-IO
//   data_length     — byte length of this sub-IO
//   prp_list_iova   — DMA IOVA of the PRP-list page for this sub-IO (LIST only)
//                     Pass 0 for SINGLE/DUAL.
//
// Returns the filled AddressDescriptor.
// -------------------------------------------------------------------------
inline AddressDescriptor fill_prp_descriptor(
    const nvm_dma_t* data_dma,
    std::uint32_t    start_page,
    std::uint32_t    pages_in_io,
    std::uint64_t    data_length,
    std::uint64_t    prp_list_iova)
{
    AddressDescriptor d{};
    d.data_length = data_length;

    // prp1 = first data page IOVA (always set).
    d.prp1 = data_dma->ioaddrs[start_page];

    PrpKind kind = classify_prp(pages_in_io);
    switch (kind) {
        case PrpKind::SINGLE:
            d.prp2 = 0;
            break;
        case PrpKind::DUAL:
            d.prp2 = data_dma->ioaddrs[start_page + 1];
            break;
        case PrpKind::LIST:
            // prp2 = DMA IOVA of the PRP-list page (not a CUDA pointer).
            // The PRP-list page content is filled separately by the caller.
            d.prp2 = prp_list_iova;
            break;
    }
    return d;
}

// -------------------------------------------------------------------------
// fill_prp_list_page — ported from main:fill_address_descriptors LIST branch.
//
// Fills a host-side buffer with the PRP-list page content for one LIST sub-IO.
// The buffer must be at least page_size bytes.
//
// Content:
//   entry[0] = data_dma->ioaddrs[start_page + 1]
//   entry[1] = data_dma->ioaddrs[start_page + 2]
//   ...
//   entry[pages_in_io - 2] = data_dma->ioaddrs[start_page + pages_in_io - 1]
//   remaining entries = 0
//
// Parameters:
//   page_buf     — host buffer to fill (at least page_size bytes)
//   data_dma     — nvm_dma_t from register_memory
//   start_page   — index of this sub-IO's first data page
//   pages_in_io  — total data pages in this sub-IO
//   page_size    — NVMe page size (typically 4096)
// -------------------------------------------------------------------------
inline void fill_prp_list_page(
    std::uint64_t*   page_buf,
    const nvm_dma_t* data_dma,
    std::uint32_t    start_page,
    std::uint32_t    pages_in_io,
    std::size_t      page_size)
{
    std::size_t entries_per_page = page_size / sizeof(std::uint64_t);
    // Zero the page first.
    for (std::size_t i = 0; i < entries_per_page; ++i) {
        page_buf[i] = 0;
    }
    // Fill entries [0 .. pages_in_io - 2] with data page IOVAs [start_page+1 .. start_page+pages_in_io-1].
    for (std::uint32_t p = 1; p < pages_in_io; ++p) {
        page_buf[p - 1] = data_dma->ioaddrs[start_page + p];
    }
}

} // namespace tutti::data_paths::local_nvme
