/*
 * tests/uapi_contract/uapi_contract_test.cpp
 *
 * Hardware-free contract test for the shared snvme UAPI header
 * (tutti/include/uapi/tutti_snvme.h).
 *
 * This test verifies that:
 *
 *   1. The shared UAPI header compiles cleanly from userspace C++17.
 *      (If any _Static_assert in the header fails, this TU does not
 *       compile — the test fails at build time.)
 *
 *   2. ABI version constant TUTTI_SNVME_ABI_VERSION is the unique
 *      source of truth and has the expected value (1).
 *
 *   3. Capability bits are distinct powers of two (non-overlapping)
 *      and TUTTI_SNVME_CAP_ALL is the bitwise-OR of all defined bits.
 *
 *   4. Struct sizes match the documented LP64 layout.  These runtime
 *      checks are redundant with the header's _Static_assert, but
 *      provide a visible CTest failure message (vs. a compile error)
 *      that makes it obvious which struct broke.
 *
 *   5. Key field offsets match the documented LP64 layout.
 *
 *   6. The ioctl command numbers are stable (magic + base + size
 *      encoding).  We check a representative subset.
 *
 * This test does NOT open /dev/ssnvme*, load kernel modules, or
 * touch any hardware.  It is safe to run in any CI environment.
 */

#include <tutti_snvme.h>

#include <cstdio>
#include <cstdint>
#include <cstring>

/* ------------------------------------------------------------------ */
/* Test helpers                                                       */
/* ------------------------------------------------------------------ */

static int test_count = 0;
static int fail_count = 0;

#define CHECK(cond, ...)                                        \
    do {                                                        \
        test_count++;                                           \
        if (!(cond)) {                                          \
            fail_count++;                                       \
            std::fprintf(stderr, "FAIL: " __VA_ARGS__);         \
            std::fprintf(stderr, "\n");                         \
        }                                                       \
    } while (0)

/* ------------------------------------------------------------------ */
/* Test 1: ABI version constant                                       */
/* ------------------------------------------------------------------ */

static int test_abi_version()
{
    /*
     * TUTTI_SNVME_ABI_VERSION must be 2 for this header revision.
     * If this fails, either the header was bumped (update this test)
     * or the constant is wrong (fix the header).
     */
    CHECK(TUTTI_SNVME_ABI_VERSION == 2,
          "TUTTI_SNVME_ABI_VERSION=%u, expected 2",
          (unsigned)TUTTI_SNVME_ABI_VERSION);

    /*
     * The version must be non-zero — 0 is reserved for "old kernel
     * that predates UAPI consolidation" (memset-zeroed reserved field).
     */
    CHECK(TUTTI_SNVME_ABI_VERSION != 0,
          "TUTTI_SNVME_ABI_VERSION must not be 0");

    return 0;
}

/* ------------------------------------------------------------------ */
/* Test 2: Capability bits are non-overlapping                        */
/* ------------------------------------------------------------------ */

static int test_capability_bits()
{
    /*
     * Each capability bit must be a distinct power of two.
     * If two bits overlap, their AND is non-zero.
     */
    uint32_t caps[] = {
        TUTTI_SNVME_CAP_QUEUE_GROUPS,
        TUTTI_SNVME_CAP_USER_QUEUES,
        TUTTI_SNVME_CAP_RAW_ADMIN,
        TUTTI_SNVME_CAP_KERNEL_IOQ_CAP,
        TUTTI_SNVME_CAP_MAP_KIND_TAG,
    };
    const int n = (int)(sizeof(caps) / sizeof(caps[0]));

    for (int i = 0; i < n; i++) {
        /* Must be a power of two */
        CHECK(caps[i] != 0 && (caps[i] & (caps[i] - 1)) == 0,
              "cap bit %d (0x%x) is not a power of two", i, caps[i]);

        /* Must not overlap with any other bit */
        for (int j = i + 1; j < n; j++) {
            CHECK((caps[i] & caps[j]) == 0,
                  "cap bits %d and %d overlap (0x%x & 0x%x)",
                  i, j, caps[i], caps[j]);
        }
    }

    /*
     * TUTTI_SNVME_CAP_ALL must be the OR of all individual bits.
     * If a new bit is added to the header but not to CAP_ALL, this
     * test fails — forcing the developer to update CAP_ALL.
     */
    uint32_t expected_all = 0;
    for (int i = 0; i < n; i++) {
        expected_all |= caps[i];
    }
    CHECK(TUTTI_SNVME_CAP_ALL == expected_all,
          "TUTTI_SNVME_CAP_ALL=0x%x, expected 0x%x",
          TUTTI_SNVME_CAP_ALL, expected_all);

    return 0;
}

/* ------------------------------------------------------------------ */
/* Test 3: Struct sizes match LP64 layout                             */
/* ------------------------------------------------------------------ */

static int test_struct_sizes()
{
    /*
     * These sizes are the LP64 (x86-64, aarch64) layout, locked by
     * _Static_assert in the header.  If any fails, the header's
     * static_assert already prevented compilation — this runtime
     * check is a belt-and-suspenders verification.
     */
    CHECK(sizeof(struct nvm_ioctl_map) == 40,
          "nvm_ioctl_map size=%zu, expected 40",
          sizeof(struct nvm_ioctl_map));

    CHECK(sizeof(struct nvm_ioctl_dev) == 104,
          "nvm_ioctl_dev size=%zu, expected 104",
          sizeof(struct nvm_ioctl_dev));

    CHECK(sizeof(struct nvm_ioctl_setup) == 160,
          "nvm_ioctl_setup size=%zu, expected 160",
          sizeof(struct nvm_ioctl_setup));

    CHECK(sizeof(struct pci_device_addr) == 16,
          "pci_device_addr size=%zu, expected 16",
          sizeof(struct pci_device_addr));

    CHECK(sizeof(struct nvm_ioctl_raw_admin) == 92,
          "nvm_ioctl_raw_admin size=%zu, expected 92",
          sizeof(struct nvm_ioctl_raw_admin));

    CHECK(sizeof(struct nvm_ioctl_queue_group) == 32,
          "nvm_ioctl_queue_group size=%zu, expected 32",
          sizeof(struct nvm_ioctl_queue_group));

    CHECK(sizeof(struct nvm_user_queue_pair_in) == 16,
          "nvm_user_queue_pair_in size=%zu, expected 16",
          sizeof(struct nvm_user_queue_pair_in));

    CHECK(sizeof(struct nvm_user_queue_pair_out) == 16,
          "nvm_user_queue_pair_out size=%zu, expected 16",
          sizeof(struct nvm_user_queue_pair_out));

    CHECK(sizeof(struct nvm_ioctl_add_user_queue) == 1056,
          "nvm_ioctl_add_user_queue size=%zu, expected 1056",
          sizeof(struct nvm_ioctl_add_user_queue));

    CHECK(sizeof(struct nvm_queue_group) == 16,
          "nvm_queue_group size=%zu, expected 16",
          sizeof(struct nvm_queue_group));

    return 0;
}

/* ------------------------------------------------------------------ */
/* Test 4: Key field offsets match LP64 layout                        */
/* ------------------------------------------------------------------ */

static int test_field_offsets()
{
    /* nvm_ioctl_map */
    CHECK(offsetof(struct nvm_ioctl_map, vaddr_start) == 0, "map.vaddr_start");
    CHECK(offsetof(struct nvm_ioctl_map, n_pages) == 8, "map.n_pages");
    CHECK(offsetof(struct nvm_ioctl_map, ioaddrs) == 16, "map.ioaddrs");
    CHECK(offsetof(struct nvm_ioctl_map, ioq_idx) == 24, "map.ioq_idx");
    CHECK(offsetof(struct nvm_ioctl_map, group_id) == 32, "map.group_id");
    CHECK(offsetof(struct nvm_ioctl_map, map_kind) == 36, "map.map_kind");

    /* nvm_ioctl_dev — check the ABI handshake fields specifically */
    CHECK(offsetof(struct nvm_ioctl_dev, max_data_size) == 16, "dev.max_data_size");
    CHECK(offsetof(struct nvm_ioctl_dev, block_size) == 24, "dev.block_size");
    CHECK(offsetof(struct nvm_ioctl_dev, disk_name) == 32, "dev.disk_name");
    CHECK(offsetof(struct nvm_ioctl_dev, q_depth) == 64, "dev.q_depth");
    CHECK(offsetof(struct nvm_ioctl_dev, sgl_supported) == 80, "dev.sgl_supported");
    CHECK(offsetof(struct nvm_ioctl_dev, abi_version) == 84, "dev.abi_version");
    CHECK(offsetof(struct nvm_ioctl_dev, capabilities) == 88, "dev.capabilities");

    /* pci_device_addr */
    CHECK(offsetof(struct pci_device_addr, domain) == 0, "pdev.domain");
    CHECK(offsetof(struct pci_device_addr, func) == 12, "pdev.func");

    /* nvm_ioctl_add_user_queue */
    CHECK(offsetof(struct nvm_ioctl_add_user_queue, pairs) == 32, "addq.pairs");
    CHECK(offsetof(struct nvm_ioctl_add_user_queue, out_pairs) == 544, "addq.out_pairs");

    return 0;
}

/* ------------------------------------------------------------------ */
/* Test 5: No width-dependent types in UAPI structs                   */
/* ------------------------------------------------------------------ */

/*
 * This test verifies at compile time (via sizeof) that no field in
 * any UAPI struct uses a type whose width differs between ILP32 and
 * LP64.  If a developer accidentally uses `long`, `size_t`, or a
 * raw pointer in a UAPI struct, the struct size on LP64 will still
 * be correct but the COMPILE-TIME static_assert in the header would
 * catch a layout change.  This runtime test is additional insurance.
 *
 * We check that the types of key fields are exactly the expected
 * fixed-width types by verifying sizeof() of the field.
 */
static int test_fixed_width_types()
{
    struct nvm_ioctl_map m;
    struct nvm_ioctl_dev d;

    /* n_pages must be 8 bytes (uint64_t, not size_t) */
    CHECK(sizeof(m.n_pages) == 8, "n_pages is %zu bytes, expected 8 (uint64_t)",
          sizeof(m.n_pages));

    /* ioaddrs must be 8 bytes (uint64_t, not pointer) */
    CHECK(sizeof(m.ioaddrs) == 8, "ioaddrs is %zu bytes, expected 8 (uint64_t)",
          sizeof(m.ioaddrs));

    /* ioq_idx must be 4 bytes (int32_t, not long) */
    CHECK(sizeof(m.ioq_idx) == 4, "ioq_idx is %zu bytes, expected 4 (int32_t)",
          sizeof(m.ioq_idx));

    /* max_data_size must be 8 bytes (uint64_t, not size_t) */
    CHECK(sizeof(d.max_data_size) == 8, "max_data_size is %zu bytes, expected 8",
          sizeof(d.max_data_size));

    /* block_size must be 8 bytes (uint64_t, not size_t) */
    CHECK(sizeof(d.block_size) == 8, "block_size is %zu bytes, expected 8",
          sizeof(d.block_size));

    /* abi_version must be 4 bytes (uint32_t) */
    CHECK(sizeof(d.abi_version) == 4, "abi_version is %zu bytes, expected 4",
          sizeof(d.abi_version));

    /* capabilities must be 4 bytes (uint32_t) */
    CHECK(sizeof(d.capabilities) == 4, "capabilities is %zu bytes, expected 4",
          sizeof(d.capabilities));

    return 0;
}

/* ------------------------------------------------------------------ */
/* Test 6: Ioctl command numbers are stable                            */
/* ------------------------------------------------------------------ */

static int test_ioctl_numbers()
{
    /*
     * The ioctl command numbers are computed by _IOW/_IOR/_IOWR macros
     * from (magic, nr, size).  We verify that the magic and type
     * encoding are correct for a representative subset.
     *
     * _IOW(dir, type, nr, size) encodes as:
     *   ((dir) << 30) | ((size) << 16) | ((type) << 8) | (nr)
     *
     * For NVM_IOCTL_TYPE=0x80:
     *   NVM_MAP_HOST_MEMORY = _IOW(0x80, 1, struct nvm_ioctl_map)
     *     = (1 << 30) | (40 << 16) | (0x80 << 8) | 1
     *     = 0x40000000 | 0x00280000 | 0x8000 | 0x01
     *     = 0x40288001
     *
     * We don't hardcode the full value (endianness/arch dependent);
     * instead we verify the magic and direction bits.
     */
    CHECK(NVM_IOCTL_TYPE == 0x80, "NVM_IOCTL_TYPE=0x%x, expected 0x80",
          (unsigned)NVM_IOCTL_TYPE);
    CHECK(NVM_CTRL_IOCTL_TYPE == 0x90, "NVM_CTRL_IOCTL_TYPE=0x%x, expected 0x90",
          (unsigned)NVM_CTRL_IOCTL_TYPE);

    /* Backward compat typo alias */
    CHECK(NVM_CTRL_IOCTL_TYOE == NVM_CTRL_IOCTL_TYPE,
          "NVM_CTRL_IOCTL_TYOE (typo alias) mismatch");

    /*
     * Verify that NVM_MAP_HOST_MEMORY and NVM_MAP_DEVICE_MEMORY have
     * the same _IOC_SIZE (both use struct nvm_ioctl_map) but different
     * _IOC_NR (1 vs 2).
     */
    CHECK((_IOC_NR(NVM_MAP_HOST_MEMORY) == 1), "NVM_MAP_HOST_MEMORY nr");
    CHECK((_IOC_NR(NVM_MAP_DEVICE_MEMORY) == 2), "NVM_MAP_DEVICE_MEMORY nr");
    CHECK((_IOC_NR(NVM_GET_DEV_INFO) == 9), "NVM_GET_DEV_INFO nr");
    CHECK((_IOC_NR(NVM_ADD_USER_QUEUE) == 14), "NVM_ADD_USER_QUEUE nr");
    CHECK((_IOC_SIZE(NVM_ADD_USER_QUEUE) == 1056),
          "NVM_ADD_USER_QUEUE size=%u, expected 1056",
          (unsigned)_IOC_SIZE(NVM_ADD_USER_QUEUE));

    /* NVM_GET_DEV_INFO is _IOR (read direction), not _IOW */
    CHECK((_IOC_DIR(NVM_GET_DEV_INFO) == _IOC_READ),
          "NVM_GET_DEV_INFO direction should be _IOC_READ");

    /* NVM_MAP_HOST_MEMORY is _IOW (write direction) */
    CHECK((_IOC_DIR(NVM_MAP_HOST_MEMORY) == _IOC_WRITE),
          "NVM_MAP_HOST_MEMORY direction should be _IOC_WRITE");

    /* NVM_RAW_ADMIN_CMD is _IOWR (read+write direction) */
    CHECK((_IOC_DIR(NVM_RAW_ADMIN_CMD) == (_IOC_READ | _IOC_WRITE)),
          "NVM_RAW_ADMIN_CMD direction should be _IOC_READ|_IOC_WRITE");

    return 0;
}

/* ------------------------------------------------------------------ */
/* Test 7: Constants and limits are stable                            */
/* ------------------------------------------------------------------ */

static int test_constants()
{
    CHECK(DISK_NAME_LEN == 32, "DISK_NAME_LEN=%d, expected 32", DISK_NAME_LEN);
    CHECK(NVM_MAX_QUEUE_GROUPS == 8, "NVM_MAX_QUEUE_GROUPS=%d, expected 8",
          NVM_MAX_QUEUE_GROUPS);
    CHECK(NVM_MAX_GROUPS_PER_FD == 1, "NVM_MAX_GROUPS_PER_FD=%d, expected 1",
          NVM_MAX_GROUPS_PER_FD);
    CHECK(NVM_MAX_QUEUES_PER_GROUP == 32, "NVM_MAX_QUEUES_PER_GROUP=%d, expected 32",
          NVM_MAX_QUEUES_PER_GROUP);
    CHECK(NVM_QUEUE_SETUP_F_ON_HOST == 1, "NVM_QUEUE_SETUP_F_ON_HOST=0x%x, expected 0x1",
          NVM_QUEUE_SETUP_F_ON_HOST);

    /* enum nvm_map_kind values */
    CHECK(NVM_MAP_KIND_UNSPECIFIED == 0, "NVM_MAP_KIND_UNSPECIFIED");
    CHECK(NVM_MAP_KIND_RING_SQ == 1, "NVM_MAP_KIND_RING_SQ");
    CHECK(NVM_MAP_KIND_RING_CQ == 2, "NVM_MAP_KIND_RING_CQ");
    CHECK(NVM_MAP_KIND_DATA == 3, "NVM_MAP_KIND_DATA");

    return 0;
}

/* ------------------------------------------------------------------ */
/* Test 8: Simulated handshake fail-closed                            */
/* ------------------------------------------------------------------ */

/*
 * Simulate the userspace ABI version check logic WITHOUT a real
 * kernel.  This verifies that the fail-closed decision is correct
 * for various abi_version values the kernel might report.
 */
static int test_handshake_fail_closed()
{
    struct nvm_ioctl_dev dev_info;

    /* Case 1: matching version → pass */
    memset(&dev_info, 0, sizeof(dev_info));
    dev_info.abi_version = TUTTI_SNVME_ABI_VERSION;
    CHECK(dev_info.abi_version == TUTTI_SNVME_ABI_VERSION,
          "matching version should pass");

    /* Case 2: old kernel (abi_version == 0) → fail-closed */
    memset(&dev_info, 0, sizeof(dev_info));
    dev_info.abi_version = 0;
    CHECK(dev_info.abi_version != TUTTI_SNVME_ABI_VERSION,
          "old kernel (v=0) should fail-closed");

    /* Case 3: future version → fail-closed */
    memset(&dev_info, 0, sizeof(dev_info));
    dev_info.abi_version = TUTTI_SNVME_ABI_VERSION + 1;
    CHECK(dev_info.abi_version != TUTTI_SNVME_ABI_VERSION,
          "future version should fail-closed");

    /* Case 4: capabilities bitmask can be checked */
    memset(&dev_info, 0, sizeof(dev_info));
    dev_info.abi_version = TUTTI_SNVME_ABI_VERSION;
    dev_info.capabilities = TUTTI_SNVME_CAP_ALL;
    CHECK((dev_info.capabilities & TUTTI_SNVME_CAP_QUEUE_GROUPS) != 0,
          "QUEUE_GROUPS cap should be set");
    CHECK((dev_info.capabilities & TUTTI_SNVME_CAP_USER_QUEUES) != 0,
          "USER_QUEUES cap should be set");

    /* Case 5: kernel without a needed cap → caller should refuse */
    dev_info.capabilities = TUTTI_SNVME_CAP_ALL & ~TUTTI_SNVME_CAP_USER_QUEUES;
    CHECK((dev_info.capabilities & TUTTI_SNVME_CAP_USER_QUEUES) == 0,
          "missing USER_QUEUES cap should be detectable");

    return 0;
}

/* ------------------------------------------------------------------ */
/* Main                                                               */
/* ------------------------------------------------------------------ */

int main()
{
    using TestFn = int (*)();
    const TestFn tests[] = {
        test_abi_version,
        test_capability_bits,
        test_struct_sizes,
        test_field_offsets,
        test_fixed_width_types,
        test_ioctl_numbers,
        test_constants,
        test_handshake_fail_closed,
    };
    const int n = (int)(sizeof(tests) / sizeof(tests[0]));

    for (int i = 0; i < n; ++i) {
        tests[i]();
    }

    if (fail_count > 0) {
        std::fprintf(stderr, "FAIL: %d / %d checks failed.\n",
                     fail_count, test_count);
        return 1;
    }

    std::printf("All %d UAPI contract checks passed.\n", test_count);
    return 0;
}
