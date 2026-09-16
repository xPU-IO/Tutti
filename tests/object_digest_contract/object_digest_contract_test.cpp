// tests/object_digest_contract/object_digest_contract_test.cpp
//
// Contract test for <tutti/spi/object_digest.h>.
//
// These digests are PERSISTENT ON-MEDIA FORMAT. Once a deployment has written
// objects, changing either algorithm invalidates every stored object:
// recovery would compute a different value than the header records and
// discard live data as corrupt. This test exists to make such a change
// impossible to land accidentally.
//
// Two classes of assertion:
//
//   1. FROZEN VALUES. CRC-32C is checked against the standard test vectors
//      (RFC 3720 / iSCSI, also used by SCTP and ext4 metadata), so an
//      independent implementation -- a debugging tool, another language, a
//      hardware CRC unit -- will agree bit for bit. object_identity is
//      checked against literals captured from this implementation; they are
//      arbitrary but must never change.
//
//   2. STRUCTURAL PROPERTIES that callers rely on: incremental == one-shot,
//      byte-order independence, avalanche, empty-input handling.
//
// Hardware-free: plain C++17, no IO, no CUDA.

#include <tutti/spi/object_digest.h>
#include <tutti/spi/storage_object_store.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool cond, const char* expr, int line) {
    if (!cond) {
        std::printf("FAIL [line %d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(cond) check((cond), #cond, __LINE__)

void check_crc(const void* data, std::size_t bytes, std::uint32_t expect,
               const char* name, int line) {
    const std::uint32_t got = tutti::crc32c(data, bytes);
    if (got != expect) {
        std::printf("FAIL [line %d]: crc32c(%s) = 0x%08X, expected 0x%08X\n",
                    line, name, got, expect);
        ++g_failures;
    }
}

#define CHECK_CRC(data, bytes, expect) \
    check_crc((data), (bytes), (expect), #data, __LINE__)

std::vector<std::uint8_t> bytes_of(const char* s) {
    return std::vector<std::uint8_t>(s, s + std::strlen(s));
}

} // namespace

int main() {
    // ==================================================================
    // 1. CRC-32C standard test vectors.
    //
    // Castagnoli polynomial 0x1EDC6F41 (reflected 0x82F63B78), init
    // 0xFFFFFFFF, final xor 0xFFFFFFFF, reflected in/out. These values are
    // fixed by the standard, not by this implementation -- if they fail,
    // either the polynomial or the convention has been changed and the
    // format is no longer CRC-32C.
    // ==================================================================
    {
        // The canonical check value: CRC-32C of "123456789".
        const auto check_value = bytes_of("123456789");
        CHECK_CRC(check_value.data(), check_value.size(), 0xE3069283u);

        const auto one = bytes_of("a");
        CHECK_CRC(one.data(), one.size(), 0xC1D04330u);

        const auto abc = bytes_of("abc");
        CHECK_CRC(abc.data(), abc.size(), 0x364B3FB7u);

        // 32 zero bytes -- the vector that catches a missing initial value.
        const std::vector<std::uint8_t> zeros(32, 0x00);
        CHECK_CRC(zeros.data(), zeros.size(), 0x8A9136AAu);

        // 32 0xFF bytes -- catches a missing final xor.
        const std::vector<std::uint8_t> ones(32, 0xFF);
        CHECK_CRC(ones.data(), ones.size(), 0x62A8AB43u);

        // 0x00..0x1F -- catches a reflected/non-reflected mix-up.
        std::vector<std::uint8_t> ramp(32);
        for (std::size_t i = 0; i < ramp.size(); ++i) {
            ramp[i] = static_cast<std::uint8_t>(i);
        }
        CHECK_CRC(ramp.data(), ramp.size(), 0x46DD794Eu);
    }

    // ==================================================================
    // 2. CRC empty-input convention.
    //
    // Zero length yields 0, not the bare final-xor value, so "no data" stays
    // distinguishable from "data whose CRC happens to be 0xFFFFFFFF".
    // ==================================================================
    {
        CHECK(tutti::crc32c(nullptr, 0) == 0u);
        const std::uint8_t byte = 0x5A;
        CHECK(tutti::crc32c(&byte, 0) == 0u);
        // A null pointer with a nonzero length must not dereference.
        CHECK(tutti::crc32c(nullptr, 16) == 0u);
    }

    // ==================================================================
    // 3. Incremental form agrees with one-shot, at every split point.
    //
    // The header codec hashes fields without staging a contiguous buffer, so
    // this equivalence is load-bearing.
    // ==================================================================
    {
        std::vector<std::uint8_t> buf(257);
        for (std::size_t i = 0; i < buf.size(); ++i) {
            buf[i] = static_cast<std::uint8_t>((i * 31u + 7u) & 0xFFu);
        }
        const std::uint32_t whole = tutti::crc32c(buf.data(), buf.size());

        for (std::size_t split = 0; split <= buf.size(); ++split) {
            std::uint32_t state = tutti::crc32c_init();
            state = tutti::crc32c_update(state, buf.data(), split);
            state = tutti::crc32c_update(state, buf.data() + split,
                                         buf.size() - split);
            if (tutti::crc32c_final(state) != whole) {
                std::printf("FAIL: incremental crc mismatch at split %zu\n",
                            split);
                ++g_failures;
                break;
            }
        }
    }

    // ==================================================================
    // 4. CRC detects the corruptions it exists to detect.
    // ==================================================================
    {
        std::vector<std::uint8_t> a(64, 0xA5);
        std::vector<std::uint8_t> b = a;
        b[37] ^= 0x01;  // single-bit flip
        CHECK(tutti::crc32c(a.data(), a.size()) !=
              tutti::crc32c(b.data(), b.size()));

        // Byte transposition -- a class of error a plain sum would miss.
        std::vector<std::uint8_t> c = {1, 2, 3, 4};
        std::vector<std::uint8_t> d = {1, 3, 2, 4};
        CHECK(tutti::crc32c(c.data(), c.size()) !=
              tutti::crc32c(d.data(), d.size()));

        // Length must matter: trailing zeros are not invisible.
        std::vector<std::uint8_t> e = {9, 9, 9};
        std::vector<std::uint8_t> f = {9, 9, 9, 0};
        CHECK(tutti::crc32c(e.data(), e.size()) !=
              tutti::crc32c(f.data(), f.size()));
    }

    // ==================================================================
    // 5. object_identity frozen values.
    //
    // Arbitrary but persistent: these are what recovery compares against a
    // header's identity field. Any change here invalidates stored objects.
    //
    // Independently reproduced from the algorithm as documented in
    // object_digest.h (FNV-1a 64 then the SplitMix64 finaliser), confirming
    // the documentation is sufficient to reimplement from -- e.g. for an
    // offline inspection tool in another language.
    // ==================================================================
    {
        CHECK(tutti::object_identity(nullptr, 0) == 0xF52A15E9A9B5E89Bull);

        const auto abc = bytes_of("abc");
        CHECK(tutti::object_identity(abc.data(), abc.size()) ==
              0x0DD490490804B508ull);

        // A 16-byte key of repeated 0xA1, the shape used by chunk keys.
        const std::vector<std::uint8_t> chunk(16, 0xA1);
        CHECK(tutti::object_identity(chunk.data(), chunk.size()) ==
              0x7ECB9653ACAF738Bull);
    }

    // ==================================================================
    // 6. object_identity avalanche: a one-bit input change must not leave
    //    the digest nearly unchanged. FNV-1a alone fails this for short
    //    keys, which is why the finalisation mix is there.
    // ==================================================================
    {
        std::vector<std::uint8_t> base(16, 0x00);
        const std::uint64_t h0 =
            tutti::object_identity(base.data(), base.size());

        int weak = 0;
        for (std::size_t byte = 0; byte < base.size(); ++byte) {
            for (int bit = 0; bit < 8; ++bit) {
                std::vector<std::uint8_t> flipped = base;
                flipped[byte] ^= static_cast<std::uint8_t>(1u << bit);
                const std::uint64_t h1 =
                    tutti::object_identity(flipped.data(), flipped.size());
                // Population count of the difference.
                std::uint64_t diff = h0 ^ h1;
                int bits = 0;
                while (diff) {
                    bits += static_cast<int>(diff & 1u);
                    diff >>= 1;
                }
                // Ideal is 32 of 64 bits; anything under 12 indicates poor
                // diffusion. Deliberately loose: this asserts the mix is
                // present, not a specific statistical quality.
                if (bits < 12) ++weak;
            }
        }
        CHECK(weak == 0);
    }

    // ==================================================================
    // 7. object_identity has no collisions across a realistic key set.
    //
    // Not a proof -- collisions are tolerable by design (recovery also
    // compares key_crc32 and payload length). This only catches a
    // catastrophically broken digest, e.g. one ignoring part of its input.
    // ==================================================================
    {
        std::set<std::uint64_t> seen;
        std::size_t total = 0;
        for (std::uint32_t i = 0; i < 4096; ++i) {
            // 18-byte keys: 16-byte chunk id plus a 2-byte layer index, the
            // real io_key shape.
            std::vector<std::uint8_t> key(18, 0x00);
            key[0] = static_cast<std::uint8_t>(i & 0xFF);
            key[1] = static_cast<std::uint8_t>((i >> 8) & 0xFF);
            key[16] = static_cast<std::uint8_t>(i & 0x7F);
            seen.insert(tutti::object_identity(key.data(), key.size()));
            ++total;
        }
        CHECK(seen.size() == total);
    }

    // ==================================================================
    // 8. Every input byte affects the digest.
    //
    // Guards against an implementation that stops early or skips a tail --
    // the failure mode that would make distinct objects share an identity.
    // ==================================================================
    {
        const std::vector<std::uint8_t> base(40, 0x00);
        const std::uint64_t h0 =
            tutti::object_identity(base.data(), base.size());
        const std::uint32_t c0 = tutti::crc32c(base.data(), base.size());
        for (std::size_t i = 0; i < base.size(); ++i) {
            std::vector<std::uint8_t> mutated = base;
            mutated[i] = 0xFF;
            CHECK(tutti::object_identity(mutated.data(), mutated.size()) != h0);
            CHECK(tutti::crc32c(mutated.data(), mutated.size()) != c0);
        }
    }

    // ==================================================================
    // 9. ObjectKey wires both digests, and distinct keys stay distinct.
    //
    // This is the boundary the store implementations actually call.
    // ==================================================================
    {
        tutti::ObjectKey key;
        key.bytes.assign(16, 0xA1);
        CHECK(key.identity() ==
              tutti::object_identity(key.bytes.data(), key.bytes.size()));
        CHECK(key.key_crc32() ==
              tutti::crc32c(key.bytes.data(), key.bytes.size()));

        // Same identity computation is reproducible across copies.
        tutti::ObjectKey copy = key;
        CHECK(copy.identity() == key.identity());
        CHECK(copy.key_crc32() == key.key_crc32());
        CHECK(copy == key);

        // A key differing only in its last byte -- the layer index position
        // -- must differ in both digests.
        tutti::ObjectKey neighbour = key;
        neighbour.bytes.back() = 0xA2;
        CHECK(neighbour != key);
        CHECK(neighbour.identity() != key.identity());
        CHECK(neighbour.key_crc32() != key.key_crc32());

        // An empty key is legal.
        tutti::ObjectKey empty;
        CHECK(empty.identity() == tutti::object_identity(nullptr, 0));
        CHECK(empty.key_crc32() == 0u);
    }

    // ==================================================================
    // 10. Digests fit the header fields that store them.
    // ==================================================================
    {
        static_assert(sizeof(tutti::ObjectKey{}.identity()) == 8,
                      "identity must fit the 8-byte header field");
        static_assert(sizeof(tutti::ObjectKey{}.key_crc32()) == 4,
                      "key_crc32 must fit the 4-byte header field");
        CHECK(tutti::ObjectHeaderLayout::kIdentityOffset + 8 <=
              tutti::ObjectHeaderLayout::kHeaderBytes);
        CHECK(tutti::ObjectHeaderLayout::kKeyCrc32Offset + 4 <=
              tutti::ObjectHeaderLayout::kHeaderBytes);
    }

    if (g_failures == 0) {
        std::printf("object_digest contract: all checks passed\n");
        return 0;
    }
    std::printf("object_digest contract: %d failure(s)\n", g_failures);
    return 1;
}
