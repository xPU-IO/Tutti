/*
 * tutti_verbose.h — TUTTI_VERBOSE log gate for bring-up / info messages.
 *
 * Usage:
 *   TUTTI_INFO("format %s\n", arg);       // printf-style
 *   if (tutti_verbose()) { std::cout << ...; }  // iostream-style
 *
 * Behaviour:
 *   - Default (TUTTI_VERBOSE not set or empty): info messages are suppressed.
 *   - TUTTI_VERBOSE=1: info messages are printed to stderr.
 *   - Error-path messages (fprintf(stderr, "...failed...")) are NEVER gated
 *     and must not use TUTTI_INFO.
 *
 * The getenv result is cached on first call (thread-safe via C++11
 * function-local static).
 *
 * SPDX-License-Identifier: GPL-2.0
 */

#ifndef TUTTI_VERBOSE_H
#define TUTTI_VERBOSE_H

#include <cstdio>
#include <cstdlib>

namespace tutti_detail {

inline bool tutti_verbose_cached()
{
    static const bool v = ([]() {
        const char* env = std::getenv("TUTTI_VERBOSE");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    })();
    return v;
}

} // namespace tutti_detail

/*
 * Returns true if TUTTI_VERBOSE env is set to a non-zero value.
 * Cached after first call.
 */
inline bool tutti_verbose()
{
    return tutti_detail::tutti_verbose_cached();
}

/*
 * TUTTI_INFO — printf-style info message, gated by TUTTI_VERBOSE.
 * Outputs to stderr (same as error messages, so log ordering is preserved
 * when both are present).
 *
 * Usage: TUTTI_INFO("device=%d pci=%s\n", dev_id, pci_str);
 */
#define TUTTI_INFO(...)                                \
    do {                                               \
        if (tutti_verbose()) {                         \
            std::fprintf(stderr, __VA_ARGS__);         \
        }                                              \
    } while (0)

/*
 * TUTTI_INFO_IF — conditional info block.
 * Usage:
 *   if (TUTTI_INFO_IF) {
 *       std::cout << "Owned devices:\n";
 *       ...
 *   }
 */
#define TUTTI_INFO_IF tutti_verbose()

#endif /* TUTTI_VERBOSE_H */
