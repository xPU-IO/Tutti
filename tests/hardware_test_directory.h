#pragma once

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <unistd.h>

namespace tutti::test_support {

// A run-scoped directory created atomically below a hardware-backed parent.
// The destructor intentionally preserves the directory. Tests call cleanup()
// only after a complete pass so failures retain their artifacts for diagnosis.
class UniqueTestDirectory {
public:
    UniqueTestDirectory() = default;
    UniqueTestDirectory(const UniqueTestDirectory&) = delete;
    UniqueTestDirectory& operator=(const UniqueTestDirectory&) = delete;
    UniqueTestDirectory(UniqueTestDirectory&&) noexcept = default;
    UniqueTestDirectory& operator=(UniqueTestDirectory&&) noexcept = default;

    static bool create(const std::string& parent,
                       const std::string& prefix,
                       UniqueTestDirectory& out,
                       std::string& error) {
        std::string pattern = parent;
        if (pattern.empty() || pattern.back() != '/') pattern += '/';
        pattern += prefix + "." + std::to_string(::getpid()) + ".XXXXXX";

        std::vector<char> writable(pattern.begin(), pattern.end());
        writable.push_back('\0');
        char* created = ::mkdtemp(writable.data());
        if (created == nullptr) {
            const int saved_errno = errno;
            error = "mkdtemp(" + pattern + ") failed: " +
                    std::strerror(saved_errno);
            return false;
        }

        out.path_ = created;
        error.clear();
        return true;
    }

    const std::string& path() const noexcept { return path_; }
    bool valid() const noexcept { return !path_.empty(); }

    bool cleanup(std::string& error) {
        if (path_.empty()) {
            error.clear();
            return true;
        }

        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
        if (ec) {
            error = "remove_all(" + path_ + ") failed: " + ec.message();
            return false;
        }

        path_.clear();
        error.clear();
        return true;
    }

private:
    std::string path_;
};

}  // namespace tutti::test_support
