#pragma once

/// @file cache.hpp
/// @brief In-memory result cache keyed by model/config/dataset hash.
///
/// v1 is intentionally small: the interface is what matters.  A file-backed
/// implementation can store the same string payload under a content hash
/// without changing callers.

#include <map>
#include <string>

namespace instrument {

class ResultCache {
public:
    bool has(const std::string& key) const { return entries_.find(key) != entries_.end(); }

    void put(const std::string& key, std::string payload) {
        entries_[key] = std::move(payload);
    }

    const std::string* get(const std::string& key) const {
        const auto it = entries_.find(key);
        return it == entries_.end() ? nullptr : &it->second;
    }

    std::size_t size() const { return entries_.size(); }
    void clear() { entries_.clear(); }

private:
    std::map<std::string, std::string> entries_;
};

}  // namespace instrument
