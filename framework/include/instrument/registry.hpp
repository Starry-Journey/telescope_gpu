#pragma once

/// @file registry.hpp
/// @brief Minimal compile-time plugin registry.
///
/// Projects register factories at startup; the CLI/config resolves by name.
/// This avoids a hard dependency on dlopen and keeps ABI simple for the MVP.

#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace instrument {

template <typename Interface>
class Registry {
public:
    using Factory = std::function<std::unique_ptr<Interface>()>;

    void add(const std::string& name, Factory factory) {
        if (name.empty() || !factory) {
            throw std::invalid_argument("Registry::add: empty name/factory");
        }
        factories_[name] = std::move(factory);
    }

    std::unique_ptr<Interface> create(const std::string& name) const {
        const auto it = factories_.find(name);
        if (it == factories_.end()) {
            throw std::out_of_range("Registry: unknown plugin '" + name + "'");
        }
        return it->second();
    }

    bool contains(const std::string& name) const {
        return factories_.find(name) != factories_.end();
    }

    std::vector<std::string> names() const {
        std::vector<std::string> out;
        for (const auto& kv : factories_) out.push_back(kv.first);
        return out;
    }

private:
    std::map<std::string, Factory> factories_;
};

}  // namespace instrument
