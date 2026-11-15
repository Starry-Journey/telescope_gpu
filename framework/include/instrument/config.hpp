#pragma once

/// @file config.hpp
/// @brief Minimal run configuration for the framework CLI/API.
///
/// v1 uses a typed struct instead of a full YAML/JSON parser; the fields are
/// deliberately the same ones a future schema-versioned config will expose.

#include <map>
#include <string>

namespace instrument {

struct FrameworkConfig {
    std::string backend = "cpu";          // "cpu", "cuda", "torch"
    unsigned seed = 42;
    std::string output_dir = "results/framework";
    std::string cache_dir = "results/framework/cache";
    int maxeval = 2000;
    double ftol = 1e-10;
    double xtol = 1e-6;
    bool verbose = false;
    std::map<std::string, std::string> options;

    std::string to_json() const {
        std::string s = "{\"backend\":\"" + backend + "\",\"seed\":" +
                        std::to_string(seed) + ",\"output_dir\":\"" + output_dir +
                        "\",\"maxeval\":" + std::to_string(maxeval) + "}";
        return s;
    }
};

}  // namespace instrument
