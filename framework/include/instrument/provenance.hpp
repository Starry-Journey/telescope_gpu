#pragma once

/// @file provenance.hpp
/// @brief Run manifest for reproducible scientific results.

#include <map>
#include <string>

namespace instrument {

struct RunManifest {
    std::string run_id;
    std::string git_commit;
    bool git_dirty = true;
    std::string config_hash;
    std::string dataset_hash;
    std::string backend;
    unsigned seed = 0;
    std::map<std::string, std::string> versions;
    std::map<std::string, double> metrics;
    double wall_time_s = 0.0;

    std::string to_json() const;
};

}  // namespace instrument
