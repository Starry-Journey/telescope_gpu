#pragma once

/// @file parameter.hpp
/// @brief Parameter/control space: names, units, bounds and values.
///
/// A ParameterSpace is the single source of truth for "what can be fitted or
/// controlled".  Applications must not hard-code Zernike indices; they declare
/// specs and let the model interpret them.

#include <string>
#include <vector>

#include "instrument/data.hpp"

namespace instrument {

struct ParameterSpec {
    std::string name;       // "z4", "uz", "distortion_k1"
    std::string unit;       // "wave", "mm", "mrad", "deg"
    std::string group;      // "aberration", "control", "environment"
    double lower = -1e9;
    double upper = 1e9;
    double prior_sigma = 0.0;  // 0 = no prior
    bool active = true;

    bool valid() const { return lower <= upper && !name.empty(); }
};

struct ParameterBlock {
    std::string space_name;
    std::vector<double> values;

    double& operator[](std::size_t i) { return values[i]; }
    const double& operator[](std::size_t i) const { return values[i]; }
    std::size_t size() const { return values.size(); }
};

class ParameterSpace {
public:
    ParameterSpace() = default;
    explicit ParameterSpace(std::vector<ParameterSpec> specs, std::string name = "default");

    const std::string& name() const { return name_; }
    std::size_t size() const { return specs_.size(); }
    bool empty() const { return specs_.empty(); }
    const ParameterSpec& spec(std::size_t i) const { return specs_.at(i); }
    const std::vector<ParameterSpec>& specs() const { return specs_; }

    int index_of(const std::string& name) const;
    std::vector<std::string> names() const;
    std::vector<std::string> units() const;

    ParameterBlock encode(const std::vector<double>& values) const;
    std::vector<double> decode(const ParameterBlock& block) const;

    /// Clamp values to [lower, upper]; returns the number of changed entries.
    int clamp(ParameterBlock& block) const;
    Status validate(const ParameterBlock& block) const;

private:
    std::vector<ParameterSpec> specs_;
    std::string name_;
};

struct ControlAxis {
    std::string name;
    std::string unit;
    int parameter_index = -1;
    double step = 1e-3;       // finite-difference step
    double lower = -1e9;
    double upper = 1e9;
};

/// Subset of a ParameterSpace that is allowed to move for control/sensitivity.
struct ControlSpace {
    std::string name;
    std::vector<ControlAxis> axes;

    std::size_t size() const { return axes.size(); }
    bool valid(const ParameterSpace& space) const {
        for (const auto& a : axes) {
            if (a.parameter_index < 0 ||
                a.parameter_index >= static_cast<int>(space.size())) return false;
        }
        return true;
    }
};

}  // namespace instrument
