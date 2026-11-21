#pragma once

/// @file sensitivity.hpp
/// @brief Sensitivity matrix S_ij = d(target_i)/d(control_j).

#include <string>
#include <vector>

#include "instrument/data.hpp"
#include "instrument/model.hpp"
#include "instrument/parameter.hpp"

namespace instrument {

struct SensitivityMatrix {
    std::vector<std::string> target_names;
    std::vector<std::string> control_names;
    std::vector<double> values;  // row-major rows x cols
    std::string unit;            // "target_unit / control_unit"
    int rank = 0;
    double condition = 0.0;
    std::vector<double> singular_values;
    std::string method;
    std::string provenance;

    int rows() const { return static_cast<int>(target_names.size()); }
    int cols() const { return static_cast<int>(control_names.size()); }
    double operator()(int r, int c) const {
        return values[static_cast<std::size_t>(r) * cols() + c];
    }
    bool valid() const {
        return rows() > 0 && cols() > 0 &&
               values.size() == static_cast<std::size_t>(rows()) * cols();
    }
};

class SensitivityBuilder {
public:
    virtual ~SensitivityBuilder() = default;
    virtual std::string name() const = 0;

    /// Build S at `base` by perturbing the axes in `controls`.
    /// The model must accept a ParameterBlock with the full ParameterSpace.
    virtual SensitivityMatrix build(const ForwardModel& model,
                                    const ParameterSpace& space,
                                    const ControlSpace& controls,
                                    const TargetExtractor& target,
                                    const ParameterBlock& base,
                                    const Context& context) const = 0;
};

/// Central finite differences with a per-axis step.
class FiniteDifferenceSensitivity : public SensitivityBuilder {
public:
    std::string name() const override { return "finite_difference"; }
    SensitivityMatrix build(const ForwardModel& model,
                            const ParameterSpace& space,
                            const ControlSpace& controls,
                            const TargetExtractor& target,
                            const ParameterBlock& base,
                            const Context& context) const override;
};

}  // namespace instrument
