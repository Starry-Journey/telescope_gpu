#pragma once

/// @file synthetic_linear.hpp
/// @brief A deterministic linear forward model for framework/control tests.
///
/// z = A u + z0, with A a 3x5 sensitivity matrix (Z4/Z8/Z9 x uz/ux/uy/uα/uβ).
/// This is not a physical optical model; it exists so the framework's
/// SensitivityBuilder/ControlSolver can be verified against a known matrix.

#include <string>
#include <vector>

#include "instrument/model.hpp"

namespace instrument {

class SyntheticLinearForward : public ForwardModel {
public:
    SyntheticLinearForward(std::vector<std::vector<double>> A,
                           std::vector<double> z0 = {0.0, 0.0, 0.0},
                           std::vector<std::string> target_names = {"Z4", "Z8", "Z9"});

    std::string name() const override { return "synthetic_linear"; }
    Observable evaluate(const ParameterBlock& params,
                        const Context& context) const override;

    const std::vector<std::vector<double>>& matrix() const { return A_; }

private:
    std::vector<std::vector<double>> A_;
    std::vector<double> z0_;
    std::vector<std::string> target_names_;
};

}  // namespace instrument
