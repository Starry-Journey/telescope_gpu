#pragma once

/// @file control.hpp
/// @brief Control solve: min ||W(dz - S u)||^2 + lambda ||u-u0||^2.

#include <string>
#include <vector>

#include "instrument/sensitivity.hpp"

namespace instrument {

struct ControlOptions {
    double lambda = 1e-6;          // Tikhonov regularization
    bool use_bounds = false;
    std::vector<double> lower;
    std::vector<double> upper;
    std::vector<double> weights;   // per-target weights, empty = 1
    std::vector<double> u0;        // reference/previous control
    bool verbose = false;
};

struct ControlSolution {
    bool ok = false;
    std::string message;
    std::vector<double> u;
    std::vector<double> residual;  // S u - dz, per target
    double cost = 0.0;
    int rank = 0;
    double condition = 0.0;
    std::vector<double> singular_values;
    int active_constraints = 0;
};

class ControlSolver {
public:
    virtual ~ControlSolver() = default;
    virtual std::string name() const = 0;
    virtual ControlSolution solve(const SensitivityMatrix& S,
                                  const std::vector<double>& dz,
                                  const ControlOptions& options) const = 0;
};

/// SVD-based weighted least squares with optional Tikhonov regularization and
/// simple bound projection.  For hard constraints/active-set QP, add a plugin;
/// this class is the reference implementation.
class WeightedLeastSquaresControl : public ControlSolver {
public:
    std::string name() const override { return "weighted_lsq_svd"; }
    ControlSolution solve(const SensitivityMatrix& S,
                          const std::vector<double>& dz,
                          const ControlOptions& options) const override;
};

}  // namespace instrument
