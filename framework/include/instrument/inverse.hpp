#pragma once

/// @file inverse.hpp
/// @brief Inverse-problem contract and result metadata.

#include <string>
#include <vector>

#include "instrument/data.hpp"
#include "instrument/parameter.hpp"

namespace instrument {

struct SolveOptions {
    int maxeval = 2000;
    int n_starts = 40;
    int n_refine = 3;
    double ftol = 1e-10;
    double xtol = 1e-6;
    unsigned seed = 12345;
    bool verbose = false;
};

struct Solution {
    bool ok = false;
    std::string message;
    std::vector<double> values;       // same layout as ParameterBlock
    std::vector<std::string> names;
    double cost = 0.0;
    int nfev = 0;
    std::vector<double> covariance;   // row-major n x n, empty if unknown
    std::map<std::string, double> diagnostics;
};

class InverseSolver {
public:
    virtual ~InverseSolver() = default;
    virtual std::string name() const = 0;
    virtual Solution solve(const Observation& observation,
                           const ParameterSpace& space,
                           const SolveOptions& options) = 0;
};

}  // namespace instrument
