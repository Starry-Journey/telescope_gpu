#pragma once

/// @file phase_adapter.hpp
/// @brief Adapt the existing CUDA phase-diversity solver to instrument::InverseSolver.
///
/// This is intentionally an adapter, not a rewrite: the numerical behavior and
/// 0.001667λ golden result stay owned by telescope::PhaseDiversitySolver.

#include "instrument/inverse.hpp"

namespace instrument {

class PhaseDiversityInverseSolver : public InverseSolver {
public:
    PhaseDiversityInverseSolver();
    std::string name() const override { return "phase_diversity_bobyqa"; }

    Solution solve(const Observation& observation,
                   const ParameterSpace& space,
                   const SolveOptions& options) override;
};

}  // namespace instrument
