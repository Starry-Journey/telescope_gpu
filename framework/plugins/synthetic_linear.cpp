#include "synthetic_linear.hpp"

#include <stdexcept>

namespace instrument {

SyntheticLinearForward::SyntheticLinearForward(std::vector<std::vector<double>> A,
                                               std::vector<double> z0,
                                               std::vector<std::string> target_names)
    : A_(std::move(A)), z0_(std::move(z0)), target_names_(std::move(target_names)) {
    if (A_.empty() || A_[0].empty()) {
        throw std::invalid_argument("SyntheticLinearForward: empty matrix");
    }
    const std::size_t n = A_[0].size();
    for (const auto& row : A_) {
        if (row.size() != n) throw std::invalid_argument("SyntheticLinearForward: ragged matrix");
    }
    if (z0_.size() != A_.size()) z0_.assign(A_.size(), 0.0);
    if (target_names_.size() != A_.size()) {
        target_names_.clear();
        for (std::size_t i = 0; i < A_.size(); ++i) target_names_.push_back("z" + std::to_string(i));
    }
}

Observable SyntheticLinearForward::evaluate(const ParameterBlock& params,
                                            const Context&) const {
    if (params.values.size() != A_[0].size()) {
        throw std::invalid_argument("SyntheticLinearForward: parameter size mismatch");
    }
    Observable out;
    out.kind = "vector";
    out.names = target_names_;
    out.values.resize(A_.size());
    for (std::size_t r = 0; r < A_.size(); ++r) {
        double v = z0_[r];
        for (std::size_t c = 0; c < A_[r].size(); ++c) v += A_[r][c] * params.values[c];
        out.values[r] = v;
    }
    return out;
}

}  // namespace instrument
