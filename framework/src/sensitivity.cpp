#include "instrument/sensitivity.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <Eigen/Dense>

namespace instrument {

SensitivityMatrix FiniteDifferenceSensitivity::build(
    const ForwardModel& model,
    const ParameterSpace& space,
    const ControlSpace& controls,
    const TargetExtractor& target,
    const ParameterBlock& base,
    const Context& context) const {
    if (!controls.valid(space)) {
        throw std::invalid_argument("Sensitivity: control index out of range");
    }
    if (base.values.size() != space.size()) {
        throw std::invalid_argument("Sensitivity: base size mismatch");
    }

    const Observable base_obs = model.evaluate(base, context);
    const Observable base_target = target.extract(base_obs);
    if (base_target.values.empty()) {
        throw std::runtime_error("Sensitivity: target extractor returned empty vector");
    }

    SensitivityMatrix S;
    S.target_names = base_target.names;
    S.control_names.reserve(controls.axes.size());
    S.method = name();
    S.provenance = "model=" + model.name() + ";extractor=" + target.name() +
                   ";context=" + context.tag;

    const int rows = static_cast<int>(base_target.values.size());
    const int cols = static_cast<int>(controls.axes.size());
    S.values.assign(static_cast<std::size_t>(rows) * cols, 0.0);

    for (int c = 0; c < cols; ++c) {
        const auto& axis = controls.axes[static_cast<std::size_t>(c)];
        S.control_names.push_back(axis.name);
        ParameterBlock plus = base;
        ParameterBlock minus = base;
        const double v = base.values[static_cast<std::size_t>(axis.parameter_index)];
        const double step = std::abs(axis.step) > 0 ? std::abs(axis.step) : 1e-3;
        const double hi = std::min(axis.upper, space.spec(axis.parameter_index).upper);
        const double lo = std::max(axis.lower, space.spec(axis.parameter_index).lower);
        double actual = 0.0;
        if (v + step <= hi && v - step >= lo) {
            plus.values[static_cast<std::size_t>(axis.parameter_index)] = v + step;
            minus.values[static_cast<std::size_t>(axis.parameter_index)] = v - step;
            actual = 2.0 * step;
        } else if (v + step <= hi) {
            // one-sided forward difference near lower bound (symmetric with the
            // upper-bound branch below): never perturb below lo
            minus.values[static_cast<std::size_t>(axis.parameter_index)] = v;
            plus.values[static_cast<std::size_t>(axis.parameter_index)] =
                std::min(hi, v + step);
            actual = plus.values[static_cast<std::size_t>(axis.parameter_index)] -
                     minus.values[static_cast<std::size_t>(axis.parameter_index)];
        } else {
            // one-sided backward difference near upper bound
            plus.values[static_cast<std::size_t>(axis.parameter_index)] = v;
            minus.values[static_cast<std::size_t>(axis.parameter_index)] = std::max(lo, v - step);
            actual = v - minus.values[static_cast<std::size_t>(axis.parameter_index)];
        }
        if (!(actual > 0)) {
            throw std::runtime_error("Sensitivity: zero step for control " + axis.name);
        }
        const Observable op = target.extract(model.evaluate(plus, context));
        const Observable om = target.extract(model.evaluate(minus, context));
        if (op.values.size() != static_cast<std::size_t>(rows) ||
            om.values.size() != static_cast<std::size_t>(rows)) {
            throw std::runtime_error("Sensitivity: target size changed under perturbation");
        }
        for (int r = 0; r < rows; ++r) {
            S.values[static_cast<std::size_t>(r) * cols + c] =
                (op.values[static_cast<std::size_t>(r)] -
                 om.values[static_cast<std::size_t>(r)]) / actual;
        }
    }

    Eigen::MatrixXd A(rows, cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            A(r, c) = S.values[static_cast<std::size_t>(r) * cols + c];
        }
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A);
    S.singular_values.assign(svd.singularValues().data(),
                             svd.singularValues().data() + svd.singularValues().size());
    S.rank = static_cast<int>(svd.rank());
    const double smin = S.singular_values.empty() ? 0.0 : S.singular_values.back();
    const double smax = S.singular_values.empty() ? 0.0 : S.singular_values.front();
    S.condition = smin > 0 ? smax / smin : std::numeric_limits<double>::infinity();
    S.unit = "target_unit / control_unit";
    return S;
}

}  // namespace instrument
