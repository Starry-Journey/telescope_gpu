#include "instrument/control.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include <Eigen/Dense>

#include "telescope/numerics.hpp"

namespace instrument {

ControlSolution WeightedLeastSquaresControl::solve(
    const SensitivityMatrix& S,
    const std::vector<double>& dz,
    const ControlOptions& options) const {
    ControlSolution out;
    if (!S.valid()) {
        out.message = "Sensitivity matrix is empty/invalid";
        return out;
    }
    if (static_cast<int>(dz.size()) != S.rows()) {
        out.message = "dz size mismatch";
        return out;
    }
    const int m = S.rows();
    const int n = S.cols();
    if (n == 0) {
        out.message = "no control axes";
        return out;
    }
    if (options.use_bounds &&
        (static_cast<int>(options.lower.size()) != n ||
         static_cast<int>(options.upper.size()) != n)) {
        out.message = "bounds size mismatch";
        return out;
    }
    // weights 是按行（每条像差目标）索引的，长度必须等于 S 的行数 m；
    // 空表示全部取 1.0。长度不匹配会在下方循环里越界读取（UB），先校验。
    if (!options.weights.empty() &&
        options.weights.size() != static_cast<std::size_t>(m)) {
        throw std::invalid_argument(
            "WeightedLeastSquaresControl::solve: weights size mismatch, "
            "expected " + std::to_string(m) + " (S.rows()) or empty, got " +
            std::to_string(options.weights.size()));
    }

    Eigen::MatrixXd A(m, n);
    Eigen::VectorXd b(m);
    for (int r = 0; r < m; ++r) {
        const double w = options.weights.empty()
                             ? 1.0
                             : options.weights[static_cast<std::size_t>(r)];
        b[r] = w * dz[static_cast<std::size_t>(r)];
        for (int c = 0; c < n; ++c) {
            A(r, c) = w * S(r, c);
        }
    }

    Eigen::VectorXd rhs = b;
    const Eigen::MatrixXd S_original = A;  // rank/condition refer to S, not augmented
    if (options.lambda > 0.0) {
        const double sl = std::sqrt(options.lambda);
        Eigen::MatrixXd aug(m + n, n);
        aug.topRows(m) = A;
        aug.bottomRows(n) = sl * Eigen::MatrixXd::Identity(n, n);
        Eigen::VectorXd br(m + n);
        br.head(m) = b;
        if (static_cast<int>(options.u0.size()) == n) {
            for (int i = 0; i < n; ++i) br[m + i] = sl * options.u0[static_cast<std::size_t>(i)];
        } else {
            br.tail(n).setZero();
        }
        A = aug;
        rhs = br;
    }

    Eigen::VectorXd u;
    try {
        u = telescope::numerics::least_squares_svd(A, rhs);
    } catch (const std::exception& e) {
        out.message = std::string("control solve failed: ") + e.what();
        return out;
    }

    if (options.use_bounds) {
        for (int i = 0; i < n; ++i) {
            const double c = std::max(options.lower[static_cast<std::size_t>(i)],
                                      std::min(options.upper[static_cast<std::size_t>(i)], u[i]));
            if (c != u[i]) {
                u[i] = c;
                ++out.active_constraints;
            }
        }
    }

    // singular values of the original S
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(S_original);
    out.singular_values.assign(svd.singularValues().data(),
                               svd.singularValues().data() + svd.singularValues().size());
    out.rank = static_cast<int>(svd.rank());
    const double smin = out.singular_values.empty() ? 0.0 : out.singular_values.back();
    const double smax = out.singular_values.empty() ? 0.0 : out.singular_values.front();
    out.condition = smin > 0 ? smax / smin : std::numeric_limits<double>::infinity();

    out.residual.resize(static_cast<std::size_t>(m));
    double cost = 0.0;
    for (int r = 0; r < m; ++r) {
        const double w = options.weights.empty()
                             ? 1.0
                             : options.weights[static_cast<std::size_t>(r)];
        double pred = 0.0;
        for (int c = 0; c < n; ++c) pred += S(r, c) * u[c];
        out.residual[static_cast<std::size_t>(r)] = pred - dz[static_cast<std::size_t>(r)];
        cost += w * w * out.residual[static_cast<std::size_t>(r)] *
                out.residual[static_cast<std::size_t>(r)];
    }
    if (options.lambda > 0.0 && static_cast<int>(options.u0.size()) == n) {
        for (int i = 0; i < n; ++i) {
            const double d = u[i] - options.u0[static_cast<std::size_t>(i)];
            cost += options.lambda * d * d;
        }
    }
    out.u.assign(u.data(), u.data() + u.size());
    out.cost = cost;
    out.ok = true;
    out.message = "ok";
    return out;
}

}  // namespace instrument
