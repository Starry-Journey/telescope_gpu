#pragma once

/// @file numerics.hpp
/// @brief Small, testable numerical core used by the engineering MVP.
///
/// The library deliberately covers only four primitives that appear again and
/// again in the telescope pipeline:
///   * dense linear systems (partial-pivot LU),
///   * linear least squares (QR, with an SVD fallback),
///   * radix-2 FFT (forward/inverse, same convention as numpy.fft),
///   * explicit RK4 ODE integration (scalar and vector right-hand sides).
///
/// It is header + one .cpp, depends only on Eigen, and is exposed to Python
/// through pybind11.  Keeping the API this small makes it possible to test
/// every function against an independent reference (Eigen / NumPy / analytic).

#include <complex>
#include <functional>
#include <vector>

#include <Eigen/Dense>

namespace telescope {
namespace numerics {

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;
using Complex = std::complex<double>;

/// Solve A x = b for a square, non-singular A (partial-pivot LU).
/// @throws std::invalid_argument if A is not square or b has wrong size.
/// @throws std::runtime_error if A is numerically singular.
Vector solve_linear(const Matrix& A, const Vector& b);

/// Solve min ||A x - b||_2 using column-pivoted Householder QR.
/// Works for tall, square and (full-column-rank) wide matrices.
Vector least_squares(const Matrix& A, const Vector& b);

/// SVD-based least squares / pseudo-inverse solution.
/// More robust than QR for rank-deficient or badly scaled systems.
Vector least_squares_svd(const Matrix& A, const Vector& b);

/// Fit y ~= c0 + c1 x + ... + cd x^d and return coefficients low-to-high.
/// @throws std::invalid_argument if x/y sizes differ or degree < 0.
Vector fit_polynomial(const std::vector<double>& x,
                      const std::vector<double>& y,
                      int degree);

/// In-place radix-2 Cooley-Tukey FFT.
/// Forward: X[k] = sum_n x[n] exp(-2*pi*i*n*k/N).
/// Inverse: x[n] = (1/N) sum_k X[k] exp(+2*pi*i*n*k/N).
/// @throws std::invalid_argument if size is not a power of two.
void fft(std::vector<Complex>& data, bool inverse = false);

/// Convenience copy-returning overload.
std::vector<Complex> fft_copy(const std::vector<Complex>& data,
                              bool inverse = false);

/// Result of an ODE integration.
/// Scalar problems fill `y`; vector problems fill `states` and leave `y` empty.
struct OdeResult {
    std::vector<double> t;
    std::vector<double> y;
    std::vector<Vector> states;
};

/// Scalar RK4: y' = f(t, y).
/// @param dt positive step; the final step is shortened to land on t1.
OdeResult solve_ode_scalar(const std::function<double(double, double)>& rhs,
                           double y0, double t0, double t1, double dt);

/// Vector RK4: y' = f(t, y), y in R^n.
OdeResult solve_ode_system(const std::function<Vector(double, const Vector&)>& rhs,
                           const Vector& y0, double t0, double t1, double dt);

/// Relative residual ||A x - b|| / max(1, ||b||), useful in tests/reports.
double relative_residual(const Matrix& A, const Vector& x, const Vector& b);

}  // namespace numerics
}  // namespace telescope
