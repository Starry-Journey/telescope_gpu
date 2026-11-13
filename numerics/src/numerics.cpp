#include "telescope/numerics.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace telescope {
namespace numerics {
namespace {

void check_finite(const Matrix& A) {
    if (!A.allFinite()) {
        throw std::invalid_argument("matrix contains NaN/Inf");
    }
}

void fft_radix2(std::vector<Complex>& a) {
    const std::size_t n = a.size();
    // bit-reversal permutation
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(a[i], a[j]);
        }
    }
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const double ang = 2.0 * M_PI / static_cast<double>(len);
        const Complex wlen(std::cos(ang), -std::sin(ang));
        for (std::size_t i = 0; i < n; i += len) {
            Complex w(1.0, 0.0);
            for (std::size_t j = 0; j < len / 2; ++j) {
                const Complex u = a[i + j];
                const Complex v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

}  // namespace

Vector solve_linear(const Matrix& A, const Vector& b) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("solve_linear: A must be square");
    }
    if (b.size() != A.rows()) {
        throw std::invalid_argument("solve_linear: b size mismatch");
    }
    check_finite(A);
    Eigen::FullPivLU<Matrix> lu(A);
    if (!lu.isInvertible()) {
        throw std::runtime_error("solve_linear: matrix is singular");
    }
    return lu.solve(b);
}

Vector least_squares(const Matrix& A, const Vector& b) {
    if (A.rows() != b.size()) {
        throw std::invalid_argument("least_squares: row/b size mismatch");
    }
    if (A.rows() == 0 || A.cols() == 0) {
        throw std::invalid_argument("least_squares: empty system");
    }
    check_finite(A);
    return A.colPivHouseholderQr().solve(b);
}

Vector least_squares_svd(const Matrix& A, const Vector& b) {
    if (A.rows() != b.size()) {
        throw std::invalid_argument("least_squares_svd: row/b size mismatch");
    }
    if (A.rows() == 0 || A.cols() == 0) {
        throw std::invalid_argument("least_squares_svd: empty system");
    }
    check_finite(A);
    return A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
}

Vector fit_polynomial(const std::vector<double>& x,
                      const std::vector<double>& y,
                      int degree) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("fit_polynomial: x/y size mismatch");
    }
    if (degree < 0) {
        throw std::invalid_argument("fit_polynomial: degree must be >= 0");
    }
    if (x.size() < static_cast<std::size_t>(degree + 1)) {
        throw std::invalid_argument("fit_polynomial: not enough samples");
    }
    Matrix A(x.size(), degree + 1);
    for (std::size_t i = 0; i < x.size(); ++i) {
        double p = 1.0;
        for (int j = 0; j <= degree; ++j) {
            A(static_cast<Eigen::Index>(i), j) = p;
            p *= x[i];
        }
    }
    Vector b(y.size());
    for (std::size_t i = 0; i < y.size(); ++i) {
        b(static_cast<Eigen::Index>(i)) = y[i];
    }
    return least_squares(A, b);
}

void fft(std::vector<Complex>& data, bool inverse) {
    const std::size_t n = data.size();
    if (n == 0) {
        return;
    }
    if ((n & (n - 1)) != 0) {
        throw std::invalid_argument("fft: size must be a power of two");
    }
    if (inverse) {
        for (auto& z : data) {
            z = std::conj(z);
        }
    }
    fft_radix2(data);
    if (inverse) {
        const double inv = 1.0 / static_cast<double>(n);
        for (auto& z : data) {
            z = std::conj(z) * inv;
        }
    }
}

std::vector<Complex> fft_copy(const std::vector<Complex>& data, bool inverse) {
    std::vector<Complex> out = data;
    fft(out, inverse);
    return out;
}

OdeResult solve_ode_scalar(const std::function<double(double, double)>& rhs,
                           double y0, double t0, double t1, double dt) {
    if (!(dt > 0.0)) {
        throw std::invalid_argument("solve_ode_scalar: dt must be positive");
    }
    if (!(t1 > t0)) {
        throw std::invalid_argument("solve_ode_scalar: require t1 > t0");
    }
    OdeResult out;
    double t = t0;
    double y = y0;
    out.t.push_back(t);
    out.y.push_back(y);
    while (t < t1 - 1e-15) {
        const double h = std::min(dt, t1 - t);
        const double k1 = rhs(t, y);
        const double k2 = rhs(t + 0.5 * h, y + 0.5 * h * k1);
        const double k3 = rhs(t + 0.5 * h, y + 0.5 * h * k2);
        const double k4 = rhs(t + h, y + h * k3);
        y += h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        t += h;
        out.t.push_back(t);
        out.y.push_back(y);
    }
    return out;
}

OdeResult solve_ode_system(const std::function<Vector(double, const Vector&)>& rhs,
                           const Vector& y0, double t0, double t1, double dt) {
    if (!(dt > 0.0)) {
        throw std::invalid_argument("solve_ode_system: dt must be positive");
    }
    if (!(t1 > t0)) {
        throw std::invalid_argument("solve_ode_system: require t1 > t0");
    }
    OdeResult out;
    double t = t0;
    Vector y = y0;
    out.t.push_back(t);
    out.states.push_back(y);
    while (t < t1 - 1e-15) {
        const double h = std::min(dt, t1 - t);
        const Vector k1 = rhs(t, y);
        const Vector k2 = rhs(t + 0.5 * h, y + 0.5 * h * k1);
        const Vector k3 = rhs(t + 0.5 * h, y + 0.5 * h * k2);
        const Vector k4 = rhs(t + h, y + h * k3);
        y += h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        t += h;
        out.t.push_back(t);
        out.states.push_back(y);
    }
    return out;
}

double relative_residual(const Matrix& A, const Vector& x, const Vector& b) {
    if (A.rows() != b.size()) {
        throw std::invalid_argument("relative_residual: size mismatch");
    }
    const Vector r = A * x - b;
    return r.norm() / std::max(1.0, b.norm());
}

}  // namespace numerics
}  // namespace telescope
