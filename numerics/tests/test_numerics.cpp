#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

#include "telescope/numerics.hpp"

using namespace telescope::numerics;

TEST(LinearSolve, Known3x3) {
    Matrix A(3, 3);
    A << 4, -2, 1,
         -2, 4, -2,
         1, -2, 4;
    Vector b(3);
    b << 11, -16, 17;
    const Vector x = solve_linear(A, b);
    EXPECT_LT(relative_residual(A, x, b), 1e-12);
    EXPECT_NEAR(x[0], 1.0, 1e-10);
    EXPECT_NEAR(x[1], -2.0, 1e-10);
    EXPECT_NEAR(x[2], 3.0, 1e-10);
}

TEST(LinearSolve, RejectsSingularAndBadShape) {
    Matrix singular(2, 2);
    singular << 1, 2, 2, 4;
    Vector b(2);
    b << 1, 2;
    EXPECT_THROW(solve_linear(singular, b), std::runtime_error);
    Matrix non_square(2, 3);
    non_square.setZero();
    EXPECT_THROW(solve_linear(non_square, b), std::invalid_argument);
}

TEST(LeastSquares, LineFit) {
    // y = 3 + 2x + small deterministic perturbation
    Matrix A(6, 2);
    Vector b(6);
    for (int i = 0; i < 6; ++i) {
        const double x = i - 2.0;
        A(i, 0) = 1.0;
        A(i, 1) = x;
        b[i] = 3.0 + 2.0 * x + 0.01 * std::sin(x);
    }
    const Vector x = least_squares(A, b);
    EXPECT_NEAR(x[0], 3.0, 0.02);
    EXPECT_NEAR(x[1], 2.0, 0.02);
    EXPECT_LT(relative_residual(A, x, b), 0.01);
}

TEST(LeastSquares, SvdMatchesQrForWellConditioned) {
    Matrix A(5, 3);
    A << 1, 0, 1,
         0, 1, 1,
         1, 1, 0,
         2, 1, 1,
         0, 2, 1;
    Vector b(5);
    b << 1, 2, 3, 4, 5;
    const Vector x_qr = least_squares(A, b);
    const Vector x_svd = least_squares_svd(A, b);
    EXPECT_LT((x_qr - x_svd).norm(), 1e-10);
}

TEST(Polynomial, FitQuadratic) {
    std::vector<double> x, y;
    for (int i = -5; i <= 5; ++i) {
        const double xi = 0.3 * i;
        x.push_back(xi);
        y.push_back(1.5 - 2.0 * xi + 0.5 * xi * xi);
    }
    const Vector c = fit_polynomial(x, y, 2);
    EXPECT_NEAR(c[0], 1.5, 1e-9);
    EXPECT_NEAR(c[1], -2.0, 1e-9);
    EXPECT_NEAR(c[2], 0.5, 1e-9);
}

TEST(Fft, ImpulseAndRoundTrip) {
    constexpr int n = 16;
    std::vector<Complex> x(n, Complex(0.0, 0.0));
    x[3] = Complex(1.0, 0.0);
    auto X = fft_copy(x);
    for (int k = 0; k < n; ++k) {
        const Complex expected = std::exp(Complex(0.0, -2.0 * M_PI * 3.0 * k / n));
        EXPECT_LT(std::abs(X[k] - expected), 1e-12);
    }
    auto xr = fft_copy(X, true);
    for (int i = 0; i < n; ++i) {
        EXPECT_LT(std::abs(xr[i] - x[i]), 1e-12);
    }
}

TEST(Fft, MatchesNaiveDft) {
    constexpr int n = 32;
    std::vector<Complex> x(n);
    for (int i = 0; i < n; ++i) {
        x[i] = Complex(std::sin(0.3 * i), 0.1 * i);
    }
    auto X = fft_copy(x);
    for (int k = 0; k < n; ++k) {
        Complex ref(0.0, 0.0);
        for (int j = 0; j < n; ++j) {
            ref += x[j] * std::exp(Complex(0.0, -2.0 * M_PI * j * k / n));
        }
        EXPECT_LT(std::abs(X[k] - ref), 1e-10);
    }
    std::vector<Complex> bad(3);
    EXPECT_THROW(fft(bad), std::invalid_argument);
}

TEST(Ode, ExponentialScalar) {
    const auto r = solve_ode_scalar([](double, double y) { return y; },
                                    1.0, 0.0, 1.0, 0.01);
    ASSERT_FALSE(r.t.empty());
    EXPECT_NEAR(r.y.back(), std::exp(1.0), 1e-7);
}

TEST(Ode, HarmonicOscillatorSystem) {
    // y'' + y = 0 -> [x, v]' = [v, -x]
    const auto r = solve_ode_system(
        [](double, const Vector& y) {
            Vector d(2);
            d[0] = y[1];
            d[1] = -y[0];
            return d;
        },
        Vector{{1.0, 0.0}}, 0.0, 2.0 * M_PI, 0.01);
    ASSERT_FALSE(r.states.empty());
    EXPECT_NEAR(r.states.back()[0], 1.0, 2e-6);
    EXPECT_NEAR(r.states.back()[1], 0.0, 2e-6);
}

TEST(Ode, RejectsBadStep) {
    EXPECT_THROW(solve_ode_scalar([](double, double y) { return y; },
                                  1.0, 0.0, 1.0, 0.0),
                 std::invalid_argument);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
