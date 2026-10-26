#include "test_framework.hpp"

#include <cmath>
#include <complex>
#include <vector>

#include "telescope/fft_cpu.hpp"

using namespace telescope;

namespace {

std::vector<Complex> naive_dft(const std::vector<Complex>& in, bool inverse) {
    const int n = static_cast<int>(in.size());
    const double pi = 3.14159265358979323846;
    const double sign = inverse ? 1.0 : -1.0;
    std::vector<Complex> out(n);
    for (int k = 0; k < n; ++k) {
        Complex sum(0.0, 0.0);
        for (int t = 0; t < n; ++t) {
            const double angle = sign * 2.0 * pi * k * t / n;
            sum += in[t] * Complex(std::cos(angle), std::sin(angle));
        }
        if (inverse) sum /= static_cast<double>(n);
        out[k] = sum;
    }
    return out;
}

}  // namespace

TEST(fft1d_matches_naive_dft) {
    std::vector<Complex> data = {
        {1.0, 0.0}, {2.0, -1.0}, {0.5, 0.25}, {-3.0, 2.0},
        {0.0, 0.0}, {4.0, 1.0}, {-2.0, -0.5}, {1.5, 0.0},
    };
    const auto expected = naive_dft(data, false);

    auto actual = data;
    fft1d(actual, false);
    for (std::size_t i = 0; i < data.size(); ++i) {
        CHECK_NEAR(actual[i].real(), expected[i].real(), 1e-10);
        CHECK_NEAR(actual[i].imag(), expected[i].imag(), 1e-10);
    }
}

TEST(fft1d_impulse) {
    std::vector<Complex> data(8, Complex(0.0, 0.0));
    data[0] = Complex(1.0, 0.0);
    fft1d(data, false);
    for (const auto& v : data) {
        CHECK_NEAR(v.real(), 1.0, 1e-12);
        CHECK_NEAR(v.imag(), 0.0, 1e-12);
    }
}

TEST(fft1d_roundtrip) {
    std::vector<Complex> data = {
        {0.2, -0.4}, {1.3, 2.1}, {-0.7, 0.0}, {3.2, -1.1},
        {0.0, 0.5}, {-2.2, 0.9}, {1.0, 1.0}, {0.3, -0.2},
    };
    const auto original = data;
    fft1d(data, false);
    fft1d(data, true);
    for (std::size_t i = 0; i < data.size(); ++i) {
        CHECK_NEAR(data[i].real(), original[i].real(), 1e-12);
        CHECK_NEAR(data[i].imag(), original[i].imag(), 1e-12);
    }
}

TEST(fft1d_parseval) {
    std::vector<Complex> data = {
        {1.0, 0.0}, {-2.0, 1.0}, {0.5, -0.5}, {3.0, 2.0},
        {0.0, 0.0}, {1.0, -1.0}, {2.0, 0.25}, {-0.5, 0.5},
    };
    const auto original = data;
    fft1d(data, false);

    double time_energy = 0.0;
    for (const auto& v : original) time_energy += std::norm(v);
    double freq_energy = 0.0;
    for (const auto& v : data) freq_energy += std::norm(v);

    CHECK_NEAR(freq_energy / static_cast<double>(data.size()), time_energy, 1e-10);
}

TEST(fft2d_roundtrip) {
    const int n = 8;
    std::vector<Complex> data(n * n);
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            data[y * n + x] = Complex(std::sin(0.3 * x) + 0.2 * y,
                                      std::cos(0.7 * y) - 0.1 * x);
        }
    }
    const auto original = data;
    fft2d(data, n, false);
    fft2d(data, n, true);
    for (std::size_t i = 0; i < data.size(); ++i) {
        CHECK_NEAR(data[i].real(), original[i].real(), 1e-12);
        CHECK_NEAR(data[i].imag(), original[i].imag(), 1e-12);
    }
}

TEST(fft1d_rejects_non_power_of_two) {
    std::vector<Complex> data(6, Complex(0.0, 0.0));
    CHECK_THROWS(fft1d(data, false));
}

TEST(fft2d_rejects_wrong_size) {
    std::vector<Complex> data(9, Complex(0.0, 0.0));
    CHECK_THROWS(fft2d(data, 4, false));
}
