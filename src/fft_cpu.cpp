#include "telescope/fft_cpu.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace telescope {
namespace {

constexpr double kPi = 3.14159265358979323846;

bool is_power_of_two(int n) { return n > 0 && (n & (n - 1)) == 0; }

}  // namespace

void fft1d(std::vector<Complex>& data, bool inverse) {
    const int n = static_cast<int>(data.size());
    if (!is_power_of_two(n)) {
        throw std::invalid_argument("fft1d: size must be a power of two");
    }
    if (n <= 1) return;

    // 位反转置换。
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(data[i], data[j]);
    }

    // Cooley-Tukey 蝶形运算。
    for (int len = 2; len <= n; len <<= 1) {
        const double angle = 2.0 * kPi / len * (inverse ? 1.0 : -1.0);
        const Complex wlen(std::cos(angle), std::sin(angle));
        for (int i = 0; i < n; i += len) {
            Complex w(1.0, 0.0);
            for (int j = 0; j < len / 2; ++j) {
                const Complex u = data[i + j];
                const Complex v = data[i + j + len / 2] * w;
                data[i + j] = u + v;
                data[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }

    if (inverse) {
        const double inv_n = 1.0 / n;
        for (Complex& x : data) x *= inv_n;
    }
}

void fft2d(std::vector<Complex>& data, int n, bool inverse) {
    if (!is_power_of_two(n)) {
        throw std::invalid_argument("fft2d: n must be a power of two");
    }
    if (static_cast<int>(data.size()) != n * n) {
        throw std::invalid_argument("fft2d: data size must be n*n");
    }

    std::vector<Complex> line(n);
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) line[x] = data[y * n + x];
        fft1d(line, inverse);
        for (int x = 0; x < n; ++x) data[y * n + x] = line[x];
    }
    for (int x = 0; x < n; ++x) {
        for (int y = 0; y < n; ++y) line[y] = data[y * n + x];
        fft1d(line, inverse);
        for (int y = 0; y < n; ++y) data[y * n + x] = line[y];
    }
}

}  // namespace telescope
