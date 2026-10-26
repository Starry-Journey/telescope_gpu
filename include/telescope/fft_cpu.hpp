#pragma once

#include <complex>
#include <vector>

namespace telescope {

// 仅用于 CPU reference / 单元测试的 radix-2 FFT。
// 要求长度是 2 的幂；不做性能优化，正确性优先。
using Complex = std::complex<double>;

void fft1d(std::vector<Complex>& data, bool inverse);

// 二维 FFT：先对每一行做 1D FFT，再对每一列做 1D FFT。
// data 按行优先存储，size = n*n，n 必须是 2 的幂。
void fft2d(std::vector<Complex>& data, int n, bool inverse);

}  // namespace telescope
