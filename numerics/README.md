# numerics：Eigen + GTest + Benchmark + pybind11

小型数值核心：线性方程组、最小二乘、radix-2 FFT、RK4。详细教程见
`../docs/tutorials/01_C++数值库.md`。

## 构建

```bash
cmake -S numerics -B build-numerics -DCMAKE_BUILD_TYPE=Release
cmake --build build-numerics -j4
ctest --test-dir build-numerics --output-on-failure
./build-numerics/numerics_bench --benchmark_min_time=0.02s
```

依赖缺失时 CMake 用 `FetchContent` 拉 Eigen/GTest/Google Benchmark/pybind11；
WSL 上默认不 `find_package(GTest)`，避免误链 Windows conda 的 `gtest.lib`。

## API

```cpp
#include "telescope/numerics.hpp"
using namespace telescope::numerics;

Vector x = solve_linear(A, b);
Vector c = least_squares(A, b);
Vector p = fit_polynomial(xs, ys, 2);
std::vector<Complex> X = fft_copy(x);
OdeResult r = solve_ode_scalar([](double t, double y){ return y; }, 1.0, 0.0, 1.0, 0.01);
```

## Python

```bash
PYTHONPATH=build-numerics/python python3 -c \
  "import telescope_numerics_py as tn, numpy as np; print(tn.solve_linear(np.eye(3), np.ones(3)))"
```

## 测试与性能

- `tests/test_numerics.cpp`：10 个 GTest（已知解、SVD/QR 对照、FFT 与朴素 DFT、RK4 解析解、异常路径）。
- `bench/bench_numerics.cpp`：solve/lstsq/FFT/RK4 复杂度。
- ASan：`-DTEL_NUMERICS_ENABLE_ASAN=ON`。
