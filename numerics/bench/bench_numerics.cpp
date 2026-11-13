#include <benchmark/benchmark.h>

#include <complex>
#include <vector>

#include "telescope/numerics.hpp"

using namespace telescope::numerics;

static void BM_SolveLinear(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    Matrix A = Matrix::Random(n, n);
    A.diagonal().array() += static_cast<double>(n);
    Vector b = Vector::Random(n);
    for (auto _ : state) {
        const Vector x = solve_linear(A, b);
        benchmark::DoNotOptimize(x.data());
    }
    state.SetComplexityN(n);
}
BENCHMARK(BM_SolveLinear)->Arg(8)->Arg(32)->Arg(128)->Complexity();

static void BM_LeastSquares(benchmark::State& state) {
    const int rows = static_cast<int>(state.range(0));
    const int cols = static_cast<int>(state.range(1));
    Matrix A = Matrix::Random(rows, cols);
    Vector b = Vector::Random(rows);
    for (auto _ : state) {
        const Vector x = least_squares(A, b);
        benchmark::DoNotOptimize(x.data());
    }
}
BENCHMARK(BM_LeastSquares)->Args({256, 16})->Args({1024, 64});

static void BM_Fft(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    std::vector<Complex> x(n);
    for (int i = 0; i < n; ++i) {
        x[i] = Complex(std::sin(0.1 * i), 0.0);
    }
    for (auto _ : state) {
        auto y = x;
        fft(y, false);
        benchmark::DoNotOptimize(y.data());
    }
    state.SetComplexityN(n);
}
BENCHMARK(BM_Fft)->Arg(256)->Arg(1024)->Arg(4096)->Complexity();

static void BM_Rk4Scalar(benchmark::State& state) {
    const int steps = static_cast<int>(state.range(0));
    for (auto _ : state) {
        auto r = solve_ode_scalar([](double, double y) { return -y; },
                                  1.0, 0.0, 1.0, 1.0 / steps);
        benchmark::DoNotOptimize(r.y.data());
    }
}
BENCHMARK(BM_Rk4Scalar)->Arg(100)->Arg(1000);

BENCHMARK_MAIN();
