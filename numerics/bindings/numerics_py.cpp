#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <complex>
#include <vector>

#include "telescope/numerics.hpp"

namespace py = pybind11;
using namespace telescope::numerics;

PYBIND11_MODULE(telescope_numerics_py, m) {
    m.doc() = "Small Eigen-based numerical core (linear solve, least squares, FFT, RK4)";

    m.def("solve_linear", &solve_linear,
          py::arg("A"), py::arg("b"),
          "Solve A x = b (partial-pivot LU).");
    m.def("least_squares", &least_squares,
          py::arg("A"), py::arg("b"),
          "Least squares via column-pivoted QR.");
    m.def("least_squares_svd", &least_squares_svd,
          py::arg("A"), py::arg("b"),
          "Least squares via SVD.");
    m.def("fit_polynomial", &fit_polynomial,
          py::arg("x"), py::arg("y"), py::arg("degree"),
          "Polynomial least-squares fit, coefficients low-to-high.");
    m.def("fft", [](const std::vector<Complex>& x, bool inverse) {
              return fft_copy(x, inverse);
          },
          py::arg("x"), py::arg("inverse") = false,
          "Radix-2 FFT; numpy-compatible sign convention.");
    m.def("solve_ode_scalar", [](py::function rhs, double y0,
                                 double t0, double t1, double dt) {
              auto f = [&rhs](double t, double y) {
                  return rhs(t, y).cast<double>();
              };
              const OdeResult r = solve_ode_scalar(f, y0, t0, t1, dt);
              py::dict out;
              out["t"] = r.t;
              out["y"] = r.y;
              return out;
          },
          py::arg("rhs"), py::arg("y0"), py::arg("t0"),
          py::arg("t1"), py::arg("dt"),
          "Classical RK4 for y' = f(t, y).");
}
