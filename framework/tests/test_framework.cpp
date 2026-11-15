/// @file test_framework.cpp
/// @brief Dependency-free framework tests (assert + CTest).

#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>

#include "instrument/control.hpp"
#include "instrument/fits_reader.hpp"
#include "instrument/parameter.hpp"
#include "instrument/registry.hpp"
#include "instrument/sensitivity.hpp"
#include "synthetic_linear.hpp"
#include "telescope/fits_io.hpp"

using namespace instrument;

namespace {
void require(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "FAIL: " << msg << "\n";
        std::abort();
    }
}

ParameterSpace make_space() {
    return ParameterSpace({
        {"uz", "mm", "control", -0.5, 0.5, 0.0, true},
        {"ux", "mm", "control", -0.5, 0.5, 0.0, true},
        {"uy", "mm", "control", -0.5, 0.5, 0.0, true},
        {"ua", "mrad", "control", -0.5, 0.5, 0.0, true},
        {"ub", "mrad", "control", -0.5, 0.5, 0.0, true},
    }, "sm5");
}

ControlSpace make_controls(const ParameterSpace& sp) {
    ControlSpace cs;
    cs.name = "sm5";
    for (int i = 0; i < static_cast<int>(sp.size()); ++i) {
        ControlAxis a;
        a.name = sp.spec(i).name;
        a.unit = sp.spec(i).unit;
        a.parameter_index = i;
        a.step = 1e-3;
        a.lower = sp.spec(i).lower;
        a.upper = sp.spec(i).upper;
        cs.axes.push_back(a);
    }
    return cs;
}
}  // namespace

int main() {
    // 1. parameter space round trip and bounds
    auto sp = make_space();
    require(sp.size() == 5, "space size");
    require(sp.index_of("ua") == 3, "index_of");
    auto b = sp.encode({0.1, -0.2, 0.3, 0.01, -0.02});
    require(b.size() == 5 && b[0] == 0.1, "encode");
    auto p = sp.encode({1.0, -1.0, 0.0, 0.0, 0.0});
    require(sp.clamp(p) == 2, "clamp count");
    require(p[0] == 0.5 && p[1] == -0.5, "clamp values");

    // 2. synthetic forward + finite-difference sensitivity == known A
    std::vector<std::vector<double>> A = {
        {4.040, 0.083, -0.083, 0.976, -0.976},
        {0.000, -4.633, 0.000, 0.000, 0.000},
        {0.000, 0.000, -4.633, 0.000, 0.000},
    };
    SyntheticLinearForward model(A);
    auto zero = sp.encode({0, 0, 0, 0, 0});
    FiniteDifferenceSensitivity fd;
    auto S = fd.build(model, sp, make_controls(sp), IdentityExtractor(), zero, {});
    require(S.valid() && S.rows() == 3 && S.cols() == 5, "S shape");
    require(S.rank == 3, "S rank");
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 5; ++c) {
            require(std::abs(S(r, c) - A[r][c]) < 1e-6, "S value");
        }
    }

    // 3. control recovers a known u from a known dz
    std::vector<double> truth = {0.20, -0.10, 0.05, 0.01, -0.02};
    auto z = model.evaluate(sp.encode(truth), {});
    ControlOptions opt;
    opt.lambda = 1e-8;
    opt.use_bounds = true;
    opt.lower.assign(5, -0.5);
    opt.upper.assign(5, 0.5);
    WeightedLeastSquaresControl solver;
    auto sol = solver.solve(S, z.values, opt);
    require(sol.ok, "control ok");
    require(sol.rank == 3, "control rank");
    // 3x5 is underdetermined: u is the minimum-norm solution, not the truth.
    // The control target must be reproduced even when u != truth.
    double max_res = 0.0;
    for (int i = 0; i < 3; ++i) max_res = std::max(max_res, std::abs(sol.residual[i]));
    require(max_res < 1e-8, "control target residual");
    require(sol.cost < 1e-12, "control cost");

    // 5x5 full-rank case: exact recovery of the control vector.
    std::vector<std::vector<double>> A5(5, std::vector<double>(5, 0.0));
    for (int i = 0; i < 5; ++i) A5[i][i] = 2.0 + i;
    SyntheticLinearForward model5(A5, {0, 0, 0, 0, 0}, {"t0", "t1", "t2", "t3", "t4"});
    auto S5 = fd.build(model5, sp, make_controls(sp), IdentityExtractor(), zero, {});
    auto z5 = model5.evaluate(sp.encode(truth), {});
    auto sol5 = solver.solve(S5, z5.values, opt);
    require(sol5.ok && sol5.rank == 5, "5x5 control rank");
    for (int i = 0; i < 5; ++i) require(std::abs(sol5.u[i] - truth[i]) < 1e-6, "5x5 recovery");

    // 4. plugin registry
    Registry<ForwardModel> reg;
    reg.add("synthetic", [&] { return std::make_unique<SyntheticLinearForward>(A); });
    require(reg.contains("synthetic"), "registry contains");
    auto created = reg.create("synthetic");
    require(created && created->name() == "synthetic_linear", "registry create");

    // 5. FITS -> Observation (upstream boundary)
    {
        telescope::FitsMetadata meta;
        meta.img_size = 8;
        meta.defocus = 1.5f;
        meta.background = 123.0f;
        meta.read_noise = 5.0f;
        meta.theta_deg = 45.0f;
        meta.zernike[2] = 0.6f;
        meta.star_x = {3.0f, 5.0f};
        meta.star_y = {4.0f, 2.0f};
        meta.star_flux = {1000.0f, 800.0f};
        std::vector<float> image(64, 10.0f);
        const auto path = std::filesystem::temp_directory_path() / "framework_test.fits";
        require(telescope::write_fits_image(path.string(), image, meta), "write FITS");
        Observation obs;
        const auto st = read_fits_observation({path.string()}, obs);
        std::filesystem::remove(path);
        require(st.ok, "read FITS observation");
        require(obs.frames.size() == 1 && obs.defocus.size() == 1, "FITS frame count");
        require(std::abs(obs.defocus[0] - 1.5) < 1e-6, "FITS defocus");
        require(obs.stars.size() == 2 && obs.frames[0].image.width == 8, "FITS stars/shape");
    }

    std::cout << "framework tests passed: rank=" << S.rank
              << " cond=" << S.condition << " max_err="
              << [&] {
                     double e = 0;
                     for (int i = 0; i < 5; ++i) e = std::max(e, std::abs(sol.u[i] - truth[i]));
                     return e;
                 }()
              << "\n";
    return 0;
}
