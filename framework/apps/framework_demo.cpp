/// @file framework_demo.cpp
/// @brief End-to-end demo: parameter space -> forward -> sensitivity -> control.
///
/// The matrix is synthetic (3x5, Z4/Z8/Z9 x uz/ux/uy/uα/uβ) so the demo has a
/// known answer.  The same code path accepts a real ForwardModel plugin.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "instrument/cache.hpp"
#include "instrument/config.hpp"
#include "instrument/control.hpp"
#include "instrument/provenance.hpp"
#include "instrument/sensitivity.hpp"
#include "synthetic_linear.hpp"

using namespace instrument;

int main(int argc, char** argv) {
    FrameworkConfig cfg;
    if (argc > 1) cfg.output_dir = argv[1];
    const std::string out_dir = cfg.output_dir;
    std::filesystem::create_directories(out_dir);

    ParameterSpace space({
        {"uz", "mm", "control", -0.5, 0.5, 0.0, true},
        {"ux", "mm", "control", -0.5, 0.5, 0.0, true},
        {"uy", "mm", "control", -0.5, 0.5, 0.0, true},
        {"ua", "mrad", "control", -0.5, 0.5, 0.0, true},
        {"ub", "mrad", "control", -0.5, 0.5, 0.0, true},
    }, "secondary_mirror_5dof");

    std::vector<std::vector<double>> A = {
        {4.040, 0.083, -0.083, 0.976, -0.976},  // Z4
        {0.000, -4.633, 0.000, 0.000, 0.000},   // Z8
        {0.000, 0.000, -4.633, 0.000, 0.000},   // Z9
    };
    SyntheticLinearForward model(A, {0, 0, 0}, {"Z4", "Z8", "Z9"});

    const std::vector<double> truth = {0.20, -0.10, 0.05, 0.01, -0.02};
    auto base = space.encode({0, 0, 0, 0, 0});
    auto truth_block = space.encode(truth);
    const auto z_target = model.evaluate(truth_block, {});

    ControlSpace controls;
    controls.name = "secondary_mirror";
    for (int i = 0; i < 5; ++i) {
        ControlAxis a;
        a.name = space.spec(i).name;
        a.unit = space.spec(i).unit;
        a.parameter_index = i;
        a.step = 1e-3;
        a.lower = space.spec(i).lower;
        a.upper = space.spec(i).upper;
        controls.axes.push_back(a);
    }

    FiniteDifferenceSensitivity builder;
    const auto S = builder.build(model, space, controls, IdentityExtractor(), base, Context{});

    ControlOptions copt;
    copt.lambda = 1e-8;
    copt.use_bounds = true;
    copt.lower = {-0.5, -0.5, -0.5, -0.5, -0.5};
    copt.upper = {0.5, 0.5, 0.5, 0.5, 0.5};
    WeightedLeastSquaresControl solver;
    const auto sol = solver.solve(S, z_target.values, copt);

    std::cout << "Sensitivity matrix (" << S.rows() << "x" << S.cols() << "), rank="
              << S.rank << ", cond=" << S.condition << "\n";
    std::cout << std::scientific << std::setprecision(6);
    for (int r = 0; r < S.rows(); ++r) {
        std::cout << S.target_names[r] << ": ";
        for (int c = 0; c < S.cols(); ++c) std::cout << S(r, c) << " ";
        std::cout << "\n";
    }
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "control ok=" << sol.ok << " rank=" << sol.rank
              << " cond=" << sol.condition << " nullspace=" << (S.cols() - sol.rank)
              << " cost=" << sol.cost << "\n";
    for (int i = 0; i < S.cols(); ++i) {
        std::cout << "  " << S.control_names[i] << " u_minnorm=" << sol.u[i]
                  << " truth=" << truth[i] << " diff=" << std::abs(sol.u[i] - truth[i]) << "\n";
    }
    std::cout << "  target residual: ";
    for (double r : sol.residual) std::cout << r << " ";
    std::cout << "\n  note: 3x5 is rank-3; u is the minimum-norm solution, "
                 "not necessarily physical truth.\n";

    std::ofstream csv(out_dir + "/sensitivity.csv");
    csv << "target";
    for (const auto& n : S.control_names) csv << "," << n;
    csv << "\n";
    for (int r = 0; r < S.rows(); ++r) {
        csv << S.target_names[r];
        for (int c = 0; c < S.cols(); ++c) csv << "," << S(r, c);
        csv << "\n";
    }

    RunManifest manifest;
    manifest.run_id = "framework-demo";
    manifest.git_commit = "local";
    manifest.backend = cfg.backend;
    manifest.seed = cfg.seed;
    manifest.config_hash = "synthetic-3x5";
    manifest.dataset_hash = "model:Z4/Z8/Z9";
    manifest.metrics["rank"] = S.rank;
    manifest.metrics["nullspace_dim"] = S.cols() - S.rank;
    manifest.metrics["condition"] = S.condition;
    manifest.metrics["control_cost"] = sol.cost;
    manifest.metrics["max_target_residual"] = [&] {
        double e = 0;
        for (double r : sol.residual) e = std::max(e, std::abs(r));
        return e;
    }();
    manifest.versions["framework"] = "1.0";
    std::ofstream mf(out_dir + "/run_manifest.json");
    mf << manifest.to_json();

    ResultCache cache;
    cache.put("framework-demo/sensitivity", manifest.config_hash);
    if (!cache.has("framework-demo/sensitivity")) {
        std::cerr << "cache self-check failed\n";
        return 2;
    }

    std::ofstream rep(out_dir + "/control_result.json");
    rep << "{\n  \"ok\": " << (sol.ok ? "true" : "false") << ",\n";
    rep << "  \"rank\": " << sol.rank << ",\n";
    rep << "  \"condition\": " << sol.condition << ",\n";
    rep << "  \"u\": [";
    for (int i = 0; i < S.cols(); ++i) {
        if (i) rep << ", ";
        rep << sol.u[i];
    }
    rep << "],\n  \"truth\": [";
    for (int i = 0; i < S.cols(); ++i) {
        if (i) rep << ", ";
        rep << truth[i];
    }
    rep << "]\n}\n";

    std::cout << "wrote " << out_dir << "/sensitivity.csv, control_result.json, run_manifest.json\n";
    return sol.ok ? 0 : 1;
}
