#include "phase_adapter.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

#include "telescope/phase_diversity.hpp"

namespace instrument {

PhaseDiversityInverseSolver::PhaseDiversityInverseSolver() = default;

Solution PhaseDiversityInverseSolver::solve(const Observation& observation,
                                            const ParameterSpace& space,
                                            const SolveOptions& options) {
    Solution out;
    if (observation.frames.empty()) {
        out.message = "phase adapter: no frames";
        return out;
    }
    if (observation.stars.empty()) {
        out.message = "phase adapter: no star table";
        return out;
    }
    const int n_frames = static_cast<int>(observation.frames.size());
    if (static_cast<int>(observation.defocus.size()) != n_frames) {
        out.message = "phase adapter: defocus size != frames";
        return out;
    }

    telescope::OpticalParams optical;
    optical.img_size = observation.frames[0].image.width;
    if (optical.img_size <= 0) {
        out.message = "phase adapter: invalid image width";
        return out;
    }

    telescope::FrameSet fs;
    fs.img_size = optical.img_size;
    fs.defocus.assign(observation.defocus.begin(), observation.defocus.end());
    fs.stars.x.reserve(observation.stars.size());
    fs.stars.y.reserve(observation.stars.size());
    fs.stars.flux.reserve(observation.stars.size());
    for (const auto& s : observation.stars) {
        fs.stars.x.push_back(s.x);
        fs.stars.y.push_back(s.y);
        fs.stars.flux.push_back(s.flux);
    }
    fs.images.reserve(static_cast<std::size_t>(n_frames));
    for (const auto& f : observation.frames) {
        if (f.image.width != optical.img_size || !f.image.valid()) {
            out.message = "phase adapter: inconsistent frame shape";
            return out;
        }
        fs.images.push_back(f.image.pixels);
    }

    fs.truth.fill(0.0f);
    for (std::size_t i = 0; i < observation.truth.size() && i < fs.truth.size(); ++i) {
        fs.truth[i] = static_cast<float>(observation.truth[i]);
    }

    // Real data has no header background; median is the conservative default.
    // If the caller supplies per-frame background, use it.
    telescope::BackgroundMode mode = telescope::BackgroundMode::kMedian;
    fs.header_backgrounds.resize(static_cast<std::size_t>(n_frames), 0.0f);
    bool has_background = true;
    for (int i = 0; i < n_frames; ++i) {
        const auto it = observation.frames[static_cast<std::size_t>(i)].scalars.find("background");
        if (it == observation.frames[static_cast<std::size_t>(i)].scalars.end()) {
            has_background = false;
            break;
        }
        fs.header_backgrounds[static_cast<std::size_t>(i)] = static_cast<float>(it->second);
    }
    if (has_background) mode = telescope::BackgroundMode::kHeader;

    std::vector<float> data;
    try {
        data = telescope::preprocess_frames(fs, mode);
    } catch (const std::exception& e) {
        out.message = std::string("phase adapter preprocess: ") + e.what();
        return out;
    }

    telescope::SolveOptions pd;
    pd.n_starts = options.n_starts;
    pd.n_refine = options.n_refine;
    pd.seed = options.seed;
    pd.maxeval = options.maxeval;
    pd.verbose = options.verbose;
    pd.bounds = telescope::ZernikeBounds::default_bounds();

    try {
        telescope::PhaseDiversitySolver solver(optical, fs.stars, data, fs.defocus,
                                                fs.truth, pd);
        const auto r = solver.solve();
        out.ok = r.ok;
        out.cost = r.cost;
        out.nfev = static_cast<int>(r.evaluations);
        out.diagnostics["cost_true"] = r.cost_true;
        out.diagnostics["cost_zero"] = r.cost_zero;
        out.diagnostics["wavefront_rms"] = r.wavefront_rms;
        out.diagnostics["pseudo_solution"] = r.pseudo_solution ? 1.0 : 0.0;

        const auto& active = telescope::active_zernike_indices();
        out.values.clear();
        out.names.clear();
        for (int idx : active) {
            out.values.push_back(static_cast<double>(r.coeffs[static_cast<std::size_t>(idx)]));
        }
        if (space.size() == active.size()) {
            out.names = space.names();
        } else {
            for (std::size_t i = 0; i < out.values.size(); ++i) {
                out.names.push_back("z" + std::to_string(i + 4));
            }
        }
        out.message = r.ok ? "ok" : "phase solver returned not-ok";
    } catch (const std::exception& e) {
        out.ok = false;
        out.message = std::string("phase adapter solve: ") + e.what();
    }
    return out;
}

}  // namespace instrument
