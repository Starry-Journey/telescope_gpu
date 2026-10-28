#include "telescope/phase_diversity.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <utility>

#include <nlopt.hpp>

#include "telescope/cuda_utils.cuh"
#include "telescope/forward_model.cuh"
#include "telescope/fits_io.hpp"
#include "telescope/zernike.hpp"

namespace telescope {

// 反演实际拟合的槽位：2/6/7（工程编号 Z4/Z8/Z9）。
const std::vector<int>& active_zernike_indices() {
    static const std::vector<int> kActive = {2, 6, 7};
    return kActive;
}

namespace {

constexpr int kBlock = 256;

__global__ void cost_reduce_kernel(const float* data,
                                   const float* model,
                                   float* costs,
                                   int total,
                                   int n_frames) {
    const int frame = blockIdx.x;
    if (frame >= n_frames || total <= 0) return;

    const float* a = data + static_cast<std::size_t>(frame) * total;
    const float* b = model + static_cast<std::size_t>(frame) * total;
    float s = 0.0f;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        const float d = a[i] - b[i];
        s += d * d;
    }

    __shared__ float smem[kBlock];
    smem[threadIdx.x] = s;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) costs[frame] = smem[0];
}

std::string frame_path(const std::string& dir, float theta_deg, float defocus) {
    char buf[512];
    const int n = std::snprintf(buf, sizeof(buf),
                                "%s/star_field_theta%03d_defocus_%+.1fwaves.fits",
                                dir.c_str(), static_cast<int>(std::lround(theta_deg)),
                                static_cast<double>(defocus));
    if (n < 0 || static_cast<std::size_t>(n) >= sizeof(buf)) {
        throw std::runtime_error("frame_path: path too long for buffer");
    }
    return std::string(buf);
}

// 真中位数：偶数个样本取中间两个的平均。原实现直接返回上中位数，
// 在 65536 像素（偶数）时会产生半个像素量级的背景偏差。
float median_of(std::vector<float> values) {
    if (values.empty()) return 0.0f;
    const std::size_t mid = values.size() / 2;
    const auto mid_it = values.begin() + static_cast<std::ptrdiff_t>(mid);
    std::nth_element(values.begin(), mid_it, values.end());
    if (values.size() % 2 != 0) return *mid_it;

    // 偶数：前半部分的最大值就是下中位数。
    const float upper = *mid_it;
    const float lower = *std::max_element(values.begin(), mid_it);
    return 0.5f * (lower + upper);
}

}  // namespace

class PhaseDiversitySolver::Impl {
public:
    Impl(const OpticalParams& params,
         const StarField& stars,
         const std::vector<float>& data,
         const std::vector<float>& defocus,
         int n_frames)
        : model(params, n_frames), host_defocus(defocus), host_costs(n_frames) {
        const int img_total = params.img_total();
        d_data.allocate(static_cast<std::size_t>(n_frames) * img_total);
        d_model.allocate(static_cast<std::size_t>(n_frames) * img_total);
        d_costs.allocate(static_cast<std::size_t>(n_frames));
        TEL_CUDA_CHECK(cudaMemcpy(d_data.get(), data.data(),
                                  data.size() * sizeof(float), cudaMemcpyHostToDevice));
        model.set_stars(stars);
    }

    ForwardModelGPU model;
    std::vector<float> host_defocus;
    std::vector<float> host_costs;  // 复用，避免每次 evaluate 都分配
    DeviceBuffer<float> d_data;
    DeviceBuffer<float> d_model;
    DeviceBuffer<float> d_costs;
    long long eval_count = 0;
};

PhaseDiversitySolver::PhaseDiversitySolver(const OpticalParams& params,
                                           const StarField& stars,
                                           const std::vector<float>& data_normalized,
                                           const std::vector<float>& defocus,
                                           const ZernikeCoeffs& truth,
                                           SolveOptions options)
    : params_(params),
      stars_(stars),
      data_(data_normalized),
      defocus_(defocus),
      truth_(truth),
      options_(options),
      n_frames_(static_cast<int>(defocus.size())),
      img_total_(params.img_total()) {
    if (n_frames_ < 2) {
        throw std::invalid_argument("phase diversity needs at least 2 frames");
    }
    if (n_frames_ > kMaxFrames) {
        throw std::invalid_argument("too many frames (kMaxFrames = 16)");
    }
    if (data_.size() != static_cast<std::size_t>(n_frames_) * img_total_) {
        throw std::invalid_argument("data size does not match frames * img_size^2");
    }
    if (!stars_.valid() || stars_.size() <= 0) {
        throw std::invalid_argument("invalid or empty star table");
    }
    if (!options_.bounds.valid()) {
        throw std::invalid_argument("invalid Zernike bounds: lower must be <= upper");
    }
    for (float d : defocus_) {
        if (!std::isfinite(d)) throw std::invalid_argument("defocus contains NaN/Inf");
    }
    for (float v : data_) {
        if (!std::isfinite(v)) throw std::invalid_argument("data contains NaN/Inf");
    }
    impl_ = new Impl(params_, stars_, data_, defocus_, n_frames_);
}

PhaseDiversitySolver::~PhaseDiversitySolver() {
    delete impl_;
}

float PhaseDiversitySolver::evaluate(const ZernikeCoeffs& coeffs) {
    ++impl_->eval_count;
    impl_->model.generate_frames(coeffs, impl_->host_defocus.data(), n_frames_,
                                 impl_->d_model.get());
    impl_->model.normalize_frames(impl_->d_model.get(), n_frames_);

    cost_reduce_kernel<<<n_frames_, kBlock>>>(
        impl_->d_data.get(), impl_->d_model.get(), impl_->d_costs.get(),
        img_total_, n_frames_);
    TEL_CUDA_CHECK(cudaGetLastError());

    TEL_CUDA_CHECK(cudaMemcpy(impl_->host_costs.data(), impl_->d_costs.get(),
                              impl_->host_costs.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
    float total = 0.0f;
    for (float c : impl_->host_costs) total += c;
    return total;
}

namespace {

struct NloptContext {
    PhaseDiversitySolver* solver;
    const std::vector<int>* active;
};

double nlopt_objective(const std::vector<double>& x,
                       std::vector<double>& /*grad*/,
                       void* data) {
    auto* ctx = static_cast<NloptContext*>(data);
    ZernikeCoeffs coeffs{};
    for (std::size_t a = 0; a < ctx->active->size(); ++a) {
        coeffs[(*ctx->active)[a]] = static_cast<float>(x[a]);
    }
    try {
        return static_cast<double>(ctx->solver->evaluate(coeffs));
    } catch (const std::exception&) {
        return 1.0e30;  // 让优化器远离不可行区域，而不是把异常抛进 C 回调。
    }
}

}  // namespace

SolveResult PhaseDiversitySolver::solve() {
    if (options_.n_starts < 1) throw std::invalid_argument("n_starts must be >= 1");
    if (options_.n_refine < 1) throw std::invalid_argument("n_refine must be >= 1");
    if (options_.maxeval < 1) throw std::invalid_argument("maxeval must be >= 1");

    SolveResult result;
    const auto& active = active_zernike_indices();
    const bool verbose = options_.verbose;

    result.cost_true = evaluate(truth_);
    result.cost_zero = evaluate(ZernikeCoeffs{});

    if (verbose) {
        std::printf("\n=== 代价函数诊断 ===\n");
        std::printf("cost(true) = %.6e\n", result.cost_true);
        std::printf("cost(zero) = %.6e\n", result.cost_zero);
    }

    struct Candidate {
        ZernikeCoeffs coeffs;
        float cost;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<std::size_t>(options_.n_starts));

    std::mt19937 rng(options_.seed);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

    if (verbose) std::printf("\n=== 多起点粗搜（%d 起点）===\n", options_.n_starts);
    for (int s = 0; s < options_.n_starts; ++s) {
        ZernikeCoeffs coeffs{};
        for (int k : active) {
            coeffs[k] = options_.bounds.lower[k] +
                        uniform(rng) * (options_.bounds.upper[k] - options_.bounds.lower[k]);
        }

        float best = evaluate(coeffs);
        float step = 0.3f;
        for (int iter = 0; iter < 200 && step >= 1e-4f; ++iter) {
            bool improved = false;
            for (int k : active) {
                for (int dir : {-1, 1}) {
                    ZernikeCoeffs trial = coeffs;
                    trial[k] = options_.bounds.clamp(
                        k, coeffs[k] + static_cast<float>(dir) * step);
                    if (trial[k] == coeffs[k]) continue;

                    const float cost = evaluate(trial);
                    if (cost < best) {
                        best = cost;
                        coeffs = trial;
                        improved = true;
                    }
                }
            }
            if (!improved) step *= 0.5f;
        }

        candidates.push_back({coeffs, best});
        if (verbose) {
            std::printf("start %2d: Z4=%.4f Z8=%.4f Z9=%.4f | cost=%.4e\n",
                        s, coeffs[2], coeffs[6], coeffs[7], best);
        }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.cost < b.cost; });

    const int refine = std::min(options_.n_refine, static_cast<int>(candidates.size()));
    if (verbose) {
        std::printf("\n=== BOBYQA 精修（前 %d 名）===\n", refine);
    }

    NloptContext ctx{this, &active};
    Candidate best_global = candidates.front();

    for (int rank = 0; rank < refine; ++rank) {
        try {
            nlopt::opt opt(nlopt::LN_BOBYQA, static_cast<unsigned>(active.size()));

            std::vector<double> lower(active.size());
            std::vector<double> upper(active.size());
            std::vector<double> x(active.size());
            for (std::size_t a = 0; a < active.size(); ++a) {
                lower[a] = options_.bounds.lower[active[a]];
                upper[a] = options_.bounds.upper[active[a]];
                x[a] = candidates[rank].coeffs[active[a]];
            }

            opt.set_lower_bounds(lower);
            opt.set_upper_bounds(upper);
            opt.set_min_objective(nlopt_objective, &ctx);
            opt.set_maxeval(options_.maxeval);
            opt.set_ftol_rel(1e-10);
            opt.set_xtol_rel(1e-6);

            double minf = 0.0;
            const nlopt::result status = opt.optimize(x, minf);

            ZernikeCoeffs coeffs{};
            for (std::size_t a = 0; a < active.size(); ++a) {
                coeffs[active[a]] = static_cast<float>(x[a]);
            }

            if (verbose) {
                std::printf("rank %d: Z4=%.4f Z8=%.4f Z9=%.4f | cost=%.4e (status=%d)\n",
                            rank, coeffs[2], coeffs[6], coeffs[7], minf,
                            static_cast<int>(status));
            }

            if (minf < best_global.cost) {
                best_global = {coeffs, static_cast<float>(minf)};
            }
        } catch (const std::exception& e) {
            // 某一起点精修失败不应让整个反演失败：保留粗搜候选继续。
            if (verbose) std::printf("rank %d: nlopt 异常: %s\n", rank, e.what());
        }
    }

    result.coeffs = best_global.coeffs;
    result.cost = best_global.cost;
    result.ok = !candidates.empty();
    result.evaluations = impl_->eval_count;

    const PupilGrid grid = make_pupil_grid(params_);
    result.wavefront_rms = wavefront_rms(result.coeffs, truth_, grid);
    // 拟合 3 个参数本身就能把噪声代价压低约 3/像素数（远小于 0.1%），
    // 因此用 1% 的相对阈值，避免把正常的噪声过拟合误报为伪解。
    constexpr float kPseudoCostTolerance = 0.01f;
    result.pseudo_solution =
        result.cost < result.cost_true * (1.0f - kPseudoCostTolerance);
    return result;
}

bool load_frame_set(const std::string& data_dir,
                    float theta_deg,
                    const std::vector<float>& defocus,
                    FrameSet& out,
                    std::string& error) {
    if (defocus.size() < 2 || defocus.size() > static_cast<std::size_t>(kMaxFrames)) {
        error = "defocus needs 2..16 frames";
        return false;
    }
    if (!std::isfinite(theta_deg)) {
        error = "theta_deg must be finite";
        return false;
    }

    out = FrameSet{};
    out.defocus = defocus;

    for (std::size_t f = 0; f < defocus.size(); ++f) {
        const std::string path = frame_path(data_dir, theta_deg, defocus[f]);
        std::vector<float> image;
        FitsMetadata meta;
        if (!read_fits_image(path, image, meta)) {
            error = "cannot read FITS: " + path;
            return false;
        }
        if (meta.img_size <= 0 ||
            image.size() != static_cast<std::size_t>(meta.img_size) * meta.img_size) {
            error = "corrupt image size in " + path;
            return false;
        }
        // 头部自描述必须和本次请求一致，否则很容易把不同角度/离焦的帧
        // 混在一起，得到“看起来收敛”的错误结果。
        if (std::fabs(meta.theta_deg - theta_deg) > 0.5f) {
            error = "theta mismatch in " + path + ": header " +
                    std::to_string(meta.theta_deg) + " vs requested " +
                    std::to_string(theta_deg);
            return false;
        }
        if (std::fabs(meta.defocus - defocus[f]) > 0.05f) {
            error = "defocus mismatch in " + path + ": header " +
                    std::to_string(meta.defocus) + " vs requested " +
                    std::to_string(defocus[f]);
            return false;
        }

        if (f == 0) {
            out.img_size = meta.img_size;
            out.truth = meta.zernike;
            out.background = meta.background;
            out.stars.x = meta.star_x;
            out.stars.y = meta.star_y;
            out.stars.flux = meta.star_flux;
        } else {
            if (meta.img_size != out.img_size) {
                error = "inconsistent image size in " + path;
                return false;
            }
            if (meta.star_x != out.stars.x || meta.star_y != out.stars.y ||
                meta.star_flux != out.stars.flux) {
                error = "star table differs between frames: " + path;
                return false;
            }
            for (int k = 0; k < kNumZernike; ++k) {
                if (std::fabs(meta.zernike[k] - out.truth[k]) > 1e-4f) {
                    error = "truth Zernike differs between frames: " + path;
                    return false;
                }
            }
        }

        out.header_backgrounds.push_back(meta.background);
        out.images.push_back(std::move(image));
    }

    if (out.img_size <= 0) {
        error = "empty frame set";
        return false;
    }
    if (!out.stars.valid() || out.stars.size() == 0) {
        error = "FITS header does not contain a valid star table "
                "(regenerate data with the v2 forward_generate)";
        return false;
    }
    return true;
}

std::vector<float> preprocess_frames(const FrameSet& frames,
                                     BackgroundMode mode,
                                     std::vector<float>* frame_backgrounds) {
    const int n = static_cast<int>(frames.images.size());
    const int img_total = frames.img_size * frames.img_size;
    if (n < 1 || img_total <= 0) {
        throw std::invalid_argument("preprocess_frames: empty frame set");
    }
    if (!frames.defocus.empty() && static_cast<int>(frames.defocus.size()) != n) {
        throw std::invalid_argument("preprocess_frames: defocus/images size mismatch");
    }

    std::vector<float> out(static_cast<std::size_t>(n) * img_total, 0.0f);
    if (frame_backgrounds) frame_backgrounds->assign(n, 0.0f);

    for (int f = 0; f < n; ++f) {
        if (static_cast<int>(frames.images[f].size()) != img_total) {
            throw std::invalid_argument("frame size mismatch in preprocess_frames");
        }

        float bg = 0.0f;
        bool have_header_bg = false;
        if (mode == BackgroundMode::kHeader) {
            if (f < static_cast<int>(frames.header_backgrounds.size()) &&
                frames.header_backgrounds[f] > 0.0f) {
                bg = frames.header_backgrounds[f];
                have_header_bg = true;
            } else if (frames.background > 0.0f) {
                bg = frames.background;
                have_header_bg = true;
            }
        }
        if (!have_header_bg) {
            std::vector<float> tmp = frames.images[f];
            bg = median_of(std::move(tmp));
        }
        if (frame_backgrounds) (*frame_backgrounds)[f] = bg;

        float sum = 0.0f;
        for (int i = 0; i < img_total; ++i) {
            // 背景扣除后必须保留负值。若在这里 clamp 到 0，泊松噪声会被
            // 翻成正偏置（约 5.6 count/像素），总量被抬高数倍，归一化后
            // 科学信号被稀释，代价函数会把解推向“最锐利”的伪解。
            const float v = frames.images[f][i] - bg;
            out[static_cast<std::size_t>(f) * img_total + i] = v;
            sum += v;
        }
        // 旧实现 sum<=0 时静默跳过归一化，导致该帧量纲和其他帧不同，
        // 代价函数被单帧主导。这里显式失败，让调用方看到坏数据。
        if (!std::isfinite(sum) || sum <= 1e-12f) {
            throw std::runtime_error("frame " + std::to_string(f) +
                                     " has non-positive total signal; check background");
        }
        const float inv = 1.0f / sum;
        for (int i = 0; i < img_total; ++i) {
            out[static_cast<std::size_t>(f) * img_total + i] *= inv;
        }
    }
    return out;
}

}  // namespace telescope
