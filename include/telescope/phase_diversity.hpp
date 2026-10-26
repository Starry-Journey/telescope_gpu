#pragma once

#include <limits>
#include <string>
#include <vector>

#include "telescope/types.hpp"

namespace telescope {

// 一次相位差观测：n 帧不同离焦量的星场图 + 星点表 + 真值。
struct FrameSet {
    int img_size = 0;
    std::vector<float> defocus;
    std::vector<std::vector<float>> images;  // [frame][pixel]
    StarField stars;
    ZernikeCoeffs truth{};
    float background = 0.0f;
    // 每帧 FITS 头部记录的背景。仿真数据里这是真值，优先使用；
    // 真实数据没有这个信息时退回中位数估计。
    std::vector<float> header_backgrounds;
};

// 从 data_dir 读取 forward_generate 写出的 FITS。
// 文件名约定：star_field_theta%03d_defocus_%+.1fwaves.fits。
bool load_frame_set(const std::string& data_dir,
                    float theta_deg,
                    const std::vector<float>& defocus,
                    FrameSet& out,
                    std::string& error);

enum class BackgroundMode {
    kHeader,  // 用 FITS 头部 BKG（仿真真值；真实数据可由定标得到）
    kMedian,  // 用每帧中位数（无头部信息时的保守估计）
};

// 每帧：扣背景 -> 按总能量归一化。负值必须保留（见 src/phase_diversity.cu
// 中的注释：clamp 会把泊松噪声翻成正偏置，导致信号被稀释和伪解）。
// 返回 [frame][pixel] 的连续数组；实际使用的背景写入 frame_backgrounds。
std::vector<float> preprocess_frames(
    const FrameSet& frames,
    BackgroundMode mode = BackgroundMode::kHeader,
    std::vector<float>* frame_backgrounds = nullptr);

struct SolveOptions {
    int n_starts = 40;
    int n_refine = 3;
    unsigned seed = 12345;
    int maxeval = 2000;
    ZernikeBounds bounds = ZernikeBounds::default_bounds();
};

struct SolveResult {
    ZernikeCoeffs coeffs{};
    float cost = std::numeric_limits<float>::infinity();
    float cost_true = std::numeric_limits<float>::infinity();
    float cost_zero = std::numeric_limits<float>::infinity();
    float wavefront_rms = 0.0f;
    bool ok = false;
    // 若 final cost 比 cost(true) 低超过 1%，说明正向模型与反演变量集合
    // 不一致，反演结果是伪解。留 1% 余量是因为噪声数据上多拟合 3 个参数
    // 本来就会让代价略微下降。该诊断来自原始项目，保留并做成显式字段。
    bool pseudo_solution = false;
};

// 只拟合 Z4 / Z8 / Z9（数组下标 2 / 6 / 7）。
// 扩展项集合时修改 src/phase_diversity.cu 的 active_indices()，并同步
// kNumZernike、边界和正向模型的 Zernike 项集合。
class PhaseDiversitySolver {
public:
    PhaseDiversitySolver(const OpticalParams& params,
                         const StarField& stars,
                         const std::vector<float>& data_normalized,
                         const std::vector<float>& defocus,
                         const ZernikeCoeffs& truth,
                         SolveOptions options);
    ~PhaseDiversitySolver();

    PhaseDiversitySolver(const PhaseDiversitySolver&) = delete;
    PhaseDiversitySolver& operator=(const PhaseDiversitySolver&) = delete;

    // 代价函数：sum_frames sum_pixels (data - model)^2。
    // model 在 GPU 上生成并按帧归一化，不产生逐帧 D2H。
    float evaluate(const ZernikeCoeffs& coeffs);

    SolveResult solve();

private:
    OpticalParams params_;
    StarField stars_;
    std::vector<float> data_;
    std::vector<float> defocus_;
    ZernikeCoeffs truth_;
    SolveOptions options_;
    int n_frames_ = 0;
    int img_total_ = 0;

    class Impl;
    Impl* impl_ = nullptr;
};

}  // namespace telescope
