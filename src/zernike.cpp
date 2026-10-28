#include "telescope/zernike.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace telescope {
namespace {

constexpr double kPi = 3.14159265358979323846;

}  // namespace

int PupilGrid::pupil_samples() const {
    int n = 0;
    for (uint8_t m : mask) n += (m != 0);
    return n;
}

PupilGrid make_pupil_grid(const OpticalParams& params) {
    if (params.img_size <= 0 || params.oversample <= 0) {
        throw std::invalid_argument("OpticalParams: img_size/oversample must be positive");
    }
    if (!(params.pupil_radius > 0.0f && params.pupil_radius <= 1.0f)) {
        throw std::invalid_argument("OpticalParams: pupil_radius must be in (0,1]");
    }

    const int n = params.fft_size();
    const int total = n * n;

    PupilGrid grid;
    grid.size = n;
    grid.rho.resize(total);
    grid.theta.resize(total);
    grid.mask.resize(total);
    grid.basis.assign(static_cast<size_t>(kNumZernike) * total, 0.0f);

    const float scale = 2.0f / static_cast<float>(n - 1);
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            const int idx = y * n + x;
            const float nx = -1.0f + static_cast<float>(x) * scale;
            const float ny = -1.0f + static_cast<float>(y) * scale;
            const float r = std::sqrt(nx * nx + ny * ny);
            grid.rho[idx] = r / params.pupil_radius;
            grid.theta[idx] = std::atan2(ny, nx);
            grid.mask[idx] = (r <= params.pupil_radius) ? 1 : 0;
        }
    }

    // 逐点计算 8 个基函数槽位。槽位 -> 真实 Noll 编号见 types.hpp 的
    // kZernikeNoll = {2,3,4,5,6,9,8,7}：前 5 项与 Noll 一致，后 3 项
    // 是 [trefoil(Noll 9), coma-x(Noll 8), coma-y(Noll 7)]。这里保留
    // 历史顺序，避免改变已有结果。
    for (int idx = 0; idx < total; ++idx) {
        if (!grid.mask[idx]) continue;
        const double r = grid.rho[idx];
        const double t = grid.theta[idx];
        const double r2 = r * r;
        const double r3 = r2 * r;

        const double z[kNumZernike] = {
            2.0 * r * std::cos(t),                                  // 槽位 0 = Noll Z2
            2.0 * r * std::sin(t),                                  // 槽位 1 = Noll Z3
            std::sqrt(3.0) * (2.0 * r2 - 1.0),                      // 槽位 2 = Noll Z4
            std::sqrt(6.0) * r2 * std::sin(2.0 * t),                // 槽位 3 = Noll Z5
            std::sqrt(6.0) * r2 * std::cos(2.0 * t),                // 槽位 4 = Noll Z6
            std::sqrt(8.0) * r3 * std::sin(3.0 * t),                // 槽位 5 = Noll Z9
            std::sqrt(8.0) * (3.0 * r3 - 2.0 * r) * std::cos(t),    // 槽位 6 = Noll Z8
            std::sqrt(8.0) * (3.0 * r3 - 2.0 * r) * std::sin(t),    // 槽位 7 = Noll Z7
        };
        for (int k = 0; k < kNumZernike; ++k) {
            grid.basis[static_cast<size_t>(k) * total + idx] = static_cast<float>(z[k]);
        }
    }
    return grid;
}

ZernikeCoeffs make_aberration_model(float theta_deg,
                                    bool include_astigmatism_trefoil) {
    // 经验弯沉模型：以 theta=90°（delta=0）为参考角，delta=|theta-90°|。
    // 注意这只是课题用的参数化模型，不是有限元/ Zemax 的物理仿真结果。
    // c[6]/c[7] 是工程编号的 Z8/Z9（真实 Noll Z8/Z7，即 x/y 彗差）。
    const double theta = theta_deg * kPi / 180.0;
    const double delta = std::abs(theta - kPi / 2.0);

    ZernikeCoeffs c{};
    c[0] = 0.0f;
    c[1] = 0.0f;
    c[2] = static_cast<float>(1.8 * std::sin(delta) + 0.2);  // 槽位 2 = Z4 离焦
    if (include_astigmatism_trefoil) {
        c[3] = static_cast<float>(0.20 * std::sin(2.0 * delta) + 0.05 * std::cos(theta));
        c[4] = static_cast<float>(0.18 * std::sin(2.0 * delta + kPi / 4.0) +
                                  0.05 * std::sin(theta));
        c[5] = static_cast<float>(0.08 * std::sin(3.0 * delta) + 0.02);
    }
    c[6] = static_cast<float>(0.20 * std::sin(delta) + 0.03);              // Z8 彗差 x
    c[7] = static_cast<float>(0.16 * std::sin(delta + kPi / 4.0) + 0.02);  // Z9 彗差 y
    return c;
}

float wavefront_rms(const ZernikeCoeffs& recovered,
                    const ZernikeCoeffs& truth,
                    const PupilGrid& grid) {
    if (grid.size <= 0 || grid.basis.empty()) return 0.0f;

    double sum_sq = 0.0;
    int count = 0;
    const int total = grid.total();
    for (int idx = 0; idx < total; ++idx) {
        if (!grid.mask[idx]) continue;
        double w = 0.0;
        for (int k = 0; k < kNumZernike; ++k) {
            const double dc = static_cast<double>(recovered[k]) - truth[k];
            w += dc * grid.basis[static_cast<size_t>(k) * total + idx];
        }
        sum_sq += w * w;
        ++count;
    }
    if (count == 0) return 0.0f;
    return static_cast<float>(std::sqrt(sum_sq / count));
}

}  // namespace telescope
