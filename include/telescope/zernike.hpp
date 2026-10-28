#pragma once

#include <cstdint>
#include <vector>

#include "telescope/types.hpp"

namespace telescope {

// 光瞳采样网格：坐标、掩膜和 8 项 Zernike 基函数。
// 基函数按列存储：basis[k * total + idx]。
struct PupilGrid {
    int size = 0;                 // fft_size
    std::vector<float> rho;       // 归一化半径，光瞳内 0..1
    std::vector<float> theta;     // 极角，弧度
    std::vector<uint8_t> mask;    // 1 = 光瞳内
    std::vector<float> basis;     // kNumZernike * size * size

    int total() const { return size * size; }
    int pupil_samples() const;
};

// 在 [-1,1]^2 的 fft_size 网格上采样光瞳。
PupilGrid make_pupil_grid(const OpticalParams& params);

// 课题使用的重力弯沉像差模型（与原始版本数值一致）。
// include_astigmatism_trefoil=true 时恢复 Z5/Z6/Z7（工程槽位）；
// 默认关闭，因为当前反演只拟合 Z4/Z8/Z9，打开后正向模型会包含
// 反演变量集合之外的项，cost(true) 不再接近 0，容易误报伪解。
// 编号约定见 types.hpp 与 docs/代码说明.md。
ZernikeCoeffs make_aberration_model(float theta_deg,
                                    bool include_astigmatism_trefoil = false);

// 光瞳内波前 RMS（单位：波长）：
// sqrt( mean_{mask} ( sum_k (rec[k]-truth[k]) * Z_k )^2 )
float wavefront_rms(const ZernikeCoeffs& recovered,
                    const ZernikeCoeffs& truth,
                    const PupilGrid& grid);

}  // namespace telescope
