#pragma once

#include <vector>

#include "telescope/types.hpp"
#include "telescope/zernike.hpp"

namespace telescope {

// 与 GPU 正向模型逐步骤对应的 CPU 参考实现。
// 用途：
//   1. 单元测试里逐像素对比 GPU 结果；
//   2. 面试时能解释每一步数学；
//   3. 排查 GPU 端问题时作为“可信基准”。
class CpuForwardModel {
public:
    explicit CpuForwardModel(const OpticalParams& params);

    const OpticalParams& params() const { return params_; }
    const PupilGrid& grid() const { return grid_; }

    // 返回归一化后的 img_size × img_size PSF（总和不严格为 1，
    // 因为最后一步降采样后再归一化；与 GPU 版本一致）。
    void compute_psf(const ZernikeCoeffs& coeffs,
                     float defocus,
                     std::vector<float>& psf) const;

    // 无噪声星场图。PSF 总归一化为 1，因此图像总和约等于 flux 总和
    // （边缘截断会损失少量能量）。
    void generate_image(const ZernikeCoeffs& coeffs,
                        const StarField& stars,
                        float defocus,
                        std::vector<float>& image) const;

private:
    OpticalParams params_;
    PupilGrid grid_;
};

}  // namespace telescope
