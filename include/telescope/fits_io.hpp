#pragma once

#include <string>
#include <vector>

#include "telescope/types.hpp"

namespace telescope {

// 本项目只使用 FITS 的最小子集：单 HDU、2D、BITPIX=-32（float32，大端）。
// 目的不是实现完整 FITS 标准，而是让数据交换不依赖外部库，便于教学。
struct FitsMetadata {
    int img_size = 0;
    float theta_deg = 0.0f;
    float defocus = 0.0f;
    ZernikeCoeffs zernike{};
    int seed = 0;
    float background = 0.0f;
    float read_noise = 0.0f;

    // 星点表随图像一起写入头部，反演端不再靠 srand(42) 猜测星点位置。
    // 这是 v2 相对原始脚本的一个重要修正：数据文件自描述、可复现。
    std::vector<float> star_x;
    std::vector<float> star_y;
    std::vector<float> star_flux;
};

bool write_fits_image(const std::string& path,
                      const std::vector<float>& image,
                      const FitsMetadata& meta);

bool read_fits_image(const std::string& path,
                     std::vector<float>& image,
                     FitsMetadata& meta);

}  // namespace telescope
