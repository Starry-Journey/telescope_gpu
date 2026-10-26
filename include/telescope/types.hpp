#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace telescope {

// Noll 编号 Z2..Z9，共 8 项。数组下标 i 对应 Noll 编号 i+2。
constexpr int kNumZernike = 8;
constexpr int kNollFirst = 2;

// 光学系统参数。fft_size = img_size * oversample。
struct OpticalParams {
    int img_size = 256;
    int oversample = 2;
    float pupil_radius = 0.45f;

    int fft_size() const { return img_size * oversample; }
    int img_total() const { return img_size * img_size; }
    int fft_total() const { return fft_size() * fft_size(); }
};

using ZernikeCoeffs = std::array<float, kNumZernike>;

// 星点表。坐标单位为像素，原点在图像左上角。
struct StarField {
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> flux;

    int size() const { return static_cast<int>(x.size()); }
    bool valid() const {
        return x.size() == y.size() && x.size() == flux.size();
    }
};

// 相位差法默认的 5 帧离焦量（单位：波长）。
inline constexpr std::array<float, 5> kDefaultDefocus = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f};

// 反演变量的物理边界。用于粗搜截断与 NLopt 边界，二者必须一致。
struct ZernikeBounds {
    ZernikeCoeffs lower{};
    ZernikeCoeffs upper{};

    static ZernikeBounds default_bounds() {
        ZernikeBounds b;
        // 只搜索 Z4/Z8/Z9，但三者都必须允许负值：重力变形模型在
        // theta > 225° 时 Z9 会变为负（见 make_aberration_model）。
        b.lower = {0.0f, 0.0f, -3.0f, 0.0f, 0.0f, 0.0f, -1.5f, -1.5f};
        b.upper = {0.0f, 0.0f, 3.0f, 0.0f, 0.0f, 0.0f, 1.5f, 1.5f};
        return b;
    }

    float clamp(int i, float v) const {
        if (v < lower[i]) return lower[i];
        if (v > upper[i]) return upper[i];
        return v;
    }
};

}  // namespace telescope
