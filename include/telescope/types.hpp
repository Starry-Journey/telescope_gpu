#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace telescope {

// 波前展开固定使用 8 个基函数槽位，数组下标 i = 槽位编号 - 2。
//
// 编号约定（重要）：
//   槽位 0..4 与 Noll Z2..Z6 完全一致；
//   槽位 5..7 实际是 [Noll Z9(trefoil), Noll Z8(coma-x), Noll Z7(coma-y)]，
//   并不是 Noll Z7/Z8/Z9。报告和旧代码沿用了“槽位号 = Z 编号”的工程叫法，
//   所以本项目的 Z8/Z9 分别指 x/y 彗差；Z7 槽位是 trefoil 而不是彗差。
//   详见 docs/代码说明.md 的编号对照表。
//   若要严格按 Noll 顺序重排，必须同时修改 active_zernike_indices()、
//   重力变形模型、FITS 头部键、测试和报告，不能只改公式。
constexpr int kNumZernike = 8;
constexpr int kNollFirst = 2;
constexpr int kMaxFrames = 16;

// 槽位标签（只用于日志/报告，不参与数值计算）。
inline constexpr std::array<const char*, kNumZernike> kZernikeLabels = {
    "Z2 tilt-x", "Z3 tilt-y",  "Z4 defocus", "Z5 astig-45",
    "Z6 astig-0", "Z7 trefoil", "Z8 coma-x",  "Z9 coma-y"};

// 槽位 -> 真实 Noll 编号，便于与 Zemax/文献对照。
inline constexpr std::array<int, kNumZernike> kZernikeNoll = {2, 3, 4, 5, 6, 9, 8, 7};

inline const char* zernike_label(int index) {
    return (index >= 0 && index < kNumZernike) ? kZernikeLabels[index] : "?";
}

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

    // 下界必须 <= 上界；NaN 参与比较时为 false，因此也能被这里拦下。
    bool valid() const {
        for (int i = 0; i < kNumZernike; ++i) {
            if (!(lower[i] <= upper[i])) return false;
        }
        return true;
    }

    float clamp(int i, float v) const {
        if (v < lower[i]) return lower[i];
        if (v > upper[i]) return upper[i];
        return v;
    }
};

}  // namespace telescope
