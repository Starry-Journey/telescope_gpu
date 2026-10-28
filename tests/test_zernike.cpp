#include "test_framework.hpp"

#include <cmath>

#include "telescope/zernike.hpp"

using namespace telescope;

namespace {
constexpr double kPi = 3.14159265358979323846;
}

TEST(pupil_grid_mask_area) {
    OpticalParams params;
    params.img_size = 128;
    params.oversample = 2;
    params.pupil_radius = 0.45f;

    const PupilGrid grid = make_pupil_grid(params);
    CHECK(grid.size == 256);
    CHECK(grid.basis.size() == static_cast<std::size_t>(kNumZernike) * 256 * 256);

    // 像素面积 = (2/(n-1))^2；光瞳面积应为 pi * r^2。
    const double pixel_area = 4.0 / (255.0 * 255.0);
    const double expected = kPi * 0.45 * 0.45;
    CHECK_NEAR(grid.pupil_samples() * pixel_area, expected, 0.02 * expected);
}

TEST(zernike_basis_orthogonality) {
    OpticalParams params;
    params.img_size = 128;
    params.oversample = 2;
    params.pupil_radius = 0.45f;
    const PupilGrid grid = make_pupil_grid(params);

    const int total = grid.total();
    const double pixel_area = 4.0 / (255.0 * 255.0);

    // 基函数用归一化半径 rho = r / pupil_radius，因此在单位圆上
    // integral(Zi * Zj) = pi * delta_ij。
    // 但这里的像素面积是 [-1,1]^2 坐标下的面积，光瞳半径是 0.45，
    // 所以实际积分 = pi * pupil_radius^2。
    const double expected_diag = kPi * 0.45 * 0.45;
    for (int i = 0; i < kNumZernike; ++i) {
        for (int j = 0; j < kNumZernike; ++j) {
            double integral = 0.0;
            for (int idx = 0; idx < total; ++idx) {
                if (!grid.mask[idx]) continue;
                integral += static_cast<double>(grid.basis[i * total + idx]) *
                            grid.basis[j * total + idx] * pixel_area;
            }
            if (i == j) {
                CHECK_NEAR(integral, expected_diag, 0.02 * expected_diag);
            } else {
                CHECK_NEAR(integral, 0.0, 0.02);
            }
        }
    }
}

TEST(aberration_model_reference_angle) {
    // theta = 90 deg 是参考角：delta = 0。
    const ZernikeCoeffs c = make_aberration_model(90.0f);
    CHECK_NEAR(c[0], 0.0, 1e-6);
    CHECK_NEAR(c[1], 0.0, 1e-6);
    CHECK_NEAR(c[2], 0.2, 1e-5);
    CHECK_NEAR(c[3], 0.0, 1e-6);
    CHECK_NEAR(c[4], 0.0, 1e-6);
    CHECK_NEAR(c[5], 0.0, 1e-6);
    CHECK_NEAR(c[6], 0.03, 1e-5);
    CHECK_NEAR(c[7], 0.16 * std::sin(kPi / 4.0) + 0.02, 1e-5);

    // 45 deg 处像散/三叶项才明显；90 deg 时 Z5 的 sin(delta) 项恰好为 0。
    const ZernikeCoeffs with_astig = make_aberration_model(45.0f, true);
    CHECK(std::fabs(with_astig[3]) > 1e-6);
    CHECK(std::fabs(with_astig[4]) > 1e-6);
    CHECK(std::fabs(with_astig[5]) > 1e-6);
}

TEST(wavefront_rms) {
    OpticalParams params;
    params.img_size = 64;
    params.oversample = 2;
    params.pupil_radius = 0.45f;
    const PupilGrid grid = make_pupil_grid(params);

    ZernikeCoeffs truth{};
    truth[2] = 1.2f;
    truth[6] = 0.2f;

    CHECK_NEAR(wavefront_rms(truth, truth, grid), 0.0, 1e-8);

    ZernikeCoeffs recovered = truth;
    recovered[2] += 0.1f;
    const float rms = wavefront_rms(recovered, truth, grid);
    CHECK(rms > 0.0f);
    CHECK(rms < 0.2f);
}

TEST(make_pupil_grid_rejects_bad_params) {
    OpticalParams bad;
    bad.img_size = 0;
    CHECK_THROWS(make_pupil_grid(bad));

    OpticalParams bad_radius;
    bad_radius.pupil_radius = 1.5f;
    CHECK_THROWS(make_pupil_grid(bad_radius));
}

TEST(zernike_bounds_validation) {
    ZernikeBounds bounds = ZernikeBounds::default_bounds();
    CHECK(bounds.valid());
    // 只允许搜索 Z4/Z8/Z9，三者都必须能取负值。
    CHECK(bounds.lower[2] < 0.0f);
    CHECK(bounds.lower[6] < 0.0f);
    CHECK(bounds.lower[7] < 0.0f);

    ZernikeBounds broken = bounds;
    broken.lower[2] = 2.0f;
    broken.upper[2] = -2.0f;
    CHECK(!broken.valid());

    // NaN 参与比较恒为 false，valid() 必须能识别。
    broken = bounds;
    broken.upper[6] = std::nanf("");
    CHECK(!broken.valid());
}
