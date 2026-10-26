#include "test_framework.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "telescope/cpu_reference.hpp"
#include "telescope/cuda_utils.cuh"
#include "telescope/forward_model.cuh"
#include "telescope/phase_diversity.hpp"

using namespace telescope;

namespace {

bool cuda_available() {
    int count = 0;
    const cudaError_t err = cudaGetDeviceCount(&count);
    return err == cudaSuccess && count > 0;
}

OpticalParams small_params() {
    OpticalParams params;
    params.img_size = 64;
    params.oversample = 2;
    params.pupil_radius = 0.45f;
    return params;
}

StarField make_stars() {
    StarField stars;
    stars.x = {18.3f, 30.7f, 45.2f, 50.1f};
    stars.y = {17.6f, 33.2f, 28.8f, 49.4f};
    stars.flux = {1200.0f, 2500.0f, 4000.0f, 800.0f};
    return stars;
}

ZernikeCoeffs make_truth() {
    ZernikeCoeffs truth{};
    truth[2] = 1.15f;  // Z4 defocus
    truth[6] = 0.18f;  // Z8 coma x
    truth[7] = 0.12f;  // Z9 coma y
    return truth;
}

std::vector<float> copy_to_host(const DeviceBuffer<float>& buffer) {
    std::vector<float> host(buffer.count());
    TEL_CUDA_CHECK(cudaMemcpy(host.data(), buffer.get(),
                              host.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return host;
}

}  // namespace

TEST(gpu_psf_matches_cpu_reference) {
    if (!cuda_available()) {
        std::printf("  (skipped: no CUDA device)\n");
        return;
    }

    const OpticalParams params = small_params();
    const ZernikeCoeffs coeffs = make_truth();

    CpuForwardModel cpu(params);
    std::vector<float> psf_cpu;
    cpu.compute_psf(coeffs, 0.7f, psf_cpu);

    ForwardModelGPU gpu(params, 1);
    DeviceBuffer<float> d_psf(params.img_total());
    gpu.compute_psf(coeffs, 0.7f, d_psf.get());
    const std::vector<float> psf_gpu = copy_to_host(d_psf);

    REQUIRE(psf_cpu.size() == psf_gpu.size());
    double max_diff = 0.0;
    double sum_cpu = 0.0;
    double sum_gpu = 0.0;
    for (std::size_t i = 0; i < psf_cpu.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(static_cast<double>(psf_cpu[i]) - psf_gpu[i]));
        sum_cpu += psf_cpu[i];
        sum_gpu += psf_gpu[i];
    }

    CHECK_NEAR(sum_cpu, 1.0, 1e-5);
    CHECK_NEAR(sum_gpu, 1.0, 1e-5);
    CHECK(max_diff < 2e-4);
}

TEST(gpu_image_matches_cpu_reference) {
    if (!cuda_available()) {
        std::printf("  (skipped: no CUDA device)\n");
        return;
    }

    const OpticalParams params = small_params();
    const ZernikeCoeffs coeffs = make_truth();
    const StarField stars = make_stars();

    CpuForwardModel cpu(params);
    std::vector<float> image_cpu;
    cpu.generate_image(coeffs, stars, -1.0f, image_cpu);

    ForwardModelGPU gpu(params, 1);
    gpu.set_stars(stars);
    DeviceBuffer<float> d_image(params.img_total());
    gpu.generate_image(coeffs, -1.0f, d_image.get());
    const std::vector<float> image_gpu = copy_to_host(d_image);

    REQUIRE(image_cpu.size() == image_gpu.size());
    double max_diff = 0.0;
    double max_value = 0.0;
    for (std::size_t i = 0; i < image_cpu.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(static_cast<double>(image_cpu[i]) - image_gpu[i]));
        max_value = std::max(max_value, static_cast<double>(image_cpu[i]));
    }

    CHECK(max_value > 0.0);
    CHECK(max_diff / max_value < 5e-4);
}

TEST(gpu_normalize_frames) {
    if (!cuda_available()) {
        std::printf("  (skipped: no CUDA device)\n");
        return;
    }

    const OpticalParams params = small_params();
    const StarField stars = make_stars();
    const ZernikeCoeffs coeffs = make_truth();
    const std::vector<float> defocus = {0.0f, 1.0f, -1.0f};
    const int n = static_cast<int>(defocus.size());

    ForwardModelGPU gpu(params, n);
    gpu.set_stars(stars);
    DeviceBuffer<float> d_frames(static_cast<std::size_t>(n) * params.img_total());
    gpu.generate_frames(coeffs, defocus.data(), n, d_frames.get());
    gpu.normalize_frames(d_frames.get(), n);

    const std::vector<float> frames = copy_to_host(d_frames);
    for (int f = 0; f < n; ++f) {
        double sum = 0.0;
        for (int i = 0; i < params.img_total(); ++i) {
            sum += frames[static_cast<std::size_t>(f) * params.img_total() + i];
        }
        CHECK_NEAR(sum, 1.0, 1e-5);
    }
}

TEST(solver_cost_is_minimal_at_truth) {
    if (!cuda_available()) {
        std::printf("  (skipped: no CUDA device)\n");
        return;
    }

    const OpticalParams params = small_params();
    const StarField stars = make_stars();
    const ZernikeCoeffs truth = make_truth();
    const std::vector<float> defocus = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f};
    const int n = static_cast<int>(defocus.size());

    // 用 GPU 正向模型生成“无噪声”观测，再交给反演器。
    ForwardModelGPU generator(params, n);
    generator.set_stars(stars);
    DeviceBuffer<float> d_frames(static_cast<std::size_t>(n) * params.img_total());
    generator.generate_frames(truth, defocus.data(), n, d_frames.get());
    generator.normalize_frames(d_frames.get(), n);
    const std::vector<float> data = copy_to_host(d_frames);

    SolveOptions options;
    options.n_starts = 1;
    options.n_refine = 1;
    options.maxeval = 10;

    PhaseDiversitySolver solver(params, stars, data, defocus, truth, options);
    const float cost_true = solver.evaluate(truth);
    const float cost_zero = solver.evaluate(ZernikeCoeffs{});

    CHECK(cost_true < 1e-9f);
    CHECK(cost_zero > cost_true);
}

TEST(solver_recovers_truth_end_to_end) {
    if (!cuda_available()) {
        std::printf("  (skipped: no CUDA device)\n");
        return;
    }

    const OpticalParams params = small_params();
    const StarField stars = make_stars();
    const ZernikeCoeffs truth = make_truth();
    const std::vector<float> defocus = {0.0f, 1.5f, -1.5f};
    const int n = static_cast<int>(defocus.size());

    ForwardModelGPU generator(params, n);
    generator.set_stars(stars);
    DeviceBuffer<float> d_frames(static_cast<std::size_t>(n) * params.img_total());
    generator.generate_frames(truth, defocus.data(), n, d_frames.get());
    generator.normalize_frames(d_frames.get(), n);
    const std::vector<float> data = copy_to_host(d_frames);

    SolveOptions options;
    options.n_starts = 20;
    options.n_refine = 2;
    options.seed = 7;
    options.maxeval = 400;

    PhaseDiversitySolver solver(params, stars, data, defocus, truth, options);
    const SolveResult result = solver.solve();

    CHECK(result.ok);
    CHECK(!result.pseudo_solution);
    CHECK_NEAR(result.coeffs[2], truth[2], 0.05);
    CHECK_NEAR(result.coeffs[6], truth[6], 0.05);
    CHECK_NEAR(result.coeffs[7], truth[7], 0.05);
    CHECK(result.wavefront_rms < 0.05f);
}

TEST(preprocess_subtracts_median_background) {
    FrameSet frames;
    frames.img_size = 4;
    frames.defocus = {0.0f, 1.0f};
    frames.images = {
        std::vector<float>(16, 10.0f),
        std::vector<float>(16, 20.0f),
    };
    frames.images[0][5] = 110.0f;  // 信号 + 背景
    frames.images[1][6] = 220.0f;

    std::vector<float> backgrounds;
    const std::vector<float> normalized =
        preprocess_frames(frames, BackgroundMode::kMedian, &backgrounds);
    REQUIRE(normalized.size() == 32);
    REQUIRE(backgrounds.size() == 2);
    CHECK_NEAR(backgrounds[0], 10.0, 1e-6);
    CHECK_NEAR(backgrounds[1], 20.0, 1e-6);

    double sum0 = 0.0;
    double sum1 = 0.0;
    for (int i = 0; i < 16; ++i) {
        sum0 += normalized[i];
        sum1 += normalized[16 + i];
    }
    CHECK_NEAR(sum0, 1.0, 1e-6);
    CHECK_NEAR(sum1, 1.0, 1e-6);
    CHECK(normalized[5] > 0.99f);
    CHECK(normalized[16 + 6] > 0.99f);
}
