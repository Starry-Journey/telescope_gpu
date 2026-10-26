#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "telescope/config.hpp"
#include "telescope/cuda_utils.cuh"
#include "telescope/fits_io.hpp"
#include "telescope/forward_model.cuh"
#include "telescope/zernike.hpp"

namespace fs = std::filesystem;
using namespace telescope;

int main(int argc, char** argv) {
    try {
        const CliArgs cli = parse_cli(argc, argv);
        if (cli.help) {
            print_usage(argv[0]);
            return 0;
        }

        const Config cfg = resolve_config(cli);
        const int img_size = cfg.optics.img_size;
        const int n_frames = static_cast<int>(cfg.defocus.size());
        const int img_total = cfg.optics.img_total();
        const std::string out_dir = cfg.resolved_data_dir();

        const ZernikeCoeffs truth = make_aberration_model(cfg.theta_deg);

        std::printf("=== forward_generate v2 ===\n");
        std::printf("theta = %.1f deg, image = %d x %d, frames = %d\n",
                    cfg.theta_deg, img_size, img_size, n_frames);
        std::printf("Z4=%.4f Z8=%.4f Z9=%.4f (Z5/Z6/Z7 = 0, see docs)\n",
                    truth[2], truth[6], truth[7]);

        // 星点表：使用 mt19937 而非 rand()，跨平台、可复现。
        StarField stars;
        {
            std::mt19937 rng(static_cast<std::mt19937::result_type>(cfg.seed));
            const float margin = std::min(50.0f, static_cast<float>(img_size) * 0.2f);
            std::uniform_real_distribution<float> pos(margin, static_cast<float>(img_size) - margin);
            std::uniform_real_distribution<float> flux(cfg.flux_min, cfg.flux_max);
            for (int i = 0; i < cfg.star_count; ++i) {
                // overlay kernel 用 roundf 把星点放到整数像素；这里先取整，
                // 保证 FITS 头部记录的坐标与生成时实际使用的坐标完全一致。
                stars.x.push_back(std::round(pos(rng)));
                stars.y.push_back(std::round(pos(rng)));
                stars.flux.push_back(flux(rng));
            }
        }

        ForwardModelGPU model(cfg.optics, n_frames);
        model.set_stars(stars);

        DeviceBuffer<float> d_frames(static_cast<std::size_t>(n_frames) * img_total);
        model.generate_frames(truth, cfg.defocus.data(), n_frames, d_frames.get());

        std::vector<float> h_frames(static_cast<std::size_t>(n_frames) * img_total);
        TEL_CUDA_CHECK(cudaMemcpy(h_frames.data(), d_frames.get(),
                                  h_frames.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost));

        fs::create_directories(out_dir);

        // 噪声模型：泊松(信号 + 背景) + 高斯读出噪声。
        std::mt19937 noise_rng(static_cast<std::mt19937::result_type>(cfg.seed) + 1000u);
        std::poisson_distribution<int> poisson;
        std::normal_distribution<float> gaussian(0.0f, cfg.read_noise);

        for (int f = 0; f < n_frames; ++f) {
            std::vector<float> image(img_total);
            for (int i = 0; i < img_total; ++i) {
                const float expected = h_frames[static_cast<std::size_t>(f) * img_total + i] +
                                       cfg.background;
                const int counts = expected > 0.0f
                                       ? poisson(noise_rng, std::poisson_distribution<int>::param_type(expected))
                                       : 0;
                const float value = static_cast<float>(counts) + gaussian(noise_rng);
                image[i] = std::max(value, 0.0f);
            }

            FitsMetadata meta;
            meta.img_size = img_size;
            meta.theta_deg = cfg.theta_deg;
            meta.defocus = cfg.defocus[f];
            meta.zernike = truth;
            meta.seed = cfg.seed;
            meta.background = cfg.background;
            meta.read_noise = cfg.read_noise;
            meta.star_x = stars.x;
            meta.star_y = stars.y;
            meta.star_flux = stars.flux;

            char name[512];
            std::snprintf(name, sizeof(name),
                          "%s/star_field_theta%03d_defocus_%+.1fwaves.fits",
                          out_dir.c_str(), static_cast<int>(std::lround(cfg.theta_deg)),
                          static_cast<double>(cfg.defocus[f]));

            if (!write_fits_image(name, image, meta)) {
                std::fprintf(stderr, "failed to write %s\n", name);
                return 1;
            }
            std::printf("frame %d: defocus=%+.1f lambda -> %s\n",
                        f, cfg.defocus[f], name);
        }

        std::printf("Done.\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "forward_generate: %s\n", e.what());
        return 1;
    }
}
