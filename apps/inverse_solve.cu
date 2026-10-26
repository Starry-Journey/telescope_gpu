#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "telescope/config.hpp"
#include "telescope/phase_diversity.hpp"

using namespace telescope;

int main(int argc, char** argv) {
    try {
        const CliArgs cli = parse_cli(argc, argv);
        if (cli.help) {
            print_usage(argv[0]);
            return 0;
        }

        const Config cfg = resolve_config(cli);
        const std::string data_dir = cfg.resolved_data_dir();

        FrameSet frames;
        std::string error;
        if (!load_frame_set(data_dir, cfg.theta_deg, cfg.defocus, frames, error)) {
            std::fprintf(stderr, "inverse_solve: %s\n", error.c_str());
            return 1;
        }

        std::printf("=== inverse_solve v2 ===\n");
        std::printf("data dir = %s\n", data_dir.c_str());
        std::printf("frames = %zu, image = %d x %d, stars = %d\n",
                    frames.images.size(), frames.img_size, frames.img_size,
                    frames.stars.size());
        std::printf("truth: Z4=%.4f Z8=%.4f Z9=%.4f\n",
                    frames.truth[2], frames.truth[6], frames.truth[7]);

        const BackgroundMode mode = cfg.background_mode == "median"
                                        ? BackgroundMode::kMedian
                                        : BackgroundMode::kHeader;
        std::vector<float> backgrounds;
        const std::vector<float> data = preprocess_frames(frames, mode, &backgrounds);
        std::printf("background estimates:");
        for (float bg : backgrounds) std::printf(" %.2f", bg);
        std::printf("\n");

        SolveOptions options;
        options.n_starts = cfg.solve_starts;
        options.n_refine = cfg.solve_refine;
        options.seed = cfg.solve_seed;
        options.maxeval = cfg.solve_maxeval;

        PhaseDiversitySolver solver(cfg.optics, frames.stars, data, frames.defocus,
                                    frames.truth, options);
        const SolveResult result = solver.solve();

        std::printf("\n=== 反演结果 ===\n");
        std::printf("          true      recovered      error\n");
        for (int k : {2, 6, 7}) {
            std::printf("Z%-2d   %+9.4f   %+9.4f   %9.4f\n",
                        k + 2, frames.truth[k], result.coeffs[k],
                        std::fabs(result.coeffs[k] - frames.truth[k]));
        }
        std::printf("wavefront RMS = %.4f lambda\n", result.wavefront_rms);
        std::printf("final cost = %.6e (cost(true) = %.6e)\n",
                    result.cost, result.cost_true);

        if (result.pseudo_solution) {
            std::printf("\nWARNING: final cost 比 cost(true) 低超过 1%%。\n"
                        "正向模型与反演变量集合不一致，结果是伪解。\n"
                        "检查 forward_generate 与 inverse_solve 的 Zernike 项集合。\n");
        }

        // 兼容旧版可视化脚本：6 个数字，一行。
        if (std::FILE* fp = std::fopen(cfg.result_file.c_str(), "w")) {
            std::fprintf(fp, "# theta = %.4f\n", cfg.theta_deg);
            std::fprintf(fp, "# true_z4 true_z8 true_z9 recovered_z4 recovered_z8 recovered_z9\n");
            std::fprintf(fp, "%.6f %.6f %.6f %.6f %.6f %.6f\n",
                         frames.truth[2], frames.truth[6], frames.truth[7],
                         result.coeffs[2], result.coeffs[6], result.coeffs[7]);
            std::fclose(fp);
            std::printf("saved: %s\n", cfg.result_file.c_str());
        } else {
            std::fprintf(stderr, "cannot write %s\n", cfg.result_file.c_str());
        }

        if (std::FILE* fp = std::fopen(cfg.report_file.c_str(), "w")) {
            std::fprintf(fp, "# telescope inverse report v2\n");
            std::fprintf(fp, "theta_deg %.4f\n", cfg.theta_deg);
            std::fprintf(fp, "frames %zu\n", frames.images.size());
            std::fprintf(fp, "cost_true %.6e\n", result.cost_true);
            std::fprintf(fp, "cost_zero %.6e\n", result.cost_zero);
            std::fprintf(fp, "cost_final %.6e\n", result.cost);
            std::fprintf(fp, "pseudo_solution %d\n", result.pseudo_solution ? 1 : 0);
            std::fprintf(fp, "wavefront_rms_lambda %.6f\n", result.wavefront_rms);
            std::fprintf(fp, "# noll truth recovered error\n");
            for (int k : {2, 6, 7}) {
                std::fprintf(fp, "%d %.6f %.6f %.6f\n", k + 2,
                             frames.truth[k], result.coeffs[k],
                             std::fabs(result.coeffs[k] - frames.truth[k]));
            }
            std::fclose(fp);
            std::printf("saved: %s\n", cfg.report_file.c_str());
        }

        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "inverse_solve: %s\n", e.what());
        return 1;
    }
}
