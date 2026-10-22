#include "forward_model.cuh"
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <functional>
#include <nlopt.hpp>

// ============================================================
// GPU 代价函数 kernel
// ============================================================
__global__ void cost_kernel(
    const float* d_data,
    const float* d_model,
    int n_frames, int img_size,
    float* d_cost_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int frame = blockIdx.y;
    int total = img_size * img_size;
    if (idx >= total || frame >= n_frames) return;

    float diff = d_data[frame * total + idx] - d_model[frame * total + idx];
    atomicAdd(d_cost_out, diff * diff);
}

// ============================================================
// NLopt 用的全局上下文（因为 nlopt 不接受捕获 lambda）
// ============================================================
static std::function<float(float, float, float)>* g_cost_fn = nullptr;

static double nlopt_obj(const std::vector<double>& x,
                        std::vector<double>& grad,
                        void* /*data*/)
{
    (void)grad;
    if (!g_cost_fn) return 0.0;
    return (double)(*g_cost_fn)((float)x[0], (float)x[1], (float)x[2]);
}

// ============================================================
// FITS 读取
// ============================================================
static bool read_fits(const std::string& path,
                      std::vector<float>& image,
                      int& img_size, float& theta, float& defocus,
                      float* true_z8)
{
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) return false;
    char card[81];
    img_size = 0; theta = 0; defocus = 0;
    memset(true_z8, 0, 8 * sizeof(float));

    while (true) {
        if (fread(card, 1, 80, fp) != 80) break;
        card[80] = '\0';
        if (strncmp(card, "NAXIS1", 6) == 0) img_size = atoi(card + 10);
        if (strncmp(card, "THETA", 5) == 0)  theta = atof(card + 10);
        if (strncmp(card, "DEFOCUS", 7) == 0) defocus = atof(card + 10);
        for (int j = 0; j < 8; ++j) {
            char key[8];
            snprintf(key, sizeof(key), "ZERN%02d", j + 2);
            if (strncmp(card, key, 6) == 0) true_z8[j] = atof(card + 10);
        }
        if (strncmp(card, "END", 3) == 0) {
            long pos = ftell(fp);
            int pad = 2880 - (int)(pos % 2880);
            if (pad < 2880) fseek(fp, pad, SEEK_CUR);
            break;
        }
    }
    if (img_size <= 0) { fclose(fp); return false; }

    int img_total = img_size * img_size;
    std::vector<float> be_data(img_total);
    fread(be_data.data(), 4, img_total, fp);
    fclose(fp);

    image.resize(img_total);
    for (int i = 0; i < img_total; ++i) {
        uint32_t v;
        memcpy(&v, &be_data[i], 4);
        v = ((v & 0xFF000000) >> 24) |
            ((v & 0x00FF0000) >> 8)  |
            ((v & 0x0000FF00) << 8)  |
            ((v & 0x000000FF) << 24);
        memcpy(&image[i], &v, 4);
    }
    return true;
}

// ------------------------------------------------------------
int main(int argc, char** argv)
{
    printf("=== inverse_solve (v4: 只优化 Z4/Z8/Z9，多起点 + BOBYQA) ===\n");
    float theta = 45.0f;
    if (argc > 1) theta = atof(argv[1]);

    OpticalParams params;
    params.img_size = 256;
    params.oversample = 2;
    params.fft_size = 512;
    params.pupil_radius = 0.45f;

    ForwardModelGPU model(params);

    const float PD_DEFOCUS[5] = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f};
    const int N_FRAMES = 5;
    const int STAR_NUM = 15;
    const int IMG_TOTAL = params.img_size * params.img_size;

    // ---- 读 FITS ----
    char dir_name[64];
    snprintf(dir_name, sizeof(dir_name), "theta_%03ddeg", (int)theta);

    std::vector<std::vector<float>> images(N_FRAMES);
    float true_z8[8] = {0};
    int img_size = 0;

    for (int f = 0; f < N_FRAMES; ++f) {
        char fname[256];
        snprintf(fname, sizeof(fname),
                 "%s/star_field_theta%03d_defocus_%+.1fwaves.fits",
                 dir_name, (int)theta, PD_DEFOCUS[f]);
        float t, d;
        if (!read_fits(fname, images[f], img_size, t, d, true_z8)) {
            fprintf(stderr, "无法读取 %s\n", fname);
            return 1;
        }
    }
    printf("读取 %d 帧，图像尺寸 %d\n", N_FRAMES, img_size);
    printf("真值: Z4=%.4f Z8=%.4f Z9=%.4f\n",
           true_z8[2], true_z8[6], true_z8[7]);

    // ---- 数据预处理：扣背景 + 归一化 ----
    std::vector<float> data_norm(N_FRAMES * IMG_TOTAL);
    for (int f = 0; f < N_FRAMES; ++f) {
        std::vector<float> tmp = images[f];
        std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
        float bg = tmp[tmp.size() / 2];

        float sum = 0.0f;
        for (int i = 0; i < IMG_TOTAL; ++i) {
            float v = images[f][i] - bg;
            if (v < 0) v = 0;
            data_norm[f * IMG_TOTAL + i] = v;
            sum += v;
        }
        if (sum > 1e-10f) {
            for (int i = 0; i < IMG_TOTAL; ++i)
                data_norm[f * IMG_TOTAL + i] /= sum;
        }
    }

    // ---- 分配 device 内存 ----
    float *d_data, *d_model, *d_cost, *d_coeffs, *d_defocus;
    CUDA_CHECK(cudaMalloc(&d_data,     N_FRAMES * IMG_TOTAL * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_model,    N_FRAMES * IMG_TOTAL * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cost,     sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_coeffs,   8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_defocus,  N_FRAMES * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_data, data_norm.data(),
                          N_FRAMES * IMG_TOTAL * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_defocus, PD_DEFOCUS,
                          N_FRAMES * sizeof(float), cudaMemcpyHostToDevice));

    // ---- 星点位置 ----
    srand(42);
    std::vector<float> h_sx(STAR_NUM), h_sy(STAR_NUM), h_sf(STAR_NUM);
    for (int i = 0; i < STAR_NUM; ++i) {
        h_sx[i] = 50.0f + (rand() / (float)RAND_MAX) * (params.img_size - 100);
        h_sy[i] = 50.0f + (rand() / (float)RAND_MAX) * (params.img_size - 100);
        h_sf[i] = 3000.0f + (rand() / (float)RAND_MAX) * 12000.0f;
    }
    float *d_sx, *d_sy, *d_sf;
    CUDA_CHECK(cudaMalloc(&d_sx, STAR_NUM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sy, STAR_NUM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sf, STAR_NUM * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_sx, h_sx.data(), STAR_NUM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sy, h_sy.data(), STAR_NUM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sf, h_sf.data(), STAR_NUM * sizeof(float), cudaMemcpyHostToDevice));

    // ---- 代价函数 ----
    auto cost_3 = [&](const float z4, const float z8, const float z9) -> float {
        float h_coeffs[8] = {0.0f, 0.0f, z4, 0.0f, 0.0f, 0.0f, z8, z9};
        CUDA_CHECK(cudaMemcpy(d_coeffs, h_coeffs,
                              8 * sizeof(float), cudaMemcpyHostToDevice));

        model.generate_frames_device(d_coeffs, d_sx, d_sy, d_sf,
                                     STAR_NUM, d_defocus, N_FRAMES,
                                     d_model);

        std::vector<float> h_model(N_FRAMES * IMG_TOTAL);
        CUDA_CHECK(cudaMemcpy(h_model.data(), d_model,
                              N_FRAMES * IMG_TOTAL * sizeof(float),
                              cudaMemcpyDeviceToHost));
        for (int f = 0; f < N_FRAMES; ++f) {
            float sum = 0.0f;
            for (int i = 0; i < IMG_TOTAL; ++i)
                sum += h_model[f * IMG_TOTAL + i];
            if (sum > 1e-10f) {
                for (int i = 0; i < IMG_TOTAL; ++i)
                    h_model[f * IMG_TOTAL + i] /= sum;
            }
        }
        CUDA_CHECK(cudaMemcpy(d_model, h_model.data(),
                              N_FRAMES * IMG_TOTAL * sizeof(float),
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemset(d_cost, 0, sizeof(float)));
        int threads = 256;
        int blocks = (IMG_TOTAL + threads - 1) / threads;
        dim3 grid(blocks, N_FRAMES);
        cost_kernel<<<grid, threads>>>(d_data, d_model, N_FRAMES, img_size, d_cost);
        CUDA_CHECK(cudaDeviceSynchronize());

        float h_cost = 0.0f;
        CUDA_CHECK(cudaMemcpy(&h_cost, d_cost, sizeof(float), cudaMemcpyDeviceToHost));
        return h_cost;
    };

    // ============================================================
    // 诊断
    // ============================================================
    printf("\n=== 代价函数诊断 ===\n");
    printf("cost(true Z4,Z8,Z9)   = %.6e\n", cost_3(true_z8[2], true_z8[6], true_z8[7]));
    printf("cost(0,0,0)           = %.6e\n", cost_3(0.0f, 0.0f, 0.0f));

    // ============================================================
    // 多起点粗搜
    // ============================================================
    printf("\n=== 多起点粗搜 ===\n");

    struct Solution { float z4, z8, z9, cost; };
    std::vector<Solution> candidates;

    const int N_STARTS = 20;
    srand(12345);

    for (int s = 0; s < N_STARTS; ++s) {
        float z4 = 0.0f + (rand() / (float)RAND_MAX) * 2.5f;
        float z8 = 0.0f + (rand() / (float)RAND_MAX) * 1.5f;
        float z9 = 0.0f + (rand() / (float)RAND_MAX) * 1.5f;

        float best = cost_3(z4, z8, z9);
        float step = 0.3f;

        for (int iter = 0; iter < 200; ++iter) {
            bool improved = false;
            for (int k = 0; k < 3; ++k) {
                for (int dir = -1; dir <= 1; dir += 2) {
                    float t4 = z4, t8 = z8, t9 = z9;
                    if (k == 0) t4 += dir * step;
                    if (k == 1) t8 += dir * step;
                    if (k == 2) t9 += dir * step;

                    if (t4 < 0.0f) t4 = 0.0f;
                    if (t8 < 0.0f) t8 = 0.0f;
                    if (t9 < 0.0f) t9 = 0.0f;

                    float c = cost_3(t4, t8, t9);
                    if (c < best) {
                        best = c;
                        z4 = t4; z8 = t8; z9 = t9;
                        improved = true;
                    }
                }
            }
            if (!improved) {
                step *= 0.5f;
                if (step < 1e-4f) break;
            }
        }

        candidates.push_back({z4, z8, z9, best});
        printf("start %2d: Z4=%.4f Z8=%.4f Z9=%.4f | cost=%.4e\n",
               s, z4, z8, z9, best);
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Solution& a, const Solution& b) {
                  return a.cost < b.cost;
              });

    // ============================================================
    // BOBYQA 精修（前 3 名）
    // ============================================================
    printf("\n=== BOBYQA 精修（前 3 名）===\n");

    std::function<float(float,float,float)> cost_fn_wrapper = cost_3;
    g_cost_fn = &cost_fn_wrapper;

    Solution best_global = candidates[0];

    for (int rank = 0; rank < 3 && rank < (int)candidates.size(); ++rank) {
        nlopt::opt opt(nlopt::LN_BOBYQA, 3);

        std::vector<double> lb = {0.0, 0.0, 0.0};
        std::vector<double> ub = {3.0, 1.5, 1.5};
        opt.set_lower_bounds(lb);
        opt.set_upper_bounds(ub);
        opt.set_min_objective(nlopt_obj, nullptr);

        opt.set_maxeval(2000);
        opt.set_ftol_rel(1e-10);
        opt.set_xtol_rel(1e-6);

        std::vector<double> x = {candidates[rank].z4,
                                 candidates[rank].z8,
                                 candidates[rank].z9};
        double minf;
        try {
            nlopt::result res = opt.optimize(x, minf);
            printf("rank %d: Z4=%.4f Z8=%.4f Z9=%.4f | cost=%.4e (status=%d)\n",
                   rank, (float)x[0], (float)x[1], (float)x[2], minf, (int)res);

            if (minf < best_global.cost) {
                best_global = {(float)x[0], (float)x[1], (float)x[2], (float)minf};
            }
        } catch (std::exception& e) {
            printf("rank %d: nlopt 异常: %s\n", rank, e.what());
        }
    }

    // ============================================================
    // 输出
    // ============================================================
    printf("\n=== 反演结果 ===\n");
    printf("        true       recovered     err\n");
    printf("Z4   %+.4f    %+.4f    %.4f\n",
           true_z8[2], best_global.z4, fabsf(best_global.z4 - true_z8[2]));
    printf("Z8   %+.4f    %+.4f    %.4f\n",
           true_z8[6], best_global.z8, fabsf(best_global.z8 - true_z8[6]));
    printf("Z9   %+.4f    %+.4f    %.4f\n",
           true_z8[7], best_global.z9, fabsf(best_global.z9 - true_z8[7]));

    float rms = sqrtf(
        (best_global.z4 - true_z8[2]) * (best_global.z4 - true_z8[2]) +
        (best_global.z8 - true_z8[6]) * (best_global.z8 - true_z8[6]) +
        (best_global.z9 - true_z8[7]) * (best_global.z9 - true_z8[7])
    ) / sqrtf(3.0f);

    printf("\nRMS error (Z4/Z8/Z9): %.4f λ\n", rms);
    printf("final cost = %.6e\n", best_global.cost);
    // ============================================================
    // 保存反演结果到文本文件，供 Python 可视化使用
    // ============================================================
    FILE* fp = fopen("inverse_result.txt", "w");
    if (fp) {
        fprintf(fp, "# theta = %.4f\n", theta);
        fprintf(fp, "# true_z4 true_z8 true_z9 recovered_z4 recovered_z8 recovered_z9\n");
        fprintf(fp, "%.6f %.6f %.6f %.6f %.6f %.6f\n",
                true_z8[2], true_z8[6], true_z8[7],
                best_global.z4, best_global.z8, best_global.z9);
        fclose(fp);
        printf("已保存: inverse_result.txt\n");
    }
    return 0;
}