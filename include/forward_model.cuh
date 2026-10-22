#ifndef FORWARD_MODEL_CUH
#define FORWARD_MODEL_CUH

#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <functional>

#define CUDA_CHECK(err) do { \
    cudaError_t _e = (err); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e)); exit(1); \
    } \
} while(0)

#define CUFFT_CHECK(err) do { \
    cufftResult _e = (err); \
    if (_e != CUFFT_SUCCESS) { \
        fprintf(stderr, "cuFFT error %s:%d: %d\n", __FILE__, __LINE__, _e); \
        exit(1); \
    } \
} while(0)

struct OpticalParams {
    int img_size;
    int fft_size;
    int oversample;
    float pupil_radius;
};

// ============================================================
// Zernike 系数表（与 Python 版一致）
// ============================================================
struct ZernikeTable {
    static constexpr float THETA_REF = 90.0f;
    static void get(float theta_deg, float* coeffs) {
        const float PI = 3.14159265358979323846f;
        float theta_rad = theta_deg * PI / 180.0f;
        float delta_rad = fabsf(theta_rad - THETA_REF * PI / 180.0f);
        coeffs[0] = 0.0f;
        coeffs[1] = 0.0f;
        coeffs[2] = 1.8f * sinf(delta_rad) + 0.2f;
        // 像散 Z5/Z6、三叶 Z7 置零。
        // 原因：inverse_solve 的正向模型是 h_coeffs = {0,0,Z4,0,0,0,Z8,Z9}，
        // 只拟合 Z4/Z8/Z9。若这里仍生成 Z5/Z6/Z7，真值就不再是代价函数的
        // 极小值（实测 cost(true)=8.67e-4 > 优化器搜到的 6.34e-4），反演必然偏。
        // 置零后实测 RMS 0.0155λ。将来若把反演扩到 6 项，把下面三行恢复即可。
        // coeffs[3] = 0.20f * sinf(2.0f * delta_rad) + 0.05f * cosf(theta_rad);
        // coeffs[4] = 0.18f * sinf(2.0f * delta_rad + PI / 4.0f) + 0.05f * sinf(theta_rad);
        // coeffs[5] = 0.08f * sinf(3.0f * delta_rad) + 0.02f;
        coeffs[3] = 0.0f;
        coeffs[4] = 0.0f;
        coeffs[5] = 0.0f;
        // 彗差 Z8/Z9。原先取 0.85 / 0.70（约 0.70λ / 0.78λ @θ=45°），
        // 比同级像散(0.20/0.18)、三叶(0.08)大出一个量级，会把对称离焦环
        // 拉成"扫把/彗尾"状。这里压到与像散同量级（约 1/4），保留彗差
        // 作为 phase-diversity 的可反演信号，但不再主导 PSF 形状。
        coeffs[6] = 0.20f * sinf(delta_rad) + 0.03f;
        coeffs[7] = 0.16f * sinf(delta_rad + PI / 4.0f) + 0.02f;
    }
};

// ============================================================
// GPU 正向模型
// ============================================================
class ForwardModelGPU {
public:
    explicit ForwardModelGPU(const OpticalParams& params);
    ~ForwardModelGPU();

    void compute_psf(const float* d_coeffs, float defocus, float* d_psf_out);

    void generate_image(const float* d_coeffs,
                        const float* d_star_x,
                        const float* d_star_y,
                        const float* d_star_flux,
                        int n_stars, float defocus,
                        float* d_image_out);

    void generate_image_host(const float* h_coeffs,
                             const float* h_star_x,
                             const float* h_star_y,
                             const float* h_star_flux,
                             int n_stars, float defocus,
                             std::vector<float>& h_image_out);

    void generate_frames_device(const float* d_coeffs,
                                const float* d_star_x,
                                const float* d_star_y,
                                const float* d_star_flux,
                                int n_stars,
                                const float* d_defocus, int n_frames,
                                float* d_frames_out);

    OpticalParams params() const { return params_; }

private:
    OpticalParams params_;
    float* d_zernikes_;
    float* d_rho_;
    float* d_theta_;
    unsigned char* d_mask_;
    cufftComplex* d_pupil_;
    cufftComplex* d_fft_;
    float* d_psf_full_;
    float* d_psf_down_;
    cufftHandle fft_plan_;

    void init_pupil_coords();
    void init_zernikes();
};

#endif