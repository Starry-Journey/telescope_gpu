#include "forward_model.cuh"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// ============================================================
// Kernel 1: Zernike 基函数
// ============================================================
__global__ void zernike_kernel(
    float* zernikes,
    const float* rho,
    const float* theta,
    const unsigned char* mask,
    int fft_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = fft_size * fft_size;
    if (idx >= total) return;

    if (!mask[idx]) {
        for (int k = 0; k < 8; ++k) zernikes[k * total + idx] = 0.0f;
        return;
    }

    float r = rho[idx];
    float t = theta[idx];
    float r2 = r * r;
    float r3 = r2 * r;

    zernikes[0 * total + idx] = 2.0f * r * cosf(t);
    zernikes[1 * total + idx] = 2.0f * r * sinf(t);
    zernikes[2 * total + idx] = sqrtf(3.0f) * (2.0f * r2 - 1.0f);
    zernikes[3 * total + idx] = sqrtf(6.0f) * r2 * sinf(2.0f * t);
    zernikes[4 * total + idx] = sqrtf(6.0f) * r2 * cosf(2.0f * t);
    zernikes[5 * total + idx] = sqrtf(8.0f) * r3 * sinf(3.0f * t);
    zernikes[6 * total + idx] = sqrtf(8.0f) * (3.0f * r3 - 2.0f * r) * cosf(t);
    zernikes[7 * total + idx] = sqrtf(8.0f) * (3.0f * r3 - 2.0f * r) * sinf(t);
}

// ============================================================
// Kernel 2: 光瞳函数 P = A * exp(j * 2π * Φ)
// ============================================================
__global__ void pupil_kernel(
    cufftComplex* pupil,
    const float* zernikes,
    const float* coeffs,
    const unsigned char* mask,
    float defocus,
    int fft_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = fft_size * fft_size;
    if (idx >= total) return;

    if (!mask[idx]) {
        pupil[idx] = make_cuFloatComplex(0.0f, 0.0f);
        return;
    }

    float phase = 0.0f;
    for (int k = 0; k < 8; ++k) {
        float c = coeffs[k];
        if (k == 2) c += defocus;
        phase += c * zernikes[k * total + idx];
    }

    float angle = 2.0f * M_PI * phase;
    pupil[idx] = make_cuFloatComplex(cosf(angle), sinf(angle));
}

// ============================================================
// Kernel 3: PSF = |FFT|^2（含 fftshift）
// ============================================================
__global__ void psf_kernel(
    float* psf,
    const cufftComplex* fft_out,
    int fft_size,
    float norm_factor)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = fft_size * fft_size;
    if (idx >= total) return;

    int y = idx / fft_size;
    int x = idx % fft_size;

    int half = fft_size / 2;
    int sy = (y + half) % fft_size;
    int sx = (x + half) % fft_size;
    int src = sy * fft_size + sx;

    float re = fft_out[src].x;
    float im = fft_out[src].y;
    psf[idx] = (re * re + im * im) * norm_factor;
}

// ============================================================
// Kernel 4: 降采样
// ============================================================
__global__ void downsample_kernel(
    float* psf_down,
    const float* psf_full,
    int img_size,
    int oversample)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = img_size * img_size;
    if (idx >= total) return;

    int oy = idx / img_size;
    int ox = idx % img_size;
    int fft_size = img_size * oversample;

    float sum = 0.0f;
    for (int dy = 0; dy < oversample; ++dy)
        for (int dx = 0; dx < oversample; ++dx)
            sum += psf_full[(oy * oversample + dy) * fft_size
                          + (ox * oversample + dx)];
    psf_down[idx] = sum;
}

// ============================================================
// Kernel 5: PSF 归一化
// ============================================================
__global__ void normalize_kernel(float* psf, int total, float inv_sum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) psf[idx] *= inv_sum;
}

// ============================================================
// Kernel 6: 星点叠加
// ============================================================
__global__ void star_overlay_kernel(
    float* image,
    const float* psf,
    const float* star_x,
    const float* star_y,
    const float* star_flux,
    int n_stars,
    int img_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = img_size * img_size;
    if (idx >= total) return;

    int py = idx / img_size;
    int px = idx % img_size;
    int half = img_size / 2;

    float sum = 0.0f;
    for (int s = 0; s < n_stars; ++s) {
        int ix = (int)roundf(star_x[s]);
        int iy = (int)roundf(star_y[s]);
        int psf_x = px - ix + half;
        int psf_y = py - iy + half;
        if (psf_x >= 0 && psf_x < img_size && psf_y >= 0 && psf_y < img_size)
            sum += star_flux[s] * psf[psf_y * img_size + psf_x];
    }
    image[idx] = sum;
}

// ============================================================
// Kernel 7: 批量星点叠加（多帧）
// ============================================================
__global__ void star_overlay_frames_kernel(
    float* d_frames,
    const float* d_psf_frames,
    const float* star_x, const float* star_y, const float* star_flux,
    int n_stars, int n_frames, int img_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int frame = blockIdx.y;
    int total = img_size * img_size;
    if (idx >= total || frame >= n_frames) return;

    int py = idx / img_size;
    int px = idx % img_size;
    int half = img_size / 2;

    const float* psf = d_psf_frames + frame * total;
    float sum = 0.0f;
    for (int s = 0; s < n_stars; ++s) {
        int ix = (int)roundf(star_x[s]);
        int iy = (int)roundf(star_y[s]);
        int psf_x = px - ix + half;
        int psf_y = py - iy + half;
        if (psf_x >= 0 && psf_x < img_size && psf_y >= 0 && psf_y < img_size)
            sum += star_flux[s] * psf[psf_y * img_size + psf_x];
    }
    d_frames[frame * total + idx] = sum;
}

// ============================================================
// ForwardModelGPU 实现
// ============================================================
ForwardModelGPU::ForwardModelGPU(const OpticalParams& params)
    : params_(params)
{
    int fft_size = params_.fft_size;
    int total = fft_size * fft_size;

    CUDA_CHECK(cudaMalloc(&d_zernikes_, 8 * total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rho_, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_theta_, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mask_, total * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_pupil_, total * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_fft_, total * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_psf_full_, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_psf_down_,
                          params_.img_size * params_.img_size * sizeof(float)));

    CUFFT_CHECK(cufftPlan2d(&fft_plan_, fft_size, fft_size, CUFFT_C2C));

    init_pupil_coords();
    init_zernikes();
}

ForwardModelGPU::~ForwardModelGPU() {
    cudaFree(d_zernikes_);
    cudaFree(d_rho_);
    cudaFree(d_theta_);
    cudaFree(d_mask_);
    cudaFree(d_pupil_);
    cudaFree(d_fft_);
    cudaFree(d_psf_full_);
    cudaFree(d_psf_down_);
    cufftDestroy(fft_plan_);
}

void ForwardModelGPU::init_pupil_coords() {
    int fft_size = params_.fft_size;
    int total = fft_size * fft_size;

    std::vector<float> h_rho(total), h_theta(total);
    std::vector<unsigned char> h_mask(total);

    for (int y = 0; y < fft_size; ++y) {
        for (int x = 0; x < fft_size; ++x) {
            int idx = y * fft_size + x;
            float nx = (2.0f * x / (fft_size - 1)) - 1.0f;
            float ny = (2.0f * y / (fft_size - 1)) - 1.0f;
            float rho = sqrtf(nx * nx + ny * ny);
            float theta = atan2f(ny, nx);

            h_rho[idx] = rho / params_.pupil_radius;
            h_theta[idx] = theta;
            h_mask[idx] = (rho <= params_.pupil_radius) ? 1 : 0;
        }
    }

    CUDA_CHECK(cudaMemcpy(d_rho_, h_rho.data(), total * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_theta_, h_theta.data(), total * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mask_, h_mask.data(), total * sizeof(unsigned char),
                          cudaMemcpyHostToDevice));
}

void ForwardModelGPU::init_zernikes() {
    int fft_size = params_.fft_size;
    int total = fft_size * fft_size;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    zernike_kernel<<<blocks, threads>>>(
        d_zernikes_, d_rho_, d_theta_, d_mask_, fft_size);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void ForwardModelGPU::compute_psf(const float* d_coeffs,
                                   float defocus,
                                   float* d_psf_out)
{
    int fft_size = params_.fft_size;
    int total = fft_size * fft_size;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    pupil_kernel<<<blocks, threads>>>(
        d_pupil_, d_zernikes_, d_coeffs, d_mask_, defocus, fft_size);

    CUFFT_CHECK(cufftExecC2C(fft_plan_, d_pupil_, d_fft_, CUFFT_FORWARD));

    float norm_factor = 1.0f / (fft_size * fft_size);
    psf_kernel<<<blocks, threads>>>(d_psf_full_, d_fft_, fft_size, norm_factor);

    int img_total = params_.img_size * params_.img_size;
    int img_blocks = (img_total + threads - 1) / threads;
    downsample_kernel<<<img_blocks, threads>>>(
        d_psf_down_, d_psf_full_, params_.img_size, params_.oversample);

    // 归一化
    std::vector<float> h_tmp(img_total);
    CUDA_CHECK(cudaMemcpy(h_tmp.data(), d_psf_down_,
                          img_total * sizeof(float), cudaMemcpyDeviceToHost));
    float sum = 0.0f;
    for (float v : h_tmp) sum += v;
    float inv_sum = (sum > 1e-20f) ? (1.0f / sum) : 1.0f;
    normalize_kernel<<<img_blocks, threads>>>(d_psf_down_, img_total, inv_sum);

    CUDA_CHECK(cudaMemcpy(d_psf_out, d_psf_down_,
                          img_total * sizeof(float), cudaMemcpyDeviceToDevice));
}

void ForwardModelGPU::generate_image(
    const float* d_coeffs,
    const float* d_star_x,
    const float* d_star_y,
    const float* d_star_flux,
    int n_stars,
    float defocus,
    float* d_image_out)
{
    int img_total = params_.img_size * params_.img_size;
    int threads = 256;
    int blocks = (img_total + threads - 1) / threads;

    float* d_psf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_psf, img_total * sizeof(float)));
    compute_psf(d_coeffs, defocus, d_psf);

    star_overlay_kernel<<<blocks, threads>>>(
        d_image_out, d_psf, d_star_x, d_star_y, d_star_flux,
        n_stars, params_.img_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_psf);
}

void ForwardModelGPU::generate_frames_device(
    const float* d_coeffs,
    const float* d_star_x, const float* d_star_y, const float* d_star_flux,
    int n_stars,
    const float* d_defocus, int n_frames,
    float* d_frames_out)
{
    int img_total = params_.img_size * params_.img_size;
    int threads = 256;
    int blocks = (img_total + threads - 1) / threads;

    float* d_psf_frames = nullptr;
    CUDA_CHECK(cudaMalloc(&d_psf_frames, n_frames * img_total * sizeof(float)));

    std::vector<float> h_defocus(n_frames);
    CUDA_CHECK(cudaMemcpy(h_defocus.data(), d_defocus,
                          n_frames * sizeof(float), cudaMemcpyDeviceToHost));

    for (int f = 0; f < n_frames; ++f) {
        float* d_psf = d_psf_frames + f * img_total;
        compute_psf(d_coeffs, h_defocus[f], d_psf);
    }

    dim3 grid(blocks, n_frames);
    star_overlay_frames_kernel<<<grid, threads>>>(
        d_frames_out, d_psf_frames,
        d_star_x, d_star_y, d_star_flux,
        n_stars, n_frames, params_.img_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_psf_frames);
}

// ============================================================
// Host 版：生成含噪声星场图
// ============================================================
void ForwardModelGPU::generate_image_host(
    const float* h_coeffs,
    const float* h_star_x,
    const float* h_star_y,
    const float* h_star_flux,
    int n_stars,
    float defocus,
    std::vector<float>& h_image_out)
{
    int img_size = params_.img_size;
    int img_total = img_size * img_size;

    float *d_coeffs, *d_star_x, *d_star_y, *d_star_flux, *d_image;
    CUDA_CHECK(cudaMalloc(&d_coeffs, 8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_star_x, n_stars * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_star_y, n_stars * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_star_flux, n_stars * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_image, img_total * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_coeffs, h_coeffs, 8 * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_star_x, h_star_x, n_stars * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_star_y, h_star_y, n_stars * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_star_flux, h_star_flux, n_stars * sizeof(float),
                          cudaMemcpyHostToDevice));

    generate_image(d_coeffs, d_star_x, d_star_y, d_star_flux,
                   n_stars, defocus, d_image);

    h_image_out.resize(img_total);
    CUDA_CHECK(cudaMemcpy(h_image_out.data(), d_image,
                          img_total * sizeof(float), cudaMemcpyDeviceToHost));

    // 加背景 + 泊松噪声 + 读出噪声
    const float BG_LEVEL = 200.0f;
    const float READ_NOISE = 5.0f;

    for (int i = 0; i < img_total; ++i) {
        float v = h_image_out[i] + BG_LEVEL;

        int iv = (int)v;
        float frac = v - iv;
        float r = (rand() / (float)RAND_MAX);
        v = (float)(iv + (r < frac ? 1 : 0));

        float u1 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
        float u2 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
        float gauss = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
        v += gauss * READ_NOISE;

        h_image_out[i] = fmaxf(v, 0.0f);
    }

    cudaFree(d_coeffs);
    cudaFree(d_star_x);
    cudaFree(d_star_y);
    cudaFree(d_star_flux);
    cudaFree(d_image);
}